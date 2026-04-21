import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from functools import partial

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock
from scipy.signal import find_peaks

try:
    import pandas_ta as ta
except ImportError:  # Python 3.14 환경에서는 pandas-ta 설치가 실패할 수 있음
    ta = None

logger = logging.getLogger("calculatechart")
logging.basicConfig(level=logging.INFO)

# ── 환경변수 설정 ────────────────────────────────
# 추천 종목 샘플링 수 (기본 220, 환경변수로 조정 가능)
RECOMMEND_SAMPLE_SIZE = int(os.environ.get("RECOMMEND_SAMPLE_SIZE", "220"))
# pykrx 호출 타임아웃 (초)
PYKRX_TIMEOUT_SEC = int(os.environ.get("PYKRX_TIMEOUT_SEC", "30"))

# pykrx는 동기 라이브러리이므로 ThreadPoolExecutor를 통해 타임아웃 제어
_executor = ThreadPoolExecutor(max_workers=4)

# 추천 결과를 메모리에 캐싱 + asyncio.Lock으로 동시 접근 보호
RECOMMEND_CACHE: dict = {
    "data": None,
    "last_updated": None,
    "as_of": None,
}
_cache_lock = asyncio.Lock()

# 종목 리스트 캐시 (1시간)
_LISTING_CACHE: dict = {
    "data": None,
    "last_updated": None,
}
_LISTING_CACHE_TTL = 3600  # 초


async def _run_with_timeout(fn, *args, timeout: int = PYKRX_TIMEOUT_SEC):
    """동기 함수(pykrx)를 ThreadPool에서 실행하고 타임아웃을 적용"""
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(_executor, partial(fn, *args)),
        timeout=timeout,
    )


app = FastAPI(title="주식 분석 시스템 API")

# CORS: allow_credentials=True 와 allow_origins=["*"] 는 브라우저 스펙 위반이므로
# 자격증명이 필요하지 않은 API는 credentials=False 로 두고 "*" 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _preload_listing():
    """서버 시작 시 종목 리스트를 미리 캐싱 — 자동완성 첫 응답 속도 개선"""
    try:
        await _run_with_timeout(_load_listing)
        logger.info("종목 리스트 사전 로드 완료")
    except Exception as e:
        logger.warning("종목 리스트 사전 로드 실패 (요청 시 재시도): %s", e)


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _best_business_day_sync(max_back_days: int = 15) -> str:
    """동기 버전 (ThreadPool에서 호출)"""
    today = date.today()
    for i in range(max_back_days):
        d = today - timedelta(days=i)
        day = _yyyymmdd(d)
        tickers = stock.get_market_ticker_list(day, market="ALL")
        if tickers:
            return day
    return _yyyymmdd(today)


async def _best_business_day(max_back_days: int = 15) -> str:
    return await _run_with_timeout(_best_business_day_sync, max_back_days)


def _load_listing() -> pd.DataFrame:
    """종목 리스트를 가져온다. 1시간 메모리 캐시 적용. fdr 실패 시 pykrx 폴백."""
    now = datetime.now()
    cached = _LISTING_CACHE["data"]
    last = _LISTING_CACHE["last_updated"]
    if cached is not None and last is not None and (now - last).total_seconds() < _LISTING_CACHE_TTL:
        return cached

    result = None

    # 1차: FinanceDataReader
    try:
        listing = fdr.StockListing("KRX")
        if listing is not None and not listing.empty:
            symbol_col = "Code" if "Code" in listing.columns else "Symbol"
            if symbol_col in listing.columns and "Name" in listing.columns:
                out = listing[[symbol_col, "Name"]].copy()
                out.columns = ["ticker", "name"]
                out["ticker"] = out["ticker"].astype(str).str.zfill(6)
                out["name"] = out["name"].astype(str)
                result = out.dropna().drop_duplicates(subset=["ticker"])
                logger.info("종목 리스트 로드 완료 (fdr): %d종목", len(result))
    except Exception as e:
        logger.warning("fdr.StockListing 실패: %s", e)

    # 2차: pykrx 폴백
    if result is None or result.empty:
        try:
            day = _best_business_day_sync()
            rows = []
            for market in ["KOSPI", "KOSDAQ"]:
                tickers = stock.get_market_ticker_list(day, market=market)
                for tk in tickers:
                    name = stock.get_market_ticker_name(tk)
                    rows.append({"ticker": str(tk).zfill(6), "name": name})
            if rows:
                result = pd.DataFrame(rows).drop_duplicates(subset=["ticker"])
                logger.info("종목 리스트 로드 완료 (pykrx 폴백): %d종목", len(result))
        except Exception as e:
            logger.error("pykrx 폴백도 실패: %s", e)

    if result is None or result.empty:
        raise ValueError("종목 리스트를 가져오지 못했습니다 (fdr + pykrx 모두 실패).")

    _LISTING_CACHE["data"] = result
    _LISTING_CACHE["last_updated"] = now
    return result


def _normalize_input_date(s: str | None, fallback: date) -> str:
    if not s:
        return _yyyymmdd(fallback)
    v = s.strip()
    if len(v) == 10 and "-" in v:
        return v.replace("-", "")
    return v


def _resolve_ticker(ticker_or_name: str) -> tuple[str, str]:
    """
    종목 검색: 코드·정확한 이름·부분 이름·초성·대소문자 무시 모두 지원.

    검색 순서 (우선순위):
    1. 6자리 숫자 → 종목코드 직접 매칭
    2. 정확한 이름 일치 (대소문자 무시)
    3. 부분 문자열 매칭 (대소문자 무시, regex 특수문자 이스케이프)
    4. 공백/특수문자 제거 후 매칭
    5. 모든 방법 실패 시 에러

    예시:
    - "sk하이닉스" → SK하이닉스 (대소문자 무시)
    - "삼성" → 삼성전자 (부분 매칭)
    - "카카오" → 카카오 (정확 매칭, 카카오뱅크보다 우선)
    - "005930" → 삼성전자 (코드 매칭)
    """
    import re
    q = ticker_or_name.strip()
    if not q:
        raise ValueError("검색어가 비어있습니다.")

    # ── 1. 6자리 숫자 → 종목코드 ──
    if q.isdigit() and len(q) == 6:
        listing = _load_listing()
        row = listing[listing["ticker"] == q]
        if not row.empty:
            return q, str(row.iloc[0]["name"])
        return q, q

    listing = _load_listing()
    names = listing["name"]

    # ── 2. 정확한 이름 일치 (대소문자 무시) ──
    q_upper = q.upper()
    exact_mask = names.str.upper() == q_upper
    if exact_mask.any():
        row = listing[exact_mask].iloc[0]
        return str(row["ticker"]), str(row["name"])

    # ── 3. 부분 문자열 매칭 (대소문자 무시) ──
    q_escaped = re.escape(q)  # regex 특수문자 이스케이프 (괄호, + 등)
    partial_mask = names.str.contains(q_escaped, na=False, case=False)
    if partial_mask.any():
        matches = listing[partial_mask]
        # 이름이 짧은 것 우선 (카카오 > 카카오뱅크, 삼성전자 > 삼성전자우)
        best = matches.loc[matches["name"].str.len().idxmin()]
        return str(best["ticker"]), str(best["name"])

    # ── 4. 공백/특수문자 제거 후 매칭 ──
    q_clean = re.sub(r'[\s\-\_\.\&]', '', q).upper()
    names_clean = names.str.replace(r'[\s\-\_\.\&]', '', regex=True).str.upper()
    clean_mask = names_clean.str.contains(re.escape(q_clean), na=False)
    if clean_mask.any():
        matches = listing[clean_mask]
        best = matches.loc[matches["name"].str.len().idxmin()]
        return str(best["ticker"]), str(best["name"])

    raise ValueError(f"종목명/코드를 찾을 수 없습니다: {q}")


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    d = df.reset_index().copy()
    d.iloc[:, 0] = pd.to_datetime(d.iloc[:, 0]).dt.strftime("%Y-%m-%d")
    if len(d.columns) == 7:
        d.columns = ["time", "open", "high", "low", "close", "volume", "fluctuation"]
    elif len(d.columns) == 8:
        d.columns = ["time", "open", "high", "low", "close", "volume", "value", "fluctuation"]
    else:
        raise ValueError(f"Unexpected columns returned from pykrx: {len(d.columns)}")
    return d


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """NaN/0값 행 제거 + 숫자 변환 — 거래정지일 등에서 발생하는 차트 끊김 방지."""
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df


def _support_resistance(
    close_values: np.ndarray,
    high_values: np.ndarray | None = None,
    low_values: np.ndarray | None = None,
    volume_values: np.ndarray | None = None,
    max_lines: int = 1,
) -> tuple[list[float], list[float]]:
    """
    지지/저항선 탐지 — 현재 주가에 가장 유의미한 레벨 1개씩만 반환.

    전문가 기준 종합 점수:
      1. 터치 횟수: 같은 가격대를 여러 번 터치할수록 강한 레벨 (40%)
      2. 거래량 집중도: 해당 가격대에서 거래량이 많을수록 강함 (30%)
      3. 최근성: 최근 형성된 레벨이 오래된 레벨보다 유의미 (20%)
      4. 현재가 근접도: 현재가에 가까울수록 즉시 영향력 큼 (10%)
    """
    lookback = len(close_values)
    if lookback < 10:
        return [], []

    last_price = float(close_values[-1])

    # high/low 없으면 close 기반으로 대체
    highs = high_values if high_values is not None else close_values
    lows = low_values if low_values is not None else close_values
    vols = volume_values if volume_values is not None else np.ones(lookback)

    # ── 1단계: 피봇 포인트 탐지 (다중 타임프레임) ──
    # 단기(5봉) + 중기(10봉) + 장기(20봉) 피봇을 모두 수집
    pivot_highs: list[tuple[int, float]] = []  # (index, price)
    pivot_lows: list[tuple[int, float]] = []

    for window in [5, 10, 20]:
        if lookback < window * 2 + 1:
            continue
        half = window
        for i in range(half, lookback - half):
            # 피봇 고점: 양쪽 window 내 최고
            if highs[i] == np.max(highs[max(0, i - half):i + half + 1]):
                pivot_highs.append((i, float(highs[i])))
            # 피봇 저점: 양쪽 window 내 최저
            if lows[i] == np.min(lows[max(0, i - half):i + half + 1]):
                pivot_lows.append((i, float(lows[i])))

    if not pivot_highs and not pivot_lows:
        # 피봇 없으면 단순 최고/최저
        return [float(np.min(lows))], [float(np.max(highs))]

    # ── 2단계: 가격대 클러스터링 (ATR 기반 허용 오차) ──
    price_range = float(np.max(highs) - np.min(lows))
    tolerance = max(last_price * 0.015, price_range * 0.02)  # 1.5% or 레인지 2% 중 큰 값

    def _cluster_pivots(pivots: list[tuple[int, float]]) -> list[dict]:
        """피봇들을 가격대별로 클러스터링. 각 클러스터의 종합 점수를 계산."""
        if not pivots:
            return []
        # 가격순 정렬
        sorted_pivots = sorted(pivots, key=lambda x: x[1])
        clusters: list[list[tuple[int, float]]] = []
        current_cluster = [sorted_pivots[0]]

        for idx_price in sorted_pivots[1:]:
            if abs(idx_price[1] - current_cluster[-1][1]) <= tolerance:
                current_cluster.append(idx_price)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx_price]
        clusters.append(current_cluster)

        scored: list[dict] = []
        for cluster in clusters:
            # 대표 가격: 거래량 가중 평균
            indices = [c[0] for c in cluster]
            prices = [c[1] for c in cluster]
            cluster_vols = [float(vols[i]) if i < len(vols) else 1.0 for i in indices]
            total_vol = sum(cluster_vols)
            if total_vol > 0:
                rep_price = sum(p * v for p, v in zip(prices, cluster_vols)) / total_vol
            else:
                rep_price = float(np.mean(prices))

            # (a) 터치 횟수 점수 (40%): 2회 이상이면 강한 레벨
            touch_count = len(cluster)
            touch_score = min(1.0, touch_count / 4.0)  # 4회면 만점

            # (b) 거래량 집중도 점수 (30%): 해당 가격대 평균 거래량 / 전체 평균 거래량
            avg_vol_at_level = np.mean(cluster_vols) if cluster_vols else 0
            global_avg_vol = float(np.mean(vols)) if len(vols) > 0 else 1.0
            vol_score = min(1.0, avg_vol_at_level / max(1, global_avg_vol))

            # (c) 최근성 점수 (20%): 가장 최근 터치가 최근일수록 높음
            most_recent_idx = max(indices)
            recency_score = most_recent_idx / max(1, lookback - 1)

            # (d) 현재가 근접도 점수 (10%): 가까울수록 높음
            dist_pct = abs(rep_price - last_price) / last_price
            proximity_score = max(0, 1.0 - dist_pct / 0.15)  # 15% 이상이면 0점

            total_score = (
                touch_score * 0.40
                + vol_score * 0.30
                + recency_score * 0.20
                + proximity_score * 0.10
            )

            scored.append({
                "price": rep_price,
                "score": total_score,
                "touches": touch_count,
                "most_recent": most_recent_idx,
            })

        return scored

    resistance_clusters = _cluster_pivots(pivot_highs)
    support_clusters = _cluster_pivots(pivot_lows)

    # ── 3단계: 현재가 기준 유효 레벨만 필터링 + 최고 점수 1개 선택 ──
    # 저항선: 현재가보다 위 (1% ~ 20% 범위)
    valid_resistance = [
        c for c in resistance_clusters
        if c["price"] > last_price * 1.005 and c["price"] < last_price * 1.20
    ]
    # 지지선: 현재가보다 아래 (1% ~ 20% 범위)
    valid_support = [
        c for c in support_clusters
        if c["price"] < last_price * 0.995 and c["price"] > last_price * 0.80
    ]

    # 점수 높은 순으로 정렬 후 max_lines개 선택
    valid_resistance.sort(key=lambda x: x["score"], reverse=True)
    valid_support.sort(key=lambda x: x["score"], reverse=True)

    resistance_out = [round(c["price"]) for c in valid_resistance[:max_lines]]
    support_out = [round(c["price"]) for c in valid_support[:max_lines]]

    # 하나도 없으면 fallback: 최근 20일 최고/최저
    if not resistance_out and lookback >= 20:
        recent_high = float(np.max(highs[-20:]))
        if recent_high > last_price * 1.005:
            resistance_out = [round(recent_high)]
    if not support_out and lookback >= 20:
        recent_low = float(np.min(lows[-20:]))
        if recent_low < last_price * 0.995:
            support_out = [round(recent_low)]

    return support_out, resistance_out


def _compute_fibonacci_levels(df: pd.DataFrame, lookback_days: int = 120) -> dict:
    """
    최근 lookback_days 중 유의미한 고점/저점 탐지 후 피보나치 되돌림 레벨 반환.
    상승 파동(저점→고점) 기준으로 되돌림 레벨 계산.
    반환값: { "swing_high": float, "swing_low": float, "levels": {ratio_str: price}, "direction": "up"|"down" }
    """
    try:
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        high  = pd.to_numeric(df["high"],  errors="coerce").dropna()
        low   = pd.to_numeric(df["low"],   errors="coerce").dropna()

        tail_len = min(lookback_days, len(close))
        close_t = close.tail(tail_len)
        high_t  = high.tail(tail_len)
        low_t   = low.tail(tail_len)

        swing_high = float(high_t.max())
        swing_low  = float(low_t.min())
        cur        = float(close.iloc[-1])

        if swing_high <= swing_low or swing_high == 0:
            return {}

        fib_range = swing_high - swing_low
        # 현재가 위치에 따라 방향 결정 (상단 40% → 하락 되돌림, 하단 60% → 상승 복귀)
        pos = (cur - swing_low) / fib_range
        direction = "down" if pos >= 0.6 else "up"

        ratios = [0.236, 0.382, 0.500, 0.618, 0.786]
        levels: dict[str, float] = {}
        for r in ratios:
            if direction == "down":
                # 고점에서 되돌림 (하락 되돌림)
                price = swing_high - fib_range * r
            else:
                # 저점에서 복귀 (상승 복귀)
                price = swing_low + fib_range * r
            levels[str(r)] = round(price, 0)

        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "levels": levels,
            "direction": direction,
        }
    except Exception:
        return {}


def _compute_ichimoku(df: pd.DataFrame) -> dict:
    """
    일목균형표 5선 계산.
    전환선(9), 기준선(26), 선행스팬1(미래26봉), 선행스팬2(미래26봉), 후행스팬(과거26봉).
    반환값: 최신값만 담은 dict (차트 오버레이용)
      tenkan, kijun, senkou_a, senkou_b, chikou,
      cloud_top, cloud_bottom, cloud_bullish (bool)
    """
    try:
        high  = pd.to_numeric(df["high"],  errors="coerce")
        low   = pd.to_numeric(df["low"],   errors="coerce")
        close = pd.to_numeric(df["close"], errors="coerce")

        def mid(h: pd.Series, l: pd.Series, n: int) -> pd.Series:
            return (h.rolling(n).max() + l.rolling(n).min()) / 2

        tenkan  = mid(high, low, 9)   # 전환선
        kijun   = mid(high, low, 26)  # 기준선
        senkou_a = ((tenkan + kijun) / 2)           # 선행스팬1 (실제로는 26봉 앞이지만 현재값으로 반환)
        senkou_b = mid(high, low, 52)               # 선행스팬2

        if tenkan.dropna().empty or kijun.dropna().empty:
            return {}

        t_val  = float(tenkan.iloc[-1])
        k_val  = float(kijun.iloc[-1])
        sa_val = float(senkou_a.iloc[-1])
        sb_val = float(senkou_b.iloc[-1])
        chikou_val = float(close.iloc[-1])  # 후행스팬 = 현재 종가 (차트에서 26봉 뒤에 표시)

        cloud_top    = max(sa_val, sb_val)
        cloud_bottom = min(sa_val, sb_val)
        cur          = float(close.iloc[-1])
        cloud_bullish = sa_val >= sb_val  # 양운(호재) vs 음운(악재)

        return {
            "tenkan":       round(t_val, 0),
            "kijun":        round(k_val, 0),
            "senkou_a":     round(sa_val, 0),
            "senkou_b":     round(sb_val, 0),
            "chikou":       round(chikou_val, 0),
            "cloud_top":    round(cloud_top, 0),
            "cloud_bottom": round(cloud_bottom, 0),
            "cloud_bullish": cloud_bullish,
            "price_above_cloud": cur > cloud_top,
            "price_below_cloud": cur < cloud_bottom,
            "tenkan_above_kijun": t_val >= k_val,
        }
    except Exception:
        return {}


def _unified_score(
    df: pd.DataFrame,
    support_lines: list[float] | None = None,
    resistance_lines: list[float] | None = None,
    box_range: dict | None = None,
) -> tuple[int, list[dict], dict]:
    """
    통합 기술적 점수 계산 — 카테고리별 캡으로 이중 계산 방지.

    카테고리 (기준점 50):
      추세   (cap ±15): 이평선 배열, Cup 패턴, 골든크로스
      모멘텀 (cap ±12): RSI, MACD, 이격도
      변동성 (cap ±10): BB 위치, 스퀴즈(수박지표)
      거래량 (cap ±15): 바닥 매집, OBV 다이버전스, 스퀴즈+거래량, 일반 거래량
      구조   (cap  ±8): 지지/저항선, 박스권, 계단지표
    반등 신호 2개+ 동시 → 시너지 보너스 +3~5
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")

    cur = float(close.iloc[-1])
    signals: list[dict] = []

    # ════════════════════════════════════════════════════
    # 지표 일괄 계산 (순서 의존 없이 미리 모두 준비)
    # ════════════════════════════════════════════════════

    sma5   = close.rolling(5).mean()
    sma20  = close.rolling(20).mean()
    sma60  = close.rolling(60).mean()
    sma120 = close.rolling(120).mean()
    sma224 = close.rolling(224).mean()

    # RSI
    if ta is not None:
        rsi_series = ta.rsi(close, length=14)
    else:
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))

    # MACD
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line

    # 볼린저 밴드
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # OBV (벡터화 — 기존 O(n) 루프 제거)
    price_diff = close.diff()
    obv_sign = pd.Series(0.0, index=close.index)
    obv_sign[price_diff > 0] = 1.0
    obv_sign[price_diff < 0] = -1.0
    obv = (vol * obv_sign).cumsum()

    # 스토캐스틱 %K / %D  (14-3-3 표준 파라미터)
    stoch_k: pd.Series | None = None
    stoch_d: pd.Series | None = None
    if len(close) >= 14:
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        denom  = (high14 - low14).replace(0, np.nan)
        raw_k  = (close - low14) / denom * 100
        stoch_k = raw_k.rolling(3).mean()   # smoothed %K
        stoch_d = stoch_k.rolling(3).mean() # %D signal

    # BB 스퀴즈 감지 (변동성/거래량 양쪽에서 사용하므로 미리 계산)
    is_squeeze = False
    if not bb_up.dropna().empty and not bb_mid.dropna().empty:
        bb_width_pct = (bb_up - bb_dn) / bb_mid * 100
        bb_width_min_20 = bb_width_pct.rolling(20).min()
        if not bb_width_pct.dropna().empty and not bb_width_min_20.dropna().empty:
            is_squeeze = float(bb_width_pct.iloc[-1]) <= float(bb_width_min_20.iloc[-1]) * 1.3

    # ════════════════════════════════════════════════════
    # 수렴→돌파→눌림목 패턴 감지 (사전 계산)
    # ════════════════════════════════════════════════════
    breakout_pullback = False  # 돌파 후 건전한 눌림목 상태
    breakout_fresh = False     # 돌파 직후 (아직 눌림목 전)

    if len(close) >= 20 and not bb_up.dropna().empty:
        bb_up_val = float(bb_up.iloc[-1])
        bb_mid_val_bp = float(bb_mid.iloc[-1])
        vol_ma20_bp = vol.rolling(20).mean()

        # 최근 10일 내 돌파 이벤트 탐색:
        # 종가가 BB 상단 또는 박스 상단을 거래량 동반으로 돌파한 날
        breakout_day = -1
        breakout_level = 0.0
        for lookback_i in range(min(10, len(close) - 1), 0, -1):
            idx = -lookback_i
            c_val = float(close.iloc[idx])
            v_val = float(vol.iloc[idx])
            v_avg = float(vol_ma20_bp.iloc[idx]) if not vol_ma20_bp.dropna().empty and idx + len(vol_ma20_bp) > 0 else 0

            # BB 상단 돌파 + 거래량 2배
            bb_up_at = float(bb_up.iloc[idx]) if idx + len(bb_up) > 0 else bb_up_val
            if c_val > bb_up_at and v_avg > 0 and v_val >= v_avg * 1.8:
                breakout_day = lookback_i
                breakout_level = bb_up_at
                break

            # 박스 상단 돌파 + 거래량 2배
            if box_range and box_range.get("is_box"):
                box_top = box_range.get("top", 0)
                if box_top > 0 and c_val > box_top and v_avg > 0 and v_val >= v_avg * 1.8:
                    breakout_day = lookback_i
                    breakout_level = box_top
                    break

        if breakout_day > 0:
            # 돌파 후 현재가가 돌파 레벨 위에 있고, 되돌림이 3% 이내
            pullback_pct = (cur - breakout_level) / breakout_level if breakout_level > 0 else 0
            # 되돌림 중 거래량 감소 확인
            post_breakout_vol = vol.tail(breakout_day).values
            vol_declining = True
            if len(post_breakout_vol) >= 2:
                vol_declining = float(np.mean(post_breakout_vol[-2:])) < float(np.mean(post_breakout_vol[:max(1, len(post_breakout_vol) // 2)])) * 1.3

            if pullback_pct >= -0.01 and breakout_day <= 3:
                breakout_fresh = True
            elif -0.03 <= pullback_pct <= 0.05 and cur > breakout_level * 0.97 and vol_declining:
                breakout_pullback = True

    # ════════════════════════════════════════════════════
    # 카테고리별 점수 + 반등 플래그
    # ════════════════════════════════════════════════════
    trend_pts     = 0   # cap ±15
    momentum_pts  = 0   # cap ±12
    volatility_pts = 0  # cap ±10
    volume_pts    = 0   # cap ±15
    structure_pts = 0   # cap  ±8

    has_cup_pattern      = False
    cup_vol_confirmed    = False
    has_golden_cross     = False
    accumulation_detected = False
    obv_divergence       = False
    squeeze_breakout     = False
    staircase_signal     = None

    # ══════════════════════════════════════════════
    # ❶ 추세 CATEGORY  (cap ±15)
    # ══════════════════════════════════════════════
    if not sma120.dropna().empty and not sma20.dropna().empty and not sma60.dropna().empty:
        s5  = float(sma5.iloc[-1]) if not sma5.dropna().empty else cur
        s20 = float(sma20.iloc[-1])
        s60 = float(sma60.iloc[-1])
        s120 = float(sma120.iloc[-1])

        # ── Cup(그릇) 패턴 + 거래량 프로파일 ──
        if len(close) >= 60:
            recent_60 = close.tail(60).values
            recent_60_vol = vol.tail(60).values
            trough_idx_60 = int(np.argmin(recent_60))
            trough_price = float(recent_60[trough_idx_60])

            if 10 <= trough_idx_60 <= 50 and trough_price > 0:
                left_high = float(np.max(recent_60[:trough_idx_60])) if trough_idx_60 > 0 else cur
                decline_depth = (left_high - trough_price) / left_high
                recovery_ratio = (cur - trough_price) / (left_high - trough_price) if left_high > trough_price else 0

                # 바닥 부근 거래량 vs 회복 구간 거래량
                base_start = max(0, trough_idx_60 - 3)
                base_end = min(len(recent_60_vol), trough_idx_60 + 3)
                vol_at_base = float(np.mean(recent_60_vol[base_start:base_end])) if base_end > base_start else 0
                recovery_start = max(trough_idx_60, len(recent_60_vol) - 10)
                vol_at_recovery = float(np.mean(recent_60_vol[recovery_start:])) if recovery_start < len(recent_60_vol) else 0
                cup_vol_confirmed = vol_at_base > 0 and vol_at_recovery > vol_at_base * 1.2

                if decline_depth >= 0.05 and recovery_ratio >= 0.5:
                    has_cup_pattern = True

        # ── 골든크로스: SMA20이 SMA60을 아래→위로 돌파 ──
        if len(sma20.dropna()) >= 5 and len(sma60.dropna()) >= 5:
            sma20_recent = sma20.dropna().tail(5).values
            sma60_recent = sma60.dropna().tail(5).values
            min_len = min(len(sma20_recent), len(sma60_recent))
            if min_len >= 5:
                sma20_r = sma20_recent[-min_len:]
                sma60_r = sma60_recent[-min_len:]
                if sma20_r[0] < sma60_r[0] and sma20_r[-1] >= sma60_r[-1]:
                    has_golden_cross = True

        # ── SMA20 방향성 ──
        sma20_rising = False
        sma20_falling = False
        if len(sma20.dropna()) >= 5:
            sma20_5ago = float(sma20.dropna().iloc[-5])
            sma20_rising  = s20 > sma20_5ago * 1.001
            sma20_falling = s20 < sma20_5ago * 0.999

        # ── 점수 부여 (elif 체인 → 하나만 선택) ──

        # A: 역배열 + Cup(거래량 확인 포함)
        if has_cup_pattern and cur < s120 and cur > s5 > s20:
            pts = 15 if cup_vol_confirmed else 12
            trend_pts += pts
            extra = " 거래량도 늘고 있어요!" if cup_vol_confirmed else ""
            signals.append({"type": "positive", "label": "바닥 반등 패턴 (Cup) 🏆",
                            "desc": f"주가가 그릇 모양으로 바닥을 찍고 올라오고 있어요.{extra}"})

        # B: 골든크로스
        elif has_golden_cross and s20 <= s60:
            trend_pts += 12
            signals.append({"type": "positive", "label": "반등 전환 신호 (골든크로스) ✨",
                            "desc": "단기 평균이 중기 평균을 올라섰어요. 전환 시작!"})

        # C: 역배열이지만 단기 반등 중
        elif cur > s5 > s20 and sma20_rising and (s60 > s20 or s120 > s20):
            trend_pts += 10
            signals.append({"type": "positive", "label": "바닥에서 올라오는 중 🌱",
                            "desc": "장기적으로는 역배열이지만, 단기 흐름이 상승으로 바뀌고 있어요."})

        # D: Cup + 이미 장기선 위로 회복
        elif has_cup_pattern and cur > s120:
            trend_pts += 8
            signals.append({"type": "positive", "label": "바닥 반등 후 회복 완료 🏆",
                            "desc": "그릇 모양 반등 후 장기 평균 위로 올라왔어요."})

        # E: 완전 정배열 (이미 많이 올라온 상태 → 낮은 가산)
        elif cur > s5 > s20 > s60 > s120:
            trend_pts += 2
            signals.append({"type": "neutral", "label": "상승 추세 유지 중 📈",
                            "desc": "꾸준히 오르고 있지만, 이미 올라온 상태예요. 추격 매수 주의."})

        # F: 부분 정배열
        elif cur > s20 > s60:
            trend_pts += 3
            signals.append({"type": "positive", "label": "중단기 흐름 괜찮음",
                            "desc": "중단기적으로 상승 중이에요."})

        # G: 완전 역배열 + 하락 가속
        elif cur < s5 < s20 < s60 < s120 and sma20_falling:
            trend_pts -= 15
            signals.append({"type": "negative", "label": "하락 추세 강함 📉",
                            "desc": "주가가 모든 평균선 아래에서 계속 내려가고 있어요."})

        # H: 역배열 + 하락 둔화
        elif cur < s20 < s60 and not sma20_falling:
            trend_pts -= 5
            signals.append({"type": "negative", "label": "하락 중이지만 둔화 🐌",
                            "desc": "아직 내려가는 중이지만, 속도가 느려지고 있어요."})

        # I: 부분 역배열
        elif cur < s20 < s60:
            trend_pts -= 8
            signals.append({"type": "negative", "label": "주가 흐름 약함 📉",
                            "desc": "중단기 흐름이 하락 중이에요."})

    # ── 일목균형표 추세 보조 (추세 카테고리 보완) ──
    ichi = _compute_ichimoku(df)
    if ichi:
        if ichi.get("price_above_cloud") and ichi.get("tenkan_above_kijun") and ichi.get("cloud_bullish"):
            trend_pts += 4
            signals.append({"type": "positive", "label": "일목균형표 강세 ☁️📈",
                            "desc": "구름 위에 있고 전환선이 기준선 위예요. 강한 상승 신호."})
        elif ichi.get("price_above_cloud"):
            trend_pts += 2
            signals.append({"type": "positive", "label": "구름 위 위치 ☁️",
                            "desc": "일목 구름 위에 있어요. 중기 상승 추세."})
        elif ichi.get("price_below_cloud") and not ichi.get("tenkan_above_kijun") and not ichi.get("cloud_bullish"):
            trend_pts -= 4
            signals.append({"type": "negative", "label": "일목균형표 약세 ☁️📉",
                            "desc": "구름 아래에서 전환선도 기준선 아래예요. 하락 추세 강함."})
        elif ichi.get("price_below_cloud"):
            trend_pts -= 2
            signals.append({"type": "negative", "label": "구름 아래 위치 ☁️",
                            "desc": "일목 구름 아래에 있어요. 중기 하락 압력."})
        else:
            # 구름 안: 방향 불확실
            if ichi.get("tenkan_above_kijun"):
                signals.append({"type": "neutral", "label": "구름 통과 중 (상승 시도) ☁️",
                                "desc": "구름 안에서 전환선이 기준선 위예요. 돌파 여부 주시."})
            else:
                signals.append({"type": "neutral", "label": "구름 통과 중 (하락 압력) ☁️",
                                "desc": "구름 안에서 전환선이 기준선 아래예요. 방향 관망."})

    # ══════════════════════════════════════════════
    # ❷ 모멘텀 CATEGORY  (cap ±12)
    # ══════════════════════════════════════════════

    # ── 이격도 ──
    if not sma20.dropna().empty:
        s20_val = float(sma20.iloc[-1])
        if s20_val > 0:
            disparity = (cur - s20_val) / s20_val * 100
            if disparity < -5:
                momentum_pts += 5
                signals.append({"type": "positive", "label": f"평균보다 많이 싸요 ({disparity:.1f}%)",
                                "desc": "20일 평균보다 많이 내려와 있어요. 반등 가능성."})
            elif disparity < -2:
                momentum_pts += 2
            elif disparity > 7:
                momentum_pts -= 5
                signals.append({"type": "negative", "label": f"평균보다 많이 비싸요 (+{disparity:.1f}%)",
                                "desc": "20일 평균보다 많이 올라와 있어요. 조정 가능."})
            elif disparity > 3:
                momentum_pts -= 2

    # ── RSI + RSI 다이버전스 ──
    rsi_val = None
    rsi_divergence = False
    if rsi_series is not None and not rsi_series.dropna().empty:
        rsi_val = float(rsi_series.iloc[-1])
        if rsi_val <= 25:
            momentum_pts += 8
            signals.append({"type": "positive", "label": f"많이 떨어진 상태 (RSI {rsi_val:.0f})",
                            "desc": "주가가 많이 내려와서 반등 가능성이 높아요."})
        elif rsi_val <= 35:
            momentum_pts += 4
            signals.append({"type": "positive", "label": f"좀 싼 편 (RSI {rsi_val:.0f})",
                            "desc": "주가가 낮은 편이에요. 매수 기회일 수 있어요."})
        elif rsi_val >= 75:
            momentum_pts -= 8
            signals.append({"type": "negative", "label": f"많이 오른 상태 (RSI {rsi_val:.0f})",
                            "desc": "단기간에 많이 올라서 내려갈 수 있어요."})
        elif rsi_val >= 65:
            momentum_pts -= 4
            signals.append({"type": "negative", "label": f"좀 비싼 편 (RSI {rsi_val:.0f})",
                            "desc": "주가가 높은 편이에요. 추가 매수는 신중하게."})
        else:
            signals.append({"type": "neutral", "label": f"보통 상태 (RSI {rsi_val:.0f})",
                            "desc": "지금은 특별히 싸지도, 비싸지도 않아요."})

        # ── RSI 다이버전스: find_peaks 기반 실제 저점/고점 쌍 비교 ──
        if len(close) >= 40 and len(rsi_series.dropna()) >= 40:
            c_arr = close.tail(60).values.astype(float)
            r_arr = rsi_series.dropna().tail(60).values.astype(float)
            # 배열 길이 맞추기
            n = min(len(c_arr), len(r_arr))
            c_arr, r_arr = c_arr[-n:], r_arr[-n:]

            # 저점 쌍 (상승 다이버전스): 가격 반전 저점을 -c_arr에서 find_peaks로 탐색
            try:
                trough_idx, _ = find_peaks(-c_arr, distance=8, prominence=c_arr.std() * 0.3)
                if len(trough_idx) >= 2:
                    i1, i2 = trough_idx[-2], trough_idx[-1]
                    # 가격: 더 낮은 저점, RSI: 더 높은 저점 → 불리쉬 다이버전스
                    if c_arr[i2] < c_arr[i1] * 0.99 and r_arr[i2] > r_arr[i1] * 1.03:
                        rsi_divergence = True
                        momentum_pts += 5
                        signals.append({"type": "positive", "label": "RSI 상승 다이버전스 📈",
                                        "desc": "주가는 더 내렸지만 힘(RSI)은 올라가고 있어요. 반전 신호!"})

                # 고점 쌍 (하락 다이버전스): 가격 고점 상승 & RSI 고점 하락
                peak_idx, _ = find_peaks(c_arr, distance=8, prominence=c_arr.std() * 0.3)
                if len(peak_idx) >= 2 and not rsi_divergence:
                    i1, i2 = peak_idx[-2], peak_idx[-1]
                    if c_arr[i2] > c_arr[i1] * 1.01 and r_arr[i2] < r_arr[i1] * 0.97:
                        momentum_pts -= 4
                        signals.append({"type": "negative", "label": "RSI 하락 다이버전스 ⚠️",
                                        "desc": "주가는 더 올랐지만 힘(RSI)이 약해지고 있어요. 상승 둔화 경고!"})
            except (ValueError, IndexError):
                pass

    # ── 스토캐스틱 %K / %D ──
    # RSI와 독립적으로 과매수/과매도 확인 → 두 지표 동시 신호면 신뢰도 2배
    stoch_signal = None
    if stoch_k is not None and stoch_d is not None:
        stoch_k_clean = stoch_k.dropna()
        stoch_d_clean = stoch_d.dropna()
        if not stoch_k_clean.empty and not stoch_d_clean.empty:
            sk = float(stoch_k_clean.iloc[-1])
            sd = float(stoch_d_clean.iloc[-1])

            if sk <= 20 and sd <= 20:
                stoch_signal = "oversold"
                if rsi_val is not None and rsi_val <= 35:
                    # RSI + 스토캐스틱 이중 과매도 → 강한 반등 신호
                    momentum_pts += 5
                    signals.append({"type": "positive", "label": f"이중 과매도 확인 (Stoch {sk:.0f} + RSI {rsi_val:.0f}) 🎯",
                                    "desc": "두 개의 지표가 동시에 '많이 내렸다'고 말하고 있어요. 강한 반등 가능성!"})
                else:
                    momentum_pts += 3
                    signals.append({"type": "positive", "label": f"스토캐스틱 과매도 (Stoch {sk:.0f})",
                                    "desc": "단기적으로 많이 내려온 상태예요. 반등 가능성 있어요."})

            elif sk >= 80 and sd >= 80:
                stoch_signal = "overbought"
                if rsi_val is not None and rsi_val >= 65:
                    momentum_pts -= 5
                    signals.append({"type": "negative", "label": f"이중 과매수 확인 (Stoch {sk:.0f} + RSI {rsi_val:.0f}) ⚠️",
                                    "desc": "두 개의 지표가 동시에 '많이 올랐다'고 말하고 있어요. 조정 가능성!"})
                else:
                    momentum_pts -= 3
                    signals.append({"type": "negative", "label": f"스토캐스틱 과매수 (Stoch {sk:.0f})",
                                    "desc": "단기적으로 많이 올라온 상태예요. 추가 매수 신중하게."})

            # %K가 %D를 아래→위로 돌파 (골든크로스): 저점에서 특히 강한 신호
            elif len(stoch_k_clean) >= 2 and len(stoch_d_clean) >= 2:
                sk_prev = float(stoch_k_clean.iloc[-2])
                sd_prev = float(stoch_d_clean.iloc[-2])
                if sk_prev < sd_prev and sk >= sd and sk < 50:
                    stoch_signal = "golden"
                    momentum_pts += 3
                    signals.append({"type": "positive", "label": f"스토캐스틱 반등 전환 (Stoch {sk:.0f}) 🔄",
                                    "desc": "단기 지표가 바닥에서 반등 신호를 보내고 있어요."})
                elif sk_prev > sd_prev and sk <= sd and sk > 50:
                    stoch_signal = "dead"
                    momentum_pts -= 2
                    signals.append({"type": "negative", "label": f"스토캐스틱 하락 전환 (Stoch {sk:.0f})",
                                    "desc": "단기 지표가 고점에서 꺾이고 있어요."})

    # ── MACD ──
    if not histogram.dropna().empty and len(histogram.dropna()) >= 2:
        hist_now  = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        macd_val  = float(macd_line.iloc[-1])
        sig_val   = float(signal_line.iloc[-1])

        if macd_val > sig_val and hist_now > hist_prev:
            momentum_pts += 6
            signals.append({"type": "positive", "label": "오를 힘이 강해지는 중 💪",
                            "desc": "주가가 위로 올라가려는 힘이 세지고 있어요."})
        elif macd_val > sig_val:
            momentum_pts += 2
        elif macd_val < sig_val and hist_now < hist_prev:
            momentum_pts -= 6
            signals.append({"type": "negative", "label": "내릴 힘이 강해지는 중 ⚠️",
                            "desc": "주가가 아래로 내려가려는 힘이 세지고 있어요."})
        elif macd_val < sig_val and hist_now >= hist_prev:
            # 데드크로스이지만 히스토그램 반등 중 → 바닥 전환의 초기 신호
            momentum_pts += 2
            signals.append({"type": "neutral", "label": "하락세 둔화 조짐 🔄",
                            "desc": "아직 약세지만 내리는 힘이 줄고 있어요. 전환 초기 신호일 수 있어요."})

        # ── MACD 다이버전스: find_peaks 기반 저점/고점 쌍 비교 ──
        if len(close) >= 40 and len(histogram.dropna()) >= 40:
            c_arr2 = close.tail(60).values.astype(float)
            h_arr  = histogram.dropna().tail(60).values.astype(float)
            n2 = min(len(c_arr2), len(h_arr))
            c_arr2, h_arr = c_arr2[-n2:], h_arr[-n2:]
            try:
                # 상승 다이버전스: 가격 저점↓ & MACD 히스토그램 저점↑
                trough_c, _ = find_peaks(-c_arr2, distance=8, prominence=c_arr2.std() * 0.3)
                if len(trough_c) >= 2:
                    i1, i2 = trough_c[-2], trough_c[-1]
                    if c_arr2[i2] < c_arr2[i1] * 0.99 and h_arr[i2] > h_arr[i1] * 1.05 and h_arr[i1] < 0:
                        momentum_pts += 4
                        signals.append({"type": "positive", "label": "MACD 상승 다이버전스 ⚡",
                                        "desc": "가격은 더 내렸지만 MACD 힘은 줄어들고 있어요. 반전 가능성!"})

                # 하락 다이버전스: 가격 고점↑ & MACD 히스토그램 고점↓
                peak_c, _ = find_peaks(c_arr2, distance=8, prominence=c_arr2.std() * 0.3)
                if len(peak_c) >= 2:
                    i1, i2 = peak_c[-2], peak_c[-1]
                    if c_arr2[i2] > c_arr2[i1] * 1.01 and h_arr[i2] < h_arr[i1] * 0.95 and h_arr[i1] > 0:
                        momentum_pts -= 4
                        signals.append({"type": "negative", "label": "MACD 하락 다이버전스 ⚠️",
                                        "desc": "가격은 더 올랐지만 MACD 힘이 약해지고 있어요. 상승 둔화 경고."})
            except (ValueError, IndexError):
                pass

    # ══════════════════════════════════════════════
    # ❸ 변동성 CATEGORY  (cap ±10)
    # ══════════════════════════════════════════════

    # ── BB 위치 ──
    if not bb_up.dropna().empty:
        up_val = float(bb_up.iloc[-1])
        dn_val = float(bb_dn.iloc[-1])
        bb_mid_val = float(bb_mid.iloc[-1])

        if up_val > dn_val:
            bb_pos = (cur - dn_val) / (up_val - dn_val)

            if bb_pos <= 0.0:
                volatility_pts += 6
                signals.append({"type": "positive", "label": "바닥 근처 도달 🔻",
                                "desc": "주가가 변동 범위의 가장 아래에 있어요. 반등 가능."})
            elif bb_pos <= 0.2:
                volatility_pts += 3
                signals.append({"type": "positive", "label": "바닥권에 가까움",
                                "desc": "주가가 변동 범위의 아래쪽에 있어요."})
            elif bb_pos >= 1.0:
                volatility_pts -= 6
                signals.append({"type": "negative", "label": "천장 근처 도달 🔺",
                                "desc": "주가가 변동 범위의 가장 위에 있어요."})
            elif bb_pos >= 0.8:
                volatility_pts -= 3
                signals.append({"type": "negative", "label": "천장권에 가까움",
                                "desc": "주가가 변동 범위의 위쪽에 있어요."})

    # ── 수박지표 (BB 스퀴즈 + 이평선 수렴) ──
    ma_convergence = None
    if not sma5.dropna().empty and not sma224.dropna().empty:
        ma_vals = [float(sma5.iloc[-1]), float(sma20.iloc[-1]),
                   float(sma60.iloc[-1]), float(sma224.iloc[-1])]
        ma_spread = max(ma_vals) - min(ma_vals)
        ma_convergence = ma_spread / cur * 100 if cur > 0 else None

    disparity_20 = None
    if not sma20.dropna().empty:
        disparity_20 = (cur / float(sma20.iloc[-1])) * 100 - 100

    if is_squeeze and ma_convergence is not None and ma_convergence < 2.0:
        if disparity_20 is not None and abs(disparity_20) < 3.0:
            if not sma224.dropna().empty and cur > float(sma224.iloc[-1]):
                volatility_pts += 6
                signals.append({"type": "positive", "label": "큰 상승 준비 중 🍉💪",
                                "desc": "장기 평균 위에서 힘을 모으고 있어요. 크게 오를 가능성."})
            else:
                volatility_pts += 4
                signals.append({"type": "positive", "label": "큰 움직임 준비 중 🍉",
                                "desc": "주가가 좁은 범위에서 모이고 있어요. 곧 크게 움직일 수 있어요."})
    elif is_squeeze:
        signals.append({"type": "neutral", "label": "주가가 쉬는 중 😴",
                        "desc": "주가 변동이 줄어들고 있어요. 곧 움직임이 커질 수 있어요."})

    # ══════════════════════════════════════════════
    # ❹ 거래량 CATEGORY  (cap ±15)
    # ══════════════════════════════════════════════

    # ── 바닥권 거래량 폭발 (매집 신호) ──
    if len(close) >= 120 and len(vol) >= 20:
        low_120 = float(low.tail(120).min())
        vol_ma20 = vol.rolling(20).mean()
        if low_120 > 0 and not vol_ma20.dropna().empty:
            near_bottom = cur <= low_120 * 1.10

            vol_explosion = False
            for v_val, v_avg in zip(vol.tail(5).values, vol_ma20.tail(5).values):
                if np.isfinite(v_val) and np.isfinite(v_avg) and v_avg > 0 and v_val >= v_avg * 3.0:
                    vol_explosion = True
                    break

            if near_bottom and vol_explosion:
                accumulation_detected = True
                volume_pts += 12
                signals.append({"type": "positive", "label": "바닥권 거래량 폭발 🐳",
                                "desc": "120일 최저가 근처에서 거래량이 크게 터졌어요. 세력 매집 가능성!"})
            elif near_bottom:
                volume_pts += 2
                signals.append({"type": "neutral", "label": "바닥권 대기 중 🔍",
                                "desc": "120일 최저가 근처이지만 아직 거래량 신호가 없어요."})

    # ── OBV 상승 다이버전스 ──
    if len(close) >= 40 and len(obv) >= 40:
        recent_close_20 = close.tail(20).values
        recent_obv_20   = obv.tail(20).values
        if len(recent_close_20) == 20 and len(recent_obv_20) == 20:
            x_arr = np.arange(20, dtype=float)
            try:
                close_slope = np.polyfit(x_arr, recent_close_20, 1)[0]
                obv_slope   = np.polyfit(x_arr, recent_obv_20, 1)[0]
                if close_slope <= 0 and obv_slope > 0:
                    obv_divergence = True
                    volume_pts += 8
                    signals.append({"type": "positive", "label": "OBV 상승 다이버전스 📊",
                                    "desc": "주가는 내리는데 거래량 흐름은 올라가고 있어요. 숨은 매수세!"})
            except (np.linalg.LinAlgError, ValueError):
                pass

    # ── BB 스퀴즈 + 거래량 유입 ──
    if is_squeeze and len(vol) >= 20:
        vol_ma20_sq = vol.rolling(20).mean()
        if not vol_ma20_sq.dropna().empty:
            avg_vol_20 = float(vol_ma20_sq.iloc[-1])
            if avg_vol_20 > 0 and any(
                np.isfinite(v) and v >= avg_vol_20 * 1.5 for v in vol.tail(3).values
            ):
                squeeze_breakout = True
                volume_pts += 8
                signals.append({"type": "positive", "label": "밴드 스퀴즈 + 거래량 유입 💥",
                                "desc": "변동성이 극도로 줄어든 상태에서 거래량이 붙고 있어요. 돌파 임박!"})

    # ── 거래량 점진적 증가(Creeping Volume) ──
    if len(vol) >= 10:
        vol_5d = vol.tail(5).values
        vol_prev5 = vol.tail(10).head(5).values
        avg_recent = float(np.mean(vol_5d[np.isfinite(vol_5d)])) if np.any(np.isfinite(vol_5d)) else 0
        avg_prev = float(np.mean(vol_prev5[np.isfinite(vol_prev5)])) if np.any(np.isfinite(vol_prev5)) else 0
        if avg_prev > 0 and avg_recent > avg_prev * 1.4:
            # 최근 5일이 이전 5일 대비 40% 증가 → 점진적 매수세 유입
            if float(close.iloc[-1]) >= float(close.iloc[-5]):
                volume_pts += 4
                signals.append({"type": "positive", "label": "거래량 점진 증가 + 상승 📈",
                                "desc": "5일간 거래량이 꾸준히 늘면서 가격도 올라가고 있어요."})

    # ── 거래량 건조(Dry-up): 하락 중 거래량 급감 → 매도세 소진 ──
    if len(vol) >= 20 and len(close) >= 20:
        vol_ma20_dry = vol.rolling(20).mean()
        if not vol_ma20_dry.dropna().empty:
            recent_vol = float(vol.tail(3).mean())
            avg_vol_20 = float(vol_ma20_dry.iloc[-1])
            price_falling = float(close.iloc[-1]) < float(close.iloc[-10]) if len(close) >= 10 else False
            if avg_vol_20 > 0 and recent_vol < avg_vol_20 * 0.4 and price_falling:
                volume_pts += 3
                signals.append({"type": "positive", "label": "매도세 소진 (거래량 건조) 🏜️",
                                "desc": "주가가 내리는데 거래량이 극도로 줄었어요. 팔 사람이 다 판 신호일 수 있어요."})

    # ── 일반 거래량 ──
    vol_ma5 = vol.rolling(5).mean()
    if len(vol) >= 5 and not vol_ma5.dropna().empty:
        cur_vol = float(vol.iloc[-1])
        avg_vol = float(vol_ma5.iloc[-1])
        if avg_vol > 0 and cur_vol >= avg_vol * 1.5:
            if float(close.iloc[-1]) > float(open_.iloc[-1]):
                volume_pts += 4
                signals.append({"type": "positive", "label": "거래량 폭발 + 양봉 🔥",
                                "desc": "많은 사람이 사면서 주가가 올랐어요."})
            else:
                volume_pts -= 4
                signals.append({"type": "negative", "label": "거래량 폭발 + 음봉 💨",
                                "desc": "많은 사람이 팔면서 주가가 내렸어요."})

    # ══════════════════════════════════════════════
    # ❺ 구조 CATEGORY  (cap ±8)
    # ══════════════════════════════════════════════

    # ── 지지/저항선 ──
    if support_lines:
        nearest_support = min(support_lines, key=lambda s: abs(cur - s) if s > 0 else float('inf'))
        if nearest_support > 0:
            dist_support = (cur - nearest_support) / nearest_support
            if -0.02 <= dist_support <= 0.03:
                structure_pts += 4
                signals.append({"type": "positive", "label": "지지선 근처 💎",
                                "desc": f"지지선({int(nearest_support):,}원) 근처에서 버티고 있어요."})

    if resistance_lines:
        nearest_resist = min(resistance_lines, key=lambda r: abs(cur - r) if r > 0 else float('inf'))
        if nearest_resist > 0:
            dist_resist = (nearest_resist - cur) / nearest_resist
            if 0 <= dist_resist <= 0.03:
                structure_pts -= 3
                signals.append({"type": "negative", "label": "저항선 근처 🧱",
                                "desc": f"저항선({int(nearest_resist):,}원) 근처에서 막힐 수 있어요."})

    # ── 박스권 (거래량 프로파일 연동) ──
    if box_range and box_range.get("is_box"):
        box_top = box_range.get("top", 0)
        box_bot = box_range.get("bottom", 0)
        if box_top > box_bot > 0:
            box_pos = (cur - box_bot) / (box_top - box_bot)
            # 최근 거래량 상태 확인
            box_vol_rising = False
            if len(vol) >= 10:
                v5 = float(vol.tail(5).mean())
                v10 = float(vol.tail(10).head(5).mean())
                box_vol_rising = v10 > 0 and v5 > v10 * 1.3

            if box_pos <= 0.2:
                # 박스 하단 = 실전에서 가장 높은 승률 진입점
                if box_vol_rising:
                    structure_pts += 8  # 캡 최대치
                    signals.append({"type": "positive", "label": "박스 하단 + 거래량 증가 📦🔥",
                                    "desc": "박스권 바닥에서 거래량이 늘고 있어요. 최적의 매수 타이밍!"})
                else:
                    structure_pts += 5
                    signals.append({"type": "positive", "label": "박스권 바닥 근처 📦",
                                    "desc": "박스권 아래쪽이에요. 거래량 증가를 확인하면 매수 신호."})
            elif box_pos >= 0.85:
                # 박스 상단: 거래량 터지면 돌파, 아니면 저항
                if box_vol_rising and float(close.iloc[-1]) > float(open_.iloc[-1]):
                    structure_pts += 4
                    signals.append({"type": "positive", "label": "박스 상단 돌파 시도 📦💥",
                                    "desc": "박스 상단에서 거래량과 함께 돌파를 시도하고 있어요!"})
                else:
                    structure_pts -= 4
                    signals.append({"type": "negative", "label": "박스권 천장 근처 📦",
                                    "desc": "박스권 위쪽이라 눌릴 수 있어요."})

    # ── 계단지표 ──
    if not atr14.dropna().empty and len(close) >= 20:
        atr_val_local = float(atr14.iloc[-1])
        grid_size = atr_val_local * 0.5
        if grid_size > 0:
            close_arr = close.tail(20).values
            grids = np.floor(close_arr / grid_size) * grid_size
            current_level = grids[-1]
            count = 1
            for idx in range(len(grids) - 2, -1, -1):
                if abs(grids[idx] - current_level) < grid_size * 0.001:
                    count += 1
                else:
                    break
            prev_level = grids[0]
            if count >= 3:
                if current_level > prev_level + grid_size * 0.5:
                    staircase_signal = "up"
                elif current_level < prev_level - grid_size * 0.5:
                    staircase_signal = "down"

    if staircase_signal == "up":
        structure_pts += 4
        signals.append({"type": "positive", "label": "한 계단씩 오르는 중 🪜",
                        "desc": "주가가 한 단계씩 올라가고 있어요."})
    elif staircase_signal == "down":
        structure_pts -= 4
        signals.append({"type": "negative", "label": "한 계단씩 내리는 중 🪜",
                        "desc": "주가가 한 단계씩 내려가고 있어요."})

    # ── 돌파 후 눌림목 (수렴→돌파→건전한 조정) ──
    if breakout_pullback:
        structure_pts += 7
        signals.append({"type": "positive", "label": "돌파 후 눌림목 진입 🎯",
                        "desc": "저항선을 돌파한 뒤 거래량이 줄며 건전하게 쉬고 있어요. 최적 매수 타이밍!"})
    elif breakout_fresh:
        structure_pts += 5
        signals.append({"type": "positive", "label": "저항 돌파 직후 🚀",
                        "desc": "거래량과 함께 저항선을 막 돌파했어요. 눌림목을 기다려보세요."})

    # ── 캔들 패턴 인식 (최근 3봉 분석) ──
    candle_pattern = None
    if len(close) >= 3 and len(open_) >= 3 and len(high) >= 3 and len(low) >= 3:
        # c0: 3봉 전, c1: 전봉, c2: 현재봉
        c0_o = float(open_.iloc[-3]) if len(open_) >= 3 else None
        c0_h = float(high.iloc[-3])  if len(high) >= 3  else None
        c0_l = float(low.iloc[-3])   if len(low) >= 3   else None
        c0_c = float(close.iloc[-3]) if len(close) >= 3 else None
        c1_o, c1_h, c1_l, c1_c = float(open_.iloc[-2]), float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])
        c2_o, c2_h, c2_l, c2_c = float(open_.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        atr_now = float(atr14.iloc[-1]) if not atr14.dropna().empty else (c2_h - c2_l)

        if atr_now > 0:
            body0 = abs(c0_c - c0_o) if c0_o is not None else 0
            body1 = abs(c1_c - c1_o)
            body2 = abs(c2_c - c2_o)
            lower_wick2 = min(c2_o, c2_c) - c2_l
            upper_wick2 = c2_h - max(c2_o, c2_c)
            lower_wick1 = min(c1_o, c1_c) - c1_l
            upper_wick1 = c1_h - max(c1_o, c1_c)

            # 망치형(Hammer): 하락 후 긴 아래꼬리 + 작은 몸통
            if (c1_c < c1_o and
                lower_wick2 > body2 * 2 and
                upper_wick2 < body2 * 0.5 and
                body2 < atr_now * 0.6):
                candle_pattern = "hammer"
                structure_pts += 3
                signals.append({"type": "positive", "label": "망치형 캔들 🔨",
                                "desc": "바닥에서 강한 매수세가 들어왔어요. 반등 신호!"})

            # 역망치형(Inverted Hammer): 하락 후 긴 위꼬리 + 작은 몸통 → 매수세 시도
            elif (c1_c < c1_o and
                  upper_wick2 > body2 * 2 and
                  lower_wick2 < body2 * 0.5 and
                  body2 < atr_now * 0.6):
                candle_pattern = "inverted_hammer"
                structure_pts += 2
                signals.append({"type": "positive", "label": "역망치형 캔들 🔼",
                                "desc": "위쪽으로 매수 시도가 있었어요. 다음 봉 확인 필요."})

            # 도지(Doji): 시가≈종가 (결정 불확실성, 추세 전환 경고)
            elif (body2 < atr_now * 0.1 and
                  (c2_h - c2_l) > atr_now * 0.3):
                candle_pattern = "doji"
                # 하락 추세 후 도지: 반등 가능성
                if c1_c < c1_o and float(close.iloc[-5]) > float(close.iloc[-2]) if len(close) >= 5 else False:
                    structure_pts += 2
                    signals.append({"type": "positive", "label": "도지 캔들 (하락 끝?) ⚖️",
                                    "desc": "사는 힘과 파는 힘이 팽팽해요. 하락 추세가 멈출 수 있어요."})
                else:
                    signals.append({"type": "neutral", "label": "도지 캔들 ⚖️",
                                    "desc": "매수/매도 힘이 비슷해요. 방향 결정 전 관망 구간."})

            # 상승장악형(Bullish Engulfing)
            elif (c1_c < c1_o and
                  c2_c > c2_o and
                  c2_o <= c1_c and
                  c2_c >= c1_o and
                  body2 > body1 * 1.2):
                candle_pattern = "engulfing"
                structure_pts += 3
                signals.append({"type": "positive", "label": "상승장악형 캔들 🟢",
                                "desc": "매도세를 완전히 압도하는 매수세가 나왔어요!"})

            # 하락장악형(Bearish Engulfing)
            elif (c1_c > c1_o and
                  c2_c < c2_o and
                  c2_o >= c1_c and
                  c2_c <= c1_o and
                  body2 > body1 * 1.2):
                candle_pattern = "bearish_engulfing"
                structure_pts -= 3
                signals.append({"type": "negative", "label": "하락장악형 캔들 🔴",
                                "desc": "매수세를 압도하는 매도세가 나왔어요. 하락 주의."})

            # 모닝스타(Morning Star): 3봉 — 큰 음봉 + 소형봉(갭) + 큰 양봉
            elif (c0_o is not None and
                  c0_c < c0_o and body0 > atr_now * 0.6 and  # 첫째: 큰 음봉
                  body1 < atr_now * 0.3 and                   # 둘째: 소형봉
                  c1_h < c0_c and                              # 둘째가 첫째 종가보다 낮음
                  c2_c > c2_o and body2 > atr_now * 0.6 and  # 셋째: 큰 양봉
                  c2_c > (c0_o + c0_c) / 2):                  # 첫째 봉 절반 이상 회복
                candle_pattern = "morning_star"
                structure_pts += 5
                signals.append({"type": "positive", "label": "모닝스타 패턴 🌟",
                                "desc": "3일 연속 바닥 반전 패턴이에요. 강한 반등 신호!"})

            # 이브닝스타(Evening Star): 3봉 — 큰 양봉 + 소형봉(갭) + 큰 음봉
            elif (c0_o is not None and
                  c0_c > c0_o and body0 > atr_now * 0.6 and  # 첫째: 큰 양봉
                  body1 < atr_now * 0.3 and                   # 둘째: 소형봉
                  c1_l > c0_c and                              # 둘째가 첫째 종가보다 높음
                  c2_c < c2_o and body2 > atr_now * 0.6 and  # 셋째: 큰 음봉
                  c2_c < (c0_o + c0_c) / 2):                  # 첫째 봉 절반 이하로 하락
                candle_pattern = "evening_star"
                structure_pts -= 5
                signals.append({"type": "negative", "label": "이브닝스타 패턴 🌙",
                                "desc": "3일 연속 천장 반전 패턴이에요. 하락 전환 경고."})

            # 십자형 샛별(Dragonfly Doji): 거의 위꼬리 없고 긴 아래꼬리 + 도지
            elif (body2 < atr_now * 0.1 and
                  lower_wick2 > atr_now * 0.6 and
                  upper_wick2 < atr_now * 0.15):
                candle_pattern = "dragonfly_doji"
                structure_pts += 3
                signals.append({"type": "positive", "label": "잠자리형 도지 🐉",
                                "desc": "아래로 많이 눌렸다가 회복했어요. 강한 반등 신호."})

            # 비석형 도지(Gravestone Doji): 긴 위꼬리 + 아래꼬리 없음 + 도지 → 천장 신호
            elif (body2 < atr_now * 0.1 and
                  upper_wick2 > atr_now * 0.6 and
                  lower_wick2 < atr_now * 0.15):
                candle_pattern = "gravestone_doji"
                structure_pts -= 3
                signals.append({"type": "negative", "label": "비석형 도지 🪦",
                                "desc": "위로 올랐다가 밀렸어요. 천장 신호일 수 있어요."})

    # ════════════════════════════════════════════════════
    # 카테고리 캡 적용
    # ════════════════════════════════════════════════════
    trend_pts      = max(-15, min(15, trend_pts))
    momentum_pts   = max(-12, min(12, momentum_pts))
    volatility_pts = max(-10, min(10, volatility_pts))
    volume_pts     = max(-15, min(15, volume_pts))
    structure_pts  = max( -8, min( 8, structure_pts))

    # ── 반등 시너지: 바닥 신호가 복수 동시 발생 시 보너스 ──
    reversal_flags = [has_cup_pattern, accumulation_detected, obv_divergence,
                      squeeze_breakout, has_golden_cross, rsi_divergence,
                      breakout_pullback, stoch_signal == "oversold"]
    reversal_count = sum(reversal_flags)
    synergy_bonus = 0
    if reversal_count >= 3:
        synergy_bonus = 5
        signals.append({"type": "positive", "label": f"반등 신호 {reversal_count}개 동시 발생 🔥🔥",
                        "desc": "여러 지표가 동시에 바닥 반등을 가리키고 있어요! 강한 매수 신호."})
    elif reversal_count >= 2:
        synergy_bonus = 3
        signals.append({"type": "positive", "label": f"반등 신호 {reversal_count}개 겹침 ✨",
                        "desc": "복수의 지표가 반등을 시사하고 있어요."})

    # ── 매집 보정: 강한 거래량 신호가 있으면 추세 감점을 절반 복구 ──
    if (accumulation_detected or obv_divergence) and trend_pts < -5:
        recovery = min(8, abs(trend_pts) // 2)
        trend_pts += recovery
        signals.append({"type": "positive", "label": "매집 신호로 하락 리스크 완화 🛡️",
                        "desc": "역배열이지만 바닥에서 매수세가 들어오고 있어 하락 위험이 줄어요."})

    # ── 최종 점수 ──
    score = 50 + trend_pts + momentum_pts + volatility_pts + volume_pts + structure_pts + synergy_bonus
    final_score = max(0, min(100, score))

    internals = {
        "rsi": rsi_val,
        "sma20": float(sma20.iloc[-1]) if not sma20.dropna().empty else None,
        "bb_upper": float(bb_up.iloc[-1]) if not bb_up.dropna().empty else None,
        "bb_lower": float(bb_dn.iloc[-1]) if not bb_dn.dropna().empty else None,
        "atr14": float(atr14.iloc[-1]) if not atr14.dropna().empty else cur * 0.02,
        "is_squeeze": is_squeeze,
        "staircase_signal": staircase_signal,
        "accumulation": accumulation_detected,
        "obv_divergence": obv_divergence,
        "breakout_pullback": breakout_pullback or breakout_fresh,
    }

    return final_score, signals, internals


def _generate_predicted_candles(
    df: pd.DataFrame,
    prediction_score: int,
    internals: dict,
    support_lines: list[float],
    resistance_lines: list[float],
    box_range: dict | None = None,
    n_days: int = 7,
) -> list[dict]:
    """
    기술적 분석 기반 주가 예측 시뮬레이션 (v3 — 몬테카를로 앙상블).

    v2 → v3 핵심 변경:
    1. 몬테카를로 앙상블: N_PATHS(50)개 경로 생성 → 중앙값으로 대표 캔들 도출
    2. ADX 기반 추세 강도 측정 (Hurst + ADX 이중 확인)
    3. 멀티타임프레임 모멘텀: 5d/10d/20d/60d 가중 혼합
    4. 피보나치 레벨 자석 효과
    5. 거래량 프로파일 기반 drift 보정
    6. 캔들 패턴 초기 드리프트 가산
    7. 적응형 GARCH(1,1): alpha/beta 레짐별 조정
    8. 변동성 스마일: 하락 시 변동성 증가 (레버리지 효과)
    """
    N_PATHS = 50  # 몬테카를로 시뮬레이션 경로 수

    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce")

    last_close = float(close.iloc[-1])
    last_date = str(df["time"].iloc[-1])[:10]
    atr_val = internals.get("atr14", last_close * 0.02)
    sma20_val = internals.get("sma20")
    bb_upper = internals.get("bb_upper")
    bb_lower = internals.get("bb_lower")
    is_squeeze = internals.get("is_squeeze", False)
    has_breakout_pullback = internals.get("breakout_pullback", False)

    # SMA 다중 계산
    sma5_val = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else last_close
    sma60_series = close.rolling(60).mean()
    sma120_series = close.rolling(120).mean()
    sma60_val = float(sma60_series.iloc[-1]) if not sma60_series.dropna().empty else None
    sma120_val = float(sma120_series.iloc[-1]) if not sma120_series.dropna().empty else None

    # ══════════════════════════════════════════════════════════════════
    # ❶ ADX 계산 — 추세 강도 측정 (14일)
    # ══════════════════════════════════════════════════════════════════
    adx_val = 25.0  # 기본값 (중립)
    if len(close) >= 28:
        tr_series = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        atr_14 = tr_series.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
        adx_series = dx.rolling(14).mean()
        if not adx_series.dropna().empty:
            adx_val = float(adx_series.iloc[-1])
            if not np.isfinite(adx_val):
                adx_val = 25.0

    # ══════════════════════════════════════════════════════════════════
    # ❷ Hurst 지수 (R/S) + ADX 종합 레짐 판단
    # ══════════════════════════════════════════════════════════════════
    hurst = 0.5
    if len(close) >= 60:
        log_ret = np.log(close.tail(60).values[1:] / close.tail(60).values[:-1])
        log_ret = log_ret[np.isfinite(log_ret)]
        if len(log_ret) >= 20:
            half = len(log_ret) // 2
            rs_vals = []
            for chunk in [log_ret[:half], log_ret[half:]]:
                cum_dev = np.cumsum(chunk - np.mean(chunk))
                r_val = float(np.max(cum_dev) - np.min(cum_dev))
                s_val = float(np.std(chunk))
                if s_val > 0:
                    rs_vals.append(r_val / s_val)
            if rs_vals:
                avg_rs = np.mean(rs_vals)
                n_obs = half
                hurst = float(np.log(avg_rs) / np.log(n_obs)) if n_obs > 1 else 0.5
                hurst = max(0.2, min(0.8, hurst))

    # ADX + Hurst 이중 확인으로 레짐 판단 정확도 향상
    is_strong_trend = adx_val >= 30 and hurst > 0.50
    is_trending = (adx_val >= 25 and hurst > 0.45) or hurst > 0.55
    is_mean_reverting = (adx_val < 20 and hurst < 0.50) or hurst < 0.45
    # 추세 방향 (+DI vs -DI)
    trend_direction = 1.0  # 기본 상승
    if len(close) >= 28:
        plus_di_val = float(plus_di.iloc[-1]) if not plus_di.dropna().empty else 50
        minus_di_val = float(minus_di.iloc[-1]) if not minus_di.dropna().empty else 50
        if np.isfinite(plus_di_val) and np.isfinite(minus_di_val):
            trend_direction = 1.0 if plus_di_val >= minus_di_val else -1.0

    # ══════════════════════════════════════════════════════════════════
    # ❸ 멀티타임프레임 모멘텀 (5d/10d/20d/60d 가중)
    # ══════════════════════════════════════════════════════════════════
    def safe_ret(period):
        if len(close) >= period + 1:
            v0 = float(close.iloc[-(period + 1)])
            return (last_close / v0 - 1) if v0 > 0 else 0
        return 0

    ret_5d = safe_ret(5)
    ret_10d = safe_ret(10)
    ret_20d = safe_ret(20)
    ret_60d = safe_ret(60)

    # n_days에 따라 단기/장기 모멘텀 가중치 조절
    if n_days <= 10:
        mom_weights = {"5d": 0.50, "10d": 0.30, "20d": 0.15, "60d": 0.05}
    elif n_days <= 20:
        mom_weights = {"5d": 0.25, "10d": 0.30, "20d": 0.30, "60d": 0.15}
    else:
        mom_weights = {"5d": 0.10, "10d": 0.15, "20d": 0.30, "60d": 0.45}

    weighted_mom = (ret_5d * mom_weights["5d"] + ret_10d * mom_weights["10d"]
                    + ret_20d * mom_weights["20d"] + ret_60d * mom_weights["60d"])

    # ══════════════════════════════════════════════════════════════════
    # ❹ 변동성 분석 (v2 유지 + 레버리지 효과 추가)
    # ══════════════════════════════════════════════════════════════════
    base_sigma = atr_val / last_close

    hist_window = min(120, len(close) - 1)
    if hist_window >= 20:
        hist_log_rets = np.diff(np.log(close.tail(hist_window + 1).values))
        hist_log_rets = hist_log_rets[np.isfinite(hist_log_rets)]
    else:
        hist_log_rets = np.array([])

    if len(hist_log_rets) >= 20:
        hist_vol = float(np.std(hist_log_rets))
        annualized_vol = hist_vol * np.sqrt(252)
        max_daily_change = float(np.max(np.abs(hist_log_rets)))
        p95_change = float(np.percentile(np.abs(hist_log_rets), 95))
        # 레버리지 효과: 하락 시 변동성 증가 비율
        neg_rets = hist_log_rets[hist_log_rets < 0]
        pos_rets = hist_log_rets[hist_log_rets > 0]
        leverage_ratio = (float(np.std(neg_rets)) / float(np.std(pos_rets))
                          if len(neg_rets) >= 5 and len(pos_rets) >= 5 and float(np.std(pos_rets)) > 0
                          else 1.0)
        leverage_ratio = max(0.8, min(1.5, leverage_ratio))
    else:
        hist_vol = base_sigma
        annualized_vol = base_sigma * np.sqrt(252)
        max_daily_change = base_sigma * 3
        p95_change = base_sigma * 2
        leverage_ratio = 1.0

    is_extreme_vol = annualized_vol > 0.60 or max_daily_change > 0.10
    is_stable = annualized_vol < 0.20

    if is_extreme_vol:
        median_change = float(np.median(np.abs(hist_log_rets))) if len(hist_log_rets) >= 20 else base_sigma * 0.5
        base_sigma = min(base_sigma, max(median_change, p95_change * 0.5))
    elif is_stable:
        base_sigma = max(base_sigma, 0.005)

    if is_extreme_vol:
        daily_ret_cap = min(0.03, p95_change * 1.0)
    elif n_days >= 21:
        daily_ret_cap = min(0.04, p95_change * 1.5)
    else:
        daily_ret_cap = min(0.06, p95_change * 2.0)

    if is_extreme_vol:
        max_total_deviation = 0.15
    elif n_days >= 21:
        max_total_deviation = 0.25
    else:
        max_total_deviation = 0.35

    if len(close) >= 5:
        recent_rets = np.abs(np.diff(np.log(close.tail(6).values)))
        recent_rets = recent_rets[np.isfinite(recent_rets)]
        if len(recent_rets) >= 3:
            recent_vol = float(np.std(recent_rets))
            vol_ratio = recent_vol / base_sigma if base_sigma > 0 else 1.0
            max_vol_ratio = 1.3 if is_extreme_vol else 1.8
            vol_persistence = 0.6 * min(max_vol_ratio, max(0.5, vol_ratio)) + 0.4
        else:
            vol_persistence = 1.0
    else:
        vol_persistence = 1.0

    # ══════════════════════════════════════════════════════════════════
    # ❺ 기대 수익률(drift) — v3: ADX+멀티TF+거래량+캔들패턴 종합
    # ══════════════════════════════════════════════════════════════════

    # score drift (v2와 동일 — 장기 감쇠)
    score_decay_factor = max(0.33, 1.0 - (n_days - 7) / 35)
    score_drift = (prediction_score - 50) / 50 * 0.003 * score_decay_factor

    # 모멘텀 drift — 멀티타임프레임 + 레짐 반영
    if is_strong_trend:
        momentum_drift = weighted_mom * 0.15 * trend_direction
    elif is_trending:
        momentum_drift = weighted_mom * 0.08
    elif is_mean_reverting:
        momentum_drift = -weighted_mom * 0.10
    else:
        momentum_drift = weighted_mom * 0.03

    # 장기 예측 시 모멘텀 감쇠
    mom_decay = max(0.2, 1.0 - (n_days - 7) / 30)
    momentum_drift *= mom_decay

    # ── 거래량 프로파일 drift 보정 ──
    vol_drift = 0.0
    if len(vol) >= 20:
        vol_ma20 = vol.rolling(20).mean()
        if not vol_ma20.dropna().empty:
            avg_vol20 = float(vol_ma20.iloc[-1])
            if avg_vol20 > 0:
                # 최근 5일 거래량 비율
                recent_vol_ratio = float(vol.tail(5).mean()) / avg_vol20
                # 거래량 증가 + 양봉 = 매수세 유입
                recent_up = sum(1 for i in range(-5, 0) if float(close.iloc[i]) > float(open_.iloc[i]))
                if recent_vol_ratio > 1.5 and recent_up >= 3:
                    vol_drift = 0.002  # 강한 매수 거래량
                elif recent_vol_ratio > 1.3 and recent_up >= 3:
                    vol_drift = 0.001
                elif recent_vol_ratio > 1.5 and recent_up <= 2:
                    vol_drift = -0.001  # 거래량은 높지만 매도 우세
                # OBV 방향성
                if len(vol) >= 10:
                    price_diff = close.diff()
                    obv_sign = pd.Series(0.0, index=close.index)
                    obv_sign[price_diff > 0] = 1.0
                    obv_sign[price_diff < 0] = -1.0
                    obv = (vol * obv_sign).cumsum()
                    obv_10 = obv.tail(10).values
                    if len(obv_10) >= 10:
                        obv_slope = (obv_10[-1] - obv_10[0]) / max(1, abs(obv_10[0])) if obv_10[0] != 0 else 0
                        vol_drift += np.sign(obv_slope) * min(0.001, abs(obv_slope) * 0.0001)

    # ── 캔들 패턴 drift 보정 ──
    pattern_drift = 0.0
    if len(close) >= 3 and len(open_) >= 3:
        c1_o, c1_c = float(open_.iloc[-2]), float(close.iloc[-2])
        c2_o, c2_h, c2_l, c2_c = float(open_.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
        body2 = abs(c2_c - c2_o)
        lower_wick2 = min(c2_o, c2_c) - c2_l
        upper_wick2 = c2_h - max(c2_o, c2_c)

        # 망치형 → 초기 상승 편향
        if c1_c < c1_o and lower_wick2 > body2 * 2 and upper_wick2 < body2 * 0.5:
            pattern_drift = 0.002
        # 상승장악형 → 강한 초기 상승
        elif c1_c < c1_o and c2_c > c2_o and c2_c >= c1_o and body2 > abs(c1_c - c1_o) * 1.2:
            pattern_drift = 0.003
        # 하락장악형 → 초기 하락 편향
        elif c1_c > c1_o and c2_c < c2_o and c2_c <= c1_o and body2 > abs(c1_c - c1_o) * 1.2:
            pattern_drift = -0.003
        # 도지(십자형) — 방향 불확실, 변동성 증가 신호
        elif body2 < atr_val * 0.1 and (lower_wick2 + upper_wick2) > body2 * 4:
            pattern_drift = 0.0  # 방향 중립이지만 변동성 팩터로 반영
            vol_persistence = min(vol_persistence * 1.2, 2.0)

    # 패턴 drift는 첫 3~5일에만 강하게 적용
    pattern_decay_days = min(5, n_days)

    daily_mu = score_drift + momentum_drift + vol_drift
    if is_extreme_vol:
        daily_mu = max(-0.0015, min(0.0015, daily_mu))

    # ══════════════════════════════════════════════════════════════════
    # ❻ 지지/저항 + 피보나치 레벨 맵 구축
    # ══════════════════════════════════════════════════════════════════
    sr_levels: list[tuple[float, str, float]] = []  # (price, type, strength)
    for s in support_lines:
        if s > 0:
            sr_levels.append((s, "support", 1.0))
    for r in resistance_lines:
        if r > 0:
            sr_levels.append((r, "resistance", 1.0))
    if bb_upper and bb_upper > 0:
        sr_levels.append((bb_upper, "resistance", 0.6))
    if bb_lower and bb_lower > 0:
        sr_levels.append((bb_lower, "support", 0.6))
    if box_range and box_range.get("is_box"):
        bt, bb_val = box_range.get("top", 0), box_range.get("bottom", 0)
        if bt > 0:
            sr_levels.append((bt, "resistance", 0.8))
        if bb_val > 0:
            sr_levels.append((bb_val, "support", 0.8))

    # 피보나치 되돌림 레벨 추가
    if len(high) >= 60 and len(low) >= 60:
        high_60 = float(high.tail(60).max())
        low_60 = float(low.tail(60).min())
        fib_range = high_60 - low_60
        if fib_range > 0:
            fib_levels = {
                0.236: low_60 + fib_range * 0.236,
                0.382: low_60 + fib_range * 0.382,
                0.500: low_60 + fib_range * 0.500,
                0.618: low_60 + fib_range * 0.618,
                0.786: low_60 + fib_range * 0.786,
            }
            for fib_pct, fib_price in fib_levels.items():
                dist = abs(fib_price - last_close) / last_close
                if dist < 0.15:  # 15% 이내만
                    ftype = "support" if fib_price < last_close else "resistance"
                    # 0.382, 0.618이 가장 강한 피보나치 레벨
                    strength = 0.7 if fib_pct in (0.382, 0.618) else 0.4
                    sr_levels.append((fib_price, ftype, strength))

    # VWAP 근사 (20일 가중평균가격) — 기관 매매 기준점
    if len(close) >= 20 and len(vol) >= 20:
        recent_vol_20 = vol.tail(20).values.astype(float)
        recent_close_20 = close.tail(20).values.astype(float)
        total_vol = np.nansum(recent_vol_20)
        if total_vol > 0:
            vwap_20 = float(np.nansum(recent_close_20 * recent_vol_20) / total_vol)
            if vwap_20 > 0:
                vtype = "support" if vwap_20 < last_close else "resistance"
                sr_levels.append((vwap_20, vtype, 0.5))

    # ══════════════════════════════════════════════════════════════════
    # ❼ GARCH(1,1) 파라미터 — 레짐별 적응
    # ══════════════════════════════════════════════════════════════════
    if is_strong_trend:
        garch_alpha, garch_beta = 0.10, 0.85  # 추세장: 관성 강함
    elif is_mean_reverting:
        garch_alpha, garch_beta = 0.12, 0.80  # 횡보장: 충격 반응 더 큼
    else:
        garch_alpha, garch_beta = 0.08, 0.90  # 기본

    # ══════════════════════════════════════════════════════════════════
    # ❽ 몬테카를로 앙상블 — N_PATHS개 경로 생성
    # ══════════════════════════════════════════════════════════════════

    # 시드: 종가 + 날짜 해시 → 매일 다른 시나리오 (v2와 동일 결정론)
    date_hash = sum(ord(c) for c in last_date)
    base_seed = (int(last_close * 100) + date_hash) % 2**31

    # 영업일 캘린더 미리 계산
    business_days: list[date] = []
    d = date.fromisoformat(last_date)
    while len(business_days) < n_days:
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        business_days.append(d)

    # 모든 경로 저장: [path_idx][day_idx] = (open, high, low, close)
    all_paths = np.zeros((N_PATHS, n_days, 4))  # O, H, L, C

    for path_idx in range(N_PATHS):
        rng = np.random.default_rng(seed=(base_seed + path_idx * 7919) % 2**31)
        prev_close_p = last_close
        trend_streak = 0
        running_sigma = base_sigma * vol_persistence
        breakout_mode = has_breakout_pullback

        for i in range(n_days):
            progress = i / max(1, n_days - 1)

            # ── 변동성 감쇠/성장 ──
            drift_decay = max(0.2, 1.0 - progress * 0.6)
            sigma_growth = 1.0 + progress * 0.15

            if is_squeeze:
                squeeze_phase = min(1.0, i / max(3, n_days * 0.25))
                sigma_scale = 0.3 + squeeze_phase * 1.5
            else:
                sigma_scale = sigma_growth

            sigma_today = min(running_sigma * sigma_scale, daily_ret_cap * 0.8)

            # ── 패턴 drift (초기 N일에만 적용) ──
            pattern_factor = pattern_drift * max(0, 1.0 - i / pattern_decay_days) if i < pattern_decay_days else 0

            # ── 지지/저항 + 피보나치 자석 효과 ──
            sr_pull = 0.0
            for level, stype, strength in sr_levels:
                dist_pct = (level - prev_close_p) / prev_close_p
                if abs(dist_pct) < 0.10:
                    if stype == "support" and dist_pct < 0:
                        sr_pull += abs(dist_pct) * 0.15 * strength
                    elif stype == "resistance" and dist_pct > 0:
                        resist_str = 0.04 if breakout_mode else 0.12
                        sr_pull -= abs(dist_pct) * resist_str * strength
                    elif stype == "support" and 0 < dist_pct < 0.03:
                        sr_pull -= dist_pct * 0.05 * strength
                    elif stype == "resistance" and -0.03 < dist_pct < 0:
                        sr_pull += abs(dist_pct) * 0.05 * strength

            # ── 다중 SMA 평균 회귀 ──
            mean_revert = 0.0
            if sma20_val and sma20_val > 0:
                dist_sma20 = (prev_close_p - sma20_val) / sma20_val
                if is_mean_reverting:
                    mean_revert += -dist_sma20 * 0.10
                elif not is_trending:
                    mean_revert += -dist_sma20 * 0.05
            if n_days >= 14 and sma60_val and sma60_val > 0:
                dist_sma60 = (prev_close_p - sma60_val) / sma60_val
                weight_60 = 0.06 if is_mean_reverting else 0.03
                mean_revert += -dist_sma60 * weight_60
            if n_days >= 21 and sma120_val and sma120_val > 0:
                dist_sma120 = (prev_close_p - sma120_val) / sma120_val
                weight_120 = 0.04 if is_mean_reverting else 0.02
                mean_revert += -dist_sma120 * weight_120

            # ── ADX 기반 추세 지속력 보정 ──
            adx_boost = 0.0
            if is_strong_trend and adx_val >= 30:
                adx_strength = min(1.0, (adx_val - 25) / 25)  # 25→0, 50→1
                adx_boost = trend_direction * 0.002 * adx_strength * drift_decay

            # ── 조정 패턴 ──
            if abs(trend_streak) >= 3 and rng.random() < 0.55:
                correction = -np.sign(trend_streak) * abs(daily_mu) * 0.4
                effective_mu = correction
                trend_streak = 0
            else:
                effective_mu = (daily_mu * drift_decay + sr_pull + mean_revert
                                + adx_boost + pattern_factor)

            # ── 수익률 생성 (레버리지 효과 반영) ──
            z = rng.normal(0, 1)
            # 하락 방향 시 변동성 비대칭 증폭
            if z < 0:
                sigma_eff = sigma_today * leverage_ratio
            else:
                sigma_eff = sigma_today

            ret_close = effective_mu + sigma_eff * z
            ret_close = max(-daily_ret_cap, min(daily_ret_cap, ret_close))

            projected = prev_close_p * (1 + ret_close)
            upper_bound = last_close * (1 + max_total_deviation)
            lower_bound = last_close * (1 - max_total_deviation)
            if projected > upper_bound:
                ret_close = (upper_bound / prev_close_p) - 1
            elif projected < lower_bound:
                ret_close = (lower_bound / prev_close_p) - 1

            # GARCH(1,1) 갱신
            running_sigma = garch_beta * running_sigma + garch_alpha * abs(ret_close)
            running_sigma = min(running_sigma, base_sigma * 2.5)

            if ret_close > 0:
                trend_streak = max(1, trend_streak + 1)
            else:
                trend_streak = min(-1, trend_streak - 1)

            if breakout_mode and i >= 5:
                breakout_mode = False

            # ── 캔들 형성 ──
            c = prev_close_p * (1 + ret_close)
            gap_scale = 0.15 + (n_days - 7) / 23 * 0.10
            gap_ret = rng.normal(0, 1) * sigma_today * gap_scale
            gap_ret = max(-daily_ret_cap * 0.5, min(daily_ret_cap * 0.5, gap_ret))
            o = prev_close_p * (1 + gap_ret)

            body = abs(c - o)
            min_body = max(atr_val * 0.15, last_close * 0.003) if is_stable else atr_val * 0.15
            if body < min_body:
                direction = 1.0 if c >= o else -1.0
                c = o + direction * min_body

            body_top = max(o, c)
            body_bot = min(o, c)
            wick_scale = min(sigma_scale, 1.5)
            upper_wick = rng.exponential(0.3) * atr_val * 0.15 * wick_scale
            lower_wick = rng.exponential(0.3) * atr_val * 0.12 * wick_scale
            h = body_top + upper_wick
            lo = body_bot - lower_wick
            h = min(h, last_close * (1 + max_total_deviation * 1.05))
            lo = max(lo, last_close * (1 - max_total_deviation * 1.05), 1)

            all_paths[path_idx, i] = [max(1, o), max(1, h), max(1, lo), max(1, c)]
            prev_close_p = c

    # ══════════════════════════════════════════════════════════════════
    # ❾ 앙상블 집계: 대표 경로 선택 (Representative Path)
    #
    # element-wise median은 각 날짜마다 서로 다른 경로에서 값을 취하므로
    # 랜덤 움직임이 서로 상쇄되어 밋밋한 직선이 됨.
    # 대신, 50개 경로 중 "최종 종가가 중앙값에 가장 가까운 단일 경로"를 채택하면
    # 실제 차트처럼 자연스러운 등락이 살아있으면서도 통계적으로 대표성 있는 예측이 됨.
    # ══════════════════════════════════════════════════════════════════

    # 각 경로의 최종 종가 (마지막 날 close)
    final_closes = all_paths[:, -1, 3]  # (N_PATHS,)
    median_final = float(np.median(final_closes))

    # 중앙값에 가장 가까운 경로 선택
    distances = np.abs(final_closes - median_final)
    best_path_idx = int(np.argmin(distances))

    candles = []
    for i in range(n_days):
        o, h, lo, c = all_paths[best_path_idx, i]
        candles.append({
            "time": business_days[i].isoformat(),
            "open": max(1, round(o)),
            "high": max(1, round(max(o, c, h))),
            "low": max(1, round(min(o, c, lo))),
            "close": max(1, round(c)),
        })

    return candles


def _detect_box_range(df: pd.DataFrame) -> dict:
    """
    박스권 탐지 조건(요청 사양):
    - 최근 40거래일 기준(40 미만이면 박스권 없음)
    - 최고 high, 최저 low
    - 진폭비율 = (top - bottom) / bottom
    - 진폭비율 <= 0.15 이고, 현재 종가가 [bottom, top] 사이면 박스권
    """
    if df is None or len(df) < 40:
        return {"is_box": False}

    last40 = df.tail(40)
    highs = pd.to_numeric(last40["high"], errors="coerce")
    lows = pd.to_numeric(last40["low"], errors="coerce")
    closes = pd.to_numeric(last40["close"], errors="coerce")
    if highs.dropna().empty or lows.dropna().empty or closes.dropna().empty:
        return {"is_box": False}

    top = float(highs.max())
    bottom = float(lows.min())
    if bottom <= 0:
        return {"is_box": False}

    amplitude = (top - bottom) / bottom
    current_close = float(closes.iloc[-1])
    if amplitude <= 0.15 and (bottom <= current_close <= top):
        return {"is_box": True, "top": top, "bottom": bottom}
    return {"is_box": False}


def _fetch_ohlcv_fdr(start_iso: str, end_iso: str, ticker: str) -> pd.DataFrame:
    """FinanceDataReader로 OHLCV 조회 (1차 소스).

    fdr은 pykrx보다 안정적이고 긴 기간 요청에서도 데이터 누락이 없음.
    반환 형식: DatetimeIndex + Open, High, Low, Close, Volume, Change (6열)
    → _standardize_ohlcv 호환 (reset_index 시 7열)
    """
    df = fdr.DataReader(ticker, start_iso, end_iso)
    if df is None or df.empty:
        return pd.DataFrame()
    # fdr 컬럼 → pykrx 호환 컬럼명으로 변환
    rename_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl == "open": rename_map[col] = "시가"
        elif cl == "high": rename_map[col] = "고가"
        elif cl == "low": rename_map[col] = "저가"
        elif cl == "close": rename_map[col] = "종가"
        elif cl == "volume": rename_map[col] = "거래량"
        elif cl == "change": rename_map[col] = "등락률"
    if rename_map:
        df = df.rename(columns=rename_map)
    # 필수 컬럼 누락 시 빈 DataFrame
    for req in ["시가", "고가", "저가", "종가", "거래량"]:
        if req not in df.columns:
            return pd.DataFrame()
    # 등락률 없으면 0으로 채움 (7열 맞추기)
    if "등락률" not in df.columns:
        df["등락률"] = 0.0
    return df[["시가", "고가", "저가", "종가", "거래량", "등락률"]]


def _fetch_ohlcv_pykrx(start: str, end: str, ticker: str) -> pd.DataFrame:
    """pykrx로 OHLCV 조회 (폴백) — 3개월 단위 분할 요청."""
    start_d = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    end_d = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    total_days = (end_d - start_d).days

    if total_days <= 90:
        return stock.get_market_ohlcv(start, end, ticker)

    chunks: list[pd.DataFrame] = []
    chunk_start = start_d
    while chunk_start <= end_d:
        chunk_end = min(chunk_start + timedelta(days=89), end_d)
        s = _yyyymmdd(chunk_start)
        e = _yyyymmdd(chunk_end)
        try:
            chunk_df = stock.get_market_ohlcv(s, e, ticker)
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
        except Exception as exc:
            logger.warning("pykrx 분할 조회 실패 (%s~%s, %s): %s", s, e, ticker, exc)
        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        return pd.DataFrame()
    combined = pd.concat(chunks)
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.sort_index()


def _fetch_ohlcv_sync(start: str, end: str, ticker: str) -> pd.DataFrame:
    """OHLCV 조회 — fdr 1차 → pykrx 폴백.

    fdr(FinanceDataReader)이 긴 기간 데이터에서 훨씬 안정적이므로 1차 소스로 사용.
    fdr 실패 시 pykrx 3개월 분할 조회로 폴백.
    """
    start_iso = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_iso = f"{end[:4]}-{end[4:6]}-{end[6:8]}"
    expected_days = (date(int(end[:4]), int(end[4:6]), int(end[6:8]))
                     - date(int(start[:4]), int(start[4:6]), int(start[6:8]))).days
    expected_trading_days = max(10, int(expected_days * 5 / 7 * 0.6))  # 보수적 기대치

    # ── 1차: FinanceDataReader ──
    try:
        df_fdr = _fetch_ohlcv_fdr(start_iso, end_iso, ticker)
        if df_fdr is not None and not df_fdr.empty and len(df_fdr) >= expected_trading_days:
            logger.info("fdr OHLCV 조회 성공 (%s): %d행", ticker, len(df_fdr))
            return df_fdr
        elif df_fdr is not None and not df_fdr.empty:
            logger.info("fdr OHLCV 부분 성공 (%s): %d행 (기대 %d+), pykrx 보충 시도",
                        ticker, len(df_fdr), expected_trading_days)
    except Exception as e:
        logger.warning("fdr OHLCV 실패 (%s): %s", ticker, e)

    # ── 2차: pykrx (분할 조회) ──
    try:
        df_pykrx = _fetch_ohlcv_pykrx(start, end, ticker)
        if df_pykrx is not None and not df_pykrx.empty:
            logger.info("pykrx OHLCV 조회 성공 (%s): %d행", ticker, len(df_pykrx))
            # fdr 부분 결과가 있었으면 병합
            if 'df_fdr' in dir() and df_fdr is not None and not df_fdr.empty:
                combined = pd.concat([df_fdr, df_pykrx])
                combined = combined[~combined.index.duplicated(keep="first")]
                logger.info("fdr+pykrx 병합 결과: %d행", len(combined))
                return combined.sort_index()
            return df_pykrx
    except Exception as e:
        logger.warning("pykrx OHLCV도 실패 (%s): %s", ticker, e)

    # fdr 부분 결과라도 반환
    if 'df_fdr' in dir() and df_fdr is not None and not df_fdr.empty:
        return df_fdr

    return pd.DataFrame()


async def _fetch_ohlcv(start: str, end: str, ticker: str) -> pd.DataFrame:
    """OHLCV 조회 비동기 래퍼 (fdr→pykrx 폴백 포함)"""
    total_days = (date(int(end[:4]), int(end[4:6]), int(end[6:8]))
                  - date(int(start[:4]), int(start[4:6]), int(start[6:8]))).days
    n_chunks = max(1, total_days // 90 + 1)
    timeout = max(PYKRX_TIMEOUT_SEC, PYKRX_TIMEOUT_SEC + n_chunks * 10)
    return await _run_with_timeout(_fetch_ohlcv_sync, start, end, ticker, timeout=timeout)


def _load_stock_for_score_sync(ticker: str, end_day: str) -> tuple[pd.DataFrame, list[float], list[float], int, dict]:
    """추천 루프에서 사용하는 동기 버전 (ThreadPool에서 호출)"""
    end_d = date.fromisoformat(f"{end_day[:4]}-{end_day[4:6]}-{end_day[6:8]}")
    start_d = end_d - timedelta(days=365)  # 365일: OBV·120일SMA·224일SMA에 충분한 데이터
    raw = _fetch_ohlcv_sync(_yyyymmdd(start_d), end_day, ticker)
    if raw.empty:
        raise ValueError("no data")
    df = _standardize_ohlcv(raw)
    # 유동성 필터: 최근 20일 평균 거래량 < 10,000이면 분석 무의미
    vol_series = pd.to_numeric(df["volume"], errors="coerce")
    if len(vol_series) >= 20 and float(vol_series.tail(20).mean()) < 10000:
        raise ValueError("low volume — skip")
    close_values = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=float)
    high_values = pd.to_numeric(df["high"], errors="coerce").dropna().to_numpy(dtype=float)
    low_values = pd.to_numeric(df["low"], errors="coerce").dropna().to_numpy(dtype=float)
    vol_values = pd.to_numeric(df["volume"], errors="coerce").dropna().to_numpy(dtype=float)
    support, resistance = _support_resistance(close_values, high_values, low_values, vol_values, max_lines=1)
    box = _detect_box_range(df)
    score, _signals, _internals = _unified_score(df, support, resistance, box)
    return df, support, resistance, score, {}


@app.get("/api/search")
async def search_stocks(q: str = Query("", min_length=1, max_length=50)):
    """
    종목 자동완성 검색 API.
    입력어에 매칭되는 종목을 최대 8개 반환.
    코드 검색, 정확 이름, 부분 매칭, 정제 매칭 순서.
    """
    import re
    query = q.strip()
    if not query:
        return {"results": []}

    try:
        # 캐시가 있으면 즉시 반환, 없으면 ThreadPool에서 로드
        cached = _LISTING_CACHE["data"]
        if cached is not None:
            listing = cached
        else:
            listing = await _run_with_timeout(_load_listing)
    except Exception:
        return {"results": []}

    results: list[dict] = []

    # 1. 숫자 → 코드 부분 매칭
    if query.isdigit():
        mask = listing["ticker"].str.startswith(query)
        for _, row in listing[mask].head(8).iterrows():
            results.append({"ticker": str(row["ticker"]), "name": str(row["name"])})
        return {"results": results}

    names = listing["name"]

    # 2. 정확 매칭 (대소문자 무시)
    q_upper = query.upper()
    exact_mask = names.str.upper() == q_upper
    if exact_mask.any():
        for _, row in listing[exact_mask].head(2).iterrows():
            results.append({"ticker": str(row["ticker"]), "name": str(row["name"])})

    # 3. 부분 매칭 (대소문자 무시)
    q_escaped = re.escape(query)
    partial_mask = names.str.contains(q_escaped, na=False, case=False)
    if partial_mask.any():
        matches = listing[partial_mask].copy()
        matches["_len"] = matches["name"].str.len()
        matches = matches.sort_values("_len")
        for _, row in matches.head(8).iterrows():
            item = {"ticker": str(row["ticker"]), "name": str(row["name"])}
            if item not in results:
                results.append(item)
            if len(results) >= 8:
                break

    # 4. 공백/특수문자 제거 후 매칭
    if len(results) < 8:
        q_clean = re.sub(r'[\s\-\_\.\&]', '', query).upper()
        names_clean = names.str.replace(r'[\s\-\_\.\&]', '', regex=True).str.upper()
        clean_mask = names_clean.str.contains(re.escape(q_clean), na=False)
        if clean_mask.any():
            matches = listing[clean_mask].copy()
            matches["_len"] = matches["name"].str.len()
            matches = matches.sort_values("_len")
            for _, row in matches.head(8).iterrows():
                item = {"ticker": str(row["ticker"]), "name": str(row["name"])}
                if item not in results:
                    results.append(item)
                if len(results) >= 8:
                    break

    return {"results": results[:8]}


@app.get("/")
def read_root():
    return {"message": "주식 분석 API 서버가 정상적으로 실행 중입니다."}


@app.get("/api/ping")
def ping():
    """Render 무료 플랜 cold start 해결용: 프론트엔드가 앱 진입 시 이 엔드포인트로 서버를 미리 깨움"""
    return {"status": "ok"}


@app.get("/api/stock/{ticker_or_name}")
async def get_stock_data(
    ticker_or_name: str,
    before_date: str | None = Query(None, description="YYYY-MM-DD — 이 날짜 이전 데이터 (스크롤 추가 로드용)"),
    days_back: int = Query(365, ge=30, le=730, description="조회할 일수 (기본 365)"),
):
    """
    OHLCV 데이터 + 점수 + 지지/저항선 조회.

    두 가지 모드:
    1. 초기 조회 (before_date 없음): 오늘 기준 최근 days_back일 데이터 + 점수/분석 반환
    2. 스크롤 추가 로드 (before_date 있음): before_date 이전 days_back일 데이터만 반환 (점수 없음)
    """
    try:
        ticker, stock_name = _resolve_ticker(ticker_or_name)

        if before_date:
            # ── 스크롤 추가 로드 모드: 지정 날짜 이전 데이터만 반환 ──
            try:
                end_d = date.fromisoformat(before_date.strip())
            except ValueError:
                return {"error": f"before_date 형식 오류: {before_date} (YYYY-MM-DD 필요)"}
            start_d = end_d - timedelta(days=days_back)
            start = _yyyymmdd(start_d)
            end = _yyyymmdd(end_d)

            raw = await _fetch_ohlcv(start, end, ticker)
            if raw.empty:
                return {"ticker": ticker, "stock_name": stock_name, "data": []}

            df = _standardize_ohlcv(raw)
            df = _clean_ohlcv(df)
            # before_date 이전(이하) 데이터만 반환
            df = df[df["time"] <= before_date.strip()]
            df = df.sort_values("time").reset_index(drop=True)
            data = df[["time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")

            return {
                "ticker": ticker,
                "stock_name": stock_name,
                "data": data,
            }
        else:
            # ── 초기 조회 모드: 최근 365일 + 점수/분석 ──
            end_d = date.today()
            start_d = end_d - timedelta(days=days_back)
            start = _yyyymmdd(start_d)
            end = _yyyymmdd(end_d)

            raw = await _fetch_ohlcv(start, end, ticker)
            if raw.empty:
                return {"error": "데이터가 없거나 종목 코드/이름이 잘못되었습니다."}

            df = _standardize_ohlcv(raw)
            df = _clean_ohlcv(df)
            df = df.sort_values("time").reset_index(drop=True)
            data = df[["time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")

            close_values = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=float)
            high_values = pd.to_numeric(df["high"], errors="coerce").dropna().to_numpy(dtype=float)
            low_values = pd.to_numeric(df["low"], errors="coerce").dropna().to_numpy(dtype=float)
            vol_values = pd.to_numeric(df["volume"], errors="coerce").dropna().to_numpy(dtype=float)
            support_lines, resistance_lines = _support_resistance(close_values, high_values, low_values, vol_values, max_lines=1)
            box_range = _detect_box_range(df)
            score, _signals, _internals = _unified_score(df, support_lines, resistance_lines, box_range)
            fibonacci = _compute_fibonacci_levels(df)
            ichimoku  = _compute_ichimoku(df)

            return {
                "ticker": ticker,
                "stock_name": stock_name,
                "data": data,
                "support_lines": support_lines,
                "resistance_lines": resistance_lines,
                "score": score,
                "score_breakdown": {},
                "box_range": box_range,
                "fibonacci": fibonacci,
                "ichimoku": ichimoku,
            }
    except asyncio.TimeoutError:
        logger.warning("주가 데이터 조회 타임아웃: %s", ticker_or_name)
        return {"error": f"주가 데이터 조회 시간 초과 ({PYKRX_TIMEOUT_SEC}초). 잠시 후 다시 시도해주세요."}
    except Exception as e:
        logger.exception("데이터 조회 실패")
        return {"error": f"데이터 조회 중 오류 발생: {str(e)}"}


def _compute_recommendations_sync(day: str, listing: pd.DataFrame, sample_size: int) -> list[dict]:
    """추천 종목 스코어링 — 시가총액 기반 필터링 + 바닥 반등 종목 발굴"""

    # ── 1. 시가총액 + 거래대금으로 유의미한 종목만 선별 ──
    cap_tickers: set[str] | None = None
    try:
        market_cap = stock.get_market_cap_by_ticker(day)
        if not market_cap.empty and "시가총액" in market_cap.columns:
            # 시가총액 500억 이상
            valid = market_cap[market_cap["시가총액"] >= 50_000_000_000]
            # 거래대금 20억 이상 (유동성 부족 종목 제거 — 실매매 불가)
            if "거래대금" in market_cap.columns:
                valid = valid[valid["거래대금"] >= 2_000_000_000]
            cap_tickers = set(valid.index.astype(str).str.zfill(6).tolist())
            logger.info("시가총액+거래대금 필터: %d → %d종목", len(market_cap), len(cap_tickers))
    except Exception as ex:
        logger.warning("시가총액 조회 실패 (리스팅 순서로 fallback): %s", ex)

    # ── 2. 리스팅과 교차 → 유효 종목 목록 ──
    all_tickers = listing["ticker"].tolist()
    if cap_tickers:
        filtered = [t for t in all_tickers if t in cap_tickers]
    else:
        filtered = all_tickers

    sample = filtered[:sample_size]
    logger.info("추천 스캔 대상: %d종목", len(sample))

    # ── 3. 개별 종목 스코어링 ──
    ranked = []
    for tk in sample:
        try:
            _df, support, resistance, score, breakdown = _load_stock_for_score_sync(tk, day)
            ranked.append(
                {
                    "ticker": tk,
                    "stock_name": str(listing.loc[listing["ticker"] == tk, "name"].iloc[0]) if (listing["ticker"] == tk).any() else tk,
                    "score": score,
                    "support_lines": support,
                    "resistance_lines": resistance,
                    "score_breakdown": breakdown,
                }
            )
        except Exception as ex:
            logger.warning("종목 로드 실패 %s: %s", tk, ex)
            continue

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


@app.get("/api/recommendations")
async def get_recommendations(limit: int = Query(10, ge=1, le=50)):
    """
    스코어 기반 추천 Top N.
    성능을 위해 코스피/코스닥 종목 중 앞쪽 N개를 샘플링해서 평가합니다.
    (N = 환경변수 RECOMMEND_SAMPLE_SIZE, 기본값 220)
    최근 계산 결과를 1시간(3600초) 동안 메모리에 캐싱하여 응답 속도를 높입니다.
    """
    global RECOMMEND_CACHE
    now = datetime.now()

    async with _cache_lock:
        # 1. 캐시 확인
        if RECOMMEND_CACHE["data"] is not None and RECOMMEND_CACHE["last_updated"] is not None:
            if (now - RECOMMEND_CACHE["last_updated"]).total_seconds() < 3600:
                cached_ranked = RECOMMEND_CACHE["data"]
                return {
                    "as_of": RECOMMEND_CACHE["as_of"],
                    "count": len(cached_ranked),
                    "top": cached_ranked[:limit],
                    "cached": True,
                    "last_updated": RECOMMEND_CACHE["last_updated"].strftime("%Y-%m-%d %H:%M:%S"),
                }

    # 2. 캐시가 없거나 만료 → 새로 계산 (lock 밖에서 실행해 다른 요청 차단 방지)
    try:
        day = await _best_business_day()
        listing = _load_listing()
        ranked = await _run_with_timeout(
            _compute_recommendations_sync, day, listing, RECOMMEND_SAMPLE_SIZE,
            timeout=PYKRX_TIMEOUT_SEC * RECOMMEND_SAMPLE_SIZE // 10,  # 전체 계산은 넉넉하게
        )

        async with _cache_lock:
            RECOMMEND_CACHE["data"] = ranked
            RECOMMEND_CACHE["last_updated"] = now
            RECOMMEND_CACHE["as_of"] = day

        return {
            "as_of": day,
            "count": len(ranked),
            "top": ranked[:limit],
            "cached": False,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except asyncio.TimeoutError:
        logger.warning("추천 계산 타임아웃")
        return {"error": "추천 종목 계산 시간 초과. 잠시 후 다시 시도해주세요."}
    except Exception as e:
        logger.exception("추천 계산 실패")
        return {"error": f"추천 계산 중 오류 발생: {str(e)}"}
    

@app.get("/api/stock/{ticker_or_name}/predict")
async def predict_stock(
    ticker_or_name: str,
    n_days: int = Query(7, ge=1, le=30, description="예측할 영업일 수 (1~30)"),
):
    try:
        try:
            ticker, stock_name = _resolve_ticker(ticker_or_name)
        except Exception:
            ticker = ticker_or_name
            stock_name = ticker_or_name

        end_d = date.today()
        start_d = end_d - timedelta(days=365)
        start_str = _yyyymmdd(start_d)
        end_str = _yyyymmdd(end_d)

        raw = await _fetch_ohlcv(start_str, end_str, ticker)
        if raw.empty:
            return {"error": "데이터 없음"}

        df = _standardize_ohlcv(raw)
        df = _clean_ohlcv(df)
        close = pd.to_numeric(df["close"], errors="coerce")
        high  = pd.to_numeric(df["high"], errors="coerce")
        cur   = float(close.iloc[-1])

        # ── 통합 점수 계산 (기본 점수와 동일한 함수) ──
        close_values = close.dropna().to_numpy(dtype=float)
        high_values = pd.to_numeric(df["high"], errors="coerce").dropna().to_numpy(dtype=float)
        low_values = pd.to_numeric(df["low"], errors="coerce").dropna().to_numpy(dtype=float)
        vol_values = pd.to_numeric(df["volume"], errors="coerce").dropna().to_numpy(dtype=float)
        support_lines, resistance_lines = _support_resistance(close_values, high_values, low_values, vol_values, max_lines=1)
        box_range = _detect_box_range(df)
        final_score, signals, internals = _unified_score(df, support_lines, resistance_lines, box_range)

        # internals에서 지표값 추출
        atr_val = internals["atr14"]
        is_squeeze = internals["is_squeeze"]
        staircase_signal = internals["staircase_signal"]
        has_accumulation = internals.get("accumulation", False)
        has_obv_div = internals.get("obv_divergence", False)

        # ── 목표가 / 손절가 — 지지선·저항선·BB·피보나치 기반 ──
        sell_short_price = None
        sell_long_price = None
        stop_loss_price = None
        sell_short_desc = None
        sell_long_desc = None
        stop_loss_desc = None

        bb_up_val = internals.get("bb_upper")
        bb_dn_val = internals.get("bb_lower")
        sma20_val = internals.get("sma20")

        # 최근 60/120일 고점·저점 (피보나치 되돌림 기준)
        if len(high) >= 60:
            high_60 = float(high.tail(60).max())
            low_60  = float(pd.to_numeric(df["low"], errors="coerce").tail(60).min())
        else:
            high_60 = cur * 1.1
            low_60  = cur * 0.9

        if len(high) >= 120:
            high_120 = float(high.tail(120).max())
            low_120  = float(pd.to_numeric(df["low"], errors="coerce").tail(120).min())
        else:
            high_120 = high_60
            low_120  = low_60

        fib_range = high_120 - low_120
        fib_382 = low_120 + fib_range * 0.382  # 되돌림 38.2%
        fib_618 = low_120 + fib_range * 0.618  # 되돌림 61.8%

        if final_score >= 65:
            # 강한 매수 신호 → 넉넉한 목표가
            # 단기: 저항선, BB상단, 60일 고점 중 현재가보다 높은 첫 번째
            short_candidates = [p for p in [
                resistance_lines[0] if resistance_lines else None,
                bb_up_val,
                fib_618 if fib_618 > cur else None,
                int(cur + atr_val * 1.5),
            ] if p is not None and p > cur * 1.01]

            if short_candidates:
                sell_short_price = int(min(short_candidates))
                pct = (sell_short_price - cur) / cur * 100
                sell_short_desc = f"1차 매도 목표 (+{pct:.1f}%)"

            # 장기: 120일 고점, 피보나치 확장
            long_candidates = [p for p in [
                high_60,
                high_120,
                int(cur + atr_val * 3.0),
            ] if p > cur * 1.03]

            if long_candidates:
                sell_long_price = int(max(long_candidates))
                pct = (sell_long_price - cur) / cur * 100
                sell_long_desc = f"최종 매도 목표 (+{pct:.1f}%)"

            # 손절: 지지선 or BB하단 or 직전 저점 아래
            stop_candidates = [p for p in [
                support_lines[0] if support_lines else None,
                bb_dn_val,
                int(cur - atr_val * 1.5),
                low_60 if low_60 < cur else None,
            ] if p is not None and 0 < p < cur * 0.99]

            if stop_candidates:
                stop_loss_price = int(max(stop_candidates))  # 가장 가까운 지지선
                pct = (cur - stop_loss_price) / cur * 100
                stop_loss_desc = f"이 가격 아래로 떨어지면 매도 ({pct:.1f}% 하락)"

        elif final_score >= 45:
            # 관망 → 보수적 목표가
            sell_short_price = int(cur + atr_val * 0.8) if cur + atr_val * 0.8 > cur else None
            if sell_short_price:
                pct = (sell_short_price - cur) / cur * 100
                sell_short_desc = f"반등 시 부분 매도 (+{pct:.1f}%)"

            stop_loss_price = int(cur - atr_val * 1.0)
            pct = (cur - stop_loss_price) / cur * 100
            stop_loss_desc = f"하락 리스크 관리 ({pct:.1f}% 하락)"

        else:
            # 약세 → 손절만
            if has_accumulation or has_obv_div:
                # 매집 신호가 있으면 약간의 목표가
                sell_short_price = int(cur + atr_val * 0.5)
                pct = (sell_short_price - cur) / cur * 100
                sell_short_desc = f"매집 신호 감안 반등 목표 (+{pct:.1f}%)"

            stop_loss_price = int(cur - atr_val * 0.8)
            pct = (cur - stop_loss_price) / cur * 100
            stop_loss_desc = f"보유 중이라면 즉시 매도 고려 ({pct:.1f}% 하락)"

        # ── 전망 — 점수 구간 세분화 + 시그널 기반 요약 ──
        positive_count = sum(1 for s in signals if s["type"] == "positive")
        negative_count = sum(1 for s in signals if s["type"] == "negative")

        if final_score >= 75:
            outlook_short = "강한 매수 신호"
            outlook_mid   = "상승 추세 기대"
            summary = "여러 지표가 동시에 상승을 가리키고 있어요. 적극적인 매수를 고려해볼 만합니다."
        elif final_score >= 60:
            outlook_short = "매수 고려"
            outlook_mid   = "반등 가능성 높음"
            if has_accumulation or has_obv_div:
                summary = "바닥에서 매수세가 들어오고 있어요. 분할 매수를 고려해보세요."
            else:
                summary = "상승 신호가 우세해요. 지지선 부근에서 매수하면 좋은 자리일 수 있어요."
        elif final_score >= 50:
            outlook_short = "관망"
            outlook_mid   = "방향성 탐색 중"
            summary = "아직 뚜렷한 방향이 없어요. 거래량 변화와 이평선 돌파 여부를 지켜보세요."
        elif final_score >= 35:
            outlook_short = "약세 주의"
            outlook_mid   = "추가 하락 가능"
            if has_accumulation:
                summary = "하락 중이지만 바닥에서 거래량이 보여요. 조금 더 지켜보세요."
            else:
                summary = "하락 압력이 있어요. 보유 중이라면 손절 라인을 꼭 지키세요."
        else:
            outlook_short = "매도 권고"
            outlook_mid   = "하락 추세 강함"
            summary = f"하락 신호가 {negative_count}개로 많아요. 보유를 줄이거나 관망하세요."

        # ── 예측 캔들 생성 ──
        predicted_candles = _generate_predicted_candles(
            df=df,
            prediction_score=final_score,
            internals=internals,
            support_lines=support_lines,
            resistance_lines=resistance_lines,
            box_range=box_range,
            n_days=n_days,
        )

        return {
            "ticker": ticker,
            "stock_name": stock_name,
            "current_price": int(cur),
            "prediction_score": final_score,
            "outlook_short": outlook_short,
            "outlook_mid": outlook_mid,
            "summary": summary,
            "signals": signals,
            "sell_targets": {
                "short_term": sell_short_price,
                "long_term": sell_long_price,
                "stop_loss": stop_loss_price,
                "short_term_desc": sell_short_desc,
                "long_term_desc": sell_long_desc,
                "stop_loss_desc": stop_loss_desc,
            },
            "predicted_candles": predicted_candles,
        }

    except asyncio.TimeoutError:
        logger.warning("예측 데이터 조회 타임아웃: %s", ticker_or_name)
        return {"error": f"예측 데이터 조회 시간 초과 ({PYKRX_TIMEOUT_SEC}초). 잠시 후 다시 시도해주세요."}
    except Exception as e:
        logger.exception("예측 실패")
        return {"error": f"예측 중 오류 발생: {str(e)}"}
