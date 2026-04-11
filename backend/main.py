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
    listing = fdr.StockListing("KRX")
    if listing is None or listing.empty:
        raise ValueError("종목 리스트를 가져오지 못했습니다.")
    symbol_col = "Code" if "Code" in listing.columns else "Symbol"
    if symbol_col not in listing.columns or "Name" not in listing.columns:
        raise ValueError("종목 리스트 컬럼 형식이 예상과 다릅니다.")
    out = listing[[symbol_col, "Name"]].copy()
    out.columns = ["ticker", "name"]
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    out["name"] = out["name"].astype(str)
    return out.dropna().drop_duplicates(subset=["ticker"])


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


def _cluster_by_price(candidates: list[tuple[float, float]], max_lines: int, tolerance_pct: float) -> list[float]:
    out: list[float] = []
    for price, _prom in sorted(candidates, key=lambda x: x[1], reverse=True):
        keep = True
        for existing in out:
            if existing != 0 and abs(price - existing) / existing <= tolerance_pct:
                keep = False
                break
        if keep:
            out.append(float(price))
        if len(out) >= max_lines:
            break
    return out


def _support_resistance(close_values: np.ndarray, max_lines: int = 1) -> tuple[list[float], list[float]]:
    # 1. 60일 고정을 없애고 전달받은 데이터 전체 길이를 사용합니다.
    lookback = len(close_values)
    if lookback < 5:
        return [], []

    # 2. 최근 60일 자르기(-lookback:)를 제거하고 전체(recent)를 사용합니다.
    recent = close_values
    y_range = float(np.max(recent) - np.min(recent))
    mean_val = float(np.mean(recent))
    
    # 3. 조회 기간(lookback)이 길어질수록 자잘한 파동을 무시하도록 동적 스케일링
    if lookback < 20:
        prominence = max(y_range * 0.001, mean_val * 0.0002)
        distance = 1
    else:
        # 전체 최고-최저 변동폭의 약 3% 이상(또는 평균가의 0.5% 이상) 큰 움직임만 유의미한 봉우리로 취급
        prominence = max(y_range * 0.03, mean_val * 0.005)
        # 캔들 개수에 비례하여 최소 간격을 넓힘 (전체 기간의 약 1/20 간격마다 하나씩)
        distance = max(3, lookback // 20)

    peaks_idx, peaks_props = find_peaks(recent, prominence=prominence, distance=distance)
    trough_idx, trough_props = find_peaks(-recent, prominence=prominence, distance=distance)
    peak_proms = peaks_props.get("prominences", np.ones(len(peaks_idx), dtype=float))
    trough_proms = trough_props.get("prominences", np.ones(len(trough_idx), dtype=float))

    resistance_candidates = [(float(recent[i]), float(p)) for i, p in zip(peaks_idx, peak_proms)]
    support_candidates = [(float(recent[i]), float(p)) for i, p in zip(trough_idx, trough_proms)]

    if not resistance_candidates and len(recent) > 0:
        resistance_candidates = [(float(np.max(recent)), 0.0)]
    if not support_candidates and len(recent) > 0:
        support_candidates = [(float(np.min(recent)), 0.0)]

    # _cluster_by_price 함수는 기존에 만드신 것을 그대로 사용합니다.
    resistance = sorted(_cluster_by_price(resistance_candidates, max_lines=max_lines, tolerance_pct=0.005), reverse=True)
    support = sorted(_cluster_by_price(support_candidates, max_lines=max_lines, tolerance_pct=0.005))
    
    return support, resistance


def _unified_score(
    df: pd.DataFrame,
    support_lines: list[float] | None = None,
    resistance_lines: list[float] | None = None,
    box_range: dict | None = None,
) -> tuple[int, list[dict], dict]:
    """
    통합 기술적 점수 계산 (기본 차트 + 예측 모두 이 함수 사용).

    기준점 50 (중립). 각 지표가 ±점수.
    반환: (최종점수, 시그널 리스트, 내부 지표값 dict)
    """
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    vol   = pd.to_numeric(df["volume"], errors="coerce")
    high  = pd.to_numeric(df["high"], errors="coerce")
    low   = pd.to_numeric(df["low"], errors="coerce")

    cur = float(close.iloc[-1])
    score = 50
    signals: list[dict] = []

    # ── 이동평균 계산 ──
    sma5   = close.rolling(5).mean()
    sma20  = close.rolling(20).mean()
    sma60  = close.rolling(60).mean()
    sma120 = close.rolling(120).mean()
    sma224 = close.rolling(224).mean()

    # ── RSI 계산 ──
    if ta is not None:
        rsi_series = ta.rsi(close, length=14)
    else:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_series = 100 - (100 / (1 + rs))

    # ── MACD 계산 ──
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line

    # ── 볼린저 밴드 계산 ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_up  = bb_mid + 2 * bb_std
    bb_dn  = bb_mid - 2 * bb_std

    # ── ATR 계산 ──
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    # ══════════════════════════════════════════════
    # 1. 이평선 배열 + Cup 패턴 + 골든크로스 (±15)
    #    - 정배열: 상승 확인 (+6~8)  ← 이미 오른 상태이므로 과도한 점수 X
    #    - 역배열 + 바닥 반등: 매수 기회! (+10~15)
    #    - 골든크로스 직전/직후: 전환점 (+8~12)
    #    - 완전 역배열 하락 중: 위험 (-8~-12)
    # ══════════════════════════════════════════════
    has_cup_pattern = False
    has_golden_cross = False

    if not sma120.dropna().empty and not sma20.dropna().empty and not sma60.dropna().empty:
        s5  = float(sma5.iloc[-1]) if not sma5.dropna().empty else cur
        s20 = float(sma20.iloc[-1])
        s60 = float(sma60.iloc[-1])
        s120 = float(sma120.iloc[-1])

        # ── Cup(그릇) 패턴 감지 ──
        # 최근 60일에서 U자형 반등: 바닥 찍고 현재가가 회복 중
        if len(close) >= 60:
            recent_60 = close.tail(60).values
            trough_idx = int(np.argmin(recent_60))
            trough_price = float(recent_60[trough_idx])

            # 바닥이 중간(10~50일 전)에 있고, 양쪽이 높은 U자형
            if 10 <= trough_idx <= 50 and trough_price > 0:
                left_high = float(np.max(recent_60[:trough_idx])) if trough_idx > 0 else cur
                right_now = cur
                decline_depth = (left_high - trough_price) / left_high  # 하락 깊이
                recovery_ratio = (right_now - trough_price) / (left_high - trough_price) if left_high > trough_price else 0

                # 하락 5% 이상 + 50% 이상 회복 = Cup 패턴
                if decline_depth >= 0.05 and recovery_ratio >= 0.5:
                    has_cup_pattern = True

        # ── 골든크로스 감지: SMA20이 SMA60을 아래→위로 돌파 ──
        if len(sma20.dropna()) >= 5 and len(sma60.dropna()) >= 5:
            sma20_recent = sma20.dropna().tail(5).values
            sma60_recent = sma60.dropna().tail(5).values
            min_len = min(len(sma20_recent), len(sma60_recent))
            if min_len >= 5:
                sma20_r = sma20_recent[-min_len:]
                sma60_r = sma60_recent[-min_len:]
                # 5일 전에는 SMA20 < SMA60 → 현재 SMA20 >= SMA60
                if sma20_r[0] < sma60_r[0] and sma20_r[-1] >= sma60_r[-1]:
                    has_golden_cross = True

        # ── SMA20 방향성: 5일 전 vs 현재 ──
        sma20_rising = False
        sma20_falling = False
        if len(sma20.dropna()) >= 5:
            sma20_5ago = float(sma20.dropna().iloc[-5])
            sma20_rising = s20 > sma20_5ago * 1.001  # 0.1% 이상 상승
            sma20_falling = s20 < sma20_5ago * 0.999

        # ── 점수 부여 (상황별) ──

        # Case A: 역배열 + Cup 패턴 (바닥 반등) — 최고 점수!
        if has_cup_pattern and cur < s120 and cur > s5 > s20:
            score += 15
            signals.append({"type": "positive", "label": "바닥 반등 패턴 (Cup) 🏆",
                            "desc": "주가가 그릇 모양으로 바닥을 찍고 올라오고 있어요. 강한 매수 신호!"})

        # Case B: 역배열이지만 골든크로스 발생 — 전환점
        elif has_golden_cross and s20 <= s60:
            score += 12
            signals.append({"type": "positive", "label": "반등 전환 신호 (골든크로스) ✨",
                            "desc": "단기 평균이 중기 평균을 올라섰어요. 하락→상승 전환 시작!"})

        # Case C: 역배열이지만 단기선 반등 중 (SMA5 > SMA20, SMA20 상승 전환)
        elif cur > s5 > s20 and sma20_rising and (s60 > s20 or s120 > s20):
            score += 10
            signals.append({"type": "positive", "label": "바닥에서 올라오는 중 🌱",
                            "desc": "장기적으로는 아직 역배열이지만, 단기 흐름이 상승으로 바뀌고 있어요."})

        # Case D: Cup 패턴 + 이미 장기선 위로 회복
        elif has_cup_pattern and cur > s120:
            score += 10
            signals.append({"type": "positive", "label": "바닥 반등 후 회복 완료 🏆",
                            "desc": "그릇 모양 반등 후 장기 평균 위로 올라왔어요."})

        # Case E: 완전 정배열 — 좋지만 이미 많이 오른 상태
        elif cur > s5 > s20 > s60 > s120:
            score += 6
            signals.append({"type": "positive", "label": "상승 추세 유지 중 📈",
                            "desc": "꾸준히 오르고 있지만, 이미 많이 올라온 상태예요."})

        # Case F: 부분 정배열 (cur > s20 > s60, s120은 아직 위)
        elif cur > s20 > s60:
            score += 4
            signals.append({"type": "positive", "label": "중단기 흐름 괜찮음",
                            "desc": "중단기적으로 상승 중이에요."})

        # Case G: 완전 역배열 + 하락 가속 (SMA20도 내려가는 중)
        elif cur < s5 < s20 < s60 < s120 and sma20_falling:
            score -= 12
            signals.append({"type": "negative", "label": "하락 추세 강함 📉",
                            "desc": "주가가 모든 평균선 아래에서 계속 내려가고 있어요."})

        # Case H: 역배열이지만 하락 둔화 (SMA20 기울기 평평)
        elif cur < s20 < s60 and not sma20_falling:
            score -= 4
            signals.append({"type": "negative", "label": "하락 중이지만 둔화 🐌",
                            "desc": "아직 내려가는 중이지만, 속도가 느려지고 있어요."})

        # Case I: 부분 역배열
        elif cur < s20 < s60:
            score -= 8
            signals.append({"type": "negative", "label": "주가 흐름 약함 📉",
                            "desc": "중단기 흐름이 하락 중이에요."})

    # ══════════════════════════════════════════════
    # 2. 이격도: 현재가 vs SMA20 (±8)
    # ══════════════════════════════════════════════
    if not sma20.dropna().empty:
        s20_val = float(sma20.iloc[-1])
        if s20_val > 0:
            disparity = (cur - s20_val) / s20_val * 100
            if disparity < -5:
                score += 8
                signals.append({"type": "positive", "label": f"평균보다 많이 싸요 ({disparity:.1f}%)",
                                "desc": "20일 평균 가격보다 많이 내려와 있어요. 반등 가능성."})
            elif disparity < -2:
                score += 3
            elif disparity > 7:
                score -= 8
                signals.append({"type": "negative", "label": f"평균보다 많이 비싸요 (+{disparity:.1f}%)",
                                "desc": "20일 평균보다 많이 올라와 있어요. 조정이 올 수 있어요."})
            elif disparity > 3:
                score -= 3

    # ══════════════════════════════════════════════
    # 3. RSI (±12) — 세분화
    # ══════════════════════════════════════════════
    rsi_val = None
    if rsi_series is not None and not rsi_series.dropna().empty:
        rsi_val = float(rsi_series.iloc[-1])
        if rsi_val <= 25:
            score += 12
            signals.append({"type": "positive", "label": f"많이 떨어진 상태 ({rsi_val:.0f}점)",
                            "desc": "주가가 많이 내려와서 반등 가능성이 높아요."})
        elif rsi_val <= 35:
            score += 6
            signals.append({"type": "positive", "label": f"좀 싼 편 ({rsi_val:.0f}점)",
                            "desc": "주가가 낮은 편이에요. 매수 기회일 수 있어요."})
        elif rsi_val >= 75:
            score -= 12
            signals.append({"type": "negative", "label": f"많이 오른 상태 ({rsi_val:.0f}점)",
                            "desc": "단기간에 많이 올라서 내려갈 수 있어요."})
        elif rsi_val >= 65:
            score -= 6
            signals.append({"type": "negative", "label": f"좀 비싼 편 ({rsi_val:.0f}점)",
                            "desc": "주가가 높은 편이에요. 추가 매수는 신중하게."})
        else:
            signals.append({"type": "neutral", "label": f"보통 상태 ({rsi_val:.0f}점)",
                            "desc": "지금은 특별히 싸지도, 비싸지도 않아요."})

    # ══════════════════════════════════════════════
    # 4. MACD (±8)
    # ══════════════════════════════════════════════
    if not histogram.dropna().empty and len(histogram.dropna()) >= 2:
        hist_now  = float(histogram.iloc[-1])
        hist_prev = float(histogram.iloc[-2])
        macd_val  = float(macd_line.iloc[-1])
        sig_val   = float(signal_line.iloc[-1])

        if macd_val > sig_val and hist_now > hist_prev:
            score += 8
            signals.append({"type": "positive", "label": "오를 힘이 강해지는 중 💪",
                            "desc": "주가가 위로 올라가려는 힘이 세지고 있어요."})
        elif macd_val > sig_val and hist_now <= hist_prev:
            score += 3  # 골든크로스 상태지만 모멘텀 둔화
        elif macd_val < sig_val and hist_now < hist_prev:
            score -= 8
            signals.append({"type": "negative", "label": "내릴 힘이 강해지는 중 ⚠️",
                            "desc": "주가가 아래로 내려가려는 힘이 세지고 있어요."})
        elif macd_val < sig_val and hist_now >= hist_prev:
            score -= 3  # 데드크로스 상태지만 모멘텀 개선 중

    # ══════════════════════════════════════════════
    # 5. 볼린저 밴드 위치 (±8)
    # ══════════════════════════════════════════════
    if not bb_up.dropna().empty:
        up_val = float(bb_up.iloc[-1])
        dn_val = float(bb_dn.iloc[-1])
        bb_mid_val = float(bb_mid.iloc[-1])

        if up_val > dn_val and (up_val - dn_val) > 0:
            # 밴드 내 위치 (0=하단, 1=상단)
            bb_pos = (cur - dn_val) / (up_val - dn_val)

            if bb_pos <= 0.0:
                score += 8
                signals.append({"type": "positive", "label": "바닥 근처 도달 🔻",
                                "desc": "주가가 변동 범위의 가장 아래에 있어요. 반등 가능."})
            elif bb_pos <= 0.2:
                score += 4
                signals.append({"type": "positive", "label": "바닥권에 가까움",
                                "desc": "주가가 변동 범위의 아래쪽에 있어요."})
            elif bb_pos >= 1.0:
                score -= 8
                signals.append({"type": "negative", "label": "천장 근처 도달 🔺",
                                "desc": "주가가 변동 범위의 가장 위에 있어요. 내려갈 수 있어요."})
            elif bb_pos >= 0.8:
                score -= 4
                signals.append({"type": "negative", "label": "천장권에 가까움",
                                "desc": "주가가 변동 범위의 위쪽에 있어요."})

    # ══════════════════════════════════════════════
    # 6. 지지선/저항선 근접 (±6)
    # ══════════════════════════════════════════════
    if support_lines:
        nearest_support = min(support_lines, key=lambda s: abs(cur - s) if s > 0 else float('inf'))
        if nearest_support > 0:
            dist_support = (cur - nearest_support) / nearest_support
            if -0.02 <= dist_support <= 0.03:
                score += 6
                signals.append({"type": "positive", "label": "지지선 근처 💎",
                                "desc": f"지지선({int(nearest_support):,}원) 근처에서 버티고 있어요."})

    if resistance_lines:
        nearest_resist = min(resistance_lines, key=lambda r: abs(cur - r) if r > 0 else float('inf'))
        if nearest_resist > 0:
            dist_resist = (nearest_resist - cur) / nearest_resist
            if 0 <= dist_resist <= 0.03:
                score -= 4
                signals.append({"type": "negative", "label": "저항선 근처 🧱",
                                "desc": f"저항선({int(nearest_resist):,}원) 근처에서 막힐 수 있어요."})

    # ══════════════════════════════════════════════
    # 7. 박스권 위치 (±5)
    # ══════════════════════════════════════════════
    if box_range and box_range.get("is_box"):
        box_top = box_range.get("top", 0)
        box_bot = box_range.get("bottom", 0)
        if box_top > box_bot > 0:
            box_pos = (cur - box_bot) / (box_top - box_bot)
            if box_pos <= 0.2:
                score += 5
                signals.append({"type": "positive", "label": "박스권 바닥 근처 📦",
                                "desc": "박스권 아래쪽에서 반등할 수 있는 자리에요."})
            elif box_pos >= 0.8:
                score -= 5
                signals.append({"type": "negative", "label": "박스권 천장 근처 📦",
                                "desc": "박스권 위쪽이라 눌릴 수 있어요."})

    # ══════════════════════════════════════════════
    # 8. 거래량 (±5)
    # ══════════════════════════════════════════════
    vol_ma5 = vol.rolling(5).mean()
    if len(vol) >= 5 and not vol_ma5.dropna().empty:
        cur_vol = float(vol.iloc[-1])
        avg_vol = float(vol_ma5.iloc[-1])
        if avg_vol > 0 and cur_vol >= avg_vol * 1.5:
            if float(close.iloc[-1]) > float(open_.iloc[-1]):
                score += 5
                signals.append({"type": "positive", "label": "거래량 폭발 + 양봉 🔥",
                                "desc": "많은 사람이 사면서 주가가 올랐어요."})
            else:
                score -= 5
                signals.append({"type": "negative", "label": "거래량 폭발 + 음봉 💨",
                                "desc": "많은 사람이 팔면서 주가가 내렸어요."})

    # ══════════════════════════════════════════════
    # 9. 수박지표 (BB 스퀴즈 + 이평선 수렴) (+5~8)
    # ══════════════════════════════════════════════
    is_squeeze = False
    staircase_signal = None

    if not bb_up.dropna().empty and not bb_mid.dropna().empty:
        bb_width_pct = (bb_up - bb_dn) / bb_mid * 100
        bb_width_min_20 = bb_width_pct.rolling(20).min()
        if not bb_width_pct.dropna().empty and not bb_width_min_20.dropna().empty:
            is_squeeze = float(bb_width_pct.iloc[-1]) <= float(bb_width_min_20.iloc[-1]) * 1.3

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
            label = "큰 움직임 준비 중 🍉"
            desc = "주가가 좁은 범위에서 모이고 있어요. 곧 크게 움직일 수 있어요."
            if not sma224.dropna().empty and cur > float(sma224.iloc[-1]):
                score += 8
                label = "큰 상승 준비 중 🍉💪"
                desc = "장기 평균 위에서 힘을 모으고 있어요. 크게 오를 가능성."
            else:
                score += 5
            signals.append({"type": "positive", "label": label, "desc": desc})
    elif is_squeeze:
        signals.append({"type": "neutral", "label": "주가가 쉬는 중 😴",
                        "desc": "주가 변동이 줄어들고 있어요. 곧 움직임이 커질 수 있어요."})

    # ══════════════════════════════════════════════
    # 10. 계단지표 (±5)
    # ══════════════════════════════════════════════
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
        score += 5
        signals.append({"type": "positive", "label": "한 계단씩 오르는 중 🪜",
                        "desc": "주가가 한 단계씩 올라가고 있어요."})
    elif staircase_signal == "down":
        score -= 5
        signals.append({"type": "negative", "label": "한 계단씩 내리는 중 🪜",
                        "desc": "주가가 한 단계씩 내려가고 있어요."})

    # ── 최종 점수 클램핑 ──
    final_score = max(0, min(100, score))

    # ── 내부 지표값 (예측용) ──
    internals = {
        "rsi": rsi_val,
        "sma20": float(sma20.iloc[-1]) if not sma20.dropna().empty else None,
        "bb_upper": float(bb_up.iloc[-1]) if not bb_up.dropna().empty else None,
        "bb_lower": float(bb_dn.iloc[-1]) if not bb_dn.dropna().empty else None,
        "atr14": float(atr14.iloc[-1]) if not atr14.dropna().empty else cur * 0.02,
        "is_squeeze": is_squeeze,
        "staircase_signal": staircase_signal,
    }

    return final_score, signals, internals


def _generate_predicted_candles(
    last_date: str,
    last_close: float,
    atr_val: float,
    prediction_score: int,
    sma20_val: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    is_squeeze: bool = False,
    staircase_signal: str | None = None,
    n_days: int = 7,
) -> list[dict]:
    """
    기하 브라운 운동(GBM) 기반 미래 일봉 생성.
    점수 방향에 따라 자연스럽게 진행 — 매도가 수렴 없음.

    핵심:
    - GBM 수익률 모델: 주가 수준 무관
    - 추세(μ)는 점수에서만 결정 → 매도가와 충돌 없음
    - 실제 차트 패턴: 3일 연속 추세 후 1일 조정 등
    """
    rng = np.random.default_rng(seed=42)

    # ── 1. 일별 기대수익률 μ ──
    score_norm = (prediction_score - 50) / 50      # -1.0 ~ +1.0
    daily_mu = score_norm * 0.005                   # ±0.5%/일

    if staircase_signal == "up":
        daily_mu += 0.002
    elif staircase_signal == "down":
        daily_mu -= 0.002

    # ── 2. 일별 변동성 σ ──
    daily_sigma = atr_val / last_close

    # ── 3. BB 가이드 ──
    bb_mid = None
    bb_half_pct = None
    if bb_upper and bb_lower and last_close > 0:
        bb_mid = (bb_upper + bb_lower) / 2
        bb_half_pct = (bb_upper - bb_lower) / 2 / last_close

    candles = []
    prev_close = last_close
    d = date.fromisoformat(last_date[:10])
    # 연속 추세 카운터 (3~4일 추세 후 조정 패턴용)
    trend_streak = 0

    for i in range(n_days):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)

        progress = i / max(1, n_days - 1)

        # ── 4. 수박 스퀴즈: 변동성 조절 ──
        if is_squeeze:
            phase = min(1.0, i / max(5, n_days * 0.3))
            vol_scale = 0.4 + phase * 1.2
        else:
            vol_scale = 0.9 + progress * 0.3

        sigma_today = daily_sigma * vol_scale

        # ── 5. 조정 패턴: 3~4일 추세 후 반대 방향 1일 ──
        if abs(trend_streak) >= 3 and rng.random() < 0.6:
            # 조정일: 추세 반대 방향
            correction_mu = -daily_mu * 0.5
            z_close = rng.normal(0, 1)
            ret_close = correction_mu + sigma_today * z_close
            trend_streak = 0
        else:
            z_close = rng.normal(0, 1)
            ret_close = daily_mu + sigma_today * z_close

        # 추세 방향 추적
        if ret_close > 0:
            trend_streak = max(1, trend_streak + 1)
        else:
            trend_streak = min(-1, trend_streak - 1)

        # ── 6. SMA20 약한 회귀 ──
        if sma20_val and sma20_val > 0:
            dist_pct = (prev_close - sma20_val) / sma20_val
            if abs(dist_pct) > 0.05:
                ret_close -= dist_pct * 0.08

        # ── 7. BB 소프트 제약 ──
        projected = prev_close * (1 + ret_close)
        if bb_mid is not None and bb_half_pct is not None:
            expand = 1.0 + progress * 0.8
            soft_upper = bb_mid * (1 + bb_half_pct * expand)
            soft_lower = bb_mid * (1 - bb_half_pct * expand)
            if projected > soft_upper:
                excess = (projected - soft_upper) / projected
                ret_close -= excess * 0.5
            elif projected < soft_lower:
                deficit = (soft_lower - projected) / projected
                ret_close += deficit * 0.5

        # ── 8. 캔들 생성 ──
        c = prev_close * (1 + ret_close)

        gap_ret = rng.normal(0, 1) * sigma_today * 0.3
        o = prev_close * (1 + gap_ret)

        # 몸통 크기 보장 (ATR 20%)
        body = abs(c - o)
        min_body = atr_val * 0.2
        if body < min_body:
            direction = 1.0 if c >= o else -1.0
            c = o + direction * min_body

        body_top = max(o, c)
        body_bot = min(o, c)

        upper_wick = rng.exponential(0.5) * atr_val * 0.3 * vol_scale
        lower_wick = rng.exponential(0.5) * atr_val * 0.25 * vol_scale
        h = body_top + upper_wick
        lo = body_bot - lower_wick

        candles.append({
            "time": d.isoformat(),
            "open": max(1, round(o)),
            "high": max(1, round(h)),
            "low": max(1, round(lo)),
            "close": max(1, round(c)),
        })
        prev_close = c

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


def _fetch_ohlcv_sync(start: str, end: str, ticker: str) -> pd.DataFrame:
    """pykrx 동기 호출 래퍼 (ThreadPool에서 실행됨)"""
    return stock.get_market_ohlcv(start, end, ticker)


async def _fetch_ohlcv(start: str, end: str, ticker: str) -> pd.DataFrame:
    """pykrx 호출에 타임아웃 적용"""
    return await _run_with_timeout(_fetch_ohlcv_sync, start, end, ticker)


def _load_stock_for_score_sync(ticker: str, end_day: str) -> tuple[pd.DataFrame, list[float], list[float], int, dict]:
    """추천 루프에서 사용하는 동기 버전 (ThreadPool에서 호출)"""
    end_d = date.fromisoformat(f"{end_day[:4]}-{end_day[4:6]}-{end_day[6:8]}")
    start_d = end_d - timedelta(days=210)
    raw = stock.get_market_ohlcv(_yyyymmdd(start_d), end_day, ticker)
    if raw.empty:
        raise ValueError("no data")
    df = _standardize_ohlcv(raw)
    close_values = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=float)
    support, resistance = _support_resistance(close_values, max_lines=1)
    box = _detect_box_range(df)
    score, _signals, _internals = _unified_score(df, support, resistance, box)
    return df, support, resistance, score, {}


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
    start_date: str | None = Query(None, description="YYYYMMDD 또는 YYYY-MM-DD"),
    end_date: str | None = Query(None, description="YYYYMMDD 또는 YYYY-MM-DD"),
):
    try:
        end_d = date.today()
        start_d = end_d - timedelta(days=90)
        start = _normalize_input_date(start_date, start_d)
        end = _normalize_input_date(end_date, end_d)

        ticker, stock_name = _resolve_ticker(ticker_or_name)
        raw = await _fetch_ohlcv(start, end, ticker)
        if raw.empty:
            return {"error": "해당 기간의 데이터가 없거나 종목 코드/이름이 잘못되었습니다."}

        df = _standardize_ohlcv(raw)
        data = df[["time", "open", "high", "low", "close", "volume"]].to_dict(orient="records")
        close_values = pd.to_numeric(df["close"], errors="coerce").dropna().to_numpy(dtype=float)
        support_lines, resistance_lines = _support_resistance(close_values, max_lines=1)
        box_range = _detect_box_range(df)
        score, _signals, _internals = _unified_score(df, support_lines, resistance_lines, box_range)
        score_breakdown = {}

        return {
            "ticker": ticker,
            "stock_name": stock_name,
            "data": data,
            "support_lines": support_lines,
            "resistance_lines": resistance_lines,
            "score": score,
            "score_breakdown": score_breakdown,
            "box_range": box_range,
        }
    except asyncio.TimeoutError:
        logger.warning("주가 데이터 조회 타임아웃: %s", ticker_or_name)
        return {"error": f"주가 데이터 조회 시간 초과 ({PYKRX_TIMEOUT_SEC}초). 잠시 후 다시 시도해주세요."}
    except Exception as e:
        logger.exception("데이터 조회 실패")
        return {"error": f"데이터 조회 중 오류 발생: {str(e)}"}


def _compute_recommendations_sync(day: str, listing: pd.DataFrame, sample_size: int) -> list[dict]:
    """추천 종목 스코어링 (동기, ThreadPool에서 실행)"""
    tickers = listing["ticker"].tolist()[:sample_size]
    ranked = []
    for tk in tickers:
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
    start_date: str | None = None,
    end_date: str | None = None,
    n_days: int = Query(7, ge=1, le=30, description="예측할 영업일 수 (1~30)"),
):
    try:
        try:
            ticker, stock_name = _resolve_ticker(ticker_or_name)
        except Exception:
            ticker = ticker_or_name
            stock_name = ticker_or_name

        end_d = datetime.today()
        start_d = end_d - timedelta(days=365)
        start_str = start_d.strftime("%Y%m%d")
        end_str = end_d.strftime("%Y%m%d")

        raw = await _fetch_ohlcv(start_str, end_str, ticker)
        if raw.empty:
            return {"error": "데이터 없음"}

        df = _standardize_ohlcv(raw)
        close = pd.to_numeric(df["close"], errors="coerce")
        high  = pd.to_numeric(df["high"], errors="coerce")
        cur   = float(close.iloc[-1])

        # ── 통합 점수 계산 (기본 점수와 동일한 함수) ──
        close_values = close.dropna().to_numpy(dtype=float)
        support_lines, resistance_lines = _support_resistance(close_values, max_lines=1)
        box_range = _detect_box_range(df)
        final_score, signals, internals = _unified_score(df, support_lines, resistance_lines, box_range)

        # internals에서 지표값 추출
        atr_val = internals["atr14"]
        is_squeeze = internals["is_squeeze"]
        staircase_signal = internals["staircase_signal"]

        # ── 목표가 / 손절가 (점수 연동) ──
        sell_short_price = None
        sell_long_price = None
        stop_loss_price = None
        sell_short_desc = None
        sell_long_desc = None
        stop_loss_desc = None

        if final_score >= 60:
            short_mult = 1.0 + (final_score - 60) / 40 * 1.0
            sell_short_price = int(cur + atr_val * short_mult)
            sell_short_desc = f"단기 목표 (ATR ×{short_mult:.1f} 상승 여력)"

            bb_up_val = internals.get("bb_upper")
            if bb_up_val and len(high) >= 60:
                highest_60 = float(high.tail(60).max())
                long_mult = 1.5 + (final_score - 60) / 40 * 1.5
                sell_long_price = int(max(bb_up_val, highest_60, cur + atr_val * long_mult))
                sell_long_desc = f"장기 목표 (주요 저항선 + ATR ×{long_mult:.1f})"

            stop_loss_price = int(cur - atr_val * 1.5)
            stop_loss_desc = "손절 기준 (ATR ×1.5 하락 시)"

        elif final_score >= 40:
            sell_short_price = int(cur + atr_val * 0.8)
            sell_short_desc = "소폭 반등 시 매도 고려"
            stop_loss_price = int(cur - atr_val * 1.0)
            stop_loss_desc = "손절 기준 (ATR ×1.0 하락 시)"

        else:
            stop_loss_price = int(cur - atr_val * 0.8)
            stop_loss_desc = "보유 중이라면 이 가격 아래로 내려가면 매도 고려"

        # ── 전망 ──
        if final_score >= 70:
            outlook_short, outlook_mid = "오를 가능성 높음 👍", "당분간 좋아 보여요"
            summary = "여러 지표가 '사도 괜찮다'고 말하고 있어요."
        elif final_score >= 40:
            outlook_short, outlook_mid = "지켜보는 게 좋아요 👀", "아직 방향을 모르겠어요"
            summary = "오를 수도, 내릴 수도 있어요. 좀 더 지켜본 후 결정하세요."
        else:
            outlook_short, outlook_mid = "내릴 수 있어요 ⚠️", "조심해야 할 시기"
            summary = "내려갈 신호가 더 많아요. 신중하게 판단하세요."

        # ── 예측 캔들 생성 ──
        last_date = str(df["time"].iloc[-1])[:10]
        predicted_candles = _generate_predicted_candles(
            last_date=last_date,
            last_close=cur,
            atr_val=atr_val,
            prediction_score=final_score,
            sma20_val=internals.get("sma20"),
            bb_upper=internals.get("bb_upper"),
            bb_lower=internals.get("bb_lower"),
            is_squeeze=is_squeeze,
            staircase_signal=staircase_signal,
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
