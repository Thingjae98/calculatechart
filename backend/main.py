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
    q = ticker_or_name.strip()
    if q.isdigit() and len(q) == 6:
        listing = _load_listing()
        row = listing[listing["ticker"] == q]
        if not row.empty:
            return q, str(row.iloc[0]["name"])
        return q, q

    listing = _load_listing()
    exact = listing[listing["name"] == q]
    if not exact.empty:
        return str(exact.iloc[0]["ticker"]), str(exact.iloc[0]["name"])
    partial = listing[listing["name"].str.contains(q, na=False)]
    if not partial.empty:
        return str(partial.iloc[0]["ticker"]), str(partial.iloc[0]["name"])
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


def _score_stock(df: pd.DataFrame, support_lines: list[float]) -> tuple[int, dict]:
    close = pd.to_numeric(df["close"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce")
    current_close = float(close.iloc[-1])

    score = 0
    breakdown = {
        "support_proximity": 0,
        "oversold_rsi": 0,
        "long_term_trend": 0,
        "volume_rebound": 0,
    }

    # 1) 지지선 근접 +40
    if support_lines:
        strongest_support = float(support_lines[0])
        if strongest_support > 0 and current_close <= strongest_support * 1.03:
            score += 40
            breakdown["support_proximity"] = 40

    # 2) RSI(14) <= 30 +30
    if ta is not None:
        rsi14 = ta.rsi(close, length=14)
    else:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi14 = 100 - (100 / (1 + rs))
    if rsi14 is not None and not rsi14.dropna().empty and float(rsi14.iloc[-1]) <= 30:
        score += 30
        breakdown["oversold_rsi"] = 30

    # 3) 종가 > SMA120 +20
    sma120 = close.rolling(120).mean()
    if not sma120.dropna().empty and current_close > float(sma120.iloc[-1]):
        score += 20
        breakdown["long_term_trend"] = 20

    # 4) 거래량 동반 반등 +10
    vol_ma5 = vol.rolling(5).mean()
    if (
        len(vol) >= 5
        and not vol_ma5.dropna().empty
        and float(vol.iloc[-1]) >= float(vol_ma5.iloc[-1]) * 1.5
        and float(close.iloc[-1]) > float(open_.iloc[-1])
    ):
        score += 10
        breakdown["volume_rebound"] = 10

    return int(score), breakdown


def _generate_predicted_candles(
    last_date: str,
    last_close: float,
    atr_val: float,
    prediction_score: int,
    sell_short_price: int | None,
    sell_long_price: int | None,
    sma20_val: float | None,
    bb_upper: float | None,
    bb_lower: float | None,
    is_squeeze: bool = False,
    staircase_signal: str | None = None,
    staircase_grid_size: float = 0.0,
    n_days: int = 7,
) -> list[dict]:
    """
    기술적 지표 기반 미래 일봉 생성 (30일까지 지원).

    **예측에 반영하는 지표:**
    1. prediction_score (MA 정배열, RSI, MACD, BB, 수박, 계단 종합)
    2. SMA20 평균 회귀 — 가격이 SMA에서 멀수록 당김
    3. BB 범위 제약 — 상/하단 밖으로 벗어나지 않음
    4. 수박지표(BB 스퀴즈) — 스퀴즈 감지 시 초기 변동성 축소 → 후반 확대 (추세 전환)
    5. 계단지표(ATR 스텝) — 상승/하락 계단 방향으로 추가 편향 + 그리드 단위 스냅
    6. 단기/장기 매도가 수렴
    7. Confidence decay — 시간이 갈수록 추세 영향 줄이고 평균 회귀 강화
    """
    # ── 기본 추세 편향 ──
    if prediction_score >= 65:
        bias = atr_val * 0.25
    elif prediction_score >= 50:
        bias = atr_val * 0.05
    elif prediction_score >= 40:
        bias = -atr_val * 0.05
    else:
        bias = -atr_val * 0.25

    # ── 계단지표 편향 추가 ──
    if staircase_signal == "up":
        bias += atr_val * 0.1
    elif staircase_signal == "down":
        bias -= atr_val * 0.1

    candles = []
    prev_close = last_close
    d = date.fromisoformat(last_date[:10])

    for i in range(n_days):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)

        remaining = n_days - i
        confidence = max(0.2, 1.0 - i * 0.03)

        # 1) 수박지표: 스퀴즈 시 변동성 축소 → 후반에 확대
        if is_squeeze:
            squeeze_phase = min(1.0, i / max(5, n_days * 0.4))
            volatility_mult = 0.4 + squeeze_phase * 0.8  # 0.4x → 1.2x
        else:
            volatility_mult = 1.0

        # 2) 추세 편향 (decay 적용)
        drift = bias * confidence * volatility_mult

        # 3) 타겟 수렴 (단기 < 15일, 장기 >= 15일)
        target = sell_short_price if i < 15 else sell_long_price
        if target is not None and remaining > 0:
            target_pull = (target - prev_close) / remaining * 0.3
            drift = drift * 0.5 + target_pull * 0.5

        # 4) SMA20 평균 회귀
        if sma20_val and sma20_val > 0:
            distance_pct = (prev_close - sma20_val) / sma20_val
            reversion_force = -distance_pct * atr_val * 0.4 * (1 - confidence)
            drift += reversion_force

        # 5) BB 범위 제약
        projected = prev_close + drift
        if bb_upper and projected > bb_upper * 1.02:
            projected = bb_upper * 1.02
            drift = projected - prev_close
        if bb_lower and projected < bb_lower * 0.98:
            projected = bb_lower * 0.98
            drift = projected - prev_close

        o = prev_close
        c = o + drift

        # 6) 계단지표: 그리드 크기가 있으면 종가를 그리드 단위에 스냅
        if staircase_grid_size > 0:
            c = round(c / staircase_grid_size) * staircase_grid_size

        # 일별 변동 범위
        range_mult = (0.3 * confidence + 0.15) * volatility_mult
        h = max(o, c) + atr_val * range_mult
        lo = min(o, c) - atr_val * (range_mult * 0.7)

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
    score, breakdown = _score_stock(df, support)
    return df, support, resistance, score, breakdown


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
        score, score_breakdown = _score_stock(df, support_lines)
        box_range = _detect_box_range(df)

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

        # pandas_ta 컬럼명으로 수동 계산 (df.ta 액세서 대신)
        close = pd.to_numeric(df["close"], errors="coerce")
        open_ = pd.to_numeric(df["open"], errors="coerce")
        vol   = pd.to_numeric(df["volume"], errors="coerce")

        # 이동평균
        sma20  = close.rolling(20).mean()
        sma60  = close.rolling(60).mean()
        sma120 = close.rolling(120).mean()

        # RSI
        if ta is not None:
            rsi_series = ta.rsi(close, length=14)
        else:
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta.clip(upper=0)).rolling(14).mean()
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

        # 추가 이동평균 (Pine Script 계단지표 연동)
        sma5   = close.rolling(5).mean()
        sma112 = close.rolling(112).mean()
        sma224 = close.rolling(224).mean()

        cur       = float(close.iloc[-1])
        signals   = []
        score     = 50

        # ──────────────────────────────────────────────
        # 1. 이동평균 (기존)
        # ──────────────────────────────────────────────
        if not sma120.dropna().empty:
            s20, s60, s120 = float(sma20.iloc[-1]), float(sma60.iloc[-1]), float(sma120.iloc[-1])
            if cur > s20 > s60 > s120:
                score += 15
                signals.append({"type": "positive", "label": "주가 흐름 좋음 📈", "desc": "최근 주가가 꾸준히 오르고 있어요. 상승 흐름이 이어지고 있습니다."})
            elif cur < s20 < s60 < s120:
                score -= 15
                signals.append({"type": "negative", "label": "주가 흐름 안 좋음 📉", "desc": "주가가 계속 내려가고 있어요. 지금 사면 손해볼 수 있습니다."})

        # ──────────────────────────────────────────────
        # 2. RSI (기존)
        # ──────────────────────────────────────────────
        if rsi_series is not None and not rsi_series.dropna().empty:
            rsi = float(rsi_series.iloc[-1])
            if rsi <= 30:
                score += 15
                signals.append({"type": "positive", "label": f"많이 떨어진 상태 ({rsi:.0f}점)", "desc": "주가가 많이 내려와서 다시 오를 가능성이 높아요."})
            elif rsi >= 70:
                score -= 15
                signals.append({"type": "negative", "label": f"많이 오른 상태 ({rsi:.0f}점)", "desc": "주가가 단기간에 많이 올라서 잠시 내려갈 수 있어요."})
            else:
                signals.append({"type": "neutral", "label": f"보통 상태 ({rsi:.0f}점)", "desc": "지금은 특별히 싸지도, 비싸지도 않은 상태예요."})

        # ──────────────────────────────────────────────
        # 3. MACD (기존)
        # ──────────────────────────────────────────────
        if not histogram.dropna().empty and len(histogram.dropna()) >= 2:
            hist_now  = float(histogram.iloc[-1])
            hist_prev = float(histogram.iloc[-2])
            if float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]) and hist_now > hist_prev:
                score += 10
                signals.append({"type": "positive", "label": "오를 힘이 강해지는 중 💪", "desc": "주가가 위로 올라가려는 힘이 점점 세지고 있어요."})
            elif float(macd_line.iloc[-1]) < float(signal_line.iloc[-1]) and hist_now < hist_prev:
                score -= 10
                signals.append({"type": "negative", "label": "내릴 힘이 강해지는 중 ⚠️", "desc": "주가가 아래로 내려가려는 힘이 점점 세지고 있어요."})

        # ──────────────────────────────────────────────
        # 4. 볼린저 밴드 (기존)
        # ──────────────────────────────────────────────
        if not bb_up.dropna().empty:
            up_val, dn_val = float(bb_up.iloc[-1]), float(bb_dn.iloc[-1])
            if cur <= dn_val * 1.02:
                score += 10
                signals.append({"type": "positive", "label": "바닥 근처 도달 🔻", "desc": "주가가 최근 범위에서 가장 낮은 쪽에 와 있어요. 반등할 수 있어요."})
            elif cur >= up_val * 0.98:
                score -= 10
                signals.append({"type": "negative", "label": "천장 근처 도달 🔺", "desc": "주가가 최근 범위에서 가장 높은 쪽에 와 있어요. 내려갈 수 있어요."})

        # ══════════════════════════════════════════════
        # 5. 수박지표 연동: BB 스퀴즈 + 이평선 수렴도
        # (Pine Script 수박지표의 핵심 로직을 Python으로 재현)
        # ══════════════════════════════════════════════
        bb_width_pct = (bb_up - bb_dn) / bb_mid * 100
        bb_width_min_20 = bb_width_pct.rolling(20).min()
        # 스퀴즈: 현재 밴드폭이 최근 20봉 최소폭의 1.3배 이내
        is_squeeze = False
        if not bb_width_pct.dropna().empty and not bb_width_min_20.dropna().empty:
            is_squeeze = float(bb_width_pct.iloc[-1]) <= float(bb_width_min_20.iloc[-1]) * 1.3

        # 이평선 수렴도: 5/20/60/224일 이평의 최대-최소 / 현재가 (%)
        ma_convergence = None
        if not sma5.dropna().empty and not sma224.dropna().empty:
            ma_vals = [float(sma5.iloc[-1]), float(sma20.iloc[-1]),
                       float(sma60.iloc[-1]), float(sma224.iloc[-1])]
            ma_spread = max(ma_vals) - min(ma_vals)
            ma_convergence = ma_spread / cur * 100

        # 이격도 (20일 기준)
        disparity_20 = (cur / float(sma20.iloc[-1])) * 100 - 100 if not sma20.dropna().empty else None

        # 수박 신호 = 스퀴즈 + 수렴 + 이격도 근접
        if is_squeeze and ma_convergence is not None and ma_convergence < 2.0:
            if disparity_20 is not None and abs(disparity_20) < 3.0:
                score += 10
                label = "큰 움직임 준비 중 🍉"
                desc = "주가가 좁은 범위에서 모이고 있어요. 곧 크게 오르거나 내릴 수 있어요."
                # 224일선 위면 강한 수박
                if not sma224.dropna().empty and cur > float(sma224.iloc[-1]):
                    score += 5
                    label = "큰 상승 준비 중 🍉💪"
                    desc = "주가가 장기 평균 위에서 힘을 모으고 있어요. 크게 오를 가능성이 높아요."
                signals.append({"type": "positive", "label": label, "desc": desc})
        elif is_squeeze:
            signals.append({"type": "neutral", "label": "주가가 쉬는 중 😴",
                            "desc": "주가 변동이 줄어들고 있어요. 곧 움직임이 커질 수 있어요."})

        # ══════════════════════════════════════════════
        # 6. 계단지표 연동: ATR 기반 계단형 지지레벨 판단
        # (Pine Script 계단지표의 핵심 로직을 Python으로 재현)
        # ══════════════════════════════════════════════
        high = pd.to_numeric(df["high"], errors="coerce")
        low  = pd.to_numeric(df["low"], errors="coerce")
        # ATR 계산 (14일)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()

        staircase_signal = None
        if not atr14.dropna().empty and len(close) >= 20:
            atr_val = float(atr14.iloc[-1])
            grid_size = atr_val * 0.5  # step_atr_mult 기본값 0.5
            if grid_size > 0:
                # 최근 20봉에서 계단 레벨 변화 추적
                close_arr = close.tail(20).values
                grids = np.floor(close_arr / grid_size) * grid_size
                # 연속 동일 레벨 카운트
                current_level = grids[-1]
                count = 1
                for i in range(len(grids) - 2, -1, -1):
                    if abs(grids[i] - current_level) < grid_size * 0.001:
                        count += 1
                    else:
                        break
                prev_level = grids[0]  # 20봉 전 레벨

                if count >= 3:  # step_confirm 기본값 3
                    if current_level > prev_level + grid_size * 0.5:
                        staircase_signal = "up"
                    elif current_level < prev_level - grid_size * 0.5:
                        staircase_signal = "down"

        if staircase_signal == "up":
            score += 5
            signals.append({"type": "positive", "label": "한 계단씩 오르는 중 🪜",
                            "desc": "주가가 한 단계씩 올라가고 있어요. 안정적인 상승 신호예요."})
        elif staircase_signal == "down":
            score -= 5
            signals.append({"type": "negative", "label": "한 계단씩 내리는 중 🪜",
                            "desc": "주가가 한 단계씩 내려가고 있어요. 조심하세요."})

        # ══════════════════════════════════════════════
        # 7. 추천 매도가 (단기/장기)
        # (Pine Script 계단지표 v2에서 추가한 로직)
        # ══════════════════════════════════════════════
        sell_short_price = None
        sell_long_price = None

        # 단기 매도가: BB 상단 (20일, 2배수)
        if not bb_up.dropna().empty:
            sell_short_price = int(float(bb_up.iloc[-1]))

        # 장기 매도가: 최근 60봉 최고가 + ATR * 1.5
        if not atr14.dropna().empty and len(high) >= 60:
            highest_60 = float(high.tail(60).max())
            sell_long_raw = highest_60 + atr_val * 1.5
            sell_long_price = int(max(sell_long_raw, cur + atr_val * 3))

        # ──────────────────────────────────────────────
        # 최종 점수 및 전망
        # ──────────────────────────────────────────────
        final_score = max(0, min(100, score))

        if final_score >= 70:
            outlook_short, outlook_mid = "오를 가능성 높음 👍", "당분간 좋아 보여요"
            summary = "여러 지표가 '사도 괜찮다'고 말하고 있어요."
        elif final_score >= 40:
            outlook_short, outlook_mid = "지켜보는 게 좋아요 👀", "아직 방향을 모르겠어요"
            summary = "오를 수도, 내릴 수도 있어요. 좀 더 지켜본 후 결정하세요."
        else:
            outlook_short, outlook_mid = "내릴 수 있어요 ⚠️", "조심해야 할 시기"
            summary = "내려갈 신호가 더 많아요. 신중하게 판단하세요."

        # 예측 캔들 생성 (미래 n 영업일)
        last_date = str(df["time"].iloc[-1])[:10]  # "YYYY-MM-DD" 형태만 취함
        atr_for_pred = float(atr14.iloc[-1]) if not atr14.dropna().empty else cur * 0.02
        sma20_last = float(sma20.iloc[-1]) if not sma20.dropna().empty else None
        bb_up_last = float(bb_up.iloc[-1]) if not bb_up.dropna().empty else None
        bb_dn_last = float(bb_dn.iloc[-1]) if not bb_dn.dropna().empty else None
        # 계단지표의 grid_size 전달
        staircase_grid = 0.0
        if not atr14.dropna().empty:
            staircase_grid = float(atr14.iloc[-1]) * 0.5

        predicted_candles = _generate_predicted_candles(
            last_date=last_date,
            last_close=cur,
            atr_val=atr_for_pred,
            prediction_score=final_score,
            sell_short_price=sell_short_price,
            sell_long_price=sell_long_price,
            sma20_val=sma20_last,
            bb_upper=bb_up_last,
            bb_lower=bb_dn_last,
            is_squeeze=is_squeeze,
            staircase_signal=staircase_signal,
            staircase_grid_size=staircase_grid,
            n_days=n_days,
        )

        result = {
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
                "short_term_desc": "단기 매도 목표 (BB 상단 기반)" if sell_short_price else None,
                "long_term_desc": "장기 매도 목표 (60일 최고가+ATR 기반)" if sell_long_price else None,
            },
            "predicted_candles": predicted_candles,
        }
        return result

    except asyncio.TimeoutError:
        logger.warning("예측 데이터 조회 타임아웃: %s", ticker_or_name)
        return {"error": f"예측 데이터 조회 시간 초과 ({PYKRX_TIMEOUT_SEC}초). 잠시 후 다시 시도해주세요."}
    except Exception as e:
        logger.exception("예측 실패")
        return {"error": f"예측 중 오류 발생: {str(e)}"}
