from datetime import date, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from pykrx import stock
from scipy.signal import find_peaks
try:
    import pandas_ta as ta
except Exception:  # Python 3.14 환경에서는 pandas-ta 설치가 실패할 수 있음
    ta = None

app = FastAPI(title="주식 분석 시스템 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _best_business_day(max_back_days: int = 15) -> str:
    today = date.today()
    for i in range(max_back_days):
        d = today - timedelta(days=i)
        day = _yyyymmdd(d)
        tickers = stock.get_market_ticker_list(day, market="ALL")
        if tickers:
            return day
    return _yyyymmdd(today)


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
    lookback = min(60, len(close_values))
    if lookback < 5:
        return [], []

    recent = close_values[-lookback:]
    y_range = float(np.max(recent) - np.min(recent))
    mean_val = float(np.mean(recent))
    if lookback < 20:
        prominence = max(y_range * 0.001, mean_val * 0.0002)
        distance = 1
    else:
        prominence = max(y_range * 0.012, mean_val * 0.001)
        distance = max(2, lookback // 25)

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


def _load_stock_for_score(ticker: str, end_day: str) -> tuple[pd.DataFrame, list[float], list[float], int, dict]:
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


@app.get("/api/stock/{ticker_or_name}")
def get_stock_data(
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
        raw = stock.get_market_ohlcv(start, end, ticker)
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
    except Exception as e:
        return {"error": f"데이터 조회 중 오류 발생: {str(e)}"}


@app.get("/api/recommendations")
def get_recommendations(limit: int = Query(10, ge=1, le=50)):
    """
    스코어 기반 추천 Top N.
    성능을 위해 코스피/코스닥 종목 중 앞쪽 220개를 샘플링해서 평가합니다.
    """
    try:
        day = _best_business_day()
        listing = _load_listing()
        tickers = listing["ticker"].tolist()[:220]

        ranked = []
        for tk in tickers:
            try:
                _df, support, resistance, score, breakdown = _load_stock_for_score(tk, day)
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
            except Exception:
                continue

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {"as_of": day, "count": len(ranked), "top": ranked[:limit]}
    except Exception as e:
        return {"error": f"추천 계산 중 오류 발생: {str(e)}"}