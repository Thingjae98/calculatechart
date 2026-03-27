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

from datetime import datetime, timedelta

# 최상단(app 선언 부근)에 전역 캐시 변수 추가
RECOMMEND_CACHE = {
    "data": None,
    "last_updated": None
}


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
    최근 계산 결과를 1시간(3600초) 동안 메모리에 캐싱하여 응답 속도를 높입니다.
    """
    global RECOMMEND_CACHE
    now = datetime.now()

    # 1. 캐시 확인: 데이터가 존재하고, 계산된 지 1시간(3600초)이 지나지 않았다면 즉시 반환
    if RECOMMEND_CACHE["data"] is not None and RECOMMEND_CACHE["last_updated"] is not None:
        if (now - RECOMMEND_CACHE["last_updated"]).total_seconds() < 3600:
            cached_ranked = RECOMMEND_CACHE["data"]
            return {
                "as_of": RECOMMEND_CACHE["as_of"],
                "count": len(cached_ranked),
                "top": cached_ranked[:limit], # 캐시된 전체 리스트에서 요청한 limit만큼만 잘라서 반환
                "cached": True,
                "last_updated": RECOMMEND_CACHE["last_updated"].strftime("%Y-%m-%d %H:%M:%S")
            }

    # 2. 캐시가 없거나 만료되었다면 새로 220개 종목 계산 진행
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

        # 점수 순으로 정렬
        ranked.sort(key=lambda x: x["score"], reverse=True)

        # 3. 계산이 끝난 전체 리스트(ranked)를 캐시에 덮어쓰기
        RECOMMEND_CACHE["data"] = ranked
        RECOMMEND_CACHE["last_updated"] = now
        RECOMMEND_CACHE["as_of"] = day

        return {
            "as_of": day,
            "count": len(ranked),
            "top": ranked[:limit],
            "cached": False,
            "last_updated": now.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": f"추천 계산 중 오류 발생: {str(e)}"}
    

@app.get("/api/stock/{ticker}/predict")
def predict_stock(ticker: str, start_date: str | None = None, end_date: str | None = None):
    try:
        try:
            _, stock_name = _resolve_ticker(ticker)
        except Exception:
            stock_name = ticker
        # 최근 1년치 데이터 강제 설정 (추세 분석을 위해)
        import datetime as dt
        end_d = dt.datetime.today()
        start_d = end_d - dt.timedelta(days=365)
        
        start_str = start_d.strftime("%Y%m%d")
        end_str = end_d.strftime("%Y%m%d")
        
        df = stock.get_market_ohlcv(start_str, end_str, ticker)
        if df.empty:
            return {"error": "데이터 없음"}
            
        df.columns = ["open", "high", "low", "close", "volume", "value", "fluctuation"] if len(df.columns) == 8 else ["open", "high", "low", "close", "volume", "fluctuation"]
        
        # 보조지표 계산 (pandas_ta 활용)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=60, append=True)
        df.ta.sma(length=120, append=True)
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        cur = latest['close']
        
        signals = []
        score = 50 # 기본점수 50점에서 가감하는 방식이 결과 분포가 더 예쁘게 나옵니다.

        # 1. 이동평균 (추세)
        s20, s60, s120 = latest.get('SMA_20'), latest.get('SMA_60'), latest.get('SMA_120')
        if pd.notna(s120):
            if cur > s20 > s60 > s120:
                score += 15
                signals.append({"type": "positive", "label": "정배열 (대세 상승)", "desc": "주가와 단기/중기/장기 이동평균선이 모두 우상향 중입니다."})
            elif cur < s20 < s60 < s120:
                score -= 15
                signals.append({"type": "negative", "label": "역배열 (하락 추세)", "desc": "완연한 하락 추세입니다. 섣부른 매수는 위험합니다."})

        # 2. RSI (과매도/과매수 심리)
        rsi = latest.get('RSI_14')
        if pd.notna(rsi):
            if rsi <= 30:
                score += 15
                signals.append({"type": "positive", "label": f"RSI 바닥 ({rsi:.1f})", "desc": "사람들이 공포에 질려 많이 팔았습니다. 반등이 나올 자리입니다."})
            elif rsi >= 70:
                score -= 15
                signals.append({"type": "negative", "label": f"RSI 과열 ({rsi:.1f})", "desc": "단기적으로 너무 올랐습니다. 조정을 주의해야 합니다."})

        # 3. MACD (모멘텀)
        macd = latest.get('MACD_12_26_9')
        signal_line = latest.get('MACDs_12_26_9')
        hist_now = latest.get('MACDh_12_26_9')
        hist_prev = prev.get('MACDh_12_26_9')
        
        if pd.notna(macd) and pd.notna(signal_line):
            if macd > signal_line and hist_now > hist_prev:
                score += 10
                signals.append({"type": "positive", "label": "MACD 매수 신호", "desc": "상승하는 힘(모멘텀)이 강해지고 있습니다."})
            elif macd < signal_line and hist_now < hist_prev:
                score -= 10
                signals.append({"type": "negative", "label": "MACD 매도 신호", "desc": "하락하는 힘이 강해지고 있습니다."})

        # 4. 볼린저 밴드 (변동성)
        bb_up, bb_dn = latest.get('BBU_20_2.0'), latest.get('BBL_20_2.0')
        if pd.notna(bb_up):
            if cur <= bb_dn * 1.02: # 하단 2% 이내
                score += 10
                signals.append({"type": "positive", "label": "볼린저 하단 지지", "desc": "통계적 바닥 구간에 도달했습니다."})
            elif cur >= bb_up * 0.98: # 상단 2% 이내
                score -= 10
                signals.append({"type": "negative", "label": "볼린저 상단 저항", "desc": "통계적 천장 구간에 도달했습니다."})

        # 점수 정규화 (0 ~ 100)
        final_score = max(0, min(100, score))

        if final_score >= 70:
            summary = "현재 기술적 지표들이 강력한 매수 신호를 보내고 있습니다."
        elif final_score >= 40:
            summary = "상승과 하락 신호가 섞여 있습니다. 확실한 방향이 나올 때까지 관망하세요."
        else:
            summary = "하락 추세가 강합니다. 지금은 매수를 피하는 것이 좋습니다."

        return {
            "ticker": ticker,
            "stock_name": ticker,  # ← 추가
            "current_price": int(cur),
            "prediction_score": final_score,
            "outlook_short": "상승 우세" if final_score >= 70 else "중립 / 관망" if final_score >= 40 else "하락 주의",  # ← 추가
            "outlook_mid": "추세 지속 가능" if final_score >= 70 else "방향성 확인 필요" if final_score >= 40 else "조정 가능성 존재",  # ← 추가
            "summary": summary,
            "signals": signals,
        }
    except Exception as e:
        return {"error": f"예측 중 오류 발생: {str(e)}"}
