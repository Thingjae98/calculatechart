"""
예측 모델 백테스트 스크립트 — 로컬 개발자 전용.

Walk-forward 검증으로 _generate_predicted_candles 정확도를 수치화.
파라미터 튜닝의 기준선(baseline) 확보 + 점수 시스템 유효성 검증이 목적.

Usage (backend/ 디렉터리에서 실행):
    python backtest.py
    python backtest.py --tickers 005930,000660,035420
    python backtest.py --horizons 1,3,7,14 --step 5
    python backtest.py --validation-days 180 --history-days 730

측정 지표:
    - MAPE: 평균 절대 오차율 (%)   — 낮을수록 좋음
    - RMSE: 제곱근 평균 제곱 오차   — 낮을수록 좋음
    - Directional Accuracy: 상승/하락 방향 맞춘 비율 (>55% 목표)
    - 점수 구간별 실제 N일 수익률: 점수 시스템이 예측력이 있는지 직접 검증

결과는 backend/backtest_results/{timestamp}.json 에 저장.
튜닝 전후 JSON을 diff하여 변경점의 실효성 판단 가능.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# Windows 콘솔에서 UTF-8 출력 강제 (cp949 기본값 회피)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# main.py 의 파이프라인 재사용
from main import (
    _clean_ohlcv,
    _detect_box_range,
    _fetch_index_sync,
    _fetch_ohlcv_sync,
    _generate_predicted_candles,
    _standardize_ohlcv,
    _support_resistance,
    _unified_score,
    _yyyymmdd,
)

# ─────────────────────────────────────────────────────────────────────
# 기본 샘플 종목 — 대형주 + 업종 대표주 다양화
# ─────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = [
    # ── 대형 기술/반도체 ──
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "035720",  # 카카오
    "066570",  # LG전자
    # ── 자동차/모빌리티 ──
    "005380",  # 현대차
    "000270",  # 기아
    "012330",  # 현대모비스
    # ── 에너지/화학 ──
    "051910",  # LG화학
    "006400",  # 삼성SDI
    "009830",  # 한화솔루션
    "011170",  # 롯데케미칼
    # ── 바이오/헬스케어 ──
    "068270",  # 셀트리온
    "207940",  # 삼성바이오로직스
    "128940",  # 한미약품
    "326030",  # SK바이오팜
    # ── 철강/소재 ──
    "005490",  # POSCO홀딩스
    "004020",  # 현대제철
    "010130",  # 고려아연
    # ── 금융/보험 ──
    "105560",  # KB금융
    "055550",  # 신한지주
    "086790",  # 하나금융지주
    "000810",  # 삼성화재
    # ── 소비재/유통 ──
    "004170",  # 신세계
    "069960",  # 현대백화점
    "282330",  # BGF리테일
    # ── 엔터/미디어 ──
    "035900",  # JYP Ent.
    "122870",  # 와이지엔터테인먼트
    # ── 건설/방산 ──
    "000720",  # 현대건설
    "047810",  # 한국항공우주
]


@dataclass
class PredictionRecord:
    """한 번의 예측 수행 결과."""
    ticker: str
    anchor_date: str         # 예측 시점 (YYYY-MM-DD) — 이 날까지의 데이터만 사용
    start_close: float       # 예측 시점 종가
    score: int               # 예측 시점에 계산된 통합 점수
    horizon: int             # 예측 대상 일수 (영업일)
    predicted_close: float   # 모델 예측 종가
    actual_close: float      # 실제 종가
    abs_pct_error: float     # |pred - actual| / actual * 100
    signed_pct_error: float  # (pred - actual) / actual * 100 — 편향 진단용
    predicted_return: float  # (pred - start) / start * 100
    actual_return: float     # (actual - start) / start * 100
    direction_correct: bool  # 방향 (+ / -) 일치 여부


@dataclass
class SRRecord:
    """지지/저항선 1개의 forward 검증 결과."""
    ticker: str
    anchor_date: str         # 레벨 탐지 시점 — 이 날까지의 데이터만 사용
    source: str              # 'algo' = _support_resistance / 'naive20' = 20일 고저 (대조군)
    kind: str                # 'support' | 'resistance'
    level: float
    start_close: float       # 탐지 시점 종가
    dist_pct: float          # 현재가 대비 거리 (%) — 멀수록 닿을 일이 없다
    touched: bool
    touch_day: int | None    # 며칠 뒤 닿았는지 (anchor 다음날 = 1)
    outcome: str             # 'bounce' | 'break' | 'unclear' | 'untouched'


@dataclass
class TickerSummary:
    ticker: str
    n_predictions: int = 0
    n_sr_levels: int = 0
    errors: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# 지지/저항 forward 검증
# ─────────────────────────────────────────────────────────────────────
# 좋은 지지선이란 "가격이 닿으면 튕겨 나오는 선"이다. 그래서 두 가지를 잰다:
#   1) touch_rate  — 애초에 가격이 닿기는 하는가 (안 닿는 선은 쓸모없다)
#   2) respect_rate — 닿았을 때 튕겼는가(bounce) vs 뚫렸는가(break)
# respect_rate가 핵심 품질 지표다. 'unclear'(횡보하다 기간 종료)는 분모에서 뺀다.

SR_TOUCH_BAND = 0.005   # 레벨의 ±0.5% 안에 들어오면 '닿았다'
SR_MOVE_PCT = 0.02      # 2% 이탈해야 bounce/break로 인정 (노이즈 배제)

# 운영이 지지/저항·점수 계산에 넘기는 봉 수 (365일 ≈ 245영업일).
# 백테스트도 이 길이로 맞춰야 _support_resistance가 운영과 같은 레벨을 낸다.
PROD_LOOKBACK = 245


def evaluate_level(
    df: pd.DataFrame,
    anchor_idx: int,
    horizon: int,
    level: float,
    kind: str,
) -> tuple[bool, int | None, str]:
    """
    anchor_idx 시점에 탐지된 level을 이후 horizon 영업일 동안 추적.
    return: (touched, touch_day, outcome)

    접촉 판정은 고가/저가로(장중에 닿았는지), 결말 판정은 **종가**로 한다.
    결말에 고가/저가를 쓰면 안 된다: 지지선까지 내려온 날은 그날 고가가 지지선보다
    위에 있는 게 당연한데(내려오는 길이므로), 그걸 '튕겼다'로 읽어 뚫린 사례까지
    bounce로 뒤집힌다. 종가는 장 마감 시점이라 접촉 이후 상태를 대표한다.
    """
    n = len(df)
    end = min(anchor_idx + horizon, n - 1)
    if end <= anchor_idx or not np.isfinite(level) or level <= 0:
        return False, None, "untouched"

    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)

    is_support = kind == "support"

    # ── 최초 접촉일 찾기 ──
    touch_k: int | None = None
    for k in range(anchor_idx + 1, end + 1):
        if is_support:
            if lows[k] <= level * (1 + SR_TOUCH_BAND):
                touch_k = k
                break
        else:
            if highs[k] >= level * (1 - SR_TOUCH_BAND):
                touch_k = k
                break
    if touch_k is None:
        return False, None, "untouched"

    touch_day = touch_k - anchor_idx

    # ── 접촉 이후 결말 판정 (종가 기준, 먼저 성립하는 쪽) ──
    for j in range(touch_k, end + 1):
        if is_support:
            if closes[j] <= level * (1 - SR_MOVE_PCT):   # 아래로 확정 이탈 → 뚫림
                return True, touch_day, "break"
            if closes[j] >= level * (1 + SR_MOVE_PCT):   # 위로 회복 → 튕김
                return True, touch_day, "bounce"
        else:
            if closes[j] >= level * (1 + SR_MOVE_PCT):   # 위로 확정 돌파 → 뚫림
                return True, touch_day, "break"
            if closes[j] <= level * (1 - SR_MOVE_PCT):   # 아래로 밀림 → 튕김
                return True, touch_day, "bounce"

    return True, touch_day, "unclear"


def naive_levels(df: pd.DataFrame, anchor_idx: int, window: int = 20) -> tuple[float, float]:
    """
    대조군: 최근 20일 최저/최고. 알고리즘이 이것보다 나은지 비교하기 위한 기준선.
    이걸 못 이기면 복잡한 점수화가 값을 못 하는 것이다.
    """
    lo_slice = pd.to_numeric(df["low"], errors="coerce").iloc[max(0, anchor_idx - window + 1): anchor_idx + 1]
    hi_slice = pd.to_numeric(df["high"], errors="coerce").iloc[max(0, anchor_idx - window + 1): anchor_idx + 1]
    return float(lo_slice.min()), float(hi_slice.max())


# ─────────────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────────────
def load_history(ticker: str, history_days: int) -> pd.DataFrame:
    """한 종목의 전체 OHLCV (학습 + 검증 + 예측대상 future)를 한 번에 로드."""
    end_d = date.today()
    start_d = end_d - timedelta(days=history_days)
    raw = _fetch_ohlcv_sync(_yyyymmdd(start_d), _yyyymmdd(end_d), ticker)
    if raw.empty:
        raise RuntimeError(f"{ticker}: OHLCV 데이터 없음")
    df = _standardize_ohlcv(raw)
    df = _clean_ohlcv(df).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# 단일 시점 예측
# ─────────────────────────────────────────────────────────────────────
def load_market_history(history_days: int) -> pd.DataFrame:
    """KOSPI 지수 전체 히스토리를 한 번 로드 (모든 종목/anchor에 재사용)."""
    end_d = date.today()
    start_d = end_d - timedelta(days=history_days)
    df = _fetch_index_sync(_yyyymmdd(start_d), _yyyymmdd(end_d), "KS11")
    if df is None or df.empty:
        print("⚠ KOSPI 지수 로드 실패 — 베타 차감 없이 진행")
        return pd.DataFrame()
    return df


def slice_market_to(market_df: pd.DataFrame, anchor_date: str) -> pd.DataFrame:
    """anchor_date 이전(포함) 데이터만 반환 — walk-forward 누수 방지."""
    if market_df is None or market_df.empty:
        return pd.DataFrame()
    try:
        anchor_ts = pd.Timestamp(anchor_date)
        return market_df.loc[market_df.index <= anchor_ts]
    except Exception:
        return pd.DataFrame()


def predict_at(df_slice: pd.DataFrame, horizons: list[int],
               market_slice: pd.DataFrame | None = None
               ) -> tuple[int, dict[int, float], list[float], list[float]] | None:
    """
    df_slice 의 마지막 날 기준으로 예측 수행.
    return: (score, {horizon: predicted_close}, support_lines, resistance_lines)
    예측 실패(데이터 부족 등) 시 None.

    지지/저항도 함께 돌려준다 — 여기서 이미 계산하므로 호출부가 다시 계산하면
    같은 값을 두 번 구하게 되고, 무엇보다 두 경로가 갈라질 수 있다.
    """
    if len(df_slice) < 60:  # 기본 지표 계산에 필요한 최소량
        return None

    # 운영과 동일 조건으로 지지/저항·점수를 계산한다.
    # 운영(/api/stock, _load_stock_for_score_sync)은 365일(≈245봉)만 넘기는데,
    # 백테스트는 history_days=730(≈479봉)을 통째로 슬라이스한다. _support_resistance는
    # lookback에 직접 의존하므로(recency 분모 = lookback-1, tolerance의 price_range =
    # 전체 구간 고저) 길이가 다르면 운영과 다른 레벨이 나온다. 마지막 PROD_LOOKBACK
    # 봉으로 잘라 조건을 맞춘다. anchor 개수·예측 로직에는 영향 없다.
    prod_slice = df_slice.iloc[-PROD_LOOKBACK:] if len(df_slice) > PROD_LOOKBACK else df_slice

    close_values = pd.to_numeric(prod_slice["close"], errors="coerce").dropna().to_numpy(dtype=float)
    high_values = pd.to_numeric(prod_slice["high"], errors="coerce").dropna().to_numpy(dtype=float)
    low_values = pd.to_numeric(prod_slice["low"], errors="coerce").dropna().to_numpy(dtype=float)
    vol_values = pd.to_numeric(prod_slice["volume"], errors="coerce").dropna().to_numpy(dtype=float)

    try:
        support, resistance = _support_resistance(
            close_values, high_values, low_values, vol_values, max_lines=1
        )
        box = _detect_box_range(prod_slice)
        score, _signals, internals = _unified_score(prod_slice, support, resistance, box)
    except Exception:
        return None

    max_h = max(horizons)
    try:
        candles = _generate_predicted_candles(
            df=df_slice,
            prediction_score=score,
            internals=internals,
            support_lines=support,
            resistance_lines=resistance,
            box_range=box,
            n_days=max_h,
            market_df=market_slice if market_slice is not None and not market_slice.empty else None,
        )
    except Exception:
        return None

    if not candles or len(candles) < max_h:
        return None

    preds: dict[int, float] = {}
    for h in horizons:
        c = candles[h - 1]
        val = c.get("close")
        if val is None or not np.isfinite(val):
            return None
        preds[h] = float(val)
    return score, preds, support, resistance


# ─────────────────────────────────────────────────────────────────────
# 한 종목 전체 walk-forward
# ─────────────────────────────────────────────────────────────────────
def backtest_ticker(
    ticker: str,
    validation_days: int,
    horizons: list[int],
    step: int,
    history_days: int,
    market_df: pd.DataFrame | None = None,
    sr_horizon: int = 20,
) -> tuple[list[PredictionRecord], list[SRRecord], TickerSummary]:
    summary = TickerSummary(ticker=ticker)
    records: list[PredictionRecord] = []
    sr_records: list[SRRecord] = []

    try:
        df = load_history(ticker, history_days=history_days)
    except Exception as e:
        summary.errors.append(f"load: {e}")
        return records, sr_records, summary

    if len(df) < 120:
        summary.errors.append(f"데이터 부족 ({len(df)}봉)")
        return records, sr_records, summary

    # 검증 구간 — 마지막 max_h 일은 실제값 확보 때문에 제외.
    # sr_horizon을 여기 섞으면 anchor 범위가 바뀌어 기존 예측 baseline과 비교가 깨진다.
    # 지지/저항은 아래 루프에서 forward 데이터가 충분한 anchor만 따로 걸러 쓴다.
    max_h = max(horizons)
    last_idx = len(df) - 1 - max_h  # 이 인덱스 이하에서만 anchor 가능
    first_idx = max(120, len(df) - validation_days - max_h)  # 초기 워밍업 + validation window

    if last_idx <= first_idx:
        summary.errors.append("검증 구간 부족")
        return records, summary

    for anchor_idx in range(first_idx, last_idx + 1, step):
        df_slice = df.iloc[: anchor_idx + 1].copy()
        start_close = float(df_slice["close"].iloc[-1])
        anchor_date = str(df_slice["time"].iloc[-1])[:10]

        market_slice = slice_market_to(market_df, anchor_date) if market_df is not None else None
        result = predict_at(df_slice, horizons, market_slice=market_slice)
        if result is None:
            continue
        score, preds, support, resistance = result

        # ── 지지/저항 forward 검증 ──
        # sr_horizon 만큼 앞이 확보된 anchor만 평가한다. 끝자락에서 창을 잘라 쓰면
        # 볼 시간이 모자라 untouched/unclear로 몰려 품질이 과소평가된다.
        if anchor_idx + sr_horizon < len(df):
            algo_pairs = [("support", support[0] if support else None),
                          ("resistance", resistance[0] if resistance else None)]
            naive_low, naive_high = naive_levels(df, anchor_idx, window=20)
            naive_pairs = [
                # 대조군도 알고리즘과 같은 쪽(현재가 위/아래) 조건을 만족할 때만 비교
                ("support", naive_low if naive_low < start_close else None),
                ("resistance", naive_high if naive_high > start_close else None),
            ]

            for src, pairs in (("algo", algo_pairs), ("naive20", naive_pairs)):
                for kind, level in pairs:
                    if level is None or not np.isfinite(level) or level <= 0:
                        continue
                    touched, touch_day, outcome = evaluate_level(
                        df, anchor_idx, sr_horizon, float(level), kind
                    )
                    sr_records.append(SRRecord(
                        ticker=ticker,
                        anchor_date=anchor_date,
                        source=src,
                        kind=kind,
                        level=float(level),
                        start_close=start_close,
                        dist_pct=float(abs(level - start_close) / start_close * 100),
                        touched=bool(touched),
                        touch_day=touch_day,
                        outcome=outcome,
                    ))

        for h in horizons:
            if anchor_idx + h >= len(df):
                continue
            actual_close = float(df.iloc[anchor_idx + h]["close"])
            if actual_close <= 0:
                continue
            predicted_close = preds[h]

            abs_pct = abs(predicted_close - actual_close) / actual_close * 100
            signed_pct = (predicted_close - actual_close) / actual_close * 100
            pred_ret = (predicted_close - start_close) / start_close * 100
            act_ret = (actual_close - start_close) / start_close * 100
            direction_ok = (pred_ret >= 0) == (act_ret >= 0)

            records.append(PredictionRecord(
                ticker=ticker,
                anchor_date=anchor_date,
                start_close=start_close,
                score=int(score),
                horizon=h,
                predicted_close=float(predicted_close),
                actual_close=float(actual_close),
                abs_pct_error=float(abs_pct),
                signed_pct_error=float(signed_pct),
                predicted_return=float(pred_ret),
                actual_return=float(act_ret),
                direction_correct=bool(direction_ok),
            ))

    summary.n_predictions = len(records)
    summary.n_sr_levels = len(sr_records)
    return records, sr_records, summary


# ─────────────────────────────────────────────────────────────────────
# 지표 집계
# ─────────────────────────────────────────────────────────────────────
def aggregate_sr(records: list[SRRecord]) -> dict[str, dict[str, Any]]:
    """source×kind 별 지지/저항 품질 집계. 키: 'algo/support' 형태."""
    out: dict[str, dict[str, Any]] = {}
    for src in ("algo", "naive20"):
        for kind in ("support", "resistance"):
            sel = [r for r in records if r.source == src and r.kind == kind]
            if not sel:
                continue
            touched = [r for r in sel if r.touched]
            bounce = sum(1 for r in touched if r.outcome == "bounce")
            broke = sum(1 for r in touched if r.outcome == "break")
            unclear = sum(1 for r in touched if r.outcome == "unclear")
            decided = bounce + broke
            out[f"{src}/{kind}"] = {
                "n_levels": len(sel),
                "touch_rate": round(len(touched) / len(sel) * 100, 1),
                "n_touched": len(touched),
                "bounce": bounce,
                "break": broke,
                "unclear": unclear,
                # 핵심 지표 — 닿았을 때 튕긴 비율. 판정 불가(unclear)는 분모에서 제외.
                "respect_rate": round(bounce / decided * 100, 1) if decided else None,
                "avg_dist_pct": round(float(np.mean([r.dist_pct for r in sel])), 2),
                "avg_touch_day": round(float(np.mean([r.touch_day for r in touched])), 1) if touched else None,
            }
    return out


def fmt_table_sr(stats: dict[str, dict[str, Any]], sr_horizon: int) -> str:
    if not stats:
        return "\n=== 지지/저항 품질 ===\n(데이터 없음)\n"
    lines = [
        f"\n=== 지지/저항 품질 (탐지 후 {sr_horizon}영업일 추적) ===",
        "┌───────────────────┬────────┬──────────┬────────┬────────┬─────────┬──────────────┬──────────┐",
        "│ 구분              │ 레벨수 │ 접촉률   │ 튕김   │ 뚫림   │ 판정불가│ 존중률★      │ 평균거리 │",
        "├───────────────────┼────────┼──────────┼────────┼────────┼─────────┼──────────────┼──────────┤",
    ]
    label = {
        "algo/support": "알고리즘 지지",
        "algo/resistance": "알고리즘 저항",
        "naive20/support": "20일저가(대조)",
        "naive20/resistance": "20일고가(대조)",
    }
    for key in ("algo/support", "naive20/support", "algo/resistance", "naive20/resistance"):
        s = stats.get(key)
        if not s:
            continue
        rr = f"{s['respect_rate']:.1f}%" if s["respect_rate"] is not None else "N/A"
        lines.append(
            f"│ {label[key]:<17} │ {s['n_levels']:>6} │ {s['touch_rate']:>7.1f}% │"
            f" {s['bounce']:>6} │ {s['break']:>6} │ {s['unclear']:>7} │ {rr:>12} │ {s['avg_dist_pct']:>7.2f}% │"
        )
    lines.append("└───────────────────┴────────┴──────────┴────────┴────────┴─────────┴──────────────┴──────────┘")
    lines.append("★ 존중률 = 튕김/(튕김+뚫림). 알고리즘이 대조군(20일 고저)보다 높아야 값을 하는 것이다.")
    return "\n".join(lines)


def aggregate_by_horizon(records: list[PredictionRecord]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    horizons = sorted({r.horizon for r in records})
    for h in horizons:
        subset = [r for r in records if r.horizon == h]
        if not subset:
            continue
        abs_errors = np.array([r.abs_pct_error for r in subset])
        signed_errors = np.array([r.signed_pct_error for r in subset])
        price_errors = np.array([abs(r.predicted_close - r.actual_close) for r in subset])
        dir_correct = np.array([r.direction_correct for r in subset])
        out[h] = {
            "n": len(subset),
            "mape_pct": float(np.mean(abs_errors)),
            "mape_median_pct": float(np.median(abs_errors)),
            "bias_pct": float(np.mean(signed_errors)),  # >0 이면 예측이 전반적으로 상향 편향
            "rmse_price": float(np.sqrt(np.mean(price_errors ** 2))),
            "direction_accuracy_pct": float(np.mean(dir_correct) * 100),
        }
    return out


def aggregate_by_score_bucket(records: list[PredictionRecord], horizon: int) -> list[dict[str, Any]]:
    """
    점수 시스템의 예측력 검증 — 점수 구간별 실제 수익률.
    점수가 높을수록 실제 수익률도 높아야 시스템이 유효함.
    """
    subset = [r for r in records if r.horizon == horizon]
    if not subset:
        return []

    buckets = [
        ("75+ 강한 매수", lambda s: s >= 75),
        ("60-75 매수",    lambda s: 60 <= s < 75),
        ("50-60 관망상",  lambda s: 50 <= s < 60),
        ("35-50 관망하",  lambda s: 35 <= s < 50),
        ("<35 매도",      lambda s: s < 35),
    ]

    out: list[dict[str, Any]] = []
    for label, fn in buckets:
        bucket = [r for r in subset if fn(r.score)]
        if not bucket:
            out.append({"bucket": label, "n": 0, "mean_return_pct": None, "hit_rate_pct": None})
            continue
        returns = np.array([r.actual_return for r in bucket])
        hit = np.mean([r.actual_return > 0 for r in bucket]) * 100  # 실제 상승 비율
        out.append({
            "bucket": label,
            "n": int(len(bucket)),
            "mean_return_pct": float(np.mean(returns)),
            "median_return_pct": float(np.median(returns)),
            "hit_rate_pct": float(hit),
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────────
def fmt_table_horizon(by_h: dict[int, dict[str, float]]) -> str:
    lines = []
    lines.append("┌────────┬────────┬──────────┬──────────┬──────────┬──────────────┐")
    lines.append("│ Horizon│   N    │   MAPE   │  Median  │   Bias   │ DirectionAcc │")
    lines.append("├────────┼────────┼──────────┼──────────┼──────────┼──────────────┤")
    for h in sorted(by_h):
        m = by_h[h]
        lines.append(
            f"│  {h:>3}일 │ {m['n']:>6} │ {m['mape_pct']:>6.2f}%  │ {m['mape_median_pct']:>6.2f}%  │"
            f" {m['bias_pct']:>+6.2f}%  │    {m['direction_accuracy_pct']:>5.1f}%    │"
        )
    lines.append("└────────┴────────┴──────────┴──────────┴──────────┴──────────────┘")
    return "\n".join(lines)


def fmt_table_buckets(buckets: list[dict[str, Any]], horizon: int) -> str:
    lines = []
    lines.append(f"\n=== 점수 구간별 {horizon}일 실제 수익률 (점수 시스템 유효성 검증) ===")
    lines.append("┌──────────────────┬────────┬──────────────┬──────────────┬──────────────┐")
    lines.append("│ 점수 구간        │   N    │ 평균 수익률  │ 중앙값 수익률│ 상승 비율    │")
    lines.append("├──────────────────┼────────┼──────────────┼──────────────┼──────────────┤")
    for b in buckets:
        mean_str = f"{b['mean_return_pct']:>+6.2f}%" if b['mean_return_pct'] is not None else "   N/A "
        median_str = f"{b.get('median_return_pct', 0):>+6.2f}%" if b['mean_return_pct'] is not None else "   N/A "
        hit_str = f"{b['hit_rate_pct']:>5.1f}%" if b['hit_rate_pct'] is not None else "  N/A"
        lines.append(
            f"│ {b['bucket']:<16} │ {b['n']:>6} │    {mean_str}   │    {median_str}   │    {hit_str}    │"
        )
    lines.append("└──────────────────┴────────┴──────────────┴──────────────┴──────────────┘")
    lines.append("✔ 점수가 높을수록 평균 수익률이 단조증가해야 점수 시스템이 유효합니다.")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="예측 모델 백테스트")
    parser.add_argument("--tickers", type=str, default="",
                        help="콤마 구분 티커 목록 (기본: 대형주 10종목)")
    parser.add_argument("--horizons", type=str, default="1,3,7",
                        help="예측 대상 영업일 (콤마 구분)")
    parser.add_argument("--step", type=int, default=5,
                        help="N영업일마다 예측 수행 (기본 5 = 주 1회)")
    parser.add_argument("--validation-days", type=int, default=180,
                        help="검증 기간 (달력일 기준, 기본 180일)")
    parser.add_argument("--history-days", type=int, default=730,
                        help="종목당 로드할 총 히스토리 (기본 730일)")
    parser.add_argument("--label", type=str, default="baseline",
                        help="결과 JSON 파일명 태그 (예: 'after_tune')")
    parser.add_argument("--sr-horizon", type=int, default=20,
                        help="지지/저항 탐지 후 추적할 영업일 (기본 20)")
    args = parser.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()] or DEFAULT_TICKERS
    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})

    print(f"▶ 백테스트 시작")
    print(f"  종목수:     {len(tickers)}  ({', '.join(tickers)})")
    print(f"  예측기간:   {horizons} 영업일")
    print(f"  Step:       {args.step}영업일마다 예측")
    print(f"  검증기간:   최근 {args.validation_days}일")
    print(f"  히스토리:   종목당 {args.history_days}일 로드\n")

    t0 = time.time()
    all_records: list[PredictionRecord] = []
    all_sr: list[SRRecord] = []
    all_summaries: list[TickerSummary] = []

    # KOSPI 지수 한 번 로드 (베타 차감용 — 모든 종목에서 재사용)
    print("▶ KOSPI 지수 로드 중...")
    market_df = load_market_history(history_days=args.history_days)
    if market_df.empty:
        print("  ⚠ 베타 차감 없이 진행\n")
    else:
        print(f"  ✔ KOSPI {len(market_df)}봉 로드\n")

    for i, ticker in enumerate(tickers, start=1):
        try:
            records, sr_records, summary = backtest_ticker(
                ticker=ticker,
                validation_days=args.validation_days,
                horizons=horizons,
                step=args.step,
                history_days=args.history_days,
                market_df=market_df,
                sr_horizon=args.sr_horizon,
            )
            all_records.extend(records)
            all_sr.extend(sr_records)
            all_summaries.append(summary)
            status = (f"{summary.n_predictions}건 / S·R {summary.n_sr_levels}개"
                      if summary.n_predictions > 0 else f"실패: {summary.errors}")
            print(f"  [{i:>2}/{len(tickers)}] {ticker}: {status}")
        except Exception as e:
            print(f"  [{i:>2}/{len(tickers)}] {ticker}: 예외 — {e}")
            traceback.print_exc()
            all_summaries.append(TickerSummary(ticker=ticker, errors=[str(e)]))

    elapsed = time.time() - t0
    print(f"\n✔ 완료 ({elapsed:.1f}s, 총 {len(all_records)}건 예측)\n")

    if not all_records:
        print("⚠ 결과 없음 — 종료")
        return 1

    # 집계 및 출력
    by_h = aggregate_by_horizon(all_records)
    print("=== Horizon 별 정확도 ===")
    print(fmt_table_horizon(by_h))

    # 주 horizon(보통 7일)로 점수 구간 분석
    main_h = 7 if 7 in horizons else horizons[-1]
    bucket_stats = aggregate_by_score_bucket(all_records, horizon=main_h)
    print(fmt_table_buckets(bucket_stats, horizon=main_h))

    # 지지/저항 품질
    sr_stats = aggregate_sr(all_sr)
    print(fmt_table_sr(sr_stats, sr_horizon=args.sr_horizon))

    # JSON 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "backtest_results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{stamp}_{args.label}.json")

    payload = {
        "timestamp": stamp,
        "label": args.label,
        "config": {
            "tickers": tickers,
            "horizons": horizons,
            "step": args.step,
            "validation_days": args.validation_days,
            "history_days": args.history_days,
            "sr_horizon": args.sr_horizon,
        },
        "summary_by_horizon": by_h,
        "score_buckets_main_horizon": {str(main_h): bucket_stats},
        "sr_quality": sr_stats,
        "per_ticker_counts": {s.ticker: s.n_predictions for s in all_summaries},
        "records": [asdict(r) for r in all_records],
        "sr_records": [asdict(r) for r in all_sr],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n▶ 결과 저장: {out_path}")
    print(f"  (튜닝 후 --label after_tune 로 재실행 → 두 JSON 비교)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
