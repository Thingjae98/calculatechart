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
    "005930",  # 삼성전자
    "000660",  # SK하이닉스
    "035420",  # NAVER
    "035720",  # 카카오
    "005380",  # 현대차
    "051910",  # LG화학
    "006400",  # 삼성SDI
    "068270",  # 셀트리온
    "207940",  # 삼성바이오로직스
    "005490",  # POSCO홀딩스
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
class TickerSummary:
    ticker: str
    n_predictions: int = 0
    errors: list[str] = field(default_factory=list)


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
               market_slice: pd.DataFrame | None = None) -> tuple[int, dict[int, float]] | None:
    """
    df_slice 의 마지막 날 기준으로 예측 수행.
    return: (score, {horizon: predicted_close})
    예측 실패(데이터 부족 등) 시 None.
    """
    if len(df_slice) < 60:  # 기본 지표 계산에 필요한 최소량
        return None

    close_values = pd.to_numeric(df_slice["close"], errors="coerce").dropna().to_numpy(dtype=float)
    high_values = pd.to_numeric(df_slice["high"], errors="coerce").dropna().to_numpy(dtype=float)
    low_values = pd.to_numeric(df_slice["low"], errors="coerce").dropna().to_numpy(dtype=float)
    vol_values = pd.to_numeric(df_slice["volume"], errors="coerce").dropna().to_numpy(dtype=float)

    try:
        support, resistance = _support_resistance(
            close_values, high_values, low_values, vol_values, max_lines=1
        )
        box = _detect_box_range(df_slice)
        score, _signals, internals = _unified_score(df_slice, support, resistance, box)
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
    return score, preds


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
) -> tuple[list[PredictionRecord], TickerSummary]:
    summary = TickerSummary(ticker=ticker)
    records: list[PredictionRecord] = []

    try:
        df = load_history(ticker, history_days=history_days)
    except Exception as e:
        summary.errors.append(f"load: {e}")
        return records, summary

    if len(df) < 120:
        summary.errors.append(f"데이터 부족 ({len(df)}봉)")
        return records, summary

    # 검증 구간 — 마지막 max_h 일은 실제값 확보 때문에 제외
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
        score, preds = result

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
    return records, summary


# ─────────────────────────────────────────────────────────────────────
# 지표 집계
# ─────────────────────────────────────────────────────────────────────
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
            records, summary = backtest_ticker(
                ticker=ticker,
                validation_days=args.validation_days,
                horizons=horizons,
                step=args.step,
                history_days=args.history_days,
                market_df=market_df,
            )
            all_records.extend(records)
            all_summaries.append(summary)
            status = f"{summary.n_predictions}건" if summary.n_predictions > 0 else f"실패: {summary.errors}"
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
        },
        "summary_by_horizon": by_h,
        "score_buckets_main_horizon": {str(main_h): bucket_stats},
        "per_ticker_counts": {s.ticker: s.n_predictions for s in all_summaries},
        "records": [asdict(r) for r in all_records],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n▶ 결과 저장: {out_path}")
    print(f"  (튜닝 후 --label after_tune 로 재실행 → 두 JSON 비교)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
