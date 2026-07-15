"""
백테스트 결과 JSON 2개를 비교한다.

CLAUDE.md의 개선 사이클("baseline 실행 → 개선 → 재실행 → 비교 → 개선 확인되면 commit")에서
'비교' 단계를 눈으로 표를 대조하며 하면 놓치기 쉬워서 스크립트로 고정했다.

사용:
  python compare_backtest.py backtest_results/<before>.json backtest_results/<after>.json
"""
from __future__ import annotations

import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_delta(before, after, higher_is_better: bool, unit: str = "") -> str:
    if before is None or after is None:
        return "     N/A    "
    d = after - before
    if abs(d) < 1e-9:
        return f"  =0.00{unit}   "
    good = (d > 0) == higher_is_better
    return f"  {d:+.2f}{unit} {'개선' if good else '악화'}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    b, a = load(sys.argv[1]), load(sys.argv[2])
    print(f"before: {b.get('label')}  ({b.get('timestamp')})")
    print(f"after : {a.get('label')}  ({a.get('timestamp')})")

    bc, ac = b.get("config", {}), a.get("config", {})
    if bc.get("tickers") != ac.get("tickers") or bc.get("step") != ac.get("step"):
        print("\n⚠ 설정이 다르다 — 같은 조건에서 비교해야 의미가 있다")
        print(f"   before: {len(bc.get('tickers', []))}종목 step={bc.get('step')} sr_horizon={bc.get('sr_horizon')}")
        print(f"   after : {len(ac.get('tickers', []))}종목 step={ac.get('step')} sr_horizon={ac.get('sr_horizon')}")

    # ── 지지/저항 품질 ──
    bs, as_ = b.get("sr_quality") or {}, a.get("sr_quality") or {}
    if bs or as_:
        print("\n=== 지지/저항 품질 ===")
        print(f"{'구분':<20}{'지표':<12}{'before':>10}{'after':>10}{'변화':>16}")
        print("─" * 70)
        for key in ("algo/support", "naive20/support", "algo/resistance", "naive20/resistance"):
            sb, sa = bs.get(key), as_.get(key)
            if not sb or not sa:
                continue
            for metric, better in (("respect_rate", True), ("touch_rate", True), ("n_touched", True)):
                vb, va = sb.get(metric), sa.get(metric)
                vbs = f"{vb:.1f}" if isinstance(vb, float) else str(vb)
                vas = f"{va:.1f}" if isinstance(va, float) else str(va)
                print(f"{key:<20}{metric:<12}{vbs:>10}{vas:>10}{fmt_delta(vb, va, better):>16}")
            print("─" * 70)

        # 핵심 판정: 알고리즘이 대조군을 이기는가
        for kind in ("support", "resistance"):
            ab = (bs.get(f"algo/{kind}") or {}).get("respect_rate")
            nb = (bs.get(f"naive20/{kind}") or {}).get("respect_rate")
            aa = (as_.get(f"algo/{kind}") or {}).get("respect_rate")
            na = (as_.get(f"naive20/{kind}") or {}).get("respect_rate")
            if None in (ab, nb, aa, na):
                continue
            print(f"[{kind}] 알고리즘 − 대조군 존중률: before {ab - nb:+.1f}%p → after {aa - na:+.1f}%p")

    # ── 예측 정확도 (회귀 확인용) ──
    bh, ah = b.get("summary_by_horizon") or {}, a.get("summary_by_horizon") or {}
    if bh and ah:
        print("\n=== 예측 정확도 (S·R 변경이 예측에 준 영향) ===")
        print(f"{'Horizon':<10}{'지표':<16}{'before':>10}{'after':>10}{'변화':>16}")
        print("─" * 66)
        for h in sorted(set(bh) & set(ah), key=lambda x: int(x)):
            for metric, better in (("mape_pct", False), ("direction_accuracy_pct", True), ("bias_pct", False)):
                vb, va = bh[h].get(metric), ah[h].get(metric)
                if vb is None or va is None:
                    continue
                # bias는 0에 가까울수록 좋다 — 절댓값으로 비교
                if metric == "bias_pct":
                    d = abs(va) - abs(vb)
                    tag = "개선" if d < 0 else ("악화" if d > 0 else "=")
                    print(f"{h + '일':<10}{metric:<16}{vb:>10.2f}{va:>10.2f}{f'  |{d:+.2f}| {tag}':>16}")
                else:
                    print(f"{h + '일':<10}{metric:<16}{vb:>10.2f}{va:>10.2f}{fmt_delta(vb, va, better):>16}")
            print("─" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
