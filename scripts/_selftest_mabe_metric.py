#!/usr/bin/env python3
"""
scripts/_selftest_mabe_metric.py
================================
Validates ``src/eval/mabe_metric`` against the *official* Kaggle metric doctests
(copied verbatim from the competition's scoring code / the XGBoost baseline
notebook). Pure Python — no pandas/numpy needed — so it runs anywhere.

Each case is (solution_records, submission_records, expected_score).
Run:  python scripts/_selftest_mabe_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.mabe_metric import mouse_fbeta_records  # noqa: E402


def S(video_id, agent_id, target_id, action, start, stop, lab_id, bl):
    return dict(video_id=video_id, agent_id=agent_id, target_id=target_id,
                action=action, start_frame=start, stop_frame=stop,
                lab_id=lab_id, behaviors_labeled=bl)


def P(video_id, agent_id, target_id, action, start, stop):
    return dict(video_id=video_id, agent_id=agent_id, target_id=target_id,
                action=action, start_frame=start, stop_frame=stop)


CASES = [
    # 1. exact match -> 1.0
    ([S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]')],
     [P(1, 1, 2, "attack", 0, 10)], 1.0),
    # 2. wrong action -> 0.0
    ([S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]')],
     [P(1, 1, 2, "mount", 0, 10)], 0.0),
    # 3. one of two label actions matched -> 0.5
    ([S(123, 1, 2, "attack", 0, 9, 1, '["1,2,attack"]'),
      S(123, 1, 2, "mount", 15, 24, 1, '["1,2,attack"]')],
     [P(123, 1, 2, "attack", 0, 9)], 0.5),
    # 4. two labs, only one has a (partial) submission -> 0.25
    ([S(123, 1, 2, "attack", 0, 9, 1, '["1,2,attack"]'),
      S(123, 1, 2, "mount", 15, 24, 1, '["1,2,attack"]'),
      S(345, 1, 2, "attack", 0, 9, 2, '["1,2,attack"]'),
      S(345, 1, 2, "mount", 15, 24, 2, '["1,2,attack"]')],
     [P(123, 1, 2, "attack", 0, 9)], 0.25),
    # 5. overlapping solution events, one covering prediction -> 1.0
    ([S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]'),
      S(1, 1, 2, "attack", 10, 20, 1, '["1,2,attack"]')],
     [P(1, 1, 2, "attack", 0, 20)], 1.0),
    # 6. prediction spanning a gap -> 0.6667
    ([S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]'),
      S(1, 1, 2, "attack", 30, 40, 1, '["1,2,attack"]')],
     [P(1, 1, 2, "attack", 0, 40)], 0.6666666666666666),
]


def main() -> int:
    ok = True
    for i, (sol, sub, expected) in enumerate(CASES, 1):
        got = mouse_fbeta_records(sol, sub)
        passed = abs(got - expected) < 1e-9
        ok &= passed
        print(f"[{i}] expected {expected:.6f}  got {got:.6f}  "
              f"{'OK' if passed else 'FAIL'}")
    print("\n" + ("ALL OFFICIAL DOCTESTS PASS" if ok else "SOME CASES FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
