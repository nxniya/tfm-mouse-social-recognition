#!/usr/bin/env python3
"""
scripts/_selftest_event_bridge.py
=================================
Validates the window->interval->score bridge (``windows_to_submission`` +
``solution_from_annotations`` + ``mouse_fbeta_records``) on synthetic data, so the
glue used by ``run_lovo_benchmark --with-event-f1`` is exercised without needing
the dataset / scipy / torch. Uses only numpy.

Run:  python scripts/_selftest_event_bridge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.mabe_metric import windows_to_submission, mouse_fbeta_records  # noqa: E402


def _solution(video, agent, target, action, start, stop, lab, triples):
    return dict(video_id=video, agent_id=agent, target_id=target, action=action,
                start_frame=start, stop_frame=stop, lab_id=lab,
                behaviors_labeled=json.dumps(triples))


def main() -> int:
    ok = True

    # One video, agent=1 target=2, behaviour "sniff" over frames 0..40, then bg.
    # 5 windows of width 10 (stride 10): [0,10) [10,20) [20,30) [30,40) [40,50)
    win_video = ["v1"] * 5
    win_agent = [1] * 5
    win_target = [2] * 5
    win_fstart = [0, 10, 20, 30, 40]
    win_fstop = [10, 20, 30, 40, 50]

    classes = ["sniff", "mount", "background"]
    solution = [_solution("v1", 1, 2, "sniff", 0, 40, "labA", ["1,2,sniff"])]

    # (a) perfect prediction: sniff on the first 4 windows (0..40), bg on the last
    preds_perfect = ["sniff", "sniff", "sniff", "sniff", "background"]
    sub = windows_to_submission(win_video, win_agent, win_target,
                                win_fstart, win_fstop, preds_perfect)
    # consecutive sniff windows should merge into ONE interval [0,40)
    assert len(sub) == 1, f"expected 1 merged interval, got {len(sub)}"
    assert (sub[0]["start_frame"], sub[0]["stop_frame"]) == (0, 40), sub[0]
    f1 = mouse_fbeta_records(solution, sub)
    print(f"[1] perfect prediction -> interval {sub[0]['start_frame']}..{sub[0]['stop_frame']}, "
          f"F1={f1:.3f}")
    assert abs(f1 - 1.0) < 1e-9, f"perfect should score 1.0, got {f1}"

    # (b) wrong action everywhere -> not in active labels for "mount"? mount IS a
    # plausible action but solution only has sniff; predicting mount yields 0 tp on
    # sniff and (mount not a labelled action) -> sniff f1 = 0.
    preds_wrong = ["mount", "mount", "mount", "mount", "background"]
    sub_w = windows_to_submission(win_video, win_agent, win_target,
                                  win_fstart, win_fstop, preds_wrong)
    # "1,2,mount" not in behaviors_labeled -> ignored; sniff gets all-fn.
    f1w = mouse_fbeta_records(solution, sub_w)
    print(f"[2] wrong action      -> F1={f1w:.3f}")
    assert abs(f1w - 0.0) < 1e-9, f"wrong action should score 0.0, got {f1w}"

    # (c) half coverage -> partial score in (0,1)
    preds_half = ["sniff", "sniff", "background", "background", "background"]
    sub_h = windows_to_submission(win_video, win_agent, win_target,
                                  win_fstart, win_fstop, preds_half)
    f1h = mouse_fbeta_records(solution, sub_h)
    print(f"[3] half coverage     -> interval {sub_h[0]['start_frame']}..{sub_h[0]['stop_frame']}, "
          f"F1={f1h:.3f}")
    assert 0.0 < f1h < 1.0, f"half coverage should be strictly between 0 and 1, got {f1h}"

    # (d) background-only -> empty submission -> score 0
    sub_bg = windows_to_submission(win_video, win_agent, win_target, win_fstart,
                                   win_fstop, ["background"] * 5)
    assert sub_bg == [], "background-only should produce no intervals"
    print("[4] background only   -> empty submission (OK)")

    print("\nALL BRIDGE SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
