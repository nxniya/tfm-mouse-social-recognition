"""
scripts/_selftest_lb_scoring.py
===============================
Checks on the leaderboard-simulation scoring (notebook 07).

A. **Identifier convention and gating.** Scores the solution against itself. It must
   come out at exactly 1.0. If the `agent_id`/`target_id` format does not match the one
   in `behaviors_labeled`, the gate in `_single_lab_f1` discards every row and the
   score falls to 0 *silently*: this test is the only defence against that failure.

B. **Windowing ceiling, decomposed.** A submission built from the TRUE window labels,
   in four variants, to separate the two sources of loss:

     full window + fixed target   <- the original scheme
     full window + routed target  <- isolates the self-directed-behaviour failure
     core        + fixed target   <- isolates the interval quantisation
     core        + routed target  <- the current scheme

   Emitting the whole window imposes a minimum interval of W frames, and 75% of the
   annotated behaviours last fewer than 64, so that variant is bounded well below 1 by
   construction alone.

C. An empty submission, all background, must yield 0 rows.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np
from scripts.run_lb_experiment import (
    load_aligned, split_videos, video_coverage, build_solution,
    build_submission, official_score, per_lab_scores, SUFFIX)
from src.eval.mabe_metric import mouse_fbeta_records


def main():
    X, y, vids, labs, classes, ev = load_aligned("raw")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    print("cache{}: {} windows | pseudo-test {} videos, {} solution rows, {} labs".format(
        SUFFIX, len(y), len(test_v), len(sol), len({r["lab_id"] for r in sol})))

    # ── A ────────────────────────────────────────────────────────────────────
    self_sub = [{k: r[k] for k in ("video_id", "agent_id", "target_id",
                                   "action", "start_frame", "stop_frame")} for r in sol]
    a = mouse_fbeta_records(sol, self_sub)
    print("\n[A] solution vs itself              = {:.6f}   (expected 1.0)".format(a))
    ok_a = abs(a - 1.0) < 1e-9

    # ── B ────────────────────────────────────────────────────────────────────
    te = np.isin(vids, test_v)
    print("\n[B] windowing ceiling (true labels)")
    res = {}
    for core in (False, True):
        for route in (False, True):
            sub = build_submission(te, vids, ev, y[te], classes,
                                   active=(active if route else None), core=core)
            res[(core, route)] = official_score(sol, sub)
            print("      {:16s} + {:6s} target = {:.4f}".format(
                "core" if core else "full window",
                "routed" if route else "fixed", res[(core, route)]))
    best = res[(True, True)]
    print("    gain from routing the target : {:+.4f}".format(
        res[(False, True)] - res[(False, False)]))
    print("    gain from emitting the core  : {:+.4f}".format(
        res[(True, False)] - res[(False, False)]))
    print("\n    per lab (core + routed):")
    sub_best = build_submission(te, vids, ev, y[te], classes, active=active, core=True)
    for lab, s in sorted(per_lab_scores(sol, sub_best, lab_of).items(), key=lambda kv: -kv[1]):
        print("      {:22s} {:.4f}".format(lab, s))
    ok_b = 0.0 < best <= 1.0

    # ── C ────────────────────────────────────────────────────────────────────
    bg = classes.index("background")
    empty = build_submission(te, vids, ev, np.full(int(te.sum()), bg), classes,
                             active=active, core=True)
    print("\n[C] all-background                  = {} rows (expected 0)".format(len(empty)))
    ok_c = len(empty) == 0

    print("\n{}".format("ALL TESTS OK" if (ok_a and ok_b and ok_c) else "FAILED"))
    return 0 if (ok_a and ok_b and ok_c) else 1


if __name__ == "__main__":
    raise SystemExit(main())
