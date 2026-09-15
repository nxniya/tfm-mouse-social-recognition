"""
scripts/lb_diagnose.py
======================
Per (lab, action) diagnostics over already-saved probabilities.

Answers the question the global score leaves open. InvincibleJellyfish (0.23) and
TranquilPanther (0.27) score far below the rest against oracle ceilings near 0.96, so
the information is present in the labels and the failure is in the prediction. What
matters is **which kind** of failure it is, because each has a different fix:

  * dead class      -> never predicted; the threshold has switched it off entirely
  * over-prediction -> predicted/actual ratio >> 1; precision collapses
  * under-prediction-> ratio << 1; recall collapses
  * confusion       -> predicted often and in roughly the right quantity, but in the
                       wrong places

Printed per action: official F1, real frames, predicted frames and their ratio.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib, json, argparse
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np

from scripts.run_lb_perlab import load_full
from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score,
    split_videos, video_coverage)
from scripts.lb_threshold import decide, build_active_mask
from scripts.run_lb_decision import min_duration_filter, duration_stats


def frames(rows):
    """Frames covered per action; the metric is frame-level."""
    out = {}
    for r in rows:
        out[r["action"]] = out.get(r["action"], 0) + int(r["stop_frame"]) - int(r["start_frame"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--pred", default="lb_ctx12")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--labs", nargs="+", required=True)
    a = ap.parse_args()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)

    d = np.load(_r / "results" / "{}_pred_{}.npz".format(a.pred, a.model))
    P = d["proba_test"].astype(np.float32)
    thr = {k: np.asarray(v, np.float32) for k, v in
           json.load(open(_r / "results" / "{}_thr_{}.json".format(a.pred, a.model))).items()}

    tlab = labs[te]
    allowed = build_active_mask(vids[te], ev["agent"][te], ev["target"][te], classes, active)
    mind, maxg = duration_stats(test_v, lab_of, cov)
    pred = np.full(len(P), bg, np.int64)
    for lab in sorted(set(tlab.tolist())):
        sel = np.flatnonzero(tlab == lab)
        pred[sel] = decide(P[sel], thr.get(lab, np.full(nC, .5, np.float32)), bg,
                           mode="residual", allowed=allowed[sel])
    sub = min_duration_filter(build_submission(te, vids, ev, pred, classes, active=active),
                              lab_of, mind, maxg)

    for lab in a.labs:
        lv = {v for v in test_v if lab_of[v] == lab}
        s_l = [r for r in sol if r["video_id"] in lv]
        p_l = [r for r in sub if r["video_id"] in lv]
        gt, pr = frames(s_l), frames(p_l)
        print("\n=== {}  ({} test videos, lab F1 = {:.4f}) ===".format(
            lab, len(lv), official_score(s_l, p_l)))
        print("    {:22s} {:>8s} {:>10s} {:>10s} {:>7s}".format(
            "action", "F1", "frames_gt", "frames_pred", "ratio"))
        for act in sorted(gt):
            f1 = official_score([r for r in s_l if r["action"] == act],
                                [r for r in p_l if r["action"] == act])
            g, p = gt.get(act, 0), pr.get(act, 0)
            print("    {:22s} {:8.4f} {:10d} {:10d} {:7s}".format(
                act, f1, g, p, "{:.2f}".format(p / g) if g else "-"))
        extra = sorted(set(pr) - set(gt))
        if extra:
            print("    predicted but not annotated in test: {}".format(
                {k: pr[k] for k in extra}))


if __name__ == "__main__":
    main()
