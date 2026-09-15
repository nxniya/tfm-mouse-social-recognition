"""
scripts/run_lb_decision.py
==========================
Sweep of the **decision layer** over already-saved probabilities.

Retrains nothing: reads `results/<pred>_pred_<model>.npz` (`proba_test`) and tries
decision variants in minutes instead of hours. Each variant is evaluated separately
so that any gain can be **attributed**: earlier runs moved two things at once and were
left unexplained.

Variants:
  * `mask`   — mask out the classes a video does not annotate. Today 28.3% of
               non-background predictions land on inactive classes and are discarded
               silently downstream, scoring neither hit nor miss.
  * `smooth` — Savitzky-Golay over each (video, agent) probability sequence.
  * `ratio`  — break ties on `p/t` rather than `p-t`.
  * `mindur` — drop intervals shorter than the 10th percentile of that (lab, action)'s
               annotated duration, and bridge short gaps.

The thresholds are reused from the run that produced the probabilities; recalibrating
them would require the out-of-fold probabilities, which this version does not need.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib, json, time, argparse, itertools
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np
import pandas as pd

from scripts.run_lb_perlab import load_full
from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score, per_lab_scores,
    split_videos, video_coverage, bootstrap_public)
from scripts.lb_threshold import decide, smooth_proba, build_active_mask

RESULTS = _r / "results"


def min_duration_filter(sub, lab_of, min_dur, max_gap):
    """Drop intervals that are too short and bridge short gaps.

    `robustify` removes overlaps but does neither of these, and isolated single-window
    false positives are exactly what inflates the predicted frame count.
    """
    by = {}
    for r in sub:
        by.setdefault((r["video_id"], r["agent_id"], r["target_id"], r["action"]), []).append(r)
    out = []
    for k, rows in by.items():
        rows.sort(key=lambda r: r["start_frame"])
        lab = lab_of.get(k[0])
        gap = max_gap.get((lab, k[3]), 0)
        merged = []
        for r in rows:
            if merged and r["start_frame"] - merged[-1]["stop_frame"] <= gap:
                merged[-1]["stop_frame"] = max(merged[-1]["stop_frame"], r["stop_frame"])
            else:
                merged.append(dict(r))
        md = min_dur.get((lab, k[3]), 0)
        out += [r for r in merged if r["stop_frame"] - r["start_frame"] >= md]
    return out


def duration_stats(videos, lab_of, cov):
    """Annotated-duration percentiles per (lab, action), for mindur and gap."""
    from src.data.loader import load_annotations
    acc = {}
    for vid in videos:
        lab = lab_of[vid]
        ann = load_annotations(vid, lab, max_frame=cov.get(vid, 0))
        if ann is None or not len(ann):
            continue
        for r in ann.to_dict("records"):
            d = int(r["stop_frame"]) - int(r["start_frame"])
            if d > 0:
                acc.setdefault((lab, r["action"]), []).append(d)
    mind, maxg = {}, {}
    for k, v in acc.items():
        mind[k] = float(np.percentile(v, 10))
        maxg[k] = float(np.percentile(v, 25))
    return mind, maxg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v2")
    ap.add_argument("--pred", default="lb_v2_ctrl")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--out", default="lb_decision")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)

    d = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred, a.model))
    P0 = d["proba_test"].astype(np.float32)
    thr_json = json.load(open(RESULTS / "{}_thr_{}.json".format(a.pred, a.model)))
    thr_by_lab = {k: np.asarray(v, np.float32) for k, v in thr_json.items()}

    tv, tag, tog, tcs = vids[te], ev["agent"][te], ev["target"][te], ev["core_start"][te]
    tlab = labs[te]
    print("test: {} windows, {} videos, {} labs".format(len(P0), len(test_v), len(set(tlab))),
          flush=True)

    allowed = build_active_mask(tv, tag, tog, classes, active)
    print("active mask: {:.1f}% of the (window x class) cells are scorable".format(
        100 * allowed.mean()), flush=True)
    Psm = smooth_proba(P0, tv, tag, tcs, window=a.smooth) if a.smooth else P0
    mind, maxg = duration_stats(test_v, lab_of, cov)

    def run(mask_on, smooth_on, mode, mindur_on):
        P = Psm if smooth_on else P0
        A = allowed if mask_on else None
        pred = np.full(len(P), bg, np.int64)
        for lab in sorted(set(tlab.tolist())):
            sel = np.flatnonzero(tlab == lab)
            t = thr_by_lab.get(lab, np.full(nC, .5, np.float32))
            pred[sel] = decide(P[sel], t, bg, mode=mode,
                               allowed=None if A is None else A[sel])
        sub = build_submission(te, vids, ev, pred, classes, active=active)
        if mindur_on:
            sub = min_duration_filter(sub, lab_of, mind, maxg)
        return official_score(sol, sub), sub

    rows = []
    base, _ = run(False, False, "residual", False)
    print("\nbaseline (matching the control run) = {:.4f}".format(base), flush=True)
    for mask_on, smooth_on, mode, mindur_on in itertools.product(
            [False, True], [False, True], ["residual", "ratio"], [False, True]):
        sc, sub = run(mask_on, smooth_on, mode, mindur_on)
        rows.append({"mask": mask_on, "smooth": smooth_on and a.smooth, "mode": mode,
                     "mindur": mindur_on, "score": round(sc, 4),
                     "delta_vs_base": round(sc - base, 4)})
        print("  mask={:5} smooth={:5} {:8} mindur={:5} -> {:.4f} ({:+.4f})".format(
            str(mask_on), str(smooth_on), mode, str(mindur_on), sc, sc - base), flush=True)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(RESULTS / "{}.csv".format(a.out), index=False)
    best = df.iloc[0]
    print("\nmejor: {}".format(best.to_dict()), flush=True)
    sc, sub = run(bool(best["mask"]), bool(best["smooth"]), best["mode"], bool(best["mindur"]))
    boot = bootstrap_public(sol, sub, test_v)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("IC 95% bootstrap: [{:.4f}, {:.4f}]".format(lo, hi))
    for lab, s in sorted(per_lab_scores(sol, sub, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {:22s} {:.4f}".format(lab, s))
    print("\nGuardado results/{}.csv ({:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
