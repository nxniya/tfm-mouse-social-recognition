"""
scripts/run_lb_ensemble.py
==========================
Blends the HistGB (tree) and CNN-LSTM (GPU) probabilities and reapplies the decision
layer.

**Aligned by key, not by index.** The row order of the caches is set by the order in
which `ProcessPoolExecutor` finishes its tasks, which is not reproducible between
runs. Aligning by position would silently pair up unrelated windows, the same bug
that already appeared when pairing raw against corrected. Here the join is on
(video, agent, core_start), which identifies a window uniquely.

The blend weight `w` is swept over a grid. `w=0` is trees only and `w=1` is the
network only, so the sweep spans both endpoints and answers whether the blend buys
anything at all.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib, time, json, argparse
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np
import pandas as pd

from scripts.run_lb_perlab import load_full
from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score, per_lab_scores,
    split_videos, video_coverage, bootstrap_public)
from scripts.lb_threshold import decide, build_active_mask
from scripts.run_lb_decision import min_duration_filter, duration_stats

RESULTS = _r / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--tree-pred", default="lb_v3_ctrl")
    ap.add_argument("--tree-model", default="histgb")
    ap.add_argument("--gpu-proba", default="lb_gpu")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    ap.add_argument("--out", default="lb_ensemble")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)

    d = np.load(RESULTS / "{}_pred_{}.npz".format(a.tree_pred, a.tree_model))
    P_tree = d["proba_test"].astype(np.float32)
    thr_json = json.load(open(RESULTS / "{}_thr_{}.json".format(a.tree_pred, a.tree_model)))
    thr_by_lab = {k: np.asarray(v, np.float32) for k, v in thr_json.items()}

    g = np.load(RESULTS / "{}_proba.npz".format(a.gpu_proba), allow_pickle=True)
    gkey = {(str(v), int(ag), int(cs)): i for i, (v, ag, cs) in
            enumerate(zip(g["vid"], g["agent"], g["core_start"]))}
    P_gpu_all = g["proba"].astype(np.float32)

    tv, tag, tog, tcs = vids[te], ev["agent"][te], ev["target"][te], ev["core_start"][te]
    tlab = labs[te]

    # Join by key; windows with no GPU counterpart keep the tree prediction
    idx = np.full(len(tv), -1, np.int64)
    for i in range(len(tv)):
        j = gkey.get((str(tv[i]), int(tag[i]), int(tcs[i])))
        if j is not None:
            idx[i] = j
    have = idx >= 0
    print("join by key: {}/{} test windows have a GPU prediction ({:.1f}%)".format(
        int(have.sum()), len(tv), 100 * have.mean()), flush=True)
    P_gpu = P_tree.copy()
    P_gpu[have] = P_gpu_all[idx[have]]

    allowed = build_active_mask(tv, tag, tog, classes, active)
    mind, maxg = duration_stats(test_v, lab_of, cov)

    def score_for(w):
        P = (1.0 - w) * P_tree + w * P_gpu
        pred = np.full(len(P), bg, np.int64)
        for lab in sorted(set(tlab.tolist())):
            sel = np.flatnonzero(tlab == lab)
            t = thr_by_lab.get(lab, np.full(nC, .5, np.float32))
            pred[sel] = decide(P[sel], t, bg, mode="residual", allowed=allowed[sel])
        sub = build_submission(te, vids, ev, pred, classes, active=active)
        sub = min_duration_filter(sub, lab_of, mind, maxg)
        return official_score(sol, sub), sub

    rows = []
    best = (None, -1, None)
    for w in a.weights:
        sc, sub = score_for(w)
        rows.append({"w_gpu": w, "score": round(sc, 4)})
        print("  w_gpu={:.1f} -> {:.4f}".format(w, sc), flush=True)
        if sc > best[1]:
            best = (w, sc, sub)
    pd.DataFrame(rows).to_csv(RESULTS / "{}.csv".format(a.out), index=False)

    w, sc, sub = best
    print("\nmejor mezcla: w_gpu={:.1f} -> {:.4f}".format(w, sc))
    boot = bootstrap_public(sol, sub, test_v)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("IC 95% bootstrap: [{:.4f}, {:.4f}]".format(lo, hi))
    for lab, s in sorted(per_lab_scores(sol, sub, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {:22s} {:.4f}".format(lab, s))
    print("\nGuardado results/{}.csv ({:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
