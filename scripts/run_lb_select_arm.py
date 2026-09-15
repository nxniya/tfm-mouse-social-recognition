"""
scripts/run_lb_select_arm.py
============================
Selects **per lab** between several runs (for instance `class_weight=balanced` versus
`class_weight=none`) and reapplies the decision layer to the result.

Why it exists: the `class_weight` ablation over the three weakest labs gave
TranquilPanther 0.2713 -> 0.3114 but CRIM13 0.3810 -> 0.3682. The best arm depends on
the lab. Taking the best of each **by looking at the test score** would leak the
evaluation set: the resulting number would no longer estimate anything.

Here the choice is made using the CALIBRATION score each run saves in
`<out>_calib_<model>.json`, computed on held-out train videos. The test score is
computed once, after the decisions are already fixed, and therefore remains an honest
estimate.

The run named first acts as the base: labs that appear in no alternative keep its
probabilities and thresholds.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib, json, time, argparse
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np

from scripts.run_lb_perlab import load_full
from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score, per_lab_scores,
    split_videos, video_coverage, bootstrap_public, subset_score,
    REAL_TEST_LABS)
from scripts.lb_threshold import decide, smooth_proba, build_active_mask
from scripts.run_lb_decision import min_duration_filter, duration_stats

RESULTS = _r / "results"


def load_run(name, model):
    """One run's test probabilities, thresholds and calibration score."""
    d = np.load(RESULTS / "{}_pred_{}.npz".format(name, model))
    thr = {k: np.asarray(v, np.float32)
           for k, v in json.load(open(RESULTS / "{}_thr_{}.json".format(name, model))).items()}
    cf = RESULTS / "{}_calib_{}.json".format(name, model)
    cal = json.load(open(cf)) if cf.exists() else {}
    return d["proba_test"].astype(np.float32), thr, cal, d["test_mask"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names; the first one is the base")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--mode", default="residual", choices=["residual", "ratio"])
    ap.add_argument("--mask", action="store_true", default=True)
    ap.add_argument("--no-mask", dest="mask", action="store_false")
    ap.add_argument("--mindur", action="store_true", default=True)
    ap.add_argument("--no-mindur", dest="mindur", action="store_false")
    ap.add_argument("--out", default="lb_armsel")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)

    runs = {}
    for name in a.runs:
        P, thr, cal, tm = load_run(name, a.model)
        if not np.array_equal(tm, te):
            raise SystemExit("different test mask in {}: runs are not comparable".format(name))
        runs[name] = (P, thr, cal)
        print("{:16s} {} windows, calib={}".format(name, P.shape, cal), flush=True)

    base = a.runs[0]
    tlab = labs[te]
    P_sel = runs[base][0].copy()
    thr_sel = dict(runs[base][1])
    choice = {}
    for lab in sorted(set(tlab.tolist())):
        best, bsc = base, runs[base][2].get(lab, -1.0)
        for name in a.runs[1:]:
            sc = runs[name][2].get(lab, None)
            if sc is not None and sc > bsc:
                best, bsc = name, sc
        choice[lab] = {"run": best, "calib": bsc}
        if best != base:
            sel = np.flatnonzero(tlab == lab)
            P_sel[sel] = runs[best][0][sel]
            if lab in runs[best][1]:
                thr_sel[lab] = runs[best][1][lab]
    print("\nper-lab choice (by CALIBRATION):")
    for lab, c in sorted(choice.items()):
        print("    {:22s} {:16s} calib={:.4f}".format(lab, c["run"], c["calib"]))

    tv, tag, tog, tcs = vids[te], ev["agent"][te], ev["target"][te], ev["core_start"][te]
    allowed = build_active_mask(tv, tag, tog, classes, active) if a.mask else None
    if a.smooth:
        P_sel = smooth_proba(P_sel, tv, tag, tcs, window=a.smooth)
    mind, maxg = duration_stats(test_v, lab_of, cov)

    pred = np.full(len(P_sel), bg, np.int64)
    for lab in sorted(set(tlab.tolist())):
        sel = np.flatnonzero(tlab == lab)
        t = thr_sel.get(lab, np.full(nC, .5, np.float32))
        pred[sel] = decide(P_sel[sel], t, bg, mode=a.mode,
                           allowed=None if allowed is None else allowed[sel])
    sub = build_submission(te, vids, ev, pred, classes, active=active)
    if a.mindur:
        sub = min_duration_filter(sub, lab_of, mind, maxg)
    sc = official_score(sol, sub)
    print("\nofficial score (per-lab selection) = {:.4f}".format(sc))
    boot = bootstrap_public(sol, sub, test_v)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print("public bootstrap 95% CI: [{:.4f}, {:.4f}]".format(lo, hi))
    _pl = per_lab_scores(sol, sub, lab_of)
    _rs = subset_score(_pl)
    # The real competition test set excludes CalMS21 and CRIM13, which are our four
    # best labs, so the mean over all 9 overstates the true standing.
    if _rs is not None:
        print("score over {} real-test-like labs = {:.4f}  (* see below)".format(
            len([k for k in _pl if k in set(REAL_TEST_LABS)]), _rs), flush=True)
    for lab, s in sorted(per_lab_scores(sol, sub, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {}{:22s} {:.4f}  [{}]".format("*" if lab in set(REAL_TEST_LABS) else " ", lab, s, choice[lab]["run"]))
    json.dump({"score": round(float(sc), 4), "choice": choice},
              open(RESULTS / "{}.json".format(a.out), "w"), indent=1)
    print("\nGuardado results/{}.json ({:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
