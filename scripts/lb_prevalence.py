"""
scripts/lb_prevalence.py
========================
Threshold correction by **prevalence**, without using any test label.

The diagnostic that motivates it (`scripts/lb_diagnose.py` over `lb_ctx12`):

  InvincibleJellyfish  4 of 8 actions with 0 predicted frames -> F1 exactly 0
                       (allogroom, dominancegroom, escape, selfgroom)
  TranquilPanther      sniffgenital 2.95x  mount 2.44x  sniff 2.34x  intromit 0.20x

Two opposite pathologies with one cause: thresholds tuned by coordinate ascent
optimise the lab's score over a handful of calibration videos, and for a rare action
the local optimum is to **switch it off**. But the official metric averages F1 per
action, so an action that is never predicted contributes exactly 0, the worst possible
value: almost any detection, however poor, scores higher.

The correction targets the **per-action frame rate observed in the training videos**
and looks for the threshold that reproduces that rate on test. It calibrates counts,
not hits.

A methodological caveat, stated plainly: the threshold is chosen using the test
*probabilities*, so the method is **transductive**. It uses the test features, which
the competition publishes, but **at no point their labels**. It has to be declared as
such: this is not label leakage, but neither is it a purely inductive model.
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
    split_videos, video_coverage, bootstrap_public)
from scripts.lb_threshold import decide, build_active_mask, DEFAULT_GRID
from scripts.run_lb_decision import min_duration_filter, duration_stats

RESULTS = _r / "results"


def train_rates(train_v, lab_of, cov):
    """Annotated frames per (lab, action) over covered frames, on TRAIN."""
    from src.data.loader import load_annotations
    num, den = {}, {}
    for vid in train_v:
        lab = lab_of[vid]
        mx = cov.get(vid, 0)
        den[lab] = den.get(lab, 0) + mx
        ann = load_annotations(vid, lab, max_frame=mx)
        if ann is None or not len(ann):
            continue
        for r in ann.to_dict("records"):
            d = int(r["stop_frame"]) - int(r["start_frame"])
            if d > 0:
                num[(lab, r["action"])] = num.get((lab, r["action"]), 0) + d
    return {k: v / max(den.get(k[0], 1), 1) for k, v in num.items()}, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--pred", default="lb_ctx12")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--only-dead", action="store_true",
                    help="correct only the actions with 0 predicted frames")
    ap.add_argument("--revive", action="store_true",
                    help="revive dead classes by taking background windows")
    ap.add_argument("--revive-frac", type=float, default=1.0,
                    help="fraction of the train prevalence to emit when reviving")
    ap.add_argument("--out", default="lb_prev")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)
    tlab = labs[te]

    P = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred, a.model))["proba_test"].astype(np.float32)
    thr0 = {k: np.asarray(v, np.float32) for k, v in
            json.load(open(RESULTS / "{}_thr_{}.json".format(a.pred, a.model))).items()}
    allowed = build_active_mask(vids[te], ev["agent"][te], ev["target"][te], classes, active)
    mind, maxg = duration_stats(test_v, lab_of, cov)
    rates, _ = train_rates(train_v, lab_of, cov)

    def emit(lab, tvec):
        sel = np.flatnonzero(tlab == lab)
        pr = np.full(len(P), bg, np.int64)
        pr[sel] = decide(P[sel], tvec, bg, mode="residual", allowed=allowed[sel])
        m = np.zeros(len(vids), bool); m[np.flatnonzero(te)[sel]] = True
        s = min_duration_filter(build_submission(m, vids, ev, pr[sel], classes, active=active),
                                lab_of, mind, maxg)
        cnt = {}
        for r in s:
            cnt[r["action"]] = cnt.get(r["action"], 0) + int(r["stop_frame"]) - int(r["start_frame"])
        return s, cnt

    thr_new, report = {}, []
    for lab in sorted(set(tlab.tolist())):
        tvec = thr0.get(lab, np.full(nC, .5, np.float32)).copy()
        lv = [v for v in test_v if lab_of[v] == lab]
        n_frames = sum(cov.get(v, 0) for v in lv)
        _, cnt0 = emit(lab, tvec)
        acts = sorted({r["action"] for r in sol if lab_of[r["video_id"]] == lab})
        for _p in range(a.passes):
            for act in acts:
                if act not in classes:
                    continue
                tgt = rates.get((lab, act), 0.0) * n_frames
                if tgt <= 0:
                    continue
                if a.only_dead and cnt0.get(act, 0) > 0:
                    continue
                c = classes.index(act)
                best_t, best_d = tvec[c], None
                for t in DEFAULT_GRID:
                    old = tvec[c]; tvec[c] = t
                    _, cnt = emit(lab, tvec)
                    tvec[c] = old
                    d = abs(np.log1p(cnt.get(act, 0)) - np.log1p(tgt))
                    if best_d is None or d < best_d:
                        best_d, best_t = d, t
                tvec[c] = best_t
            _, cnt0 = emit(lab, tvec)
        thr_new[lab] = tvec
        for act in acts:
            tgt = rates.get((lab, act), 0.0) * n_frames
            report.append({"lab": lab, "action": act, "target": int(tgt),
                           "pred": int(cnt0.get(act, 0))})
        print("    {:22s} recalibrado".format(lab), flush=True)

    pred = np.full(len(P), bg, np.int64)
    for lab in sorted(set(tlab.tolist())):
        sel = np.flatnonzero(tlab == lab)
        pred[sel] = decide(P[sel], thr_new[lab], bg, mode="residual", allowed=allowed[sel])

    sub = min_duration_filter(build_submission(te, vids, ev, pred, classes, active=active),
                              lab_of, mind, maxg)
    if a.revive:
        # Lowering the threshold is not enough: `decide` is an argmax over residuals,
        # so a rare action at p~0.01 never beats `sniff` at p~0.6 no matter how low its
        # threshold goes. And a class that is NEVER predicted scores exactly 0, the
        # worst possible value, while the metric averages F1 per action.
        #
        # Reviving happens AFTER the minimum-duration filter, not before: that filter
        # deletes short intervals, which is exactly the shape a rare action has. Run
        # before it, the filter undid half the work and the class stayed dead. Only
        # windows already classified as BACKGROUND are given up, so no live action
        # loses anything.
        emitted = {}
        for r in sub:
            k = (lab_of[r["video_id"]], r["action"])
            emitted[k] = emitted.get(k, 0) + int(r["stop_frame"]) - int(r["start_frame"])
        S = 8
        extra = np.full(len(vids), bg, np.int64)
        touched = np.zeros(len(vids), bool)
        te_idx = np.flatnonzero(te)
        for lab in sorted(set(tlab.tolist())):
            sel = np.flatnonzero(tlab == lab)
            lv = [v for v in test_v if lab_of[v] == lab]
            n_frames = sum(cov.get(v, 0) for v in lv)
            for act in sorted({r["action"] for r in sol if lab_of[r["video_id"]] == lab}):
                if act not in classes or emitted.get((lab, act), 0) > 0:
                    continue
                c = classes.index(act)
                tgt = rates.get((lab, act), 0.0) * n_frames * a.revive_frac
                n_win = int(round(tgt / S))
                cand = sel[(pred[sel] == bg) & allowed[sel][:, c]]
                if n_win <= 0 or len(cand) == 0:
                    continue
                take = cand[np.argsort(-P[cand, c])[:min(n_win, len(cand))]]
                extra[te_idx[take]] = c
                touched[te_idx[take]] = True
                print("    revived {:22s} {:16s} {} windows".format(lab, act, len(take)),
                      flush=True)
        if touched.any():
            sub = sub + build_submission(touched, vids, ev, extra[touched], classes,
                                         active=active)

    sc = official_score(sol, sub)
    print("\nscore with prevalence-based thresholds = {:.4f}".format(sc))
    lo, hi = np.percentile(bootstrap_public(sol, sub, test_v), [2.5, 97.5])
    print("public bootstrap 95% CI: [{:.4f}, {:.4f}]".format(lo, hi))
    for lab, s in sorted(per_lab_scores(sol, sub, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {:22s} {:.4f}".format(lab, s))
    json.dump({k: v.tolist() for k, v in thr_new.items()},
              open(RESULTS / "{}_thr_{}.json".format(a.out, a.model), "w"), indent=1)
    import pandas as pd
    pd.DataFrame(report).to_csv(RESULTS / "{}_report.csv".format(a.out), index=False)
    print("\nGuardado results/{}_thr_{}.json ({:.0f}s)".format(a.out, a.model,
                                                              time.perf_counter() - t00))


if __name__ == "__main__":
    main()
