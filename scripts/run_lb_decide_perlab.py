"""
scripts/run_lb_decide_perlab.py
===============================
Chooses the **decision layer per lab**, using calibration probabilities.

The earlier factorial sweep (`run_lb_decision.py`) picked ONE global configuration by
looking at the test score. That has two problems: it looks at test, and it imposes the
same rule on nine labs whose pathologies are opposite — InvincibleJellyfish suffers
from classes that are never predicted, TranquilPanther from severe over-prediction,
and there is no reason the same tie-break or the same filter should suit both.

Here each lab picks its variant using its own **calibration videos**, whose
probabilities `run_lb_perlab.py` now saves as `proba_calib`. The thresholds are
retuned **within each variant**, because a threshold calibrated for `residual` is not
valid for `ratio`. Test is touched exactly once, at the end.

Selecting by calibration is legitimate here because every variant shares the same
model and the same feature set: what is being compared are decision rules of equal
capacity, which is precisely the condition required for this criterion to be
unbiased.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import sys, pathlib, json, time, argparse, itertools
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np

from scripts.run_lb_perlab import load_full
from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score, per_lab_scores,
    split_videos, video_coverage, bootstrap_public, subset_score,
    REAL_TEST_LABS)
from scripts.lb_threshold import decide, tune_thresholds, smooth_proba, build_active_mask
from scripts.lb_viterbi import transition_logprobs, decode_lab
from scripts.run_lb_decision import min_duration_filter, duration_stats
from scripts.lb_prevalence import train_rates

RESULTS = _r / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--pred", default="lb_ctx12c")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--labs", nargs="+", default=None,
                    help="restrict to these labs; the rest are reused as-is")
    ap.add_argument("--viterbi-weights", nargs="*", type=float,
                    default=[0.5, 1.0, 2.0, 4.0],
                    help="emission weights to try; [] disables Viterbi")
    ap.add_argument("--trans-alpha", type=float, default=1.0,
                    help="Laplace smoothing of the transition matrix")
    ap.add_argument("--with-revive", action="store_true",
                    help="include reviving dead classes as a variant")
    ap.add_argument("--out", default="lb_decperlab")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)

    d = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred, a.model))
    P_te = d["proba_test"].astype(np.float32)
    cal_m = d["calib_mask"]
    P_cal_c = d["proba_calib"].astype(np.float32)
    P_cal = np.zeros((len(y), nC), np.float32)
    P_cal[cal_m] = P_cal_c
    cal_vids = sorted(set(vids[cal_m].tolist()))
    print("calibration: {} windows across {} videos".format(int(cal_m.sum()), len(cal_vids)),
          flush=True)

    sol_c, active_c = build_solution(cal_vids, lab_of, cov)
    mind_c, maxg_c = duration_stats(cal_vids, lab_of, cov)
    mind_t, maxg_t = duration_stats(test_v, lab_of, cov)

    all_c = build_active_mask(vids[cal_m], ev["agent"][cal_m], ev["target"][cal_m],
                              classes, active_c)
    A_cal = np.zeros((len(y), nC), bool); A_cal[cal_m] = all_c
    all_t = build_active_mask(vids[te], ev["agent"][te], ev["target"][te], classes, active)
    A_te = np.zeros((len(y), nC), bool); A_te[te] = all_t

    Ps_cal = smooth_proba(P_cal[cal_m], vids[cal_m], ev["agent"][cal_m],
                          ev["core_start"][cal_m], window=a.smooth)
    S_cal = np.zeros_like(P_cal); S_cal[cal_m] = Ps_cal
    Ps_te = smooth_proba(P_te, vids[te], ev["agent"][te], ev["core_start"][te],
                         window=a.smooth)
    S_te = np.zeros((len(y), nC), np.float32); S_te[te] = Ps_te
    Pt_full = np.zeros((len(y), nC), np.float32); Pt_full[te] = P_te

    rates, _ = train_rates(train_v, lab_of, cov)

    def revive(sub_in, pred_v, sel, lab, n_frames, Pv, sol_l):
        """Give up BACKGROUND windows to the lab's actions that emit nothing."""
        em = {}
        for r in sub_in:
            em[r["action"]] = em.get(r["action"], 0) + int(r["stop_frame"]) - int(r["start_frame"])
        extra = np.full(len(vids), bg, np.int64)
        touched = np.zeros(len(vids), bool)
        for act in sorted({r["action"] for r in sol_l}):
            if act not in classes or em.get(act, 0) > 0:
                continue
            c_ = classes.index(act)
            n_win = int(round(rates.get((lab, act), 0.0) * n_frames / 8))
            cand = sel[(pred_v[sel] == bg) & A_cal[sel][:, c_]]
            if n_win <= 0 or len(cand) == 0:
                continue
            take = cand[np.argsort(-Pv[cand, c_])[:min(n_win, len(cand))]]
            extra[take] = c_; touched[take] = True
        if touched.any():
            sub_in = sub_in + build_submission(touched, vids, ev, extra[touched],
                                               classes, active=active_c)
        return sub_in

    _rev = [False, True] if a.with_revive else [False]
    variants = list(itertools.product(["residual", "ratio"], [False, True], [False, True],
                                      _rev))
    choice, thr_pick, rows_out = {}, {}, []
    tlab = labs[te]

    for lab in sorted(set(tlab.tolist())):
        if a.labs and lab not in a.labs:
            continue
        lab_cal = cal_m & (labs == lab)
        lv = [v for v in cal_vids if lab_of[v] == lab]
        sol_l = [r for r in sol_c if r["video_id"] in set(lv)]
        if not lab_cal.sum() or not sol_l:
            continue
        best = (None, -1.0, None)
        thr_cache = {}
        for mode, sm, md, rv in variants:
            Pv = S_cal if sm else P_cal
            thr = tune_thresholds(
                Pv, lab_cal, vids, labs, ev, classes, sol_l, active_c, lab_of,
                score_fn=official_score,
                submission_fn=lambda mk, pr: build_submission(
                    mk, vids, ev, pr, classes, active=active_c),
                mode=mode, allowed=A_cal, verbose=False)
            t = thr.get(lab, np.full(nC, .5, np.float32))
            sel = np.flatnonzero(lab_cal)
            pr = decide(Pv[sel], t, bg, mode=mode, allowed=A_cal[sel])
            sub = build_submission(lab_cal, vids, ev, pr, classes, active=active_c)
            if md:
                sub = min_duration_filter(sub, lab_of, mind_c, maxg_c)
            if rv:
                prv = np.full(len(y), bg, np.int64); prv[sel] = pr
                nf = sum(cov.get(v, 0) for v in lv)
                sub = revive(sub, prv, sel, lab, nf, Pv, sol_l)
            sc = official_score(sol_l, sub)
            thr_cache[(mode, bool(sm))] = t
            if sc > best[1]:
                best = ({"decoder": "argmax", "mode": mode, "smooth": bool(sm),
                         "mindur": bool(md), "revive": bool(rv),
                         "calib": round(float(sc), 4)}, sc, t)

        # ── variantes Viterbi ────────────────────────────────────────────────
        # These reuse the thresholds tuned for `ratio`: with uniform transitions and
        # w=1 the log(p/t) emission reproduces that rule exactly (self-test (a) in
        # lb_viterbi), so they are the thresholds consistent with this decoder.
        # Retuning them by coordinate ascent inside Viterbi would multiply the cost by
        # the grid length with no justification.
        best_vit = (-1.0, None)
        if a.viterbi_weights:
            tr_m = np.isin(vids, train_v) & (labs == lab)
            logA = transition_logprobs(y, vids, ev["agent"], ev["core_start"],
                                       tr_m, nC, alpha=a.trans_alpha)
            sel = np.flatnonzero(lab_cal)
            for w in a.viterbi_weights:
                for sm in (False, True):
                    t = thr_cache.get(("ratio", sm))
                    if t is None:
                        continue
                    Pv = S_cal if sm else P_cal
                    pr = decode_lab(Pv[sel], t, bg, vids[sel], ev["agent"][sel],
                                    ev["core_start"][sel], logA, w=w,
                                    allowed=A_cal[sel])
                    for md in (False, True):
                        sub = build_submission(lab_cal, vids, ev, pr, classes,
                                               active=active_c)
                        if md:
                            sub = min_duration_filter(sub, lab_of, mind_c, maxg_c)
                        sc = official_score(sol_l, sub)
                        if sc > best_vit[0]:
                            best_vit = (sc, (w, bool(sm), bool(md)))
                        if sc > best[1]:
                            best = ({"decoder": "viterbi", "mode": "ratio",
                                     "smooth": bool(sm), "mindur": bool(md),
                                     "revive": False, "w": float(w),
                                     "calib": round(float(sc), 4)}, sc, t)

        c_, sc, t = best
        if best_vit[1] is not None:
            # Printed whether it wins or loses: without this line, a Viterbi bug and
            # a legitimate loss to argmax are indistinguishable from the log.
            print("      [best viterbi: {:.4f} at w={} smooth={} mindur={}]".format(
                best_vit[0], *best_vit[1]), flush=True)
        choice[lab] = c_
        thr_pick[lab] = t
        print("    {:22s} {:8s} smooth={:5s} mindur={:5s} revive={:5s} w={:>4} calib={:.4f}"
              .format(lab, c_["decoder"] if c_["decoder"] == "viterbi" else c_["mode"],
                      str(c_["smooth"]), str(c_["mindur"]), str(c_["revive"]),
                      c_.get("w", "-"), sc), flush=True)

    # ── apply to test, exactly once ─────────────────────────────────────────
    sub_all = []
    for lab in sorted(set(tlab.tolist())):
        c = choice.get(lab)
        if c is None:
            continue
        m = te & (labs == lab)
        sel = np.flatnonzero(m)
        Pv = S_te if c["smooth"] else Pt_full
        if c.get("decoder") == "viterbi":
            tr_m = np.isin(vids, train_v) & (labs == lab)
            logA = transition_logprobs(y, vids, ev["agent"], ev["core_start"],
                                       tr_m, nC, alpha=a.trans_alpha)
            pr = decode_lab(Pv[sel], thr_pick[lab], bg, vids[sel], ev["agent"][sel],
                            ev["core_start"][sel], logA, w=c["w"], allowed=A_te[sel])
        else:
            pr = decide(Pv[sel], thr_pick[lab], bg, mode=c["mode"], allowed=A_te[sel])
        s = build_submission(m, vids, ev, pr, classes, active=active)
        if c["mindur"]:
            s = min_duration_filter(s, lab_of, mind_t, maxg_t)
        sub_all += s

    sc = official_score(sol, sub_all)
    print("\nscore with a per-lab decision layer = {:.4f}".format(sc))
    lo, hi = np.percentile(bootstrap_public(sol, sub_all, test_v), [2.5, 97.5])
    print("public bootstrap 95% CI: [{:.4f}, {:.4f}]".format(lo, hi))
    _pl = per_lab_scores(sol, sub_all, lab_of)
    _rs = subset_score(_pl)
    # The real competition test set excludes CalMS21 and CRIM13, which are our four
    # best labs, so the mean over all 9 overstates the true standing.
    if _rs is not None:
        print("score over {} real-test-like labs = {:.4f}  (* see below)".format(
            len([k for k in _pl if k in set(REAL_TEST_LABS)]), _rs), flush=True)
    for lab, s in sorted(per_lab_scores(sol, sub_all, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {}{:22s} {:.4f}  [{}]".format("*" if lab in set(REAL_TEST_LABS) else " ", lab, s, choice.get(lab, {}).get("mode", "-")))
    json.dump({"score": round(float(sc), 4), "choice": choice},
              open(RESULTS / "{}.json".format(a.out), "w"), indent=1)
    json.dump({k: v.tolist() for k, v in thr_pick.items()},
              open(RESULTS / "{}_thr_{}.json".format(a.out, a.model), "w"), indent=1)
    print("\nGuardado results/{}.json ({:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
