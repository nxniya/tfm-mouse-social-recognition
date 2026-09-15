"""
scripts/run_lb_finalize.py
==========================
Combines the two final improvements, which until now had only been measured
separately:

  * the **per-lab decision layer** (`run_lb_decide_perlab.py`, 0.4943), chosen on
    calibration videos: 7 of 9 labs prefer `ratio` and 8 of 9 prefer smoothing, the
    exact opposite of what the global sweep selected.
  * **reviving dead classes** by prevalence (`lb_prevalence.py`, 0.4899), which lifts
    InvincibleJellyfish from 0.2377 to 0.2830 by reviving its 4 never-predicted
    actions.

It neither retrains nor retunes thresholds: it reuses the ones the per-lab tuning
saved, rebuilds the submission with each lab's chosen variant, and applies the revival
on top. The revival runs **after** the minimum-duration filter, because that filter
deletes short intervals, which is the typical shape of a rare action.
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
from scripts.lb_viterbi import transition_logprobs, decode_lab
from scripts.run_lb_decision import min_duration_filter, duration_stats
from scripts.lb_prevalence import train_rates

RESULTS = _r / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--pred", default="lb_ctx12c")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--dec", default="lb_decperlab")
    ap.add_argument("--pred-extra", default=None,
                    help="second probability source; its labs override --pred, for "
                         "example the 5 bagged labs over the 9 base ones")
    ap.add_argument("--dec-extra", default=None,
                    help="second decision source; its labs override --dec")
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--trans-alpha", type=float, default=1.0)
    ap.add_argument("--revive-frac", type=float, default=1.0)
    ap.add_argument("--no-revive", dest="revive", action="store_false", default=True)
    ap.add_argument("--tune-revive", action="store_true",
                    help="decide PER LAB whether to revive, using calibration")
    ap.add_argument("--out", default="lb_best")
    a = ap.parse_args()
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix); del X
    nC = len(classes); bg = classes.index("background")
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)
    tlab = labs[te]

    P_te = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred, a.model))["proba_test"].astype(np.float32)
    if a.pred_extra:
        # Substitution PER LAB, not by position: both runs share the same test mask,
        # so the rows do correspond, but this is checked before merging so that
        # unrelated windows are never paired up silently.
        _d = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred_extra, a.model))
        if not np.array_equal(_d["test_mask"], te):
            raise SystemExit("different test mask in --pred-extra: not comparable")
        _P = _d["proba_test"].astype(np.float32)
        _has = np.flatnonzero(_P.sum(axis=1) > 0)
        _labs_x = sorted(set(labs[te][_has].tolist()))
        for _l in _labs_x:
            _s = np.flatnonzero(labs[te] == _l)
            P_te[_s] = _P[_s]
        print("probabilities replaced for: {}".format(_labs_x), flush=True)
    choice = json.load(open(RESULTS / "{}.json".format(a.dec)))["choice"]
    thr = {k: np.asarray(v, np.float32) for k, v in
           json.load(open(RESULTS / "{}_thr_{}.json".format(a.dec, a.model))).items()}
    if a.dec_extra:
        # The small labs were retuned with the revival inside their variant space;
        # their decisions and thresholds replace the base run's.
        ce = json.load(open(RESULTS / "{}.json".format(a.dec_extra)))["choice"]
        te_ = json.load(open(RESULTS / "{}_thr_{}.json".format(a.dec_extra, a.model)))
        choice.update(ce)
        thr.update({k: np.asarray(v, np.float32) for k, v in te_.items()})
        print("decisions replaced for: {}".format(sorted(ce)), flush=True)

    A = np.zeros((len(y), nC), bool)
    A[te] = build_active_mask(vids[te], ev["agent"][te], ev["target"][te], classes, active)
    Pf = np.zeros((len(y), nC), np.float32); Pf[te] = P_te
    Sf = np.zeros((len(y), nC), np.float32)
    Sf[te] = smooth_proba(P_te, vids[te], ev["agent"][te], ev["core_start"][te],
                          window=a.smooth)
    mind, maxg = duration_stats(test_v, lab_of, cov)

    pred = np.full(len(y), bg, np.int64)
    sub = []
    for lab in sorted(set(tlab.tolist())):
        c = choice.get(lab)
        if c is None:
            continue
        m = te & (labs == lab)
        sel = np.flatnonzero(m)
        Pv = Sf if c["smooth"] else Pf
        _t = thr.get(lab, np.full(nC, .5, np.float32))
        if c.get("decoder") == "viterbi":
            tr_m = np.isin(vids, train_v) & (labs == lab)
            logA = transition_logprobs(y, vids, ev["agent"], ev["core_start"],
                                       tr_m, nC, alpha=a.trans_alpha)
            pr = decode_lab(Pv[sel], _t, bg, vids[sel], ev["agent"][sel],
                            ev["core_start"][sel], logA, w=c["w"], allowed=A[sel])
        else:
            pr = decide(Pv[sel], _t, bg, mode=c["mode"], allowed=A[sel])
        pred[sel] = pr
        s = build_submission(m, vids, ev, pr, classes, active=active)
        if c["mindur"]:
            s = min_duration_filter(s, lab_of, mind, maxg)
        sub += s
    print("score sin reactivacion = {:.4f}".format(official_score(sol, sub)), flush=True)

    rates, _ = train_rates(train_v, lab_of, cov)

    def revive_into(sub_in, pred_v, mask_v, Pv, sol_v, act_v, frames_of, scale=1.0):
        """Give up BACKGROUND windows to actions that emit nothing. Returns the
        extended submission and the list of revived (lab, action, n)."""
        emitted = {}
        for r in sub_in:
            k = (lab_of[r["video_id"]], r["action"])
            emitted[k] = emitted.get(k, 0) + int(r["stop_frame"]) - int(r["start_frame"])
        extra = np.full(len(vids), bg, np.int64)
        touched = np.zeros(len(vids), bool)
        done = []
        for lab in sorted(set(labs[mask_v].tolist())):
            sel = np.flatnonzero(mask_v & (labs == lab))
            n_frames = frames_of(lab)
            for act in sorted({r["action"] for r in sol_v if lab_of[r["video_id"]] == lab}):
                if act not in classes or emitted.get((lab, act), 0) > 0:
                    continue
                c_ = classes.index(act)
                n_win = int(round(rates.get((lab, act), 0.0) * n_frames * scale / 8))
                cand = sel[(pred_v[sel] == bg) & A[sel][:, c_]]
                if n_win <= 0 or len(cand) == 0:
                    continue
                take = cand[np.argsort(-Pv[cand, c_])[:min(n_win, len(cand))]]
                extra[take] = c_; touched[take] = True
                done.append((lab, act, len(take)))
        if touched.any():
            sub_in = sub_in + build_submission(touched, vids, ev, extra[touched],
                                               classes, active=active)
        return sub_in, done

    if a.tune_revive:
        # Decide per lab, on CALIBRATION, whether reviving is worth it. The decision
        # layer's own tuning could not account for this, because the revival was not
        # in its variant space; in fact for InvincibleJellyfish it picked a variant in
        # which the rare classes do emit something, and so no longer count as dead,
        # but score worse.
        d2 = np.load(RESULTS / "{}_pred_{}.npz".format(a.pred, a.model))
        cal_m = d2["calib_mask"]
        Pc = np.zeros((len(y), nC), np.float32); Pc[cal_m] = d2["proba_calib"].astype(np.float32)
        cvids = sorted(set(vids[cal_m].tolist()))
        sol_c, active_c = build_solution(cvids, lab_of, cov)
        mind_c, maxg_c = duration_stats(cvids, lab_of, cov)
        Ac = np.zeros((len(y), nC), bool)
        Ac[cal_m] = build_active_mask(vids[cal_m], ev["agent"][cal_m], ev["target"][cal_m],
                                      classes, active_c)
        Sc = np.zeros((len(y), nC), np.float32)
        Sc[cal_m] = smooth_proba(Pc[cal_m], vids[cal_m], ev["agent"][cal_m],
                                 ev["core_start"][cal_m], window=a.smooth)
        A_save = A.copy(); A[:] = Ac
        for lab in sorted(set(tlab.tolist())):
            c = choice.get(lab)
            m = cal_m & (labs == lab)
            lv = [v for v in cvids if lab_of[v] == lab]
            sol_l = [r for r in sol_c if r["video_id"] in set(lv)]
            if c is None or not m.sum() or not sol_l:
                continue
            sel = np.flatnonzero(m)
            Pv = Sc if c["smooth"] else Pc
            prc = np.full(len(y), bg, np.int64)
            prc[sel] = decide(Pv[sel], thr.get(lab, np.full(nC, .5, np.float32)), bg,
                              mode=c["mode"], allowed=Ac[sel])
            sc0 = build_submission(m, vids, ev, prc[sel], classes, active=active_c)
            if c["mindur"]:
                sc0 = min_duration_filter(sc0, lab_of, mind_c, maxg_c)
            base = official_score(sol_l, sc0)
            nf = sum(cov.get(v, 0) for v in lv)
            sc1, _d = revive_into(sc0, prc, m, Pc, sol_l, None, lambda _l: nf,
                                  scale=a.revive_frac)
            alt = official_score(sol_l, sc1)
            c["revive"] = bool(alt > base + 1e-9)
            print("    {:22s} calib without={:.4f} with={:.4f} -> revive={}".format(
                lab, base, alt, c["revive"]), flush=True)
        A[:] = A_save

    if a.revive:
        emitted = {}
        for r in sub:
            k = (lab_of[r["video_id"]], r["action"])
            emitted[k] = emitted.get(k, 0) + int(r["stop_frame"]) - int(r["start_frame"])
        extra = np.full(len(vids), bg, np.int64)
        touched = np.zeros(len(vids), bool)
        for lab in sorted(set(tlab.tolist())):
            _c = choice.get(lab, {})
            if "revive" in _c and not _c["revive"]:
                continue          # calibration says it is not worth it for this lab
            sel = np.flatnonzero(te & (labs == lab))
            n_frames = sum(cov.get(v, 0) for v in test_v if lab_of[v] == lab)
            for act in sorted({r["action"] for r in sol if lab_of[r["video_id"]] == lab}):
                if act not in classes or emitted.get((lab, act), 0) > 0:
                    continue
                c_ = classes.index(act)
                n_win = int(round(rates.get((lab, act), 0.0) * n_frames * a.revive_frac / 8))
                cand = sel[(pred[sel] == bg) & A[sel][:, c_]]
                if n_win <= 0 or len(cand) == 0:
                    continue
                take = cand[np.argsort(-Pf[cand, c_])[:min(n_win, len(cand))]]
                extra[take] = c_; touched[take] = True
                print("    revived {:22s} {:16s} {} windows".format(lab, act, len(take)),
                      flush=True)
        if touched.any():
            sub = sub + build_submission(touched, vids, ev, extra[touched], classes,
                                         active=active)

    sc = official_score(sol, sub)
    print("\nSCORE FINAL = {:.4f}".format(sc))
    lo, hi = np.percentile(bootstrap_public(sol, sub, test_v), [2.5, 97.5])
    print("public bootstrap 95% CI: [{:.4f}, {:.4f}]".format(lo, hi))
    _pl = per_lab_scores(sol, sub, lab_of)
    _rs = subset_score(_pl)
    # The real competition test set excludes CalMS21 and CRIM13, which are our four
    # best labs, so the mean over all 9 overstates the true standing.
    if _rs is not None:
        print("score over {} real-test-like labs = {:.4f}  (* see below)".format(
            len([k for k in _pl if k in set(REAL_TEST_LABS)]), _rs), flush=True)
    for lab, s in sorted(per_lab_scores(sol, sub, lab_of).items(), key=lambda kv: -kv[1]):
        print("    {}{:22s} {:.4f}  [{} smooth={} mindur={}]".format(
            "*" if lab in set(REAL_TEST_LABS) else " ", lab, s, choice[lab]["mode"], choice[lab]["smooth"], choice[lab]["mindur"]))
    json.dump({"score": round(float(sc), 4), "choice": choice},
              open(RESULTS / "{}.json".format(a.out), "w"), indent=1)
    print("\nGuardado results/{}.json ({:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
