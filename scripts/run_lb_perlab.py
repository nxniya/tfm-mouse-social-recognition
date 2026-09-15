"""
scripts/run_lb_perlab.py
========================
**Per-lab** models over the full corpus, with out-of-fold threshold calibration and
temporal smoothing of the probabilities.

Context: a single 21-class softmax had to cover labs recording at 10 to 30 fps, with
different arenas and different vocabularies. One model per lab, plus ten times more
data, took the score from 0.2770 to 0.3429. The per (lab, action) diagnostic says what
remains to fix is **precision**: the small labs over-predict by up to 7.8x. Against
that:

* **Out-of-fold calibration.** The thresholds used to be tuned on 25% of held-out
  videos; in TranquilPanther that is about 4 videos, and the thresholds overfit. Here
  the small labs use k-fold over their training videos and calibrate on the pooled
  out-of-fold probabilities. The three large labs (task1, task2, supplemental) keep
  the holdout, which at 50 to 200 videos is already enough: k-fold on them would
  multiply the cost for nothing.
* **Smoothing.** Each (video, agent) probability sequence is smoothed before deciding,
  and **also before calibrating** — calibrating on raw probabilities and deciding on
  smoothed ones leaves the thresholds in the wrong place.
* **Persisted probabilities.** The previous run saved only predictions, which forced a
  full retrain just to retune the thresholds.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys, pathlib, time, json, argparse
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np
import pandas as pd

from scripts.run_lb_experiment import (
    build_solution, build_submission, official_score, per_lab_scores,
    split_videos, video_coverage, bootstrap_public, SEED, PUBLIC_FRAC)
from scripts.lb_threshold import (decide, tune_thresholds, smooth_proba, proba_tree)

FEAT = _r / "dataset" / "features"
RESULTS = _r / "results"


def load_full(suffix="_full", cond="raw"):
    d = np.load(FEAT / "lb_{}{}.npz".format(cond, suffix), allow_pickle=True)
    # With no frame cap the corpus is about 7 GB, so every copy counts: chaining
    # `astype` and `nan_to_num` kept three arrays alive at once, over 20 GB, and the
    # process died while loading. Cleaned in place instead.
    X = d["X"]
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    ev = {"agent": d["win_agent"].astype(np.int64), "target": d["win_target"].astype(np.int64),
          "fstart": d["win_fstart"].astype(np.int64), "fstop": d["win_fstop"].astype(np.int64),
          "core_start": d["win_core_start"].astype(np.int64),
          "core_stop": d["win_core_stop"].astype(np.int64)}
    agg = bool(d["aggregated"]) if "aggregated" in d.files else False
    return (X, d["y"].astype(np.int64), d["video_ids"].astype(str),
            d["lab_ids"].astype(str), [str(c) for c in d["classes"]], ev, agg)


def fit_lab(name, Xtr, ytr, seed=SEED, class_weight="balanced"):
    """Fit one lab's model, tolerating classes with very few samples.

    HistGB performs an internal STRATIFIED split for early stopping, which blows up
    when a class in the subset has a single sample; that happens in small labs and when
    subsampling. `validation_fraction=None` makes early stopping use the training data
    itself, avoiding that split without discarding any row.
    """
    from src.models.baseline import RandomForestBaseline, GradientBoostingBaseline
    if name == "rf":
        clf = RandomForestBaseline(seed=seed, class_weight=class_weight)
        clf.fit(Xtr, ytr)
        return clf
    try:
        clf = GradientBoostingBaseline(seed=seed, class_weight=class_weight)
        clf.fit(Xtr, ytr)
        return clf
    except ValueError:
        clf = GradientBoostingBaseline(seed=seed, validation_fraction=None,
                                       class_weight=class_weight)
        clf.fit(Xtr, ytr)
        return clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_full")
    ap.add_argument("--models", nargs="+", default=["histgb"])
    ap.add_argument("--cap-per-lab", type=int, default=250000)
    ap.add_argument("--test-frac", type=float, default=None,
                    help="fraction of videos held out as test. Default 0.30, which is "
                         "right for MEASURING but wastes 30 per cent of the data in a "
                         "real submission; 0.02 leaves 1-2 videos per lab.")
    ap.add_argument("--pooled-cap-per-lab", type=int, default=90000)
    ap.add_argument("--save-models", default=None,
                    help="directory to serialise, per lab, the final model plus its "
                         "thresholds and metadata (required for the submission)")
    ap.add_argument("--bag-folds", action="store_true",
                    help="average the fold models, currently discarded, with the "
                         "final model; only affects k-fold-calibrated labs")
    ap.add_argument("--context-multi", nargs="+", type=int, default=None)
    ap.add_argument("--context", type=int, default=0,
                    help="past/future context windows (0 disables it)"
                    )
    ap.add_argument("--tune-frac", type=float, default=0.25)
    ap.add_argument("--cv-folds", type=int, default=4)
    ap.add_argument("--cv-max-videos", type=int, default=40,
                    help="labs with at most this many train videos calibrate by k-fold")
    ap.add_argument("--smooth", type=int, default=9, help="0 disables the smoothing")
    ap.add_argument("--mode", default="residual", choices=["residual", "ratio"])
    ap.add_argument("--pooled", action="store_true",
                    help="a single model over ALL labs (cross-lab)")
    ap.add_argument("--labs", nargs="+", default=None,
                    help="restrict to these labs, for quick ablations")
    ap.add_argument("--class-weight", default="balanced",
                    choices=["balanced", "none"])
    ap.add_argument("--out", default="lb_perlab")
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t00 = time.perf_counter()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix)
    nC = len(classes); bg = classes.index("background")
    print("Corpus: {} windows, {} videos, {} labs, {} classes, agg={}".format(
        len(y), len(set(vids)), len(set(labs)), nC, agg), flush=True)

    train_v, test_v, lab_of = (split_videos(vids, labs, test_frac=a.test_frac)
                               if a.test_frac else split_videos(vids, labs))

    # Cache metadata the submission needs. `n_base` has to be computed HERE, before
    # the context is appended: afterwards X.shape[1] is no longer 4F.
    _dm = np.load(FEAT / "lb_raw{}.npz".format(a.suffix), allow_pickle=True)
    lab_kp = (json.loads(str(_dm["lab_keypoints"]))
              if "lab_keypoints" in _dm.files else {})
    win_size = int(_dm["window_size"]) if "window_size" in _dm.files else 64
    win_stride = int(_dm["stride"]) if "stride" in _dm.files else 8
    n_base_cache = X.shape[1] // 4 if agg else X.shape[1]
    pre_ctx_width = int(X.shape[1])      # width BEFORE the context is appended
    del _dm

    if a.context or a.context_multi:
        from scripts.lb_context import add_context, add_context_multi
        if not agg:
            raise SystemExit("--context requires an aggregated cache")
        _nb = X.shape[1] // 4
        if a.context_multi:
            X = add_context_multi(X, vids, ev["agent"], ev["core_start"],
                                  n_base=_nb, ks=tuple(a.context_multi))
        else:
            # `add_context` used to concatenate [X | past | future], which keeps four
            # large arrays alive at once. Over the full corpus that is about 20 GB and
            # does not fit. The destination is allocated once and filled in place.
            _D = X.shape[1]
            _out = np.empty((len(X), _D + 2 * _nb), np.float32)
            _out[:, :_D] = X
            add_context(X, vids, ev["agent"], ev["core_start"], n_base=_nb,
                        k=a.context, out=_out)
            del X
            X = _out

    cov = video_coverage(vids, ev["fstop"])
    _mind = _maxg = _rates = {}
    if a.save_models:
        from scripts.run_lb_decision import duration_stats
        from scripts.lb_prevalence import train_rates
        print("  computing durations and prevalences per (lab, action)...", flush=True)
        _mind, _maxg = duration_stats(train_v, lab_of, cov)
        _rates, _ = train_rates(train_v, lab_of, cov)
    sol_te, active_te = build_solution(test_v, lab_of, cov)
    te = np.isin(vids, test_v)
    print("train {} videos | test {} videos ({} windows) | smooth={} mode={}".format(
        len(train_v), len(test_v), int(te.sum()), a.smooth, a.mode), flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for mname in a.models:
        pred_g = np.full(len(y), bg, np.int64)
        pred_arg_g = np.full(len(y), bg, np.int64)
        P_test_g = np.zeros((len(y), nC), np.float32)
        thr_store = {}
        calib_store = {}
        P_cal_g = np.zeros((len(y), nC), np.float32)
        calib_g = np.zeros(len(y), bool)
        t0 = time.perf_counter()
        _cw = None if a.class_weight == "none" else a.class_weight
        _groups = ([("__pooled__", sorted(a.labs or set(labs.tolist())))] if a.pooled
                   else [(l, [l]) for l in sorted(set(labs.tolist()))])
        for lab, _members in _groups:
            if a.labs and not a.pooled and lab not in a.labs:
                continue
            _mem = set(_members)
            lab_tr = [v for v in train_v if lab_of[v] in _mem]
            lab_te = [v for v in test_v if lab_of[v] in _mem]
            if not lab_tr or not lab_te:
                continue
            te_m = np.isin(vids, lab_te)
            use_cv = (not a.pooled) and len(lab_tr) <= a.cv_max_videos

            def _cap(idx):
                # In pooled mode the sampling CANNOT be uniform over the corpus:
                # TranquilPanther contributes about 13k of 1.29M windows, so a global
                # cap of 250k would leave it about 2.5k, FEWER than its own per-lab
                # model already uses. Quota per lab instead: small labs enter whole and
                # large ones are trimmed, which is what makes the comparison fair.
                if a.pooled:
                    out = []
                    for _l in sorted(set(labs[idx].tolist())):
                        s = idx[labs[idx] == _l]
                        out.append(np.sort(rng.choice(s, a.pooled_cap_per_lab,
                                                      replace=False))
                                   if len(s) > a.pooled_cap_per_lab else s)
                    return np.sort(np.concatenate(out)) if out else idx
                return (np.sort(rng.choice(idx, a.cap_per_lab, replace=False))
                        if len(idx) > a.cap_per_lab else idx)

            # ── out-of-fold probabilities, for calibration ───────────────────
            calib_m = np.zeros(len(y), bool)
            P_cal = np.zeros((len(y), nC), np.float32)
            fold_models = []
            if use_cv:
                folds = np.array_split(rng.permutation(len(lab_tr)), a.cv_folds)
                for f in folds:
                    hold = [lab_tr[i] for i in f]
                    if not hold:
                        continue
                    hm = np.isin(vids, hold)
                    fm = np.isin(vids, [v for v in lab_tr if v not in set(hold)])
                    fi = _cap(np.flatnonzero(fm))
                    if len(fi) == 0 or hm.sum() == 0:
                        continue
                    c = fit_lab(mname, X[fi], y[fi], class_weight=_cw)
                    P_cal[hm] = proba_tree(c, X[hm], nC, aggregated=agg)
                    calib_m |= hm
                    if a.bag_folds:
                        # These models are already trained: they were used only for
                        # the out-of-fold probabilities and then thrown away. Keeping
                        # them gives an ensemble at no extra training cost, precisely
                        # in the 7-to-23-video labs where variance hurts most.
                        fold_models.append(c)
                cal_vids = lab_tr
            else:
                k = max(1, int(round(a.tune_frac * len(lab_tr))))
                perm = rng.permutation(len(lab_tr))
                cal_vids = [lab_tr[i] for i in perm[:k]]
                fit_v = [lab_tr[i] for i in perm[k:]] or lab_tr
                fi = _cap(np.flatnonzero(np.isin(vids, fit_v)))
                c = fit_lab(mname, X[fi], y[fi], class_weight=_cw)
                calib_m = np.isin(vids, cal_vids)
                P_cal[calib_m] = proba_tree(c, X[calib_m], nC, aggregated=agg)

            # ── final model: ALL of the lab's training videos ────────────────
            fi = _cap(np.flatnonzero(np.isin(vids, lab_tr)))
            clf = fit_lab(mname, X[fi], y[fi], class_weight=_cw)
            P_te = proba_tree(clf, X[te_m], nC, aggregated=agg)
            if fold_models:
                # Mean probability over [full model] plus the fold models.
                # NOTE: P_cal is still the out-of-fold output of a SINGLE model, noisier
                # than this average, so the thresholds end up calibrated
                # conservadora. Se documenta; corregirlo exigiria CV repetida.
                acc = P_te.astype(np.float64)
                for _m in fold_models:
                    acc = acc + proba_tree(_m, X[te_m], nC, aggregated=agg)
                P_te = (acc / (1.0 + len(fold_models))).astype(np.float32)
                print("      bagging: {} fold models + 1 full".format(
                    len(fold_models)), flush=True)
            pred_arg_g[te_m] = P_te.argmax(1)

            # ── smoothing, applied identically when calibrating and deciding ─
            if a.smooth:
                if calib_m.sum():
                    P_cal[calib_m] = smooth_proba(
                        P_cal[calib_m], vids[calib_m], ev["agent"][calib_m],
                        ev["core_start"][calib_m], window=a.smooth)
                P_te = smooth_proba(P_te, vids[te_m], ev["agent"][te_m],
                                    ev["core_start"][te_m], window=a.smooth)
            P_test_g[te_m] = P_te

            thr = {}
            if calib_m.sum():
                sol_c, active_c = build_solution(cal_vids, lab_of, cov)
                if sol_c:
                    thr = tune_thresholds(
                        P_cal, calib_m, vids, labs, ev, classes, sol_c, active_c, lab_of,
                        score_fn=official_score,
                        submission_fn=lambda mk, pr: build_submission(
                            mk, vids, ev, pr, classes, active=active_c),
                        mode=a.mode, verbose=False)
            if a.pooled:
                # The model is shared, but the threshold is still per lab: the metric
                # averages per lab and each one has its own prevalence
                for _l in sorted(set(labs[te_m].tolist())):
                    _s = np.flatnonzero(te_m & (labs == _l))
                    _t = thr.get(_l, np.full(nC, 0.5, np.float32))
                    pred_g[_s] = decide(P_test_g[_s], _t, bg, mode=a.mode)
            else:
                tvec = thr.get(lab, np.full(nC, 0.5, np.float32))
                pred_g[te_m] = decide(P_te, tvec, bg, mode=a.mode)
            P_cal_g[calib_m] = P_cal[calib_m]
            calib_g |= calib_m
            # Scored on CALIBRATION, not on test: it is the only thing a
            # hyperparameter such as class_weight can be chosen with, without leaking
            # the evaluation set. Stored per lab alongside the thresholds.
            if calib_m.sum() and thr:
                for _l in sorted(set(labs[calib_m].tolist())):
                    _cm = calib_m & (labs == _l)
                    if not _cm.sum():
                        continue
                    _t = thr.get(_l, np.full(nC, 0.5, np.float32))
                    _pr = np.full(len(y), bg, np.int64)
                    _pr[_cm] = decide(P_cal[_cm], _t, bg, mode=a.mode)
                    _sc, _ac = build_solution(
                        [v for v in cal_vids if lab_of[v] == _l], lab_of, cov)
                    if _sc:
                        calib_store[_l] = round(float(official_score(
                            _sc, build_submission(_cm, vids, ev, _pr[_cm],
                                                  classes, active=_ac))), 4)

            if a.pooled:
                thr_store.update({k: v.tolist() for k, v in thr.items()})
            else:
                thr_store[lab] = thr.get(
                    lab, np.full(nC, 0.5, np.float32)).tolist()

            if a.save_models and not a.pooled:
                # The submission needs the model; until now `clf` died with the loop.
                import joblib
                _md = pathlib.Path(a.save_models); _md.mkdir(parents=True, exist_ok=True)
                joblib.dump({
                    "lab": lab, "model_name": mname, "clf": clf,
                    "classes": list(classes),
                    "thresholds": thr.get(lab, np.full(nC, 0.5, np.float32)).astype(np.float32),
                    # Inference must rebuild EXACTLY the same vector: the lab's
                    # features, zero-padded out to the global width, and only then the
                    # context with n_base = global_width / 4.
                    "padded_width": int(pre_ctx_width),
                    "n_base": int(n_base_cache),
                    "context_k": int(a.context or 0),
                    "aggregated": bool(agg),
                    "lab_keypoints": lab_kp.get(lab),
                    "window": int(win_size), "stride": int(win_stride),
                    # What the decision layer needs at inference time and cannot
                    # cheaply recompute there: typical durations per action, for the
                    # minimum-duration filter, and prevalence per action, for reviving
                    # dead classes. Both come ONLY from this lab's training
                    # annotations.
                    "min_dur": {a: v for (l_, a), v in _mind.items() if l_ == lab},
                    "max_gap": {a: v for (l_, a), v in _maxg.items() if l_ == lab},
                    "rates": {a: v for (l_, a), v in _rates.items() if l_ == lab},
                }, _md / "{}__{}.joblib".format(lab, mname), compress=3)
            print("    {:22s} {:3d} tr vids  calib={:6d} win ({})  test={:6d}".format(
                lab, len(lab_tr), int(calib_m.sum()), "cv" if use_cv else "holdout",
                int(te_m.sum())), flush=True)

        el = time.perf_counter() - t0
        sub_a = build_submission(te, vids, ev, pred_arg_g[te], classes, active=active_te)
        sub_t = build_submission(te, vids, ev, pred_g[te], classes, active=active_te)
        sc_a, sc_t = official_score(sol_te, sub_a), official_score(sol_te, sub_t)
        boot = bootstrap_public(sol_te, sub_t, test_v)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"model": mname, "smooth": a.smooth, "mode": a.mode,
                     "f1_event_argmax": round(sc_a, 4), "f1_event_official": round(sc_t, 4),
                     "gain_thresholds": round(sc_t - sc_a, 4),
                     "boot_ci_lo": round(float(lo), 4), "boot_ci_hi": round(float(hi), 4),
                     "fit_predict_s": round(el, 1)})
        print("  {:8s} argmax={:.4f} -> thresholds={:.4f}  ({:.0f}s)".format(
            mname, sc_a, sc_t, el), flush=True)
        for lab, s in sorted(per_lab_scores(sol_te, sub_t, lab_of).items(), key=lambda kv: -kv[1]):
            print("      {:22s} {:.4f}".format(lab, s), flush=True)
        json.dump(thr_store, open(RESULTS / "{}_thr_{}.json".format(a.out, mname), "w"), indent=1)
        json.dump(calib_store, open(RESULTS / "{}_calib_{}.json".format(
            a.out, mname), "w"), indent=1)
        print("  per-lab calibration score: {}".format(calib_store), flush=True)
        np.savez_compressed(RESULTS / "{}_pred_{}.npz".format(a.out, mname),
                            pred=pred_g[te], pred_argmax=pred_arg_g[te],
                            proba_test=P_test_g[te], test_mask=te,
                            proba_calib=P_cal_g[calib_g],
                            calib_mask=calib_g)

    pd.DataFrame(rows).to_csv(RESULTS / "{}.csv".format(a.out), index=False)
    print("\nGuardado results/{}.csv  (total {:.0f}s)".format(a.out, time.perf_counter() - t00))


if __name__ == "__main__":
    main()
