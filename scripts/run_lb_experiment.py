"""
scripts/run_lb_experiment.py
============================
MABe leaderboard simulation over the multi-lab corpus (`lb_raw` / `lb_corr`).

A leaderboard-shaped protocol, not LOVO: the model is trained once on the training
videos and evaluated on held-out videos (a pseudo-test), scored with the **official
metric** (`mouse_fbeta_records`: F1 per action -> mean per lab -> mean across labs).
Within the pseudo-test the real leaderboard's public/private split (32% / 68%) is
simulated, and the uncertainty is bounded by bootstrapping over videos.

Everything runs twice: `raw` (raw tracking) and `corr` (MouseSkeleton correction).

Two decisions set the metric's ceiling, and therefore everything else:

* **Core emission.** The interval emitted is `[core_start, core_stop)`, the S central
  frames of the window, not the whole window. Emitting the whole window imposes a
  minimum interval of W frames, and since 75% of annotated behaviours last under 64
  frames, that only inflates the false positives: the measured ceiling goes from 0.49
  with the full window to about 0.95 with the core at S=8.
* **Target routing.** `win_target` always points at the other mouse, but the solution
  encodes self-directed behaviours as `self`. The `behaviors_labeled` filter inside
  `_single_lab_f1` discards those predictions **silently**. Here the target chosen is
  whichever one exists in the video's active set, worth +0.31 on CRIM13 and
  TranquilPanther.

`src/eval/mabe_metric.py` is NOT touched: it reproduces the six official doctests and
must keep doing so. Everything above lives in this bridging layer.

Stages (`--stage`): models | bootstrap | crosslab | al | geometry | all
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys, pathlib, time, json, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np
import pandas as pd

FEAT = _r / "dataset" / "features"
RESULTS = _r / "results"
SEED = 42
TEST_FRAC = 0.30
PUBLIC_FRAC = 0.32          # the public leaderboard uses about 32% of the test set
N_BOOT = 500
CONDITIONS = ("raw", "corr")
MODELS = ["rf", "histgb", "tcn", "bilstm"]
TREE_MODELS = {"rf", "histgb"}
SUFFIX = os.environ.get("LB_SUFFIX", "_s8")


def _cache(cond):
    return FEAT / "lb_{}{}.npz".format(cond, SUFFIX)


def _pair_idx_path():
    return FEAT / "lb_pair_index{}.npz".format(SUFFIX)


# ───────────────────────────── data ─────────────────────────────────────────
def load_lb_cache(cond):
    d = np.load(_cache(cond), allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    ev = {"agent": d["win_agent"].astype(np.int64), "target": d["win_target"].astype(np.int64),
          "fstart": d["win_fstart"].astype(np.int64), "fstop": d["win_fstop"].astype(np.int64)}
    # Emitted core: present in the newer caches; falls back to the window if absent.
    ev["core_start"] = (d["win_core_start"].astype(np.int64)
                        if "win_core_start" in d.files else ev["fstart"])
    ev["core_stop"] = (d["win_core_stop"].astype(np.int64)
                       if "win_core_stop" in d.files else ev["fstop"])
    return (X, d["y"].astype(np.int64), d["video_ids"].astype(str),
            d["lab_ids"].astype(str), [str(c) for c in d["classes"]], ev)


def build_pair_index():
    """Index of the windows common to both caches.

    The correction imputes missing keypoints, so `min_fill` lets a different set of
    windows through in each variant. To keep the comparison paired they are intersected
    on (video, agent, start frame): same windows, same labels, and the only thing that
    differs is the features.
    """
    p = _pair_idx_path()
    a = load_lb_cache("raw")
    b = load_lb_cache("corr")
    ka = list(zip(a[2].tolist(), a[5]["agent"].tolist(), a[5]["fstart"].tolist()))
    kb = {k: i for i, k in enumerate(zip(
        b[2].tolist(), b[5]["agent"].tolist(), b[5]["fstart"].tolist()))}
    ia, ib = [], []
    for i, k in enumerate(ka):
        j = kb.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
    ia = np.asarray(ia, np.int64)
    ib = np.asarray(ib, np.int64)
    if not np.array_equal(a[1][ia], b[1][ib]):
        raise SystemExit("Labels disagree across the paired intersection")
    np.savez_compressed(p, raw=ia, corr=ib)
    print("Paired intersection: {} windows (raw {} / corr {})".format(
        len(ia), len(a[1]), len(b[1])), flush=True)
    return ia, ib


def load_aligned(cond):
    """Caches restricted to the common windows, aligned 1:1 across variants."""
    p = _pair_idx_path()
    if not p.exists():
        build_pair_index()
    idx = np.load(p)[cond]
    X, y, vids, labs, classes, ev = load_lb_cache(cond)
    return (X[idx], y[idx], vids[idx], labs[idx], classes,
            {k: v[idx] for k, v in ev.items()})


def split_videos(vids, labs, test_frac=TEST_FRAC, seed=SEED):
    """Split by video, stratified by lab."""
    rng = np.random.default_rng(seed)
    lab_of = {v: l for v, l in zip(vids, labs)}
    train, test = [], []
    for lab in sorted(set(labs)):
        lv = sorted(v for v in set(vids) if lab_of[v] == lab)
        n_test = max(1, int(round(test_frac * len(lv))))
        sel = set(rng.permutation(len(lv))[:n_test].tolist())
        for i, v in enumerate(lv):
            (test if i in sel else train).append(v)
    return sorted(train), sorted(test), lab_of


def video_coverage(vids, fstop):
    cov = {}
    for v, f in zip(vids, fstop):
        cov[v] = max(cov.get(v, 0), int(f))
    return cov


# ─────────────────────────── official metric ───────────────────────────────
def build_solution(videos, lab_of, cov):
    """Solution rows plus the per-video active set, used to route the target."""
    from src.data.loader import load_annotations
    from src.eval.mabe_metric import solution_from_annotations
    sol, active = [], {}
    for vid in videos:
        lab = lab_of[vid]
        limit = cov.get(vid, 0)
        if limit <= 0:
            continue
        ann = load_annotations(vid, lab, max_frame=limit)
        if ann is None or not len(ann):
            continue
        ann = ann[ann["start_frame"] < limit].copy()
        if not len(ann):
            continue
        ann["stop_frame"] = ann["stop_frame"].clip(upper=limit)
        triples = set()
        for r in ann.to_dict("records"):
            a = int(r["agent_id"])
            t = "self" if r["target_id"] == r["agent_id"] else int(r["target_id"])
            triples.add("{},{},{}".format(a, t, r["action"]))
        active[str(vid)] = triples
        sol += solution_from_annotations(ann, str(vid), lab, json.dumps(sorted(triples)))
    return sol, active


def build_submission(mask, vids, ev, pred, classes, active=None, core=True):
    """Submission with core emission and target routing.

    `core=True` emits `[core_start, core_stop)`. `active` allows self-directed
    behaviours to be routed to `target='self'` so the `behaviors_labeled` filter does
    not discard them. With `active=None` the old behaviour is kept, always the other
    mouse, which is what reproduces the stride-32 run.
    """
    from src.eval.mabe_metric import windows_to_submission
    names = np.array([classes[p] for p in pred], dtype=object)
    vv, ag = vids[mask], ev["agent"][mask]
    other = ev["target"][mask]
    if active is None:
        tg = other
    else:
        tg = np.empty(len(vv), dtype=object)
        for i in range(len(vv)):
            key = "{},self,{}".format(int(ag[i]), names[i])
            tg[i] = ("self" if key in active.get(str(vv[i]), ())
                     else str(int(other[i])))
    fs = ev["core_start"][mask] if core else ev["fstart"][mask]
    fe = ev["core_stop"][mask] if core else ev["fstop"][mask]
    return windows_to_submission(win_video=vv, win_agent=ag, win_target=tg,
                                 win_fstart=fs, win_fstop=fe, y_pred_names=names)


def official_score(sol, sub, videos=None):
    from src.eval.mabe_metric import mouse_fbeta_records
    if videos is not None:
        videos = set(videos)
        sol = [r for r in sol if r["video_id"] in videos]
        sub = [r for r in sub if r["video_id"] in videos]
    if not sol or not sub:
        return 0.0
    return float(mouse_fbeta_records(sol, sub))


# The labs that DO resemble the real competition test set. The official test set
# excludes CalMS21 and CRIM13, which are 4 of our 9 labs and also the 4 with the best
# scores (mean 0.5987 against 0.4107 for these five). Averaging all nine produces an
# optimistic figure that does not estimate the real leaderboard position, so both
# means should always be reported.
REAL_TEST_LABS = ("CautiousGiraffe", "ElegantMink", "InvincibleJellyfish",
                  "JovialSwallow", "TranquilPanther")


def subset_score(per_lab, labs=REAL_TEST_LABS):
    """Mean score over a subset of labs, or None when the subset is empty.

    The official metric averages per lab, so restricting the average to a subset is
    exactly what the metric would do if the test set contained only those
    laboratorios.
    """
    vals = [v for k, v in per_lab.items() if k in set(labs)]
    return float(np.mean(vals)) if vals else None


def report_scores(per_lab, prefix="    "):
    """Print the global score, the real-test-like score, and the per-lab breakdown."""
    g = float(np.mean(list(per_lab.values()))) if per_lab else float("nan")
    r = subset_score(per_lab)
    print("{}score {} labs = {:.4f}".format(prefix, len(per_lab), g), flush=True)
    if r is not None:
        print("{}score {} labs tipo test real = {:.4f}".format(
            prefix, len([k for k in per_lab if k in set(REAL_TEST_LABS)]), r), flush=True)
    for lab, s in sorted(per_lab.items(), key=lambda kv: -kv[1]):
        mark = "*" if lab in set(REAL_TEST_LABS) else " "
        print("{}  {} {:22s} {:.4f}".format(prefix, mark, lab, s), flush=True)
    print("{}  (* = lab present in the real competition test set)".format(prefix),
          flush=True)


def per_lab_scores(sol, sub, lab_of):
    out = {}
    for lab in sorted({r["lab_id"] for r in sol}):
        vs = {r["video_id"] for r in sol if r["lab_id"] == lab}
        out[lab] = official_score(sol, sub, vs)
    return out


def bootstrap_public(sol, sub, test_videos, frac=PUBLIC_FRAC, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    tv = np.array(sorted(test_videos), dtype=object)
    k = max(1, int(round(frac * len(tv))))
    return np.array([official_score(sol, sub, rng.choice(tv, size=k, replace=False).tolist())
                     for _ in range(n_boot)])


# ─────────────────────────── stage: models ─────────────────────────────────
def stage_models(epochs, models, train_subsample, tune_frac=0.25, reuse_thr=None):
    """Train, calibrate per (lab, action) thresholds, and score with and without."""
    from sklearn.metrics import f1_score
    from src.evaluation.lovo import run_tree, run_seq
    from scripts.lb_threshold import (proba_tree, proba_seq, tune_thresholds,
                                      apply_thresholds)

    build_pair_index()
    # Load one condition at a time: each cache takes about 2.3 GB.
    _, y, vids, labs, classes, ev = load_aligned("raw")
    train_v, test_v, lab_of = split_videos(vids, labs)

    # Sub-split of train: `fit` trains, `tune` calibrates the thresholds. They must
    # be different videos, or the thresholds overfit to what the model already saw.
    rng = np.random.default_rng(SEED)
    fit_v, tune_v = [], []
    for lab in sorted({lab_of[v] for v in train_v}):
        lv = sorted(v for v in train_v if lab_of[v] == lab)
        k = max(1, int(round(tune_frac * len(lv))))
        sel = set(rng.permutation(len(lv))[:k].tolist())
        for i, v in enumerate(lv):
            (tune_v if i in sel else fit_v).append(v)

    cov = video_coverage(vids, ev["fstop"])
    sol_te, active_te = build_solution(test_v, lab_of, cov)
    sol_tu, active_tu = build_solution(tune_v, lab_of, cov)
    fit = np.isin(vids, fit_v)
    tu = np.isin(vids, tune_v)
    te = np.isin(vids, test_v)
    tr = np.isin(vids, train_v)
    print("fit {} vid ({} win) | tune {} vid ({} win) | test {} vid ({} win)".format(
        len(fit_v), int(fit.sum()), len(tune_v), int(tu.sum()),
        len(test_v), int(te.sum())), flush=True)

    # With `reuse_thr` the thresholds come already calibrated from an earlier run
    # that had a clean fit/tune split, so here training uses ALL the train videos: the
    # split only existed in order to calibrate, and giving it up cost more than the
    # umbrales aportaban (-0,052 frente a +0,014 en tcn).
    base = tr if reuse_thr is not None else fit
    fit_idx = np.flatnonzero(base)
    if train_subsample and len(fit_idx) > train_subsample:
        fit_idx = np.sort(rng.choice(fit_idx, train_subsample, replace=False))
        print("Training subsample: {} windows".format(len(fit_idx)), flush=True)

    bg = classes.index("background")
    nC = len(classes)
    rows, preds, thr_store = [], {}, {}
    for cond in CONDITIONS:
        X, y_c, vids_c, labs_c, classes_c, ev_c = load_aligned(cond)
        for m in models:
            t0 = time.perf_counter()
            if m in TREE_MODELS:
                _, fitted = run_tree(m, X[fit_idx], y_c[fit_idx], X[:1], seed=SEED)
                P_tu = None if reuse_thr is not None else proba_tree(fitted, X[tu], nC)
                P_te = proba_tree(fitted, X[te], nC)
            else:
                _, fitted = run_seq(m, X[fit_idx], y_c[fit_idx], X[:1], nC,
                                    epochs=epochs, seed=SEED)
                P_tu = None if reuse_thr is not None else proba_seq(fitted, X[tu], nC)
                P_te = proba_seq(fitted, X[te], nC)
            el = time.perf_counter() - t0

            # (a) Plain argmax: the reference, equivalent to the previous run
            pred_arg = P_te.argmax(axis=1).astype(np.int64)
            sc_arg = official_score(
                sol_te, build_submission(te, vids_c, ev_c, pred_arg, classes_c,
                                         active=active_te))

            # (b) Per (lab, action) thresholds
            if reuse_thr is not None:
                thr = {l: np.asarray(v, np.float32)
                       for l, v in reuse_thr.get("{}|{}".format(cond, m), {}).items()}
            else:
                Pf_tu = np.zeros((len(vids), nC), np.float32); Pf_tu[tu] = P_tu
                thr = tune_thresholds(
                    Pf_tu, tu, vids_c, labs_c, ev_c, classes_c, sol_tu, active_tu, lab_of,
                    score_fn=official_score,
                    submission_fn=lambda mk, pr: build_submission(
                        mk, vids_c, ev_c, pr, classes_c, active=active_tu),
                    verbose=False)
            Pf_te = np.zeros((len(vids), nC), np.float32); Pf_te[te] = P_te
            pred_thr = apply_thresholds(Pf_te, te, labs_c, thr, classes_c)[te]
            sc_thr = official_score(
                sol_te, build_submission(te, vids_c, ev_c, pred_thr, classes_c,
                                         active=active_te))

            wf1 = float(f1_score(y_c[te], pred_thr, average="macro", zero_division=0))
            preds["{}|{}".format(cond, m)] = pred_thr
            preds["{}|{}|argmax".format(cond, m)] = pred_arg
            np.savez_compressed(
                RESULTS / "lb_proba_{}_{}{}.npz".format(cond, m, SUFFIX),
                proba_test=P_te,
                proba_tune=(P_te[:0] if P_tu is None else P_tu),
                thr=np.stack([thr.get(l, np.full(nC, .5, np.float32))
                              for l in sorted(thr)]) if thr else np.zeros((0, nC)),
                thr_labs=np.asarray(sorted(thr), dtype=object))
            thr_store["{}|{}".format(cond, m)] = {k: v.tolist() for k, v in thr.items()}
            rows.append({"condition": cond, "model": m,
                         "f1_event_argmax": round(sc_arg, 4),
                         "f1_event_official": round(sc_thr, 4),
                         "gain_thresholds": round(sc_thr - sc_arg, 4),
                         "f1_macro_window": round(wf1, 4), "fit_predict_s": round(el, 2),
                         "n_train_win": int(len(fit_idx)), "n_test_win": int(te.sum()),
                         "epochs": epochs if m not in TREE_MODELS else None})
            print("  {:5s} {:9s} argmax={:.4f} -> umbrales={:.4f} ({:+.4f})  ({:.0f}s)".format(
                cond, m, sc_arg, sc_thr, sc_thr - sc_arg, el), flush=True)
        del X

    pd.DataFrame(rows).to_csv(RESULTS / "lb_models{}.csv".format(SUFFIX), index=False)
    json.dump(thr_store, open(RESULTS / "lb_thresholds{}.json".format(SUFFIX), "w"), indent=1)
    np.savez_compressed(RESULTS / "lb_predictions{}.npz".format(SUFFIX),
                        test_mask=te, video_ids=vids, lab_ids=labs,
                        classes=np.asarray(classes),
                        win_agent=ev["agent"], win_target=ev["target"],
                        win_fstart=ev["fstart"], win_fstop=ev["fstop"],
                        win_core_start=ev["core_start"], win_core_stop=ev["core_stop"],
                        **preds)
    json.dump({"train_videos": train_v, "fit_videos": fit_v, "tune_videos": tune_v,
               "test_videos": test_v},
              open(RESULTS / "lb_split{}.json".format(SUFFIX), "w"), indent=1)
    print("Guardado lb_models{}.csv + umbrales + probabilidades".format(SUFFIX), flush=True)


# ──────────────────── stage: public/private bootstrap ──────────────────────
def stage_bootstrap():
    d = np.load(RESULTS / "lb_predictions{}.npz".format(SUFFIX), allow_pickle=True)
    te = d["test_mask"]; vids = d["video_ids"].astype(str); labs = d["lab_ids"].astype(str)
    classes = [str(c) for c in d["classes"]]
    split = json.load(open(RESULTS / "lb_split{}.json".format(SUFFIX)))
    test_v = split["test_videos"]
    lab_of = {v: l for v, l in zip(vids, labs)}
    ev = {k: d["win_" + k].astype(np.int64)
          for k in ("agent", "target", "fstart", "fstop", "core_start", "core_stop")}
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(test_v, lab_of, cov)

    rng = np.random.default_rng(SEED)
    tv = np.array(sorted(test_v), dtype=object)
    k = max(1, int(round(PUBLIC_FRAC * len(tv))))
    pub_fixed = set(rng.choice(tv, size=k, replace=False).tolist())
    priv_fixed = [v for v in tv if v not in pub_fixed]

    rows = []
    # The "cond|model|argmax" keys are the no-threshold reference: not rescored
    for key in [k_ for k_ in d.files if "|" in k_ and not k_.endswith("|argmax")]:
        cond, model = key.split("|")
        sub = build_submission(te, vids, ev, d[key], classes, active=active)
        boot = bootstrap_public(sol, sub, test_v)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "condition": cond, "model": model,
            "score_full": round(official_score(sol, sub), 4),
            "score_public_32": round(official_score(sol, sub, pub_fixed), 4),
            "score_private_68": round(official_score(sol, sub, priv_fixed), 4),
            "boot_mean": round(float(boot.mean()), 4), "boot_std": round(float(boot.std()), 4),
            "boot_ci_lo": round(float(lo), 4), "boot_ci_hi": round(float(hi), 4),
            "boot_range": round(float(hi - lo), 4),
            **{"lab_" + kk: round(vv, 4) for kk, vv in per_lab_scores(sol, sub, lab_of).items()},
        })
        print("  {:5s} {:9s} full={:.4f} pub32={:.4f} priv68={:.4f} CI=[{:.3f},{:.3f}]".format(
            cond, model, rows[-1]["score_full"], rows[-1]["score_public_32"],
            rows[-1]["score_private_68"], lo, hi), flush=True)
    df = pd.DataFrame(rows).sort_values("score_full", ascending=False)
    df.to_csv(RESULTS / "lb_bootstrap{}.csv".format(SUFFIX), index=False)
    print("Guardado lb_bootstrap{}.csv".format(SUFFIX), flush=True)
    return df


# ─────────────────────────── stage: cross-lab ──────────────────────────────
_CL = {}


def _cl_init(cond):
    X, y, vids, labs, classes, ev = load_aligned(cond)
    from src.evaluation.lovo import aggregate_windows
    _CL["A"] = np.nan_to_num(aggregate_windows(X))
    _CL.update(y=y, vids=vids, labs=labs, classes=classes, ev=ev, cond=cond)


def _cl_cell(args):
    tr_lab, te_lab = args
    from src.models.baseline import RandomForestBaseline
    A, y, vids, labs = _CL["A"], _CL["y"], _CL["vids"], _CL["labs"]
    classes, ev, cond = _CL["classes"], _CL["ev"], _CL["cond"]
    tr = labs == tr_lab
    te = labs == te_lab
    if tr_lab == te_lab:  # intra-lab: a 70/30 split by video
        uv = sorted(set(vids[tr]))
        rng = np.random.default_rng(SEED)
        cut = max(1, int(0.7 * len(uv)))
        keep = set(np.array(uv, dtype=object)[rng.permutation(len(uv))][:cut].tolist())
        te = tr & ~np.isin(vids, list(keep))
        tr = tr & np.isin(vids, list(keep))
    if tr.sum() == 0 or te.sum() == 0:
        return None
    t0 = time.perf_counter()
    clf = RandomForestBaseline(seed=SEED)
    clf.fit(A[tr], y[tr])
    pred = clf.predict(A[te])
    el = time.perf_counter() - t0
    lab_of = {v: l for v, l in zip(vids, labs)}
    cov = video_coverage(vids, ev["fstop"])
    sol, active = build_solution(sorted(set(vids[te])), lab_of, cov)
    sub = build_submission(te, vids, ev, pred, classes, active=active)
    tr_acts = {classes[c] for c in set(y[tr].tolist())} - {"background"}
    te_acts = {classes[c] for c in set(y[te].tolist())} - {"background"}
    return {"condition": cond, "train_lab": tr_lab, "test_lab": te_lab,
            "official": round(official_score(sol, sub), 4),
            "n_common_actions": len(tr_acts & te_acts),
            "n_train_win": int(tr.sum()), "n_test_win": int(te.sum()),
            "fit_predict_s": round(el, 2)}


def stage_crosslab(workers):
    from scripts.build_lb_cache import LABS
    rows = []
    for cond in CONDITIONS:
        cells = [(a, b) for a in LABS for b in LABS]
        print("\n=== cross-lab {} — {} celdas ===".format(cond, len(cells)), flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_cl_init,
                                 initargs=(cond,)) as ex:
            done = 0
            for fut in as_completed([ex.submit(_cl_cell, c) for c in cells]):
                r = fut.result()
                done += 1
                if r:
                    rows.append(r)
                    if done % 20 == 0:
                        print("  [{}/{}] {} ...".format(done, len(cells), cond), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "lb_crosslab{}.csv".format(SUFFIX), index=False)
    print("Guardado lb_crosslab{}.csv".format(SUFFIX), flush=True)


# ──────────────────────── stage: active learning ───────────────────────────
_AL = {}


def _al_init(cond, pool_cap, val_cap):
    X, y, vids, labs, classes, ev = load_aligned(cond)
    split = json.load(open(RESULTS / "lb_split{}.json".format(SUFFIX)))
    tr = np.isin(vids, split["train_videos"])
    va = np.isin(vids, split["test_videos"])
    rng = np.random.default_rng(SEED)
    ti = np.flatnonzero(tr); vi = np.flatnonzero(va)
    if len(ti) > pool_cap:
        ti = rng.choice(ti, pool_cap, replace=False)
    if len(vi) > val_cap:
        vi = rng.choice(vi, val_cap, replace=False)
    _AL.update(Xp=X[ti], yp=y[ti], Xv=X[vi], yv=y[vi], nC=len(classes), cond=cond)


def _al_run(args):
    strat_name, seed = args
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.rnn import BehaviorGRU
    from src.active_learning import query_strategies as QS

    Xp, yp, Xv, yv, nC = _AL["Xp"], _AL["yp"], _AL["Xv"], _AL["yv"], _AL["nC"]
    dev = torch.device("cpu")

    def model_factory():
        return BehaviorGRU(input_size=Xp.shape[-1], n_classes=nC)

    def fit_fn(model, Xt, yt, Xva, yva, device):
        model.to(device).train()
        cnt = np.bincount(yt, minlength=nC).astype(np.float64)
        w = np.zeros(nC, np.float32); pres = cnt > 0
        w[pres] = cnt[pres].sum() / (pres.sum() * cnt[pres])
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        dl = DataLoader(TensorDataset(torch.tensor(np.nan_to_num(Xt)), torch.tensor(yt)),
                        batch_size=32, shuffle=True)
        for _ in range(15):
            for xb, yb in dl:
                opt.zero_grad()
                loss = crit(model(xb.float()), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return model

    strat = {"random": QS.RandomStrategy(seed=seed),
             "least_confidence": QS.LeastConfidenceStrategy(),
             "margin": QS.MarginSamplingStrategy(),
             "entropy": QS.EntropySamplingStrategy(),
             "coreset": QS.CoresetStrategy(),
             "badge": QS.BADGEStrategy()}[strat_name]
    t0 = time.perf_counter()
    h = QS.run_al_cycle(strat, model_factory, Xp, yp, Xp, yp, Xv, yv,
                        n_initial=30, n_query_per_round=20, n_rounds=8, device=dev,
                        fit_fn=fit_fn, random_state=seed, verbose=False)
    return {"condition": _AL["cond"], "strategy": strat_name, "seed": seed,
            "aulc": round(float(QS.aulc(np.array(h["n_labeled"]), np.array(h["val_f1"]))), 4),
            "final_f1": round(float(h["val_f1"][-1]), 4),
            "best_f1": round(float(max(h["val_f1"])), 4),
            "elapsed_s": round(time.perf_counter() - t0, 1)}


def stage_al(workers, seeds, strategies, pool_cap, val_cap):
    rows = []
    for cond in CONDITIONS:
        tasks = [(s, sd) for s in strategies for sd in seeds]
        print("\n=== AL {} — {} runs ===".format(cond, len(tasks)), flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_al_init,
                                 initargs=(cond, pool_cap, val_cap)) as ex:
            for fut in as_completed([ex.submit(_al_run, t) for t in tasks]):
                r = fut.result()
                rows.append(r)
                print("  {:5s} {:17s} seed={:3d} AULC={:.4f} final={:.4f} ({:.0f}s)".format(
                    cond, r["strategy"], r["seed"], r["aulc"], r["final_f1"],
                    r["elapsed_s"]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "lb_al_runs{}.csv".format(SUFFIX), index=False)
    from scipy.stats import wilcoxon
    summ = []
    for cond, g in df.groupby("condition"):
        piv = g.pivot(index="seed", columns="strategy", values="aulc")
        base = piv["random"] if "random" in piv else None
        for s in piv.columns:
            p = np.nan
            if base is not None and s != "random":
                try:
                    p = float(wilcoxon(piv[s], base).pvalue)
                except Exception:
                    p = np.nan
            summ.append({"condition": cond, "strategy": s,
                         "aulc_mean": round(float(piv[s].mean()), 4),
                         "aulc_std": round(float(piv[s].std()), 4),
                         "delta_vs_random": (round(float(piv[s].mean() - base.mean()), 4)
                                             if base is not None and s != "random" else 0.0),
                         "wilcoxon_p": None if np.isnan(p) else round(p, 4)})
    pd.DataFrame(summ).to_csv(RESULTS / "lb_al_summary{}.csv".format(SUFFIX), index=False)
    print("Guardado lb_al_runs{}.csv + lb_al_summary{}.csv".format(SUFFIX, SUFFIX), flush=True)


# ───────────────────── stage: geometry (Objective 1) ───────────────────────
def _geom_video(args):
    lab, vid, frame_limit = args
    from src.data.loader import load_tracking
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton
    from src.skeleton.config import skeleton_smooth_kwargs, load_skeleton_cfg
    from src.data.correction_cache import load_or_correct
    from src.skeleton.kinematic_constraints import compute_constraint_violations
    from src.data.features import _to_wide
    try:
        raw = load_tracking(vid, lab)
        raw = raw[raw["video_frame"] < frame_limit].reset_index(drop=True)
        kps = sorted(raw["bodypart"].astype(str).unique())
        sk = build_skeleton(lab, keypoints_hint=kps)
        fit_skeleton(sk, raw)
        kw = skeleton_smooth_kwargs(); kw.pop("skeleton", None)
        t0 = time.perf_counter()
        corr, _m = load_or_correct(raw, lab, str(vid), sk, kw)
        el = time.perf_counter() - t0
        # NOTE: `length_threshold` (1.84) is a sigma multiplier used to CORRECT; the
        # violation report uses a relative-error threshold (0.25) instead. Confusing
        # the two makes the correction look far better than it is.
        _thr = float(load_skeleton_cfg().outlier.violation_report_thr)
        out = {}
        for tag, df in (("raw", raw), ("corr", corr)):
            wide = _to_wide(df)
            kp_index = {k: i for i, k in enumerate(sk.keypoints)}
            lb = wide.filter(like="nose_x").iloc[:, 0] * 0 + sk.L_body_median_px
            v = compute_constraint_violations(wide, sk, kp_index, lb, threshold=_thr)
            out[tag] = float(v["violation_rate"].mean())
        return {"lab": lab, "video_id": vid, "viol_raw": round(out["raw"], 4),
                "viol_corr": round(out["corr"], 4),
                "delta_pp": round((out["corr"] - out["raw"]) * 100, 2),
                "correct_s": round(el, 2)}
    except Exception as e:  # noqa: BLE001
        return {"lab": lab, "video_id": vid, "error": "{}: {}".format(type(e).__name__, e)}


def stage_geometry(workers, frame_limit):
    from scripts.build_lb_cache import select_videos
    split = json.load(open(RESULTS / "lb_split{}.json".format(SUFFIX)))
    test_v = set(split["test_videos"])
    pairs = [(l, v) for l, v in select_videos(1000) if v in test_v]
    print("\n=== Geometry (O1) — {} pseudo-test videos ===".format(len(pairs)), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_geom_video, (l, v, frame_limit)) for l, v in pairs]):
            rows.append(fut.result())
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "lb_geometry.csv", index=False)
    ok = df[df["viol_raw"].notna()] if "viol_raw" in df else df
    if len(ok):
        print("Media viol raw={:.4f}  corr={:.4f}  delta={:+.2f} p.p.".format(
            ok.viol_raw.mean(), ok.viol_corr.mean(), ok.delta_pp.mean()))
    print("Guardado results/lb_geometry.csv", flush=True)


# ─────────────────────────────── main ────────────────────────────────────────
def main():
    global SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["models", "bootstrap", "crosslab", "al", "geometry", "all"])
    ap.add_argument("--suffix", default=SUFFIX)
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("LB_EPOCHS", "15")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LB_WORKERS", "6")))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--train-subsample", type=int, default=60000)
    ap.add_argument("--reuse-thresholds", default=None,
                    help="JSON of already-calibrated thresholds; trains on all of train")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 7, 99])
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "least_confidence", "margin", "entropy", "coreset", "badge"])
    ap.add_argument("--pool-cap", type=int, default=2500)
    ap.add_argument("--val-cap", type=int, default=900)
    ap.add_argument("--frame-limit", type=int, default=int(os.environ.get("LB_FRAME_LIMIT", "9000")))
    a = ap.parse_args()
    SUFFIX = a.suffix
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if a.stage in ("models", "all"):
        _thr = json.load(open(a.reuse_thresholds)) if a.reuse_thresholds else None
        stage_models(a.epochs, a.models, a.train_subsample, reuse_thr=_thr)
    if a.stage in ("bootstrap", "all"):
        stage_bootstrap()
    if a.stage in ("crosslab", "all"):
        stage_crosslab(a.workers)
    if a.stage in ("al", "all"):
        stage_al(a.workers, a.seeds, a.strategies, a.pool_cap, a.val_cap)
    if a.stage == "geometry":
        stage_geometry(a.workers, a.frame_limit)

    print("\nTiempo total etapa '{}': {:.0f}s".format(a.stage, time.perf_counter() - t0))


if __name__ == "__main__":
    main()
