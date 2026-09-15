"""
scripts/lb_threshold.py
=======================
Per-window probabilities, **temporal smoothing** and **per (lab, action) thresholds**.

An argmax over 21 classes with `background` dominating under-predicts the short, rare
actions: the official metric averages F1 **per action**, so losing a rare action costs
exactly as much as losing `sniff`. The recipe the leading teams use is a threshold
`t[lab, action]` calibrated on held-out data, deciding by comparing each class against
its own requirement rather than by raw probability.

The per (lab, action) diagnostic showed the dominant failure is NOT dead classes, only
5 of 56 score 0, but **over-prediction**: TranquilPanther predicts `sniff` on 27,520
frames against 3,548 real ones, a factor of 7.8. Recall is fine, 0.4 to 0.7; what
collapses is precision. The two new pieces here target that:

* `smooth_proba` — smooths each (video, agent) probability sequence in temporal order.
  An isolated window firing above threshold is switched off when its neighbours do not
  agree, and that isolated firing is exactly the noise inflating the false positives.
* `decide(mode="ratio")` — breaks ties on `p/t` instead of `p-t`. With residuals, a
  class with a high threshold wins on absolute margin even when it is proportionally
  less well supported; the ratio compares each class against its own scale.
"""
from __future__ import annotations
import numpy as np

DEFAULT_GRID = np.round(np.concatenate([np.arange(0.02, 0.30, 0.02),
                                        np.arange(0.30, 0.95, 0.05)]), 3)


# ───────────────────────── per-window probabilities ─────────────────────────
def _expand(proba, model_classes, n_classes):
    """sklearn returns columns only for classes it saw; expand out to n_classes."""
    full = np.zeros((len(proba), n_classes), dtype=np.float32)
    for j, c in enumerate(np.asarray(model_classes, dtype=int)):
        full[:, int(c)] = proba[:, j]
    return full


def proba_tree(clf, X, n_classes, aggregated=False):
    """`aggregated=True` when X already holds window statistics, shape (N, 4F)."""
    from src.evaluation.lovo import aggregate_windows
    A = X if aggregated else np.nan_to_num(aggregate_windows(X))
    p = clf.predict_proba(A)
    return _expand(p, clf.pipeline.classes_, n_classes)


def proba_seq(fitted, X, n_classes, batch=1024):
    import torch
    model, scaler = fitted
    model.eval()
    Xs = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch):
            t = torch.nan_to_num(torch.tensor(Xs[i:i + batch], dtype=torch.float32))
            out.append(torch.softmax(model(t), dim=1).cpu().numpy())
    p = np.vstack(out).astype(np.float32)
    if p.shape[1] != n_classes:
        q = np.zeros((len(p), n_classes), np.float32)
        q[:, :p.shape[1]] = p[:, :n_classes]
        p = q
    return p


# ─────────────────────────── temporal smoothing ─────────────────────────────
def smooth_proba(P, vids, agents, core_start, window=9, polyorder=2):
    """Savitzky-Golay over each (video, agent) probability sequence.

    Cache windows arrive neither time-ordered nor grouped, so they must be grouped by
    (video, agent) and sorted by `core_start` before filtering. Reuses `_smooth_1d`
    from `src/data/features.py`, which tolerates NaN and returns the signal untouched
    when it is shorter than the filter window.
    """
    from src.data.features import _smooth_1d
    out = P.copy()
    keys = {}
    for i, (v, a) in enumerate(zip(vids.tolist(), agents.tolist())):
        keys.setdefault((v, a), []).append(i)
    for idx in keys.values():
        idx = np.asarray(idx)
        if len(idx) < window:
            continue
        idx = idx[np.argsort(core_start[idx])]
        blk = P[idx]
        for c in range(P.shape[1]):
            out[idx, c] = _smooth_1d(blk[:, c].astype(np.float64),
                                     window_length=window, polyorder=polyorder)
    return np.nan_to_num(out, nan=0.0)


# ─────────────────────────────── decision ───────────────────────────────────
def build_active_mask(vids, agents, others, classes, active):
    """An (N, C) mask: which classes each window is allowed to score.

    `_single_lab_f1` SILENTLY discards any prediction whose
    "{agent},{target},{action}" is not in that video's `behaviors_labeled`. Without
    this mask, a window whose winning class is inactive produces nothing at all:
    neither hit nor miss, it is simply lost. Measured on the current test set that
    affects 28.3% of non-background predictions, 43,043 of 151,889, and 45% of those
    had an alternative active class at p>0.05.

    This is legitimate: `behaviors_labeled` ships in `test.csv`, and filtering on it is
    exactly what the leading teams describe doing.
    """
    M = np.zeros((len(vids), len(classes)), dtype=bool)
    cache = {}
    for i in range(len(vids)):
        key = (str(vids[i]), int(agents[i]), int(others[i]))
        row = cache.get(key)
        if row is None:
            s_ = active.get(key[0], ())
            row = np.array([("{},self,{}".format(key[1], c) in s_) or
                            ("{},{},{}".format(key[1], key[2], c) in s_)
                            for c in classes], dtype=bool)
            cache[key] = row
        M[i] = row
    return M


def decide(P, thr, bg_idx, mode="residual", allowed=None):
    """Pick a class by comparing each one against its own threshold.

    `residual`: argmax of `p - t`.  `ratio`: argmax of `p / t`.
    Either way, `background` is emitted when no class clears its threshold.
    """
    t = np.maximum(thr[None, :], 1e-6)
    R = (P / t) if mode == "ratio" else (P - thr[None, :])
    R = R.copy()
    R[:, bg_idx] = -np.inf
    if allowed is not None:
        # Without this the winning class can be one the video does not annotate, and
        # the prediction is discarded downstream without a trace.
        R = np.where(allowed, R, -np.inf)
    best = R.argmax(axis=1)
    passes = P[np.arange(len(P)), best] > thr[best]
    return np.where(passes, best, bg_idx).astype(np.int64)


# ──────────────────────────── threshold tuning ──────────────────────────────
def tune_thresholds(P, mask, vids, labs, ev, classes, sol, active, lab_of,
                    score_fn, submission_fn, grid=DEFAULT_GRID, passes=2,
                    init=0.5, mode="residual", allowed=None, verbose=True):
    """Coordinate ascent on `t[lab, action]` against the official metric.

    Optimised lab by lab, because a lab's score depends only on its own videos, so the
    coordinates of different labs are separable. `P` must already be smoothed if
    smoothing will be used at inference: calibrating on raw probabilities and then
    deciding on smoothed ones leaves the thresholds in the wrong place.
    """
    bg = classes.index("background")
    nC = len(classes)
    out = {}
    idx_all = np.flatnonzero(mask)
    for lab in sorted(set(labs[mask].tolist())):
        sel = idx_all[labs[idx_all] == lab]
        if len(sel) == 0:
            continue
        lab_vids = set(vids[sel].tolist())
        sol_lab = [r for r in sol if r["video_id"] in lab_vids]
        if not sol_lab:
            continue
        acts = sorted({r["action"] for r in sol_lab})
        aidx = [classes.index(a) for a in acts if a in classes]
        thr = np.full(nC, float(init), dtype=np.float32)
        submask = np.zeros(len(vids), bool); submask[sel] = True
        Psub = P[sel]
        Asub = None if allowed is None else allowed[sel]

        def _score(t):
            sub = submission_fn(submask, decide(Psub, t, bg, mode=mode, allowed=Asub))
            return score_fn(sol_lab, sub)

        best = _score(thr)
        for _ in range(passes):
            improved = False
            for c in aidx:
                cur = thr[c]; bs = best
                for t in grid:
                    thr[c] = t
                    s = _score(thr)
                    if s > bs + 1e-9:
                        bs, cur = s, t
                thr[c] = cur
                if bs > best + 1e-9:
                    best = bs; improved = True
            if not improved:
                break
        out[lab] = thr
        if verbose:
            print("    umbrales {:22s} score_tune={:.4f}".format(lab, best), flush=True)
    return out


def apply_thresholds(P, mask, labs, thr_by_lab, classes, default=0.5, mode="residual",
                     allowed=None):
    """Apply each lab's thresholds to its own windows."""
    bg = classes.index("background")
    pred = np.full(len(labs), bg, dtype=np.int64)
    idx = np.flatnonzero(mask)
    for lab in sorted(set(labs[idx].tolist())):
        sel = idx[labs[idx] == lab]
        thr = thr_by_lab.get(lab, np.full(len(classes), default, np.float32))
        pred[sel] = decide(P[sel], thr, bg, mode=mode,
                           allowed=None if allowed is None else allowed[sel])
    return pred
