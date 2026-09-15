"""
src/submit/predict.py
=====================
Inference for the real submission: from a tracking parquet to `submission.csv`
rows.

Three things this module has to respect that are not obvious:

* **Identifier format.** The annotations use integers and encode a self-directed
  behaviour as `agent == target` (`2,2,rear`), but `behaviors_labeled` and
  `sample_submission.csv` use strings: `mouse1`, `mouse2` and `self`. The scorer
  builds the key `"{agent_id},{target_id},{action}"` and looks it up in
  `behaviors_labeled`; a key that does not match is **discarded silently**,
  scoring neither a hit nor a miss. So only triplets appearing literally in
  `behaviors_labeled` are emitted here: whatever convention the file uses, the
  key matches. That also applies the active-class masking, which was already a
  measured improvement on its own.

* **Feature layout identical to training.** The cache pads each lab out to the
  global width **per block** (mean, std, min, max), not at the end, and only then
  appends the temporal context with `n_base = global_width / 4`. This has to be
  reproduced exactly, or the columns stop meaning what the model was trained on.

* **The lab's canonical keypoints.** AdaptableSnail mixes two schemas, of 10 and
  18 keypoints; the bundle records the canonical one used at training time.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import numpy as np

BACKGROUND = "background"


# ───────────────────────────── loading the models ────────────────────────────
def load_bundles(models_dir) -> Dict[str, dict]:
    """Load the serialised bundles, keyed by lab."""
    import joblib
    out = {}
    for p in sorted(pathlib.Path(models_dir).glob("*.joblib")):
        b = joblib.load(p)
        out[b["lab"]] = b
    return out


def parse_behaviors(raw) -> List[str]:
    """`behaviors_labeled` arrives as a JSON string of "agent,target,action"."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    try:
        return [str(x) for x in json.loads(str(raw))]
    except Exception:
        return []


# ─────────────────────────────── features ────────────────────────────────────
def video_windows(tracking_df, lab_id, bundle, frame_limit=None):
    """Aggregated windows per ordered pair, in the exact training layout.

    Returns (X, meta); each `meta` row carries the agent, the target and the
    core frame span.
    """
    from src.data.features import extract_features
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton
    from src.eval.mabe_metric import window_frame_spans

    W = int(bundle.get("window", 64))
    S = int(bundle.get("stride", 8))
    raw = tracking_df
    if frame_limit:
        raw = raw[raw["video_frame"] < frame_limit].reset_index(drop=True)

    kps = sorted(raw["bodypart"].astype(str).unique())
    hint = bundle.get("lab_keypoints")
    if hint:
        sel = [k for k in hint if k in set(kps)]
        if sel:
            kps = sel
    sk = build_skeleton(lab_id, keypoints_hint=kps)
    fit_skeleton(sk, raw)

    res = extract_features(raw, sk, None, window_size=W, stride=S,
                           min_fill=0.0, center="none", pairs=True)

    n_base = int(bundle["n_base"])
    agg = bool(bundle.get("aggregated", True))
    nb = 4 if agg else 1
    k = int(bundle.get("context_k") or 0)
    # Width BEFORE the context is added. It is derived from n_base rather than
    # read from `padded_width`, because that field was captured after the context
    # was appended and so holds 444, not 296. Padding to 444 and then expanding
    # would give 592 columns and the model would fail. The consistency check
    # below catches a bundle where these disagree.
    padded = n_base * nb
    stored = int(bundle.get("padded_width", 0))
    if stored and stored not in (padded, padded + 2 * n_base):
        raise ValueError(
            "inconsistent width: bundle says {}, derived {} (context k={})"
            .format(stored, padded, k))
    off = (W - S) // 2

    rows, meta = [], []
    for key in sorted(res.keys(), key=str):
        r = res[key]
        Xm, fdf = r["X"], r["features_df"]
        if Xm is None or len(Xm) == 0:
            continue
        spans = window_frame_spans(fdf, W, S, 0.0)
        if len(spans) != len(Xm):
            continue
        Xc = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0)
        if agg:
            flat = np.concatenate([Xc.mean(1), Xc.std(1), Xc.min(1), Xc.max(1)],
                                  axis=1).astype(np.float32)
        else:
            flat = Xc.reshape(len(Xc), -1).astype(np.float32)
        # Pad PER BLOCK out to the global width, exactly as build_lb_cache does.
        if flat.shape[1] < padded:
            f = flat.shape[1] // nb
            buf = np.zeros((len(flat), padded), np.float32)
            for j in range(nb):
                buf[:, j * n_base:j * n_base + f] = flat[:, j * f:(j + 1) * f]
            flat = buf
        rows.append(flat)
        for i in range(len(flat)):
            fstart = int(spans[i][0])
            meta.append({"agent": int(r["agent"]),
                         "target": None if r["target"] is None else int(r["target"]),
                         "core_start": fstart + off, "core_stop": fstart + off + S})
    if not rows:
        return np.zeros((0, padded), np.float32), []
    X = np.concatenate(rows, axis=0)

    if k:
        from scripts.lb_context import add_context
        X = add_context(X, np.array(["v"] * len(X), dtype=object),
                        np.array([m["agent"] for m in meta]),
                        np.array([m["core_start"] for m in meta]),
                        n_base=n_base, k=k, verbose=False)
    return X, meta


def _proba(clf, X, n_classes):
    """Probabilities expanded to the full class vocabulary."""
    p = clf.predict_proba(X)
    full = np.zeros((len(X), n_classes), np.float32)
    for j, c in enumerate(np.asarray(clf.pipeline.classes_, dtype=int)):
        full[:, int(c)] = p[:, j]
    return full


# ─────────────────────────────── prediction ──────────────────────────────────
def predict_video(lab_id, video_id, tracking_df, behaviors_labeled, bundle,
                  frame_limit=None) -> List[dict]:
    """Submission rows for one video; an empty list if there is nothing to emit."""
    from scripts.lb_threshold import decide
    from src.eval.mabe_metric import windows_to_submission

    active = set(parse_behaviors(behaviors_labeled))
    if not active:
        return []

    X, meta = video_windows(tracking_df, lab_id, bundle, frame_limit=frame_limit)
    if len(X) == 0:
        return []

    classes = list(bundle["classes"])
    bg = classes.index(BACKGROUND)
    P = _proba(bundle["clf"], X, len(classes))

    # Active-class mask, already in the REAL format of `behaviors_labeled`.
    # `tgt_name[i][c]` records which target class c must be emitted against on
    # row i ("self" or "mouseN"), or None when that triplet is not annotated in
    # this video.
    allowed = np.zeros((len(meta), len(classes)), bool)
    tgt_name = []
    for i, m in enumerate(meta):
        a = "mouse{}".format(m["agent"])
        o = None if m["target"] is None else "mouse{}".format(m["target"])
        names = []
        for j, c in enumerate(classes):
            if "{},self,{}".format(a, c) in active:
                allowed[i, j] = True
                names.append("self")
            elif o is not None and "{},{},{}".format(a, o, c) in active:
                allowed[i, j] = True
                names.append(o)
            else:
                names.append(None)
        tgt_name.append(names)
    allowed[:, bg] = True

    thr = np.asarray(bundle["thresholds"], np.float32)
    choice = bundle.get("choice") or {}

    # Temporal smoothing, if the calibration selected it for this lab.
    if choice.get("smooth"):
        from scripts.lb_threshold import smooth_proba
        P = smooth_proba(P, np.array([str(video_id)] * len(meta), dtype=object),
                         np.array([m["agent"] for m in meta]),
                         np.array([m["core_start"] for m in meta]),
                         window=int(choice.get("smooth_window", 9)))

    pred = decide(P, thr, bg, mode=choice.get("mode", "residual"), allowed=allowed)

    def _emit(sel_idx, pr):
        """Windows to intervals, with ids in the behaviors_labeled format."""
        if len(sel_idx) == 0:
            return []
        return windows_to_submission(
            win_video=np.array([str(video_id)] * len(sel_idx), dtype=object),
            win_agent=np.array(["mouse{}".format(meta[i]["agent"]) for i in sel_idx],
                               dtype=object),
            win_target=np.array([tgt_name[i][pr[i]] for i in sel_idx], dtype=object),
            win_fstart=np.array([meta[i]["core_start"] for i in sel_idx], np.int64),
            win_fstop=np.array([meta[i]["core_stop"] for i in sel_idx], np.int64),
            y_pred_names=np.array([classes[pr[i]] for i in sel_idx], dtype=object))

    sub = _emit(np.flatnonzero(pred != bg), pred)

    # Minimum-duration filter plus gap bridging. `min_duration_filter` is keyed
    # by (lab, action); the bundle stores the per-action statistics, computed
    # over THIS lab's own training annotations.
    if choice.get("mindur", True) and bundle.get("min_dur"):
        from scripts.run_lb_decision import min_duration_filter
        lab_of = {str(video_id): lab_id}
        mind = {(lab_id, a): v for a, v in bundle["min_dur"].items()}
        maxg = {(lab_id, a): v for a, v in (bundle.get("max_gap") or {}).items()}
        sub = min_duration_filter(sub, lab_of, mind, maxg)

    # Revive dead classes, and do it AFTER the filter: the filter removes short
    # intervals, which is the typical shape of a rare action, so a class revived
    # before it would just be deleted again. Only windows already classified as
    # background are given up, so no live action loses anything, and the revived
    # class's F1 moves from exactly 0 to at least 0.
    rates = bundle.get("rates") or {}
    if rates:
        emitted = {}
        for r in sub:
            emitted[r["action"]] = (emitted.get(r["action"], 0)
                                    + int(r["stop_frame"]) - int(r["start_frame"]))
        n_frames = int(tracking_df["video_frame"].max()) + 1
        S = int(bundle.get("stride", 8))
        extra_idx, extra_pred = [], np.array(pred, copy=True)
        for j, c in enumerate(classes):
            if c not in rates or emitted.get(c, 0) > 0:
                continue
            n_win = int(round(rates[c] * n_frames / max(S, 1)))
            cand = np.flatnonzero((pred == bg) & allowed[:, j])
            if n_win <= 0 or len(cand) == 0:
                continue
            take = cand[np.argsort(-P[cand, j])[:min(n_win, len(cand))]]
            extra_pred[take] = j
            extra_idx += take.tolist()
        if extra_idx:
            sub = sub + _emit(np.asarray(sorted(set(extra_idx))), extra_pred)

    # Final guard: never emit a triplet absent from `behaviors_labeled`.
    return [r for r in sub
            if "{},{},{}".format(r["agent_id"], r["target_id"], r["action"]) in active]
