"""
src/eval/mabe_metric.py
=======================
Official MABe-Mouse-Behavior-Detection competition metric (event/interval F-beta)
plus the helpers needed to score this project's models with it.

Why this exists
---------------
The project reports *window-macro-F1* (one label per fixed 64-frame window). The
competition — and the literature it should be compared against — scores **events**:
predicted vs. labelled frame-intervals per ``(video, agent, target, action)`` key,
F-beta per action, averaged per action then per lab. The two are not comparable;
a thesis that wants to position its numbers needs the official metric.

The scoring core (`mouse_fbeta_records` / `_single_lab_f1`) is a faithful,
dependency-free re-implementation of the Kaggle metric (the original used polars;
this uses only the stdlib so it can be unit-tested anywhere). It reproduces the
official doctests exactly — see ``scripts/_selftest_mabe_metric.py``.

The remaining helpers bridge this project's per-window predictions to the event
format:
  * ``window_frame_spans`` — recover each kept window's [start, stop] frame span,
    replaying ``features.build_windows``' min-fill filter so spans align 1:1 with X.
  * ``windows_to_submission`` — merge consecutive same-action windows per
    ``(video, agent, target)`` into intervals (a submission DataFrame).
  * ``solution_from_annotations`` — build the official solution DataFrame from this
    project's annotation parquet + the video's ``behaviors_labeled`` active set.
  * ``decode_intervals`` / ``robustify`` — ports of the notebook's
    ``predict_multiclass`` / ``robustify`` for the per-frame-probability path.

Adapted from the public Kaggle metric and the community XGBoost baseline notebook
("Social action recognition in mice — XGBoost").
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

BACKGROUND = "background"


# ════════════════════════════════════════════════════════════════════════════
#  Scoring core — pure Python, no third-party deps (so it is trivially testable)
# ════════════════════════════════════════════════════════════════════════════
def _single_lab_f1(lab_solution: List[dict], lab_submission: List[dict],
                   beta: float = 1.0) -> float:
    """F-beta for one lab. ``*_solution``/``*_submission`` are lists of dicts with
    keys video_id, agent_id, target_id, action, start_frame, stop_frame; solution
    rows additionally carry ``behaviors_labeled`` (JSON list of "agent,target,action").

    Mirrors the official ``single_lab_f1``: frame-set intersection per key, per
    action, with predictions outside a video's active-label set ignored.
    """
    def _key(r) -> str:
        return f"{r['video_id']}_{r['agent_id']}_{r['target_id']}_{r['action']}"

    label_frames: defaultdict = defaultdict(set)
    for r in lab_solution:
        label_frames[_key(r)].update(range(int(r["start_frame"]), int(r["stop_frame"])))

    videos = {r["video_id"] for r in lab_solution}
    # active-label set per video (from behaviors_labeled)
    active_by_video: Dict[object, set] = {}
    for v in videos:
        bl = next((r.get("behaviors_labeled") for r in lab_solution
                   if r["video_id"] == v), None)
        active_by_video[v] = set(json.loads(bl)) if isinstance(bl, str) else set()

    prediction_frames: defaultdict = defaultdict(set)
    for v in videos:
        active = active_by_video.get(v, set())
        for r in lab_submission:
            if r["video_id"] != v:
                continue
            if f"{r['agent_id']},{r['target_id']},{r['action']}" not in active:
                continue  # cannot be scored — not an active label for this video
            new = set(range(int(r["start_frame"]), int(r["stop_frame"])))
            new -= prediction_frames[_key(r)]  # ignore truly-redundant frames
            prediction_frames[_key(r)].update(new)

    tps: defaultdict = defaultdict(int)
    fns: defaultdict = defaultdict(int)
    fps: defaultdict = defaultdict(int)
    for key, pred in prediction_frames.items():
        action = key.split("_")[-1]
        lab = label_frames.get(key, set())
        tps[action] += len(pred & lab)
        fns[action] += len(lab - pred)
        fps[action] += len(pred - lab)

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split("_")[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    b2 = beta * beta
    f1s = []
    for action in distinct_actions:
        denom = (1 + b2) * tps[action] + b2 * fns[action] + fps[action]
        f1s.append(0.0 if denom == 0 else (1 + b2) * tps[action] / denom)
    return sum(f1s) / len(f1s) if f1s else 0.0


def mouse_fbeta_records(solution: List[dict], submission: List[dict],
                        beta: float = 1.0) -> float:
    """Official MABe F-beta over record-lists (lab-averaged). See module docstring."""
    if not solution or not submission:
        raise ValueError("Missing solution or submission data")
    solution_videos = {r["video_id"] for r in solution}
    submission = [r for r in submission if r["video_id"] in solution_videos]

    labs = {r.get("lab_id") for r in solution}
    lab_scores = []
    for lab in labs:
        lab_sol = [r for r in solution if r.get("lab_id") == lab]
        lab_vids = {r["video_id"] for r in lab_sol}
        lab_sub = [r for r in submission if r["video_id"] in lab_vids]
        lab_scores.append(_single_lab_f1(lab_sol, lab_sub, beta=beta))
    return sum(lab_scores) / len(lab_scores) if lab_scores else 0.0


# ════════════════════════════════════════════════════════════════════════════
#  pandas-facing wrappers (match the official signatures)
# ════════════════════════════════════════════════════════════════════════════
def mouse_fbeta(solution, submission, beta: float = 1.0) -> float:
    """DataFrame wrapper around :func:`mouse_fbeta_records`.

    ``solution`` needs columns video_id, agent_id, target_id, action, start_frame,
    stop_frame, lab_id, behaviors_labeled; ``submission`` the first six.
    """
    req = ["video_id", "agent_id", "target_id", "action", "start_frame", "stop_frame"]
    for col in req:
        if col not in solution.columns:
            raise ValueError(f"Solution is missing column {col}")
        if col not in submission.columns:
            raise ValueError(f"Submission is missing column {col}")
    return mouse_fbeta_records(solution.to_dict("records"),
                               submission.to_dict("records"), beta=beta)


def score(solution, submission, row_id_column_name: str = "row_id",
          beta: float = 1.0) -> float:
    """Competition entry point (drops the row-id column, then scores)."""
    solution = solution.drop(columns=[row_id_column_name], errors="ignore")
    submission = submission.drop(columns=[row_id_column_name], errors="ignore")
    return mouse_fbeta(solution, submission, beta=beta)


# ════════════════════════════════════════════════════════════════════════════
#  Window → interval bridge (this project's models predict one label per window)
# ════════════════════════════════════════════════════════════════════════════
def window_frame_spans(feature_df, window_size: int, stride: int,
                       min_fill: float):
    """Replay ``features.build_windows`` to recover each *kept* window's frame span.

    Returns a list of (start_frame, stop_frame) using the DataFrame's index
    (``video_frame``) — aligned 1:1 and in the same order as the X array that
    ``build_windows`` produced, including the same min-fill skipping.
    ``stop_frame`` is exclusive (frame after the window's last frame).
    """
    import numpy as np

    arr = feature_df.values.astype(np.float32)
    frames = np.asarray(feature_df.index)
    T = arr.shape[0]
    spans = []
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        chunk = arr[start:end]
        valid_rows = np.isfinite(chunk).all(axis=1)
        if valid_rows.mean() < min_fill:
            continue  # identical filter to build_windows
        spans.append((int(frames[start]), int(frames[end - 1]) + 1))
    return spans


def windows_to_submission(win_video, win_agent, win_target,
                          win_fstart, win_fstop, y_pred_names,
                          background: str = BACKGROUND):
    """Merge consecutive same-action windows per (video, agent, target) into
    interval rows. Returns a list of submission dicts (drop background)."""
    import numpy as np

    win_video = np.asarray(win_video)
    win_agent = np.asarray(win_agent)
    win_target = np.asarray(win_target)
    win_fstart = np.asarray(win_fstart, dtype=int)
    win_fstop = np.asarray(win_fstop, dtype=int)
    y_pred_names = np.asarray(y_pred_names, dtype=object)

    order = np.lexsort((win_fstart, win_target, win_agent, win_video))
    rows: List[dict] = []
    cur = None
    for i in order:
        act = str(y_pred_names[i])
        keytuple = (win_video[i], win_agent[i], win_target[i])
        if act == background:
            cur = None
            continue
        if (cur is not None and cur["_k"] == keytuple and cur["action"] == act
                and win_fstart[i] <= cur["stop_frame"]):
            cur["stop_frame"] = max(cur["stop_frame"], int(win_fstop[i]))
            continue
        cur = {"_k": keytuple, "video_id": win_video[i], "agent_id": win_agent[i],
               "target_id": win_target[i], "action": act,
               "start_frame": int(win_fstart[i]), "stop_frame": int(win_fstop[i])}
        rows.append(cur)
    for r in rows:
        r.pop("_k", None)
    return rows


def solution_from_annotations(ann_df, video_id, lab_id, behaviors_labeled,
                              agent_fmt=lambda a: int(a),
                              target_fmt=None):
    """Build official-solution rows for one video from this project's annotation
    parquet (columns agent_id, target_id, action, start_frame, stop_frame).

    ``behaviors_labeled`` is a JSON string of "agent,target,action" entries, as in
    train.csv. ``agent_fmt``/``target_fmt`` control the id representation so that
    solution and submission use the *same* convention (otherwise keys won't match).
    Default keeps integer ids and maps target==agent to 'self'.
    """
    rows = []
    for r in ann_df.to_dict("records"):
        a = agent_fmt(r["agent_id"])
        if target_fmt is not None:
            t = target_fmt(r["target_id"])
        else:
            t = "self" if r["target_id"] == r["agent_id"] else int(r["target_id"])
        rows.append({
            "video_id": video_id, "agent_id": a, "target_id": t,
            "action": r["action"], "start_frame": int(r["start_frame"]),
            "stop_frame": int(r["stop_frame"]), "lab_id": lab_id,
            "behaviors_labeled": behaviors_labeled,
        })
    return rows


# ════════════════════════════════════════════════════════════════════════════
#  Per-frame-probability path (ports of the XGBoost notebook helpers)
# ════════════════════════════════════════════════════════════════════════════
def decode_intervals(prob_df, meta, thresholds: Optional[Dict[str, float]] = None,
                     default_threshold: float = 0.27):
    """Decode per-frame action probabilities into interval rows (port of the
    notebook's ``predict_multiclass``).

    ``prob_df`` : DataFrame (frames x actions) of probabilities.
    ``meta``    : DataFrame with columns video_id, agent_id, target_id, video_frame
                  aligned row-for-row with ``prob_df``.
    Returns a submission DataFrame.
    """
    import numpy as np
    import pandas as pd

    thresholds = thresholds or {}
    proba = prob_df.values
    ama = np.argmax(proba, axis=1)
    max_p = proba.max(axis=1)
    thr = np.array([thresholds.get(c, default_threshold) for c in prob_df.columns])
    ama = np.where(max_p >= thr[ama], ama, -1)
    ama = pd.Series(ama, index=meta["video_frame"].values)

    changes = (ama != ama.shift(1)).values
    ama_ch = ama[changes]
    meta_ch = meta[changes]
    mask = ama_ch.values >= 0
    if len(mask) == 0:
        return pd.DataFrame(columns=["video_id", "agent_id", "target_id",
                                     "action", "start_frame", "stop_frame"])
    mask[-1] = False
    sub = pd.DataFrame({
        "video_id": meta_ch["video_id"].values[mask],
        "agent_id": meta_ch["agent_id"].values[mask],
        "target_id": meta_ch["target_id"].values[mask],
        "action": prob_df.columns[ama_ch[mask].values],
        "start_frame": ama_ch.index[mask],
        "stop_frame": ama_ch.index[1:][mask[:-1]],
    })
    return sub


def robustify(submission):
    """Clean a submission for scoring (port of the notebook's ``robustify`` core):
    drop start>=stop and remove overlapping predictions per (video, agent, target).
    The empty-video fill is omitted (it needs the tracking files); add separately
    if a fully-dense submission is required."""
    import pandas as pd

    sub = submission[submission["start_frame"] < submission["stop_frame"]].copy()
    parts = []
    for _, g in sub.groupby(["video_id", "agent_id", "target_id"]):
        g = g.sort_values("start_frame")
        keep = []
        last_stop = -1
        for _, r in g.iterrows():
            if r["start_frame"] < last_stop:
                keep.append(False)
            else:
                keep.append(True)
                last_stop = r["stop_frame"]
        parts.append(g[keep])
    out = pd.concat(parts) if parts else sub
    return out.reset_index(drop=True)
