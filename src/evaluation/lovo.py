"""
src/evaluation/lovo.py
======================
Leave-One-Video-Out (LOVO) benchmark helpers (A1 + A2 + B1).

One honest evaluation protocol for every model: LOVO over the videos in the
multi-video cache (≡ leave-one-group-out), a fixed label space, and the same
macro-F1 metric for trees and sequence models. The notebooks (`04_baseline`,
`05_deep_model`) import these helpers and run the benchmark in-cell.

Primary metric ``f1_macro_eval`` = macro-F1 over the cross-video-evaluable
classes (those present in ≥2 videos); ``f1_macro_full`` = sklearn-default macro
(kept for continuity, deflated by single-video structural-zero classes).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO / "dataset" / "features" / "CalMS21_task1_multi_5_features.npz"

TREE_MODELS = {"rf", "histgb"}
SEQ_MODELS = {"bilstm", "cnn_lstm", "gru", "tcn"}
ALL_MODELS = ["rf", "histgb", "bilstm", "cnn_lstm", "gru", "tcn"]
EVAL_TARGETS = ("sniff", "mount", "sniffgenital")  # documented; recomputed from data


# ───────────────────────────── data ──────────────────────────────────────────
def load_cache(path=DEFAULT_CACHE):
    """Return (X, y, video_ids, classes) from the multi-video npz cache."""
    d = np.load(path, allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = d["y"].astype(np.int64)
    vids = d["video_ids"].astype(str)
    classes = [str(c) for c in d["classes"]]
    return X, y, vids, classes


EVENT_CACHE = REPO / "dataset" / "features" / "CalMS21_task1_multi_5_event.npz"


def load_event_cache(path=EVENT_CACHE):
    """Return (X, y, video_ids, classes, ev) from the *event* cache (D5).

    Identical to :func:`load_cache` but also returns ``ev`` — the per-window
    metadata (``agent``/``target``/``fstart``/``fstop``) the official interval
    F-beta needs, as written by ``scripts/build_event_cache.py``. ``ev`` is
    ``None`` when the cache lacks that metadata (e.g. the plain feature cache),
    so callers can fall back to window-F1 only. Keeps :func:`load_cache`'s
    4-tuple signature untouched for the existing notebook cells.
    """
    d = np.load(path, allow_pickle=True)
    X = np.nan_to_num(d["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = d["y"].astype(np.int64)
    vids = d["video_ids"].astype(str)
    classes = [str(c) for c in d["classes"]]
    ev = None
    if all(k in d.files for k in ("win_agent", "win_target", "win_fstart", "win_fstop")):
        ev = {"agent": d["win_agent"].astype(np.int64),
              "target": d["win_target"].astype(np.int64),
              "fstart": d["win_fstart"].astype(np.int64),
              "fstop": d["win_fstop"].astype(np.int64)}
    return X, y, vids, classes, ev


def event_f1_for_fold(lab, test_vid, te_idx, pred, classes, vids, ev):
    """Official interval F-beta for one LOVO fold (D5).

    Builds a submission from the fold's window predictions + per-window
    (agent, target, frame-span) metadata, a solution from the test video's
    annotations, and scores with the competition metric. Returns NaN if the
    needed pieces are unavailable.
    """
    import json as _json
    from src.data.loader import load_annotations
    from src.eval.mabe_metric import (
        windows_to_submission, solution_from_annotations, mouse_fbeta_records)

    try:
        ann = load_annotations(test_vid, lab)
    except Exception:  # noqa: BLE001
        return float("nan")
    if ann is None or len(ann) == 0:
        return float("nan")

    names = np.array([classes[p] for p in pred], dtype=object)
    sub = windows_to_submission(
        win_video=vids[te_idx], win_agent=ev["agent"][te_idx],
        win_target=ev["target"][te_idx], win_fstart=ev["fstart"][te_idx],
        win_fstop=ev["fstop"][te_idx], y_pred_names=names)
    if not sub:
        return 0.0

    # Active-label set = (agent,target,action) triples observed in this video,
    # using the same id convention as the submission (agent int, target int/'self').
    triples = set()
    for r in ann.to_dict("records"):
        a = int(r["agent_id"])
        t = "self" if r["target_id"] == r["agent_id"] else int(r["target_id"])
        triples.add(f"{a},{t},{r['action']}")
    behaviors_labeled = _json.dumps(sorted(triples))

    sol = solution_from_annotations(ann, str(test_vid), lab, behaviors_labeled)
    return mouse_fbeta_records(sol, sub)


def evaluable_class_ids(y, vids, classes):
    """Class ids present in ≥2 distinct videos (cross-video-evaluable)."""
    ids = []
    for c in range(len(classes)):
        if len(np.unique(vids[y == c])) >= 2:
            ids.append(c)
    return ids


def aggregate_windows(X):
    """(N, W, F) -> (N, 4F): per-feature mean/std/min/max over the time axis."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)],
                          axis=1).astype(np.float32)


# ──────────────────────────── metrics ────────────────────────────────────────
def fold_metrics(y_true, y_pred, n_classes, eval_ids):
    """Return (per_class_f1, support, f1_macro_eval, f1_macro_full) for one fold."""
    per_class = f1_score(y_true, y_pred, labels=list(range(n_classes)),
                         average=None, zero_division=0)
    support = np.bincount(y_true, minlength=n_classes)
    present_eval = [c for c in eval_ids if support[c] > 0]
    f1_eval = float(np.mean([per_class[c] for c in present_eval])) if present_eval else np.nan
    f1_full = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return per_class, support, f1_eval, f1_full


# ──────────────────────────── tree path ──────────────────────────────────────
def run_tree(name, Xtr, ytr, Xte, seed=42):
    """Fit a tree model on temporally-aggregated windows; return (pred, model)."""
    from src.models.baseline import RandomForestBaseline, GradientBoostingBaseline
    Xtr_a = np.nan_to_num(aggregate_windows(Xtr))
    Xte_a = np.nan_to_num(aggregate_windows(Xte))
    clf = (RandomForestBaseline(seed=seed) if name == "rf"
           else GradientBoostingBaseline(seed=seed))
    clf.fit(Xtr_a, ytr)
    return clf.predict(Xte_a), clf


# ──────────────────────────── deep path ──────────────────────────────────────
def build_seq_model(name, input_size, n_classes):
    from src.models.rnn import BehaviorLSTM, BehaviorCNNLSTM, BehaviorGRU, BehaviorTCN
    if name == "bilstm":
        return BehaviorLSTM(input_size=input_size, n_classes=n_classes)
    if name == "cnn_lstm":
        return BehaviorCNNLSTM(input_size=input_size, n_classes=n_classes)
    if name == "gru":
        return BehaviorGRU(input_size=input_size, n_classes=n_classes)
    if name == "tcn":
        return BehaviorTCN(input_size=input_size, n_classes=n_classes)
    raise ValueError(name)


def run_seq(name, Xtr, ytr, Xte, n_classes, epochs=60, seed=42, device=None):
    """Train a sequence model under the LOVO protocol; return (pred, (model, scaler)).

    Per-fold StandardScaler (fit on train), stratified 15% early-stopping val from
    the training videos, class-weighted CE. Mirrors the harness exactly."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from src.data.dataset import MouseBehaviorDataset
    from src.train import fit

    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[-1]))
    Xtr_s = scaler.transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape).astype(np.float32)
    Xte_s = scaler.transform(Xte.reshape(-1, Xte.shape[-1])).reshape(Xte.shape).astype(np.float32)

    idx = np.arange(len(ytr))
    tr_i, va_i = train_test_split(idx, test_size=0.15, random_state=seed, stratify=ytr)
    train_ds = MouseBehaviorDataset(Xtr_s[tr_i], ytr[tr_i], include_background=True)
    val_ds = MouseBehaviorDataset(Xtr_s[va_i], ytr[va_i], include_background=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float32)
    pres = counts > 0
    w[pres] = counts[pres].sum() / (pres.sum() * counts[pres])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w))

    model = build_seq_model(name, Xtr.shape[-1], n_classes)
    fit(model, train_loader, val_loader, epochs=epochs, lr=1e-3, patience=12,
        device=device, criterion=criterion, verbose=False)

    model.eval()
    with torch.no_grad():
        t = torch.nan_to_num(torch.tensor(Xte_s, dtype=torch.float32).to(device))
        pred = model(t).argmax(dim=1).cpu().numpy()
    return pred, (model, scaler)


# ──────────────────────────── checkpoints (C3) ───────────────────────────────
def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def save_checkpoint(name, fi, test_vid, fitted, classes, input_size,
                    seed=42, epochs=60, features="all-64", tag=None,
                    out_dir=None):
    """Persist a fold's model with a provenance manifest (C3)."""
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "checkpoints_lovo")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_fold{fi}_{test_vid}" + (f"_{tag}" if tag else "")
    manifest = {
        "model": name, "fold": fi, "test_video": str(test_vid),
        "protocol": "LOVO (leave-one-video-out)", "classes": classes,
        "input_size": int(input_size), "seed": seed, "epochs": epochs,
        "features": features, "git_commit": _git_commit(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if name in TREE_MODELS:
        import pickle
        with open(out_dir / f"{stem}.pkl", "wb") as fh:
            pickle.dump(fitted, fh)
    else:
        import torch
        model, scaler = fitted
        torch.save({"model_state_dict": model.state_dict(),
                    "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                    "manifest": manifest}, out_dir / f"{stem}.pt")
    (out_dir / f"{stem}.json").write_text(json.dumps(manifest, indent=2))
    return manifest
