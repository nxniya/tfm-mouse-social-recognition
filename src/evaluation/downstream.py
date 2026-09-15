"""
src/evaluation/downstream.py
============================
A4 (detector calibration) + A3 (downstream-F1 re-measurement) helpers.

`02d_statistics` imports these and runs the analysis in-cell:
  * ``calibrate_detector`` — sweep the outlier-detection params and recommend the
    set that matches the EDA target (~25-30% flagged, length-dominated).
  * ``score_downstream_f1`` — reproduce §16 (raw vs corrected, same BiLSTM
    checkpoint, bootstrap CI) with a chosen set of detection params, so the
    effect of A4 on the downstream task can be measured.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Detection-param grid + EDA-derived targets (A4)
GRID = {
    "speed_sigma":           [3.0, 4.0, 5.0, 6.0],
    "detection_n_sigma":     [3.0, 4.0, 5.0, 6.0],
    "adaptive_thresholds_k": [2.5, 3.5, 4.5, 6.0],
}
CURRENT = {"speed_sigma": 3.0, "detection_n_sigma": 3.0, "adaptive_thresholds_k": 2.5}
TARGET_OVERALL, SPEED_CAP, LENGTH_BAND = 0.30, 0.03, (0.18, 0.40)


# ──────────────────────────── A4 — calibration ───────────────────────────────
def calibrate_detector(lab, video_ids, grid=GRID, target=TARGET_OVERALL,
                       speed_cap=SPEED_CAP, length_band=LENGTH_BAND):
    """Sweep detection params over videos; return (sweep_df, recommendation_dict)."""
    import pandas as pd
    from src.data.loader import load_tracking
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton
    from src.skeleton.keypoint_smoother import detect_outliers_for_video

    loaded = []
    for vid in video_ids:
        try:
            trk = load_tracking(vid, lab)
            sk = build_skeleton(lab)
            fit_skeleton(sk, trk)
            loaded.append((vid, trk, sk))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipping {vid}: {e}")
    if not loaded:
        raise SystemExit("No videos could be loaded.")

    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())]
    if CURRENT not in combos:
        combos.insert(0, CURRENT)

    rows = []
    for combo in combos:
        per_video = [detect_outliers_for_video(
            trk, sk, speed_sigma=combo["speed_sigma"],
            detection_n_sigma=combo["detection_n_sigma"],
            adaptive_thresholds_k=combo["adaptive_thresholds_k"], use_adaptive=True)
            for _v, trk, sk in loaded]
        agg = {k: float(np.mean([d[k] for d in per_video]))
               for k in ("frac_flagged", "frac_speed", "frac_length",
                         "frac_accel", "frac_swap", "frac_near_zero")}
        rows.append({**combo, **agg, "is_current": combo == CURRENT})

    df = pd.DataFrame(rows).sort_values("frac_flagged").reset_index(drop=True)
    cand = df[(df["frac_speed"] <= speed_cap) & df["frac_length"].between(*length_band)].copy()
    if cand.empty:
        cand = df[df["frac_speed"] <= speed_cap].copy()
    if cand.empty:
        cand = df.copy()
    cand["cost"] = (cand["frac_flagged"] - target).abs()
    best = cand.sort_values("cost").iloc[0]
    rec = {"recommended": {k: float(best[k]) for k in grid},
           "frac_flagged": float(best["frac_flagged"]),
           "frac_length": float(best["frac_length"]),
           "frac_speed": float(best["frac_speed"]),
           "videos": [v for v, _t, _s in loaded]}
    return df, rec


# ──────────────────────────── A3 — downstream F1 ──────────────────────────────
def smooth_kwargs_from_cfg(cfg):
    """Build smooth_video kwargs from a skeleton cfg.

    Thin wrapper kept for backward compatibility (``02d_statistics`` imports it);
    the canonical mapping now lives in :func:`src.skeleton.config.skeleton_smooth_kwargs`
    so every correction call site shares one definition.
    """
    from src.skeleton.config import skeleton_smooth_kwargs
    return skeleton_smooth_kwargs(cfg)


def _bootstrap_delta(y, pr, pc, seed=42, n_boot=1000):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    d = [f1_score(y[i], pc[i], average="macro", zero_division=0)
         - f1_score(y[i], pr[i], average="macro", zero_division=0)
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return tuple(np.percentile(d, [2.5, 97.5]))


def score_downstream_f1(cfg, lab="CalMS21_task1", video="13638642",
                        det_params=None, ckpt_path=None, seed=42):
    """Reproduce §16 raw-vs-corrected F1 with the given detection params.

    det_params=None → use the shipped config (baseline, reproduces the old number).
    Returns a dict: f1_raw, f1_corr, delta_f1, ci, frames_flagged_pct, n_windows.
    """
    import torch
    from sklearn.metrics import f1_score
    from src.data.loader import load_tracking, load_annotations
    from src.data.features import extract_features
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton
    from src.skeleton.keypoint_smoother import smooth_video
    from src.models.rnn import BehaviorLSTM

    kwargs = smooth_kwargs_from_cfg(cfg)
    if det_params:
        kwargs.update(det_params)

    raw = load_tracking(video, lab)
    try:
        ann = load_annotations(video, lab)
        ann = ann if ann is not None and len(ann) else None
    except Exception:
        ann = None
    skeleton = build_skeleton(lab)
    fit_skeleton(skeleton, raw)
    corrected, report = smooth_video(raw, lab, skeleton=skeleton, **kwargs)
    flagged = 100.0 * report.get("n_outlier_frames", 0) / max(report.get("n_total_frames", 1), 1)

    res_raw, res_corr = extract_features(raw, skeleton, ann), extract_features(corrected, skeleton, ann)
    mouse = max(res_corr, key=lambda m: int(np.sum(
        [str(v) != "background" for v in res_corr[m].get("y", [])])) if res_corr[m].get("y") is not None else -1)
    Xr, Xc, yc = res_raw[mouse]["X"], res_corr[mouse]["X"], res_corr[mouse]["y"]

    ckpt_path = Path(ckpt_path) if ckpt_path else (
        REPO / "results" / "checkpoints" / f"bilstm_{lab}_{video}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    n_classes = sd["head.weight"].shape[0]
    lstm_key = next(k for k in sd if "lstm" in k and "weight_ih" in k)
    input_size = sd[lstm_key].shape[1]
    model = BehaviorLSTM(input_size=input_size, n_classes=n_classes, bidirectional=True)
    model.load_state_dict(sd); model.eval()

    def infer(X):
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)[:, :, :input_size]
            return model(t).argmax(dim=1).numpy()

    labels = sorted(set(yc.tolist()))
    l2i = {l: i for i, l in enumerate(labels)}
    yi = np.array([l2i.get(l, -1) for l in yc]); valid = yi >= 0
    yt = yi[valid]
    pr, pc = infer(Xr)[valid], infer(Xc)[valid]
    f1_raw = float(f1_score(yt, pr, average="macro", zero_division=0))
    f1_corr = float(f1_score(yt, pc, average="macro", zero_division=0))
    lo, hi = _bootstrap_delta(yt, pr, pc, seed=seed)
    return {"f1_raw": round(f1_raw, 4), "f1_corr": round(f1_corr, 4),
            "delta_f1": round(f1_corr - f1_raw, 4), "ci": (round(lo, 4), round(hi, 4)),
            "frames_flagged_pct": round(flagged, 1), "n_windows": int(valid.sum())}
