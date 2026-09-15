"""
src/skeleton/config.py
======================
Single source of truth for skeleton-correction parameters.

Every call site that produces *corrected* tracking for the downstream pipeline —
the feature pipeline (`feature_pipeline.extract_corrected`), the correction cache,
the event cache builder (`scripts/build_event_cache.py`), the downstream
re-measurement (`evaluation/downstream.score_downstream_f1`) and notebook 04 §2b —
builds its ``smooth_video`` kwargs from **configs/skeleton.yaml** via
:func:`skeleton_smooth_kwargs`. This closes the long-standing divergence where the
correction params were defined three incompatible ways (bare code defaults,
``default.yaml``'s ``skeleton:`` block, and ``skeleton.yaml``), so "corrected" now
means exactly one thing and the raw-vs-corrected comparison is well-posed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
SKELETON_CFG_PATH = REPO / "configs" / "skeleton.yaml"


def load_skeleton_cfg(path: Optional[Path] = None):
    """Load ``configs/skeleton.yaml`` (OmegaConf), the correction source of truth."""
    from omegaconf import OmegaConf
    return OmegaConf.load(str(Path(path) if path else SKELETON_CFG_PATH))


def skeleton_smooth_kwargs(cfg=None, path=None) -> dict:
    """Build ``smooth_video`` kwargs from a skeleton cfg.

    Maps the optimiser weights, solver settings, outlier-detection thresholds and
    per-keypoint adhesion floors onto the ``smooth_video`` signature. With no args
    it reads ``configs/skeleton.yaml`` (the canonical corrected definition); pass
    ``path=`` to build kwargs from an alternative config (e.g. ``skeleton_gentle.yaml``
    for the A3 aggressiveness sweep) or ``cfg=`` for an already-loaded OmegaConf.
    """
    if cfg is None:
        cfg = load_skeleton_cfg(path)
    w, s, o = cfg.weights, cfg.solver, cfg.outlier
    kp_floor = (dict(cfg.keypoints.adhesion_floor)
                if "keypoints" in cfg and "adhesion_floor" in cfg.keypoints else None)
    return dict(
        auto_edge_lambdas=True,
        lambda_l=float(w.L),
        lambda_t=float(w.get("T", w.get("S", 0.8))),
        lambda_acc=float(w.acc),
        lambda_angle=float(w.angle),
        lambda_jerk=float(w.jerk),
        lambda_angle_cont=float(w.get("angle_cont", 5.0)),
        outlier_adhesion=float(s.outlier_adhesion),
        n_passes=int(s.n_passes),
        confidence_sigma=float(s.get("confidence_sigma", 0.35)),
        speed_mad_k=float(o.get("speed_mad_k", 3.0)),
        max_displacement_factor=float(s.get("max_disp_factor", 0.0)),
        non_outlier_adhesion_scale=float(s.get("non_outlier_adhesion_scale", 1.0)),
        keypoint_adhesion_floor=kp_floor,
        length_threshold=float(o.get("length_threshold", 0.20)),
        detection_n_sigma=float(o.get("detection_n_sigma", 3.0)),
        adaptive_thresholds_k=float(s.get("adaptive_k", 2.5)),
        # A2: detector aggressiveness is now config-driven (calibrated set flags
        # ~48% vs the old ~78%). Default 3.0 preserves the pre-calibration value.
        speed_sigma=float(o.get("speed_sigma", 3.0)),
    )
