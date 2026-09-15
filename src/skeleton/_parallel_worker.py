"""
Worker function for parallel multi-video skeleton correction.

Kept in a standalone module so it is importable by spawned subprocesses on Windows
(ProcessPoolExecutor uses 'spawn' by default, which requires pickleable callables
defined in importable modules — not in notebook cells).
"""
from __future__ import annotations


def process_one_video(args: tuple) -> dict:
    """Load, fit skeleton, and correct a single video. Returns per-video metrics.

    Args:
        args: tuple of
            (video_id, lab_id, frame_limit, length_threshold,
             lambda_l, lambda_t, speed_mad_k, repo_root_str)

    Returns:
        dict with keys: video_id, n_frames, rate_before, rate_after,
        reduction_%, elapsed_s, ms_per_frame  — or {video_id, error} on failure.
    """
    vid_id, lab, frame_lim, length_thr, lambda_l, lambda_t, speed_mad_k, repo_root = args
    try:
        import sys
        import time

        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        import numpy as np
        from src.data.loader import load_tracking
        from src.skeleton import (
            build_skeleton,
            fit_skeleton,
            smooth_video,
            compute_constraint_violations,
        )
        from src.skeleton.mouse_skeleton import _to_wide, _segment_length

        df = load_tracking(vid_id, lab)
        if df is None or df.empty:
            return {"video_id": vid_id, "error": "empty tracking"}

        df_sub = df[df["video_frame"] < frame_lim].copy()
        skl = build_skeleton(lab)
        fit_skeleton(skl, df_sub)

        mouse_id = int(df_sub["mouse_id"].iloc[0])
        wide_b = _to_wide(df_sub[df_sub["mouse_id"] == mouse_id])
        l_b = _segment_length(wide_b, "nose", "tail_base")
        viol_b = compute_constraint_violations(wide_b, skl, {}, l_b, threshold=0.25)  # biological quality threshold
        rate_b = float(viol_b["violation_rate"].mean())

        t0 = time.perf_counter()
        df_corr, _ = smooth_video(
            df_sub,
            lab_id=lab,
            skeleton=skl,
            speed_sigma=speed_mad_k,
            length_threshold=length_thr,
            lambda_l=lambda_l,
            lambda_t=lambda_t,
            lambda_acc=2.0,          # acceleration regularization (suppresses jitter)
            lambda_angle=5.0,        # body-axis collinearity constraint
            lambda_jerk=0.5,         # jerk suppression (3rd-order temporal penalty)
            lambda_angle_cont=2.0,   # orientation continuity (swap detection)
            auto_edge_lambdas=True,
            pre_smooth=True,
            outlier_adhesion=0.6,    # prioritise geometry over corrupted observations
            n_passes=2,              # multi-pass: warm-start refinement
        )
        elapsed_s = time.perf_counter() - t0

        wide_a = _to_wide(df_corr[df_corr["mouse_id"] == mouse_id])
        l_a = _segment_length(wide_a, "nose", "tail_base")
        viol_a = compute_constraint_violations(wide_a, skl, {}, l_a, threshold=0.25)  # biological quality threshold
        rate_a = float(viol_a["violation_rate"].mean())

        n_frames = df_sub["video_frame"].nunique()
        return {
            "video_id": vid_id,
            "n_frames": n_frames,
            "rate_before": round(rate_b, 4),
            "rate_after": round(rate_a, 4),
            "reduction_%": round((1 - rate_a / max(rate_b, 1e-9)) * 100, 2),
            "elapsed_s": round(elapsed_s, 3),
            "ms_per_frame": round(elapsed_s / max(n_frames, 1) * 1000, 3),
        }
    except Exception as exc:
        return {"video_id": vid_id, "error": str(exc)}


def process_one_video_population(
    video_id: str,
    lab_id: str,
    df_tracking,
    *,
    frame_limit: int = 100,
    skeleton=None,
    lambda_l: float = 12.0,
    lambda_t: float = 0.8,
    lambda_acc: float = 0.2,
    lambda_angle: float = 10.0,
    lambda_jerk: float = 0.05,
    lambda_angle_cont: float = 5.0,
    outlier_adhesion: float = 0.6,
    n_passes: int = 3,
    length_threshold: float = 1.84,
    violation_report_thr: float = 0.25,
) -> dict:
    """Run the correction pipeline over one video and return population metrics.

    Unlike :func:`process_one_video`, which is built for ``ProcessPoolExecutor``,
    this works on an already-loaded DataFrame and accepts a pre-fitted
    ``skeleton``. That makes it the right choice for sequential loops in a
    notebook, where the kernel keeps state between videos.

    Args:
        video_id:   Video identifier.
        lab_id:     Lab identifier.
        df_tracking: An already-loaded long-format tracking DataFrame.
        frame_limit: Maximum number of frames to process.
        skeleton:   A pre-fitted ``MouseSkeleton``; built and fitted if None.
        lambda_l through lambda_angle_cont: optimiser weights.
        outlier_adhesion: the optimiser's adhesion for outlier keypoints.
        n_passes:   Number of optimiser passes.
        length_threshold: outlier-detection threshold for the solver. It is well
            above 1, so only catastrophic errors trigger it. Not used for the
            violation report.
        violation_report_thr: biological-quality threshold for the violation
            report, as a fraction of relative deviation. Defaults to 0.25, 25%.
            Keeping this separate from length_threshold matters: reporting
            violations at the solver's own permissive threshold would make the
            correction look far better than it is.

    Returns:
        A dict with keys: video_id, n_frames, viol_rate_before, viol_rate_after,
        viol_reduction_pct, jerk_nose, ms_per_frame, error (None on success).
    """
    import time
    import numpy as np
    import pandas as pd

    try:
        from src.skeleton.keypoint_smoother import smooth_video
        from src.skeleton.mouse_skeleton import _to_wide, _segment_length
        from src.skeleton.kinematic_constraints import compute_constraint_violations
        from src.skeleton.metrics import jerk_energy
        from src.skeleton import build_skeleton, fit_skeleton

        frames = sorted(df_tracking["video_frame"].unique())[:frame_limit]
        df = df_tracking[df_tracking["video_frame"].isin(frames)].copy()
        mouse = int(df["mouse_id"].iloc[0])

        # Build/fit skeleton if not provided
        if skeleton is None:
            _kp_hint = df["bodypart"].unique().tolist()
            skeleton = build_skeleton(lab_id, keypoints_hint=_kp_hint)
            fit_skeleton(skeleton, df)

        t0 = time.perf_counter()
        df_corr, _ = smooth_video(
            df,
            lab_id=lab_id,
            skeleton=skeleton,
            lambda_l=lambda_l,
            lambda_t=lambda_t,
            lambda_acc=lambda_acc,
            lambda_angle=lambda_angle,
            lambda_jerk=lambda_jerk,
            lambda_angle_cont=lambda_angle_cont,
            auto_edge_lambdas=True,
            outlier_adhesion=outlier_adhesion,
            n_passes=n_passes,
            length_threshold=length_threshold,  # pass per-lab threshold to detector
            pre_smooth_window=9,
        )
        elapsed = time.perf_counter() - t0

        wide_b = _to_wide(df[df["mouse_id"] == mouse])
        wide_a = _to_wide(df_corr[df_corr["mouse_id"] == mouse])

        # Per-video L_body for normalisation
        lb = _segment_length(wide_b, "nose", "tail_base")
        l_body_vid = float(lb[lb > 10].median()) if len(lb[lb > 10]) > 5 else 1.0

        def _vr(wide) -> float:
            lb = _segment_length(wide, "nose", "tail_base")
            # Detect per-keypoint dropout frames: keypoint at exactly (0, 0).
            # Partial dropouts (only nose=0,0 while tail_base is valid) inflate
            # L_body → deflate apparent violations → misleading before/after metric.
            _drop = pd.Series(False, index=wide.index)
            for _kp in ("nose", "tail_base"):
                _xc, _yc = f"{_kp}_x", f"{_kp}_y"
                if _xc in wide.columns and _yc in wide.columns:
                    _drop |= (wide[_xc] == 0) & (wide[_yc] == 0)
            # Robust per-video L_body from dropout-free frames
            _lb_clean = lb[~_drop & (lb > 10)]
            _l_body_ref = float(_lb_clean.median()) if len(_lb_clean) > 5 \
                else float(lb[lb > 10].median() if (lb > 10).any() else lb.median())
            if not np.isfinite(_l_body_ref) or _l_body_ref <= 0:
                _l_body_ref = 1.0
            # Replace dropout-frame L_body with per-video reference
            lb_final = lb.where(~_drop, _l_body_ref)
            v = compute_constraint_violations(
                wide, skeleton, {},
                lb_final,
                threshold=violation_report_thr,
            )
            return float(v["violation_rate"].mean())

        vr_b = _vr(wide_b)
        vr_a = _vr(wide_a)
        # Avoid astronomical percentages when vr_b ≈ 0
        if vr_b < 1e-6:
            reduction = 0.0 if vr_a < 1e-6 else float("nan")
        else:
            reduction = (vr_b - vr_a) / vr_b * 100

        jerk_df = jerk_energy(df_corr, mouse, l_body_vid, keypoints=["nose"])
        jerk_nose = float(jerk_df["jerk_energy_norm"].iloc[0]) if len(jerk_df) > 0 else np.nan

        return {
            "video_id": video_id,
            "n_frames": len(frames),
            "viol_rate_before": round(vr_b, 5),
            "viol_rate_after": round(vr_a, 5),
            "viol_reduction_pct": round(reduction, 1),
            "jerk_nose": round(jerk_nose, 4),
            "ms_per_frame": round(elapsed / max(len(frames), 1) * 1000, 2),
            "error": None,
        }

    except Exception as exc:
        return {"video_id": video_id, "error": str(exc)}
