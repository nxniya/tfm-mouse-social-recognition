"""
src/skeleton/pose_lifting.py
==============================
Estimating the vertical component dz from the length constraints of the
skeletal segments, which lifts a 2D pose into an implicit 3D one.

The idea
--------
If two keypoints i and j are joined by a segment of reference length L_ij, and
the observed 2D distance between them is d_2D < L_ij, then the inferred depth
difference is:

    dz_ij = sqrt(max(0, L_ij^2 - d_2D^2))

In other words, how much relative depth would account for the segment appearing
shortened in the 2D projection. Note the sign is unrecoverable: this gives the
magnitude of the depth difference, never its direction.

The behaviours this is meant to capture:
  - ``rear``: the mouse rears up, so dz across nose-neck and neck-body_center
    both grow.
  - ``mount``: one mouse climbs onto the other, making the dz between their
    bodies detectable.
  - ``dominancemount``: like mount, with an additional lateral displacement.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.skeleton.mouse_skeleton import MouseSkeleton, _to_wide


# ---------------------------------------------------------------------------
# Estimating dz across a whole video
# ---------------------------------------------------------------------------

def estimate_delta_z(
    tracking_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    clip_to_positive: bool = True,
) -> pd.DataFrame:
    """Add dz columns to a long-format tracking DataFrame.

    For each skeleton edge (src, dst), computes dz_src_dst frame by frame and
    adds it as a column of a wide-format DataFrame.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Long-format tracking for a single video.
    skeleton : MouseSkeleton
        A skeleton with L_ref fitted and ``L_body_median_px`` estimated.
    kp_index : dict[str, int]
        Keypoint name to index. Unused here; kept so the signature matches the
        other modules.
    clip_to_positive : bool
        When True, negative dz values are clipped to 0. They arise when the
        observed 2D distance exceeds L_ref, which means a tracking error rather
        than a real pose.

    Returns
    -------
    pd.DataFrame
        Wide format, one row per (frame, mouse), with an extra
        ``dz_{src}_{dst}`` column per skeleton edge plus ``dz_sum``, the sum of
        all dz values, which serves as a global proxy for how upright the mouse
        is.
    """
    wide = _to_wide(tracking_df)
    l_body = skeleton.L_body_median_px

    dz_cols = {}

    for edge in skeleton.edges:
        sx, sy = f"{edge.src}_x", f"{edge.src}_y"
        dx, dy = f"{edge.dst}_x", f"{edge.dst}_y"

        if sx not in wide.columns or dx not in wide.columns:
            continue

        d2d = np.sqrt((wide[sx] - wide[dx]) ** 2 + (wide[sy] - wide[dy]) ** 2)
        L_ref_px = edge.L_ref * l_body

        if L_ref_px <= 0:
            continue

        dz2 = L_ref_px ** 2 - d2d ** 2
        if clip_to_positive:
            dz2 = dz2.clip(lower=0)

        col_name = f"dz_{edge.src}_{edge.dst}"
        dz_cols[col_name] = np.sqrt(dz2.abs()) * np.sign(dz2).clip(lower=0)

    for col, values in dz_cols.items():
        wide[col] = values

    dz_col_names = list(dz_cols.keys())
    if dz_col_names:
        wide["dz_sum"] = wide[dz_col_names].sum(axis=1)
    else:
        wide["dz_sum"] = 0.0

    return wide


# ---------------------------------------------------------------------------
# Verticality features, per behavioural segment
# ---------------------------------------------------------------------------

def dz_segment_features(
    dz_wide: pd.DataFrame,
    start_frame: int,
    stop_frame: int,
    agg_fns: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Extract dz statistics for one behavioural segment.

    Parameters
    ----------
    dz_wide : pd.DataFrame
        Output of ``estimate_delta_z()``.
    start_frame, stop_frame : int
        Frame range of the segment.
    agg_fns : list of str, optional
        Aggregation functions applied to each dz_* column. Defaults to
        ['mean', 'max', 'std'].

    Returns
    -------
    dict
        Maps each key to its statistic, for example
        ``{'dz_nose_neck_mean': 12.3, ...}``.
    """
    if agg_fns is None:
        agg_fns = ["mean", "max", "std"]

    seg = dz_wide[
        (dz_wide["video_frame"] >= start_frame) &
        (dz_wide["video_frame"] <= stop_frame)
    ]

    dz_cols = [c for c in dz_wide.columns if c.startswith("dz_")]
    features: Dict[str, float] = {}

    for col in dz_cols:
        values = seg[col].dropna()
        if len(values) == 0:
            for fn in agg_fns:
                features[f"{col}_{fn}"] = 0.0
            continue
        for fn in agg_fns:
            if fn == "mean":
                features[f"{col}_{fn}"] = float(values.mean())
            elif fn == "max":
                features[f"{col}_{fn}"] = float(values.max())
            elif fn == "std":
                features[f"{col}_{fn}"] = float(values.std(ddof=0))
            elif fn == "median":
                features[f"{col}_{fn}"] = float(values.median())

    return features


# ---------------------------------------------------------------------------
# Diagnostic: mean dz per behaviour, used to sanity-check the estimate
# ---------------------------------------------------------------------------

def dz_by_behavior(
    dz_wide: pd.DataFrame,
    annotations_df: pd.DataFrame,
    dz_key: str = "dz_sum",
) -> pd.DataFrame:
    """Mean dz per behaviour class.

    If the dz estimate is meaningful, the upright behaviours (rear, mount)
    should rank above the rest here.

    Parameters
    ----------
    dz_wide : pd.DataFrame
        Output of ``estimate_delta_z()``.
    annotations_df : pd.DataFrame
        Annotations, with ``action``, ``start_frame`` and ``stop_frame`` columns.
    dz_key : str
        Which dz column to use; defaults to ``dz_sum``.

    Returns
    -------
    pd.DataFrame
        Columns ``action``, ``dz_mean``, ``dz_max``, ``n_frames``, sorted by
        ``dz_mean`` descending.
    """
    rows = []
    for _, ann in annotations_df.iterrows():
        seg = dz_wide[
            (dz_wide["video_frame"] >= ann["start_frame"]) &
            (dz_wide["video_frame"] <= ann["stop_frame"])
        ]
        if seg.empty:
            continue
        vals = seg[dz_key].dropna()
        rows.append({
            "action": ann["action"],
            "dz_mean": float(vals.mean()),
            "dz_max": float(vals.max()),
            "n_frames": len(vals),
        })

    if not rows:
        return pd.DataFrame(columns=["action", "dz_mean", "dz_max", "n_frames"])

    return (
        pd.DataFrame(rows)
        .groupby("action")
        .agg(
            dz_mean=("dz_mean", "mean"),
            dz_max=("dz_max", "max"),
            n_frames=("n_frames", "sum"),
        )
        .reset_index()
        .sort_values("dz_mean", ascending=False)
    )
