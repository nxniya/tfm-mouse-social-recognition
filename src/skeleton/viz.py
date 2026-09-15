"""src/skeleton/viz.py — skeleton visualisation helpers.

Pure functions for rendering skeletons, finding the frames that violate a
constraint, and picking representative frames for the figures. Extracted from
notebooks/02_skeleton_correction.ipynb.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.skeleton.mouse_skeleton import _to_wide as to_wide, _segment_length as seg_len


# ---------------------------------------------------------------------------
# Skeleton rendering
# ---------------------------------------------------------------------------

def draw_skeleton(
    ax: plt.Axes,
    kp_xy: dict[str, tuple[float, float]],
    edges,
    title: str,
    color: str = "steelblue",
    violated_edges: Optional[set] = None,
) -> None:
    """Draw a skeleton on *ax* from a {keypoint: (x, y)} dict.

    Edges listed in *violated_edges* are highlighted in red.

    Args:
        ax: the matplotlib Axes to draw on.
        kp_xy: dict mapping a keypoint name to its (x, y).
        edges: list of ``SkeletonEdge``, using their .src and .dst attributes.
        title: subplot title.
        color: base colour for the edges and the valid points.
        violated_edges: set of (src, dst) pairs that violate their constraint.
    """
    if violated_edges is None:
        violated_edges = set()

    for e in edges:
        if e.src in kp_xy and e.dst in kp_xy:
            xs = [kp_xy[e.src][0], kp_xy[e.dst][0]]
            ys = [kp_xy[e.src][1], kp_xy[e.dst][1]]
            is_viol = (e.src, e.dst) in violated_edges or (e.dst, e.src) in violated_edges
            ax.plot(xs, ys, "-",
                    color="tomato" if is_viol else color,
                    lw=3 if is_viol else 2,
                    alpha=0.9)

    # Offset the labels when two keypoints sit closer than 8 px, or they overlap
    kp_list = list(kp_xy.items())
    offsets: dict[str, tuple[float, float]] = {kp: (0, 7) for kp, _ in kp_list}
    for i, (kp_i, (xi, yi)) in enumerate(kp_list):
        for j, (kp_j, (xj, yj)) in enumerate(kp_list):
            if i < j and math.hypot(xi - xj, yi - yj) < 8:
                offsets[kp_i] = (0, 14)
                offsets[kp_j] = (0, -14)

    for kp, (x, y) in kp_xy.items():
        ax.scatter(x, y, s=60, zorder=5, color=color)
        dx, dy = offsets[kp]
        ax.text(x + dx, y + dy, kp, fontsize=6, ha="center", va="bottom")

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)


# ---------------------------------------------------------------------------
# Extracting keypoints from a long DataFrame
# ---------------------------------------------------------------------------

def frame_kp(
    df_long: pd.DataFrame,
    frame: int,
    mouse_id: int = 0,
) -> dict[str, tuple[float, float]]:
    """Extract the {bodypart: (x, y)} dict for one frame and one mouse.

    Args:
        df_long: long DataFrame with columns [video_frame, mouse_id, bodypart, x, y].
        frame: frame index.
        mouse_id: mouse identifier.

    Returns:
        A dict mapping each keypoint name to its (x, y).
    """
    sub = df_long[(df_long["video_frame"] == frame) & (df_long["mouse_id"] == mouse_id)]
    return {row["bodypart"]: (row["x"], row["y"]) for _, row in sub.iterrows()}


# ---------------------------------------------------------------------------
# Detecting violations in individual frames
# ---------------------------------------------------------------------------

def get_violated_edges(
    wide: pd.DataFrame,
    frame: int,
    skeleton,
    threshold: float,
) -> set[tuple[str, str]]:
    """Return the set of edges whose length is violated in one frame.

    Args:
        wide: wide DataFrame with {kp}_x / {kp}_y columns.
        frame: index of the frame to inspect.
        skeleton: a fitted ``MouseSkeleton``.
        threshold: relative violation threshold, e.g. 0.25 for 25%.

    Returns:
        A set of (src, dst) tuples that violate their constraint.
    """
    fr_df = wide[wide["video_frame"] == frame]
    violated: set[tuple[str, str]] = set()
    for e in skeleton.edges:
        l_e = seg_len(fr_df, e.src, e.dst)
        if len(l_e) == 0:
            continue
        expected = e.L_ref * skeleton.L_body_median_px
        if expected > 0 and abs(l_e.iloc[0] - expected) / expected > threshold:
            violated.add((e.src, e.dst))
    return violated


def _hip_dist(wide: pd.DataFrame, frame: int) -> float:
    """Distance in pixels between hip_left and hip_right in the given frame."""
    rows = wide[wide["video_frame"] == frame]
    if len(rows) == 0:
        return 0.0
    r = rows.iloc[0]
    try:
        return math.hypot(
            r["hip_left_x"] - r["hip_right_x"],
            r["hip_left_y"] - r["hip_right_y"],
        )
    except KeyError:
        return float("inf")  # no hips in this lab, so the criterion cannot apply


def _frame_clean(
    wide: pd.DataFrame,
    frame: int,
    skeleton,
    threshold: float,
) -> bool:
    """True when the frame violates no length constraint at all."""
    fr_df = wide[wide["video_frame"] == frame]
    if len(fr_df) == 0:
        return False
    for e in skeleton.edges:
        l_e = seg_len(fr_df, e.src, e.dst)
        if len(l_e) == 0:
            continue
        expected = e.L_ref * skeleton.L_body_median_px
        if expected > 0 and abs(l_e.iloc[0] - expected) / expected > threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Automatic frame selection for the figures
# ---------------------------------------------------------------------------

def find_best_frames(
    wide_before: pd.DataFrame,
    wide_after: pd.DataFrame,
    skeleton,
    threshold: float,
    hip_min_dist: float = 30.0,
) -> tuple[Optional[int], Optional[int]]:
    """Pick the (f_viol, f_ok) pair for a before/after comparison figure.

    - ``f_viol``: a frame with a hip violation in BEFORE. Among those, the one
      whose AFTER has the largest hip_left to hip_right distance, since that is
      where the correction is most visible.
    - ``f_ok``: a frame clean in BOTH, with the hips well separated.

    Args:
        wide_before: wide DataFrame (``_to_wide`` format) BEFORE the correction.
        wide_after:  wide DataFrame AFTER the correction.
        skeleton:    a fitted ``MouseSkeleton``.
        threshold:   relative violation threshold.
        hip_min_dist: minimum hip separation in pixels for a frame to count as
            representative.

    Returns:
        The (f_viol, f_ok) integer pair, or (None, None) if no valid frame is
        found.
    """
    hip_edges = [e for e in skeleton.edges if "hip" in e.src or "hip" in e.dst]
    hip_viol_set: set[int] = set()
    for e in hip_edges:
        l_e = seg_len(wide_before, e.src, e.dst)
        expected = e.L_ref * skeleton.L_body_median_px
        if expected > 0:
            mask = (l_e - expected).abs() / expected > threshold
            hip_viol_set.update(
                wide_before.loc[mask[mask].index, "video_frame"].values.tolist()
            )

    common_frames = (
        set(wide_before["video_frame"].values) & set(wide_after["video_frame"].values)
    )
    hip_viol_frames = sorted(f for f in hip_viol_set if f in common_frames)

    if hip_viol_frames:
        def viol_score(f: int) -> tuple:
            d_after = _hip_dist(wide_after, f)
            after_ok = _frame_clean(wide_after, f, skeleton, threshold)
            return (-d_after, not after_ok)

        hip_viol_frames.sort(key=viol_score)
        f_viol = int(hip_viol_frames[0])

        d_bef = _hip_dist(wide_before, f_viol)
        d_aft = _hip_dist(wide_after, f_viol)
        clean_aft = _frame_clean(wide_after, f_viol, skeleton, threshold)
        print(f"Violation frame (hip): {f_viol}")
        print(
            f"  hips BEFORE: {d_bef:.1f} px  |  hips AFTER: {d_aft:.1f} px"
            f"  |  AFTER clean: {clean_aft}"
        )
    else:
        # Fallback: a body-length violation
        l_b = seg_len(wide_before, "nose", "tail_base")
        viol_mask = (l_b - skeleton.L_body_median_px).abs() / skeleton.L_body_median_px > threshold
        body_viol = wide_before.loc[viol_mask, "video_frame"].values
        if len(body_viol) > 0:
            body_viol_common = sorted(f for f in body_viol if f in common_frames)
            f_viol = int(max(body_viol_common, key=lambda f: _hip_dist(wide_after, f)))
            print(f"Violation frame (body length): {f_viol}")
        else:
            print("No frames with a violation were found.")
            return None, None

    # Look for a frame clean in BOTH, with the hips well separated
    frames_sorted = sorted(common_frames, key=lambda f: abs(int(f) - f_viol))

    for candidate_frames, label in [
        # Level 1: clean, and hips >= hip_min_dist in both
        (
            [
                f for f in frames_sorted
                if f != f_viol
                and _frame_clean(wide_before, f, skeleton, threshold)
                and _frame_clean(wide_after, f, skeleton, threshold)
                and _hip_dist(wide_before, f) >= hip_min_dist
                and _hip_dist(wide_after, f) >= hip_min_dist
            ],
            "clean in BEFORE and AFTER",
        ),
        # Level 2: hips >= hip_min_dist in both, cleanliness not required
        (
            [
                f for f in frames_sorted
                if f != f_viol
                and _hip_dist(wide_before, f) >= hip_min_dist
                and _hip_dist(wide_after, f) >= hip_min_dist
            ],
            "hips ok in BEFORE and AFTER",
        ),
        # Level 3: hips ok in BEFORE at least
        (
            [
                f for f in frames_sorted
                if f != f_viol and _hip_dist(wide_before, f) >= hip_min_dist
            ],
            "hips ok in BEFORE",
        ),
    ]:
        if candidate_frames:
            f_ok = int(candidate_frames[0])
            print(f"Clean frame: {f_ok}  [{label}]")
            break
    else:
        f_ok = int(next(f for f in frames_sorted if f != f_viol))
        print(f"Clean frame (fallback): {f_ok}")

    return f_viol, f_ok
