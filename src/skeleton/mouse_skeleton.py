"""
src/skeleton/mouse_skeleton.py
===============================
The articulated mouse graph, and the statistical estimation of each segment's
reference length from the MABe dataset.

The skeleton is an undirected graph G = (V, E), where:
  - V = the keypoints available in the lab being worked on
  - E = the pairs of keypoints joined by a bone segment

Reference lengths are in pixels normalised by the median nose-to-tail_base
length of the video. Without that normalisation, lengths fitted in one lab are
meaningless in another, because the labs film at different camera resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures: the graph's nodes and edges
# ---------------------------------------------------------------------------

@dataclass
class SkeletonEdge:
    """An edge of the skeletal graph, with its biomechanical parameters.

    Attributes
    ----------
    src, dst : str
        Names of the two endpoint keypoints.
    L_ref : float
        Normalised reference length, that is, the median divided by L_body.
        Left at 0 until it has been estimated.
    L_std : float
        Standard deviation of the normalised length.
    angle_range : tuple[float, float]
        Valid joint-angle range in degrees, as (min, max).
        (-180, 180) means the angle is unconstrained.
    """
    src: str
    dst: str
    L_ref: float = 0.0
    L_std: float = 0.05
    angle_range: Tuple[float, float] = (-180.0, 180.0)


@dataclass
class MouseSkeleton:
    """The articulated mouse graph.

    Two keypoint sets are supported:
    - **CalMS21** (7 kp): nose, neck, ear_left, ear_right,
      hip_left, hip_right, tail_base
    - **MABe22_keypoints** (12 kp): the above plus body_center,
      forepaw_left, forepaw_right, hindpaw_left, hindpaw_right,
      tail_midpoint, tail_tip

    The length parameters are either estimated by calling ``fit()`` on a
    tracking DataFrame, or loaded from the EDA metadata CSV.
    """

    lab_id: str
    keypoints: List[str] = field(default_factory=list)
    edges: List[SkeletonEdge] = field(default_factory=list)
    # Length of the main body segment, in pixels, used as the scale
    L_body_median_px: float = 1.0

    # ── Predefined graphs, one per lab type ─────────────────────────────────

    # The minimum keypoint set shared by every lab that has nose and tail_base
    _CORE_KEYPOINTS: List[str] = field(default_factory=lambda: [
        "nose", "neck", "ear_left", "ear_right", "tail_base",
    ], init=False, repr=False)

    _CALMS21_KEYPOINTS: List[str] = field(default_factory=lambda: [
        "nose", "neck", "ear_left", "ear_right",
        "hip_left", "hip_right", "tail_base",
    ], init=False, repr=False)

    _MABE22_KEYPOINTS: List[str] = field(default_factory=lambda: [
        "nose", "neck", "ear_left", "ear_right", "body_center",
        "hip_left", "hip_right",
        "forepaw_left", "forepaw_right",
        "hindpaw_left", "hindpaw_right",
        "tail_base", "tail_midpoint", "tail_tip",
    ], init=False, repr=False)

    # ── Canonical edges, before L_ref has been estimated ────────────────────
    _CALMS21_EDGES: List[Tuple] = field(default_factory=lambda: [
        # (src, dst, angle_range_deg)
        ("nose",      "neck",       (-90,  90)),
        ("neck",      "ear_left",   (-90,  90)),
        ("neck",      "ear_right",  (-90,  90)),
        ("neck",      "hip_left",   (-45,  45)),
        ("neck",      "hip_right",  (-45,  45)),
        ("hip_left",  "tail_base",  (-45,  45)),
        ("hip_right", "tail_base",  (-45,  45)),
    ], init=False, repr=False)

    _MABE22_EDGES: List[Tuple] = field(default_factory=lambda: [
        ("nose",         "neck",          (-90, 90)),
        ("neck",         "ear_left",      (-90, 90)),
        ("neck",         "ear_right",     (-90, 90)),
        ("neck",         "body_center",   (-30, 30)),
        ("body_center",  "hip_left",      (-45, 45)),
        ("body_center",  "hip_right",     (-45, 45)),
        ("body_center",  "forepaw_left",  (-90, 90)),
        ("body_center",  "forepaw_right", (-90, 90)),
        ("hip_left",     "hindpaw_left",  (-60, 60)),
        ("hip_right",    "hindpaw_right", (-60, 60)),
        ("body_center",  "tail_base",     (-15, 15)),
        ("tail_base",    "tail_midpoint", (-45, 45)),
        ("tail_midpoint","tail_tip",      (-45, 45)),
    ], init=False, repr=False)

    # ── Default L_ref values, taken from the EDA ────────────────────────────
    # Normalised as L_ref / L_body_median_px, where L_body = nose to tail_base
    # Source: the exploratory analysis in notebook 01, over the CalMS21 videos.
    # They are only a starting point; fit_skeleton overwrites them per video.
    _DEFAULT_L_REF: Dict[str, float] = field(default_factory=lambda: {
        "nose-neck":           0.22,
        "neck-ear_left":       0.14,
        "neck-ear_right":      0.14,
        "neck-hip_left":       0.55,
        "neck-hip_right":      0.55,
        "hip_left-tail_base":  0.30,
        "hip_right-tail_base": 0.30,
        # MABe22_keypoints extras
        "neck-body_center":        0.40,
        "body_center-hip_left":    0.18,
        "body_center-hip_right":   0.18,
        "body_center-forepaw_left":  0.25,
        "body_center-forepaw_right": 0.25,
        "hip_left-hindpaw_left":     0.22,
        "hip_right-hindpaw_right":   0.22,
        "body_center-tail_base":     0.28,
        "tail_base-tail_midpoint":   0.32,
        "tail_midpoint-tail_tip":    0.30,
    }, init=False, repr=False)


# ---------------------------------------------------------------------------
# Factory constructor
# ---------------------------------------------------------------------------

def build_skeleton(
    lab_id: str,
    keypoints_hint: Optional[List[str]] = None,
) -> MouseSkeleton:
    """Build the predefined skeleton for a given lab.

    Parameters
    ----------
    lab_id : str
        Lab identifier, for example "CalMS21_task1" or "MABe22_keypoints".
    keypoints_hint : list[str], optional
        The keypoints actually present in the data. When supplied, they are used
        to recognise a 7-keypoint CalMS21 schema even if ``lab_id`` is not one of
        the names listed explicitly below.

    Returns
    -------
    MouseSkeleton, with its edges initialised to the default L_ref values.
    Call ``fit()`` on real data to replace them with fitted parameters.
    """
    sk = MouseSkeleton(lab_id=lab_id)
    # pylint: disable=protected-access

    _calms21_set = set(sk._CALMS21_KEYPOINTS)
    _hint_set = set(keypoints_hint) if keypoints_hint else set()

    if lab_id in ("CalMS21_task1", "CalMS21_task2", "CalMS21_supplemental"):
        keypoints = sk._CALMS21_KEYPOINTS
        edge_defs = sk._CALMS21_EDGES
    elif _hint_set and _hint_set.issubset(_calms21_set):
        # Unknown lab, but its 7-keypoint schema is CalMS21-compatible
        keypoints = sk._CALMS21_KEYPOINTS
        edge_defs = sk._CALMS21_EDGES
    else:
        # MABe22_keypoints, and any lab with the extended keypoint set
        keypoints = sk._MABE22_KEYPOINTS
        edge_defs = sk._MABE22_EDGES

    sk.keypoints = keypoints
    sk.edges = [
        SkeletonEdge(
            src=src,
            dst=dst,
            L_ref=sk._DEFAULT_L_REF.get(f"{src}-{dst}",
                  sk._DEFAULT_L_REF.get(f"{dst}-{src}", 0.0)),
            L_std=0.05,
            angle_range=angle_range,
        )
        for src, dst, angle_range in edge_defs
    ]
    return sk


# ---------------------------------------------------------------------------
# Statistical estimation of L_ref from real data (fit)
# ---------------------------------------------------------------------------

def fit_skeleton(
    skeleton: MouseSkeleton,
    tracking_df: pd.DataFrame,
    l_body_col: Optional[str] = None,
    percentile: float = 50.0,
) -> MouseSkeleton:
    """Estimate the skeleton's biomechanical parameters from tracking data.

    For each edge (src, dst), builds the distribution of that segment's length
    across every frame and every mouse in the DataFrame, normalised by the body
    length (nose to tail_base).

    Parameters
    ----------
    skeleton : MouseSkeleton
        An instance created by ``build_skeleton()``.
    tracking_df : pd.DataFrame
        Long-format DataFrame with columns ``video_frame``, ``mouse_id``,
        ``bodypart``, ``x``, ``y``. May span several videos.
    l_body_col : str, optional
        Name of a precomputed per-frame body-length column. When None, the body
        length is computed as the nose-to-tail_base distance.
    percentile : float
        Percentile taken as L_ref; the default, 50, is the median.

    Returns
    -------
    The same MouseSkeleton, updated in place and returned.
    """
    wide = _to_wide(tracking_df)

    # Body length, per frame and per mouse
    if l_body_col and l_body_col in wide.columns:
        l_body = wide[l_body_col]
    else:
        l_body = _segment_length(wide, "nose", "tail_base")

    # Store the global median body length, in pixels
    valid_l_body = l_body[(l_body > 0) & l_body.notna()]
    if len(valid_l_body) == 0:
        return skeleton
    skeleton.L_body_median_px = float(np.percentile(valid_l_body, percentile))

    # Normalise, then compute L_ref per edge
    # NOTE: divide by unfiltered l_body, which may contain zeros (dropout frames
    # where nose/tail_base are at (0,0)).  x/0 → inf, and inf > 0 is True, so
    # inf values would survive the old filter and corrupt std() → NaN.
    # Fix: use np.isfinite() to exclude both NaN and inf before computing stats.
    for edge in skeleton.edges:
        raw_lengths = _segment_length(wide, edge.src, edge.dst)
        norm_lengths = raw_lengths / l_body
        valid = norm_lengths[np.isfinite(norm_lengths) & (norm_lengths > 0)]
        if len(valid) < 10:
            continue
        edge.L_ref = float(np.percentile(valid, percentile))
        edge.L_std = float(valid.std())

    return skeleton


def fit_skeleton_per_mouse(
    lab_id: str,
    tracking_df: pd.DataFrame,
    keypoints_hint: Optional[List[str]] = None,
    percentile: float = 50.0,
) -> Dict[int, MouseSkeleton]:
    """Fit one independent skeleton per ``mouse_id`` in the video.

    Where ``fit_skeleton`` pools every mouse's lengths into a single shared
    ``L_ref``, this estimates ``L_ref`` and ``L_std`` per edge from **only** that
    mouse's rows, and returns a ``{mouse_id: MouseSkeleton}`` dict. The graph
    topology and the angle ranges are identical across mice, since they are the
    same species; only the reference bone lengths and their spread differ.

    Its purpose is to isolate the pooled-versus-per-mouse reference as an
    experimental variable in the generalisation study; see the correction
    ablation under LOVO, which found that a per-mouse reference does not rescue
    the correction.

    Parameters
    ----------
    lab_id : str
        Lab identifier, which determines the keypoint schema.
    tracking_df : pd.DataFrame
        Long-format tracking for a single video, with a ``mouse_id`` column.
    keypoints_hint : list[str], optional
        Keypoints present in the data, used to auto-detect the schema.
    percentile : float
        Percentile taken as ``L_ref``; the default, 50, is the median.

    Returns
    -------
    dict[int, MouseSkeleton]
        One fitted skeleton per mouse.
    """
    out: Dict[int, MouseSkeleton] = {}
    for mid, mdf in tracking_df.groupby("mouse_id"):
        sk = build_skeleton(lab_id, keypoints_hint=keypoints_hint)
        fit_skeleton(sk, mdf, percentile=percentile)
        out[int(mid)] = sk
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long format to wide: one {bodypart}_x, {bodypart}_y pair per column."""
    required = {"video_frame", "mouse_id", "bodypart", "x", "y"}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"Missing columns in tracking_df: {missing}")

    wide = df_long.pivot_table(
        index=["video_frame", "mouse_id"],
        columns="bodypart",
        values=["x", "y"],
        aggfunc="first",
    )
    wide.columns = [f"{bp}_{coord}" for coord, bp in wide.columns]
    return wide.reset_index()


def _segment_length(wide: pd.DataFrame, src: str, dst: str) -> pd.Series:
    """Euclidean length of one segment, for every row."""
    sx, sy = f"{src}_x", f"{src}_y"
    dx, dy = f"{dst}_x", f"{dst}_y"

    if sx not in wide.columns or dx not in wide.columns:
        return pd.Series(np.nan, index=wide.index)

    return np.sqrt(
        (wide[sx] - wide[dx]) ** 2 + (wide[sy] - wide[dy]) ** 2
    )


def get_edge_key(src: str, dst: str) -> str:
    """Canonical key for an edge, ordered lexicographically."""
    return f"{min(src, dst)}-{max(src, dst)}"


def skeleton_to_adjacency(skeleton: MouseSkeleton) -> Dict[str, List[str]]:
    """Return the graph as an adjacency dict."""
    adj: Dict[str, List[str]] = {kp: [] for kp in skeleton.keypoints}
    for edge in skeleton.edges:
        if edge.src in adj:
            adj[edge.src].append(edge.dst)
        if edge.dst in adj:
            adj[edge.dst].append(edge.src)
    return adj
