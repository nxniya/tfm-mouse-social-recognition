"""
src/data/features.py
=====================
2D and implicit-3D feature engineering for mouse behaviour classification.

Extraction pipeline:
  1. Normalisation: group centroid plus a per-video reference body length
  2. Per-animal kinematics: speed, acceleration, heading, joint angles
  3. Per-segment Δz: implicit depth from pose lifting
  4. Relational features: inter-mouse distances and angles
  5. Sliding windows: 64 frames, per-window statistics

Window stride is set in ``configs/default.yaml``. At stride 16 consecutive windows
overlap by 75%, which is convenient for data volume and is the leakage source that
any random train/validation split over windows will pick up.
Implicaciones Skeleton: Skeleton_implicaciones_fases_siguientes.md §2
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as _savgol
from scipy.stats import skew as _sp_skew, kurtosis as _sp_kurt

from src.skeleton.mouse_skeleton import MouseSkeleton, _to_wide, _segment_length
from src.skeleton.pose_lifting import estimate_delta_z


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

WINDOW_SIZE: int = 64       # frames per window, about 2 s at 25-30 fps
STRIDE: int = 32            # 50% overlap, raised from 16 to reduce leakage
MIN_WINDOW_FILL: float = 0.75  # minimum fraction of valid frames in a window


# ---------------------------------------------------------------------------
# Step 1: normalisation and cleaning
# ---------------------------------------------------------------------------

def _filter_zero_frames(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where every keypoint sits at (0, 0).

    CalMS21 uses (0, 0) to mark absent tracking rather than a missing value, so
    these rows would otherwise be read as a mouse pinned to the corner of the arena.
    """
    coord_cols_x = [c for c in wide_df.columns if c.endswith("_x")]
    coord_cols_y = [c for c in wide_df.columns if c.endswith("_y")]
    if not coord_cols_x:
        return wide_df
    zero_x = (wide_df[coord_cols_x] == 0).all(axis=1)
    zero_y = (wide_df[coord_cols_y] == 0).all(axis=1)
    return wide_df[~(zero_x & zero_y)].copy()


def normalize_coordinates(
    wide_df: pd.DataFrame,
    l_body_px: float,
    mice_ids: Optional[List[int]] = None,
    center: str = "centroid",
) -> pd.DataFrame:
    """Normalise coordinates by the group centroid and the body length.

    .. warning::
       ``center="centroid"`` subtracts the centroid of *the rows it is given*. Called
       per mouse, as ``extract_features`` does, every mouse ends up centred on
       itself and **all inter-mouse geometry collapses**: ``dist_centroid``,
       ``approach_rate`` and ``speed_relative`` fall to about 1e-7 and the centroid
       speeds stop measuring locomotion at all. Use ``center="none"`` to keep
       absolute positions in body lengths, which is what the relational and
       displacement features need.

    Transformation:
        coords_norm = (coords - group_centroid_per_frame) / l_body_px

    The group centroid is the mean of every keypoint of every mouse in the frame.
    Subtracting it removes global arena translation without needing to know the
    arena dimensions, which vary by lab and are not always recorded.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide-format tracking with ``{kp}_x``, ``{kp}_y``, ``video_frame`` and
        ``mouse_id`` columns.
    l_body_px : float
        Reference body length for the video, in pixels. Use
        ``skeleton.L_body_median_px`` from the fitted skeleton.
    mice_ids : list[int], optional
        Mice to include in the centroid. Defaults to all of them.

    Returns
    -------
    pd.DataFrame
        The same schema, with the coordinates normalised in place.
    """
    df = wide_df.copy()
    coord_cols_x = [c for c in df.columns if c.endswith("_x")]
    coord_cols_y = [c for c in df.columns if c.endswith("_y")]

    if center == "none":
        # Scale only: keeps the translation, and with it the inter-mouse geometry.
        df[coord_cols_x] = df[coord_cols_x].divide(l_body_px)
        df[coord_cols_y] = df[coord_cols_y].divide(l_body_px)
        return df

    if mice_ids is not None:
        mask = df["mouse_id"].isin(mice_ids)
        cx = df.loc[mask, coord_cols_x].replace(0, np.nan).mean(axis=1)
        cy = df.loc[mask, coord_cols_y].replace(0, np.nan).mean(axis=1)
    else:
        cx = df[coord_cols_x].replace(0, np.nan).mean(axis=1)
        cy = df[coord_cols_y].replace(0, np.nan).mean(axis=1)

    # Align the centroid index with the full dataframe (it may be a
    # sub-conjunto si mice_ids fue especificado)
    if not cx.index.equals(df.index):
        cx = cx.reindex(df.index)
        cy = cy.reindex(df.index)

    df[coord_cols_x] = df[coord_cols_x].subtract(cx, axis=0).divide(l_body_px)
    df[coord_cols_y] = df[coord_cols_y].subtract(cy, axis=0).divide(l_body_px)
    return df


# ---------------------------------------------------------------------------
# Step 2: per-animal kinematics
# ---------------------------------------------------------------------------

def _centroid_xy(wide_row: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame centroid: the mean of every keypoint of every mouse."""
    xcols = [c for c in wide_row.columns if c.endswith("_x")]
    ycols = [c for c in wide_row.columns if c.endswith("_y")]
    cx = wide_row[xcols].replace(0, np.nan).mean(axis=1).values
    cy = wide_row[ycols].replace(0, np.nan).mean(axis=1).values
    return cx, cy


def _finite_diff(series: np.ndarray, order: int = 1) -> np.ndarray:
    """Centred finite difference, NaN at the edges."""
    out = np.full_like(series, np.nan)
    if order == 1:
        out[1:-1] = (series[2:] - series[:-2]) / 2.0
    elif order == 2:
        out[1:-1] = series[2:] - 2 * series[1:-1] + series[:-2]
    return out


def _smooth_1d(arr: np.ndarray, window_length: int = 9, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay smoothing of a 1D signal that may contain NaNs.

    NaNs are linearly interpolated before filtering and restored afterwards, so
    the filter never spreads a gap across its window. Signals shorter than the
    window are returned unchanged.
    """
    n = len(arr)
    if n < window_length:
        return arr
    finite_mask = np.isfinite(arr)
    if finite_mask.sum() < window_length:
        return arr
    x_all = np.arange(n)
    x_valid = x_all[finite_mask]
    y_valid = arr[finite_mask]
    interp = np.interp(x_all, x_valid, y_valid)
    wl = window_length if window_length % 2 == 1 else window_length + 1
    smoothed = _savgol(interp, window_length=wl, polyorder=polyorder, mode="nearest")
    smoothed[~finite_mask] = np.nan
    return smoothed


def compute_kinematics(
    wide_df: pd.DataFrame,
    keypoints: List[str],
    smooth: bool = True,
    smooth_window: int = 9,
) -> pd.DataFrame:
    """Per-frame kinematic features for one mouse.

    With ``smooth=True``, positions are Savitzky-Golay smoothed before
    differentiating. Differentiating raw tracking amplifies its noise, and the
    second and third derivatives are unusable without this.

    Features per frame:
      - centroid_x, centroid_y
      - centroid_vx, centroid_vy   centroid velocity
      - centroid_speed             its magnitude
      - centroid_ax, centroid_ay   acceleration
      - centroid_accel             its magnitude
      - centroid_jerk              jerk magnitude, the third derivative
      - {kp}_vx, {kp}_vy, {kp}_speed, {kp}_accel, per keypoint
      - body_angle                 angle of the nose-to-tail_base vector, radians
      - body_angle_vel             angular velocity
      - body_length                normalised body length

    Parameters
    ----------
    wide_df : pd.DataFrame
        One row per frame, with ``{kp}_x`` and ``{kp}_y`` columns.
        Already filtered of zero frames and normalised.
    keypoints : list[str]
        Skeleton keypoints, for example ``skeleton.keypoints``.
    smooth : bool
        If True, smooth positions before differentiating.
    smooth_window : int
        Savitzky-Golay window length. Must be odd; an even value is incremented.

    Returns
    -------
    pd.DataFrame
        Indexed by ``video_frame``, one column per feature.
    """
    df = wide_df.set_index("video_frame").sort_index() if "video_frame" in wide_df.columns else wide_df.copy()

    feats: Dict[str, np.ndarray] = {}

    # Smooth positions with Savitzky-Golay before differentiating
    if smooth:
        df = df.copy()
        pos_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
        for col in pos_cols:
            df[col] = _smooth_1d(df[col].values.astype(float), window_length=smooth_window)

    # Centroide
    xcols = [f"{kp}_x" for kp in keypoints if f"{kp}_x" in df.columns]
    ycols = [f"{kp}_y" for kp in keypoints if f"{kp}_y" in df.columns]
    cx = df[xcols].mean(axis=1).values
    cy = df[ycols].mean(axis=1).values
    feats["centroid_x"] = cx
    feats["centroid_y"] = cy

    cvx = _finite_diff(cx)
    cvy = _finite_diff(cy)
    feats["centroid_vx"] = cvx
    feats["centroid_vy"] = cvy
    feats["centroid_speed"] = np.sqrt(cvx ** 2 + cvy ** 2)
    feats["centroid_ax"] = _finite_diff(cx, order=2)
    feats["centroid_ay"] = _finite_diff(cy, order=2)
    feats["centroid_accel"] = np.sqrt(feats["centroid_ax"] ** 2 + feats["centroid_ay"] ** 2)

    # Centroid jerk, the third derivative of position
    feats["centroid_jerk"] = np.sqrt(
        _finite_diff(feats["centroid_ax"]) ** 2 +
        _finite_diff(feats["centroid_ay"]) ** 2
    )

    # Per-keypoint kinematics: velocity and acceleration
    for kp in keypoints:
        xc, yc = f"{kp}_x", f"{kp}_y"
        if xc not in df.columns:
            continue
        xs = df[xc].values.astype(float)
        ys = df[yc].values.astype(float)
        vx = _finite_diff(xs)
        vy = _finite_diff(ys)
        ax = _finite_diff(xs, order=2)
        ay = _finite_diff(ys, order=2)
        feats[f"{kp}_vx"] = vx
        feats[f"{kp}_vy"] = vy
        feats[f"{kp}_speed"] = np.sqrt(vx ** 2 + vy ** 2)
        feats[f"{kp}_accel"] = np.sqrt(ax ** 2 + ay ** 2)

    # Body orientation, from nose to tail_base
    if "nose_x" in df.columns and "tail_base_x" in df.columns:
        dx = df["tail_base_x"].values - df["nose_x"].values
        dy = df["tail_base_y"].values - df["nose_y"].values
        angle = np.arctan2(dy, dx)
        feats["body_angle"] = angle
        # Angular velocity: angle difference wrapped to [-pi, pi]
        raw_dangle = _finite_diff(angle)
        feats["body_angle_vel"] = (raw_dangle + np.pi) % (2 * np.pi) - np.pi

    # Normalised body length, which should sit near 1 after normalisation
    if "nose_x" in df.columns and "tail_base_x" in df.columns:
        body_len = np.sqrt(
            (df["nose_x"].values - df["tail_base_x"].values) ** 2 +
            (df["nose_y"].values - df["tail_base_y"].values) ** 2
        )
        feats["body_length"] = body_len

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# Step 2b: inter-segment angles
# ---------------------------------------------------------------------------

def compute_segment_angles(
    wide_df: pd.DataFrame,
    skeleton: MouseSkeleton,
) -> pd.DataFrame:
    """Joint angles at each skeleton node.

    For each node with at least two incident edges, the angle between its two
    most relevant outgoing segments.

    Features:
      - neck_spine_angle      nose-neck-tail_base, that is, dorsal curvature
      - hip_tail_angle_left   neck-hip_left-tail_base
      - hip_tail_angle_right  neck-hip_right-tail_base

    Returns
    -------
    pd.DataFrame indexed by video_frame.
    """
    df = wide_df.set_index("video_frame").sort_index() if "video_frame" in wide_df.columns else wide_df.copy()
    feats: Dict[str, np.ndarray] = {}

    def _angle_at_vertex(
        ax: np.ndarray, ay: np.ndarray,
        bx: np.ndarray, by: np.ndarray,
        cx: np.ndarray, cy: np.ndarray,
    ) -> np.ndarray:
        """Angle ABC with its vertex at B, in radians over [0, pi]."""
        v1x, v1y = ax - bx, ay - by
        v2x, v2y = cx - bx, cy - by
        dot = v1x * v2x + v1y * v2y
        n1 = np.sqrt(v1x ** 2 + v1y ** 2)
        n2 = np.sqrt(v2x ** 2 + v2y ** 2)
        cos_a = dot / (n1 * n2 + 1e-8)
        return np.arccos(np.clip(cos_a, -1.0, 1.0))

    kps = set(skeleton.keypoints)
    req = {"nose", "neck", "tail_base"}
    if req.issubset(kps) and all(f"{k}_x" in df.columns for k in req):
        feats["neck_spine_angle"] = _angle_at_vertex(
            df["nose_x"].values,      df["nose_y"].values,
            df["neck_x"].values,      df["neck_y"].values,
            df["tail_base_x"].values, df["tail_base_y"].values,
        )

    if {"neck", "hip_left", "tail_base"}.issubset(kps) and \
            all(f"{k}_x" in df.columns for k in ["neck", "hip_left", "tail_base"]):
        feats["hip_tail_angle_left"] = _angle_at_vertex(
            df["neck_x"].values,      df["neck_y"].values,
            df["hip_left_x"].values,  df["hip_left_y"].values,
            df["tail_base_x"].values, df["tail_base_y"].values,
        )

    if {"neck", "hip_right", "tail_base"}.issubset(kps) and \
            all(f"{k}_x" in df.columns for k in ["neck", "hip_right", "tail_base"]):
        feats["hip_tail_angle_right"] = _angle_at_vertex(
            df["neck_x"].values,       df["neck_y"].values,
            df["hip_right_x"].values,  df["hip_right_y"].values,
            df["tail_base_x"].values,  df["tail_base_y"].values,
        )

    return pd.DataFrame(feats, index=df.index)


# ---------------------------------------------------------------------------
# Step 3: per-segment Δz, the implicit depth
# ---------------------------------------------------------------------------

def compute_dz_features(
    tracking_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    mouse_id: int,
) -> pd.DataFrame:
    """Extract the Δz features estimated by pose lifting, for one mouse.

    Filters to the given mouse_id, calls ``estimate_delta_z`` and returns the
    ``dz_*`` columns indexed by ``video_frame``.

    Returns
    -------
    pd.DataFrame indexed by video_frame, with dz_{src}_{dst} columns plus dz_sum.
    """
    mouse_df = tracking_df[tracking_df["mouse_id"] == mouse_id].copy()
    dz_wide = estimate_delta_z(mouse_df, skeleton, kp_index)
    dz_cols = [c for c in dz_wide.columns if c.startswith("dz_")]
    if "video_frame" in dz_wide.columns:
        return dz_wide.set_index("video_frame")[dz_cols]
    return dz_wide[dz_cols]


# ---------------------------------------------------------------------------
# Step 4: relational features between mice
# ---------------------------------------------------------------------------

def compute_relational_features(
    wide_dfs: Dict[int, pd.DataFrame],
    keypoints: List[str],
    dz_dfs: Optional[Dict[int, pd.DataFrame]] = None,
    agent: Optional[int] = None,
    target: Optional[int] = None,
) -> pd.DataFrame:
    """Relational features between mice.

    Features per (mouse_a, mouse_b) pair:
      - dist_centroid          distance between centroids
      - dist_nose_nose         nose to nose
      - dist_nose_tail_{ab/ba} nose of a to tail of b, and the reverse
      - angle_relative         heading of a relative to b
      - speed_relative         difference in speed magnitudes
      - dz_centroid_diff       difference in dz_sum between mice, when available

    Parameters
    ----------
    wide_dfs : dict[int, pd.DataFrame]
        One normalised frame per mouse_id, indexed by video_frame.
    keypoints : list[str]
    dz_dfs : dict[int, pd.DataFrame], optional
        Δz frames indexed by video_frame and mouse_id.

    Returns
    -------
    pd.DataFrame indexed by video_frame.
    """
    ids = sorted(wide_dfs.keys())
    if len(ids) < 2:
        # Single mouse: no relational block to build
        return pd.DataFrame(index=next(iter(wide_dfs.values())).index)

    # Alinear frames comunes
    common_frames = wide_dfs[ids[0]].index
    for mid in ids[1:]:
        common_frames = common_frames.intersection(wide_dfs[mid].index)

    feats: Dict[str, np.ndarray] = {}
    # ORDERED pair. The relational block is asymmetric: facing, sniff_site and
    # nose-of-A-in-B's-frame all depend on which mouse is the agent. Computing it
    # once with ids[0] as agent and sharing it between both mice would be wrong for
    # half the windows.
    a = ids[0] if agent is None else agent
    b = ids[1] if target is None else target
    da = wide_dfs[a].loc[common_frames]
    db = wide_dfs[b].loc[common_frames]

    def _get(df: pd.DataFrame, col: str) -> np.ndarray:
        return df[col].values.astype(float) if col in df.columns else np.full(len(df), np.nan)

    # Centroides
    xcols_a = [f"{kp}_x" for kp in keypoints if f"{kp}_x" in da.columns]
    ycols_a = [f"{kp}_y" for kp in keypoints if f"{kp}_y" in da.columns]
    xcols_b = [f"{kp}_x" for kp in keypoints if f"{kp}_x" in db.columns]
    ycols_b = [f"{kp}_y" for kp in keypoints if f"{kp}_y" in db.columns]

    cax = da[xcols_a].mean(axis=1).values
    cay = da[ycols_a].mean(axis=1).values
    cbx = db[xcols_b].mean(axis=1).values
    cby = db[ycols_b].mean(axis=1).values

    feats["dist_centroid"] = np.sqrt((cax - cbx) ** 2 + (cay - cby) ** 2)

    # Distancias nose-nose, nose-tail
    if "nose_x" in da.columns and "nose_x" in db.columns:
        nax, nay = _get(da, "nose_x"), _get(da, "nose_y")
        nbx, nby = _get(db, "nose_x"), _get(db, "nose_y")
        feats["dist_nose_nose"] = np.sqrt((nax - nbx) ** 2 + (nay - nby) ** 2)

        if "tail_base_x" in db.columns:
            tbx_b, tby_b = _get(db, "tail_base_x"), _get(db, "tail_base_y")
            feats["dist_nose_a_tail_b"] = np.sqrt((nax - tbx_b) ** 2 + (nay - tby_b) ** 2)

    if "nose_x" in db.columns and "tail_base_x" in da.columns:
        nbx2, nby2 = _get(db, "nose_x"), _get(db, "nose_y")
        tbx_a, tby_a = _get(da, "tail_base_x"), _get(da, "tail_base_y")
        feats["dist_nose_b_tail_a"] = np.sqrt((nbx2 - tbx_a) ** 2 + (nby2 - tby_a) ** 2)

    # ── Features that separate sniffface, sniffbody and sniffgenital ──
    # The distinction that matters is WHICH REGION of B the agent is sniffing:
    #   sniffface    A's nose near B's head or neck   small dist_nose_a_neck_b
    #   sniffbody    A's nose near B's mid-body        no distance especially small
    #   sniffgenital A's nose near B's hip or tail     small dist_nose_a_hip_b
    #                                                  or dist_nose_a_tail_b
    if "nose_x" in da.columns:
        nax_r, nay_r = _get(da, "nose_x"), _get(da, "nose_y")

        # Nose of A to neck of B: the head region
        if "neck_x" in db.columns:
            nkbx, nkby = _get(db, "neck_x"), _get(db, "neck_y")
            feats["dist_nose_a_neck_b"] = np.sqrt((nax_r - nkbx) ** 2 + (nay_r - nkby) ** 2)

        # Nose of A to mid-ear of B: the most frontal point available
        ear_cols = [(c.replace("_x", ""), c) for c in db.columns if c in ("ear_left_x", "ear_right_x")]
        if ear_cols:
            ear_xs = np.stack([_get(db, f"{kp}_x") for kp, _ in ear_cols], axis=1)
            ear_ys = np.stack([_get(db, f"{kp}_y") for kp, _ in ear_cols], axis=1)
            ear_mx = np.nanmean(ear_xs, axis=1)
            ear_my = np.nanmean(ear_ys, axis=1)
            feats["dist_nose_a_ear_b"] = np.sqrt((nax_r - ear_mx) ** 2 + (nay_r - ear_my) ** 2)

        # Nose of A to mid-hip of B: the genital region
        hip_cols = [kp for kp in ("hip_left", "hip_right") if f"{kp}_x" in db.columns]
        if hip_cols:
            hip_xs = np.stack([_get(db, f"{kp}_x") for kp in hip_cols], axis=1)
            hip_ys = np.stack([_get(db, f"{kp}_y") for kp in hip_cols], axis=1)
            hip_mx = np.nanmean(hip_xs, axis=1)
            hip_my = np.nanmean(hip_ys, axis=1)
            feats["dist_nose_a_hip_b"] = np.sqrt((nax_r - hip_mx) ** 2 + (nay_r - hip_my) ** 2)

        # sniff_site_ratio: position of A's nose along B's body axis, normalised.
        # Near -1 is B's face or neck (sniffface), near 0 its mid-body
        # (sniffbody), near +1 its tail or genitals (sniffgenital). One scalar
        # therefore orders the three sniff variants that the confusion matrix
        # keeps mixing up.
        if "dist_nose_a_neck_b" in feats and "dist_nose_a_tail_b" in feats:
            d_head = feats["dist_nose_a_neck_b"]
            d_tail = feats["dist_nose_a_tail_b"]
            feats["sniff_site_ratio"] = (d_tail - d_head) / (d_tail + d_head + 1e-6)

    # ── Fin features sniff discriminantes ─────────────────────────────────

    # Relative heading
    def _body_angle(df: pd.DataFrame) -> Optional[np.ndarray]:
        if "nose_x" in df.columns and "tail_base_x" in df.columns:
            dx = _get(df, "tail_base_x") - _get(df, "nose_x")
            dy = _get(df, "tail_base_y") - _get(df, "nose_y")
            return np.arctan2(dy, dx)
        return None

    ang_a = _body_angle(da)
    ang_b = _body_angle(db)
    if ang_a is not None and ang_b is not None:
        rel = ang_a - ang_b
        feats["angle_relative"] = (rel + np.pi) % (2 * np.pi) - np.pi

    # Approach rate: the derivative of the inter-centroid distance
    # Negativo = acercamiento; positivo = alejamiento
    feats["approach_rate"] = _finite_diff(feats["dist_centroid"])

    # Angle between A's heading and the A-to-B vector.
    # Near 0 rad, A faces B directly; near pi, A faces away.
    if ang_a is not None:
        vec_ab_angle = np.arctan2(cby - cay, cbx - cax)
        facing = ang_a - vec_ab_angle
        feats["facing_b"] = np.cos((facing + np.pi) % (2 * np.pi) - np.pi)

    # Relative speed
    if "centroid_speed" in da.columns and "centroid_speed" in db.columns:
        feats["speed_relative"] = da["centroid_speed"].values - db["centroid_speed"].values
    else:
        # Derive from the centroid when no kinematic frame was supplied
        va = np.sqrt(_finite_diff(cax) ** 2 + _finite_diff(cay) ** 2)
        vb = np.sqrt(_finite_diff(cbx) ** 2 + _finite_diff(cby) ** 2)
        feats["speed_relative"] = va - vb

    # Δz difference between mice: a relative 3D posture signal
    if dz_dfs is not None and a in dz_dfs and b in dz_dfs:
        dz_a = dz_dfs[a].reindex(common_frames)
        dz_b = dz_dfs[b].reindex(common_frames)
        if "dz_sum" in dz_a.columns and "dz_sum" in dz_b.columns:
            feats["dz_centroid_diff"] = dz_a["dz_sum"].values - dz_b["dz_sum"].values

    # ── Geometria inter-raton adicional ───────────────────────────────────
    # Minimum distance over every keypoint pair. When the mice overlap this
    # measures real proximity far better than the centroid distance, which stays
    # large while the animals are already touching.
    kps_a = [kp for kp in keypoints if f"{kp}_x" in da.columns]
    kps_b = [kp for kp in keypoints if f"{kp}_x" in db.columns]
    if kps_a and kps_b:
        Ax = np.stack([_get(da, f"{kp}_x") for kp in kps_a], axis=1)
        Ay = np.stack([_get(da, f"{kp}_y") for kp in kps_a], axis=1)
        Bx = np.stack([_get(db, f"{kp}_x") for kp in kps_b], axis=1)
        By = np.stack([_get(db, f"{kp}_y") for kp in kps_b], axis=1)
        D3 = np.sqrt((Ax[:, :, None] - Bx[:, None, :]) ** 2 +
                     (Ay[:, :, None] - By[:, None, :]) ** 2)
        D = D3.reshape(len(Ax), -1)
        with np.errstate(all="ignore"):
            feats["min_kp_dist"] = np.nanmin(D, axis=1)
            Ds = np.sort(np.where(np.isfinite(D), D, np.inf), axis=1)[:, :3]
            feats["mean_k3_kp_dist"] = np.where(np.isfinite(Ds).all(1), Ds.mean(1), np.nan)

        # ── Solape y oclusion ─────────────────────────────────────────────
        # OFF BY DEFAULT. Measured with a clean A/B (same cache, same labs, same
        # RNG sequence): 0.4020 with them against 0.4006 without, so +0.0014, which
        # is noise. They add five columns per block and buy nothing. Kept because
        # the A/B is the publishable result, not the code. MABE_OCCLUSION=1 turns
        # them back on.
        if os.environ.get("MABE_OCCLUSION", "0") != "0":
            # `mount` and `intromit` are defined by one mouse being ON TOP of the
            # other, and nothing above measures that: only distances, angles and
            # closing speed. From overhead, "on top" shows up as heavily overlapping
            # bounding boxes and as apparent shortening of the mouse underneath.
            # The diagnostic quantified the cost of missing it: mount is
            # over-predicted 5.73x and intromit sits at 0.46x, which is exactly the
            # confusion these features target.
            # Coordinates are already divided by body length, so thresholds
            # expressed in body lengths are comparable across labs.
            with np.errstate(all="ignore"):
                axmin, axmax = np.nanmin(Ax, 1), np.nanmax(Ax, 1)
                aymin, aymax = np.nanmin(Ay, 1), np.nanmax(Ay, 1)
                bxmin, bxmax = np.nanmin(Bx, 1), np.nanmax(Bx, 1)
                bymin, bymax = np.nanmin(By, 1), np.nanmax(By, 1)
                iw = np.clip(np.minimum(axmax, bxmax) - np.maximum(axmin, bxmin), 0, None)
                ih = np.clip(np.minimum(aymax, bymax) - np.maximum(aymin, bymin), 0, None)
                inter = iw * ih
                area_a = (axmax - axmin) * (aymax - aymin)
                area_b = (bxmax - bxmin) * (bymax - bymin)
                union = area_a + area_b - inter
                feats["bbox_iou"] = np.where(union > 0, inter / union, 0.0)
                feats["area_ratio"] = np.where(area_b > 0, area_a / area_b, np.nan)
                dia_a = np.sqrt((axmax - axmin) ** 2 + (aymax - aymin) ** 2)
                dia_b = np.sqrt((bxmax - bxmin) ** 2 + (bymax - bymin) ** 2)
                # Below 1 means the agent looks shorter than the target, the
                # foreshortening of being above or below it. That is the postural
                # signal that separates mount.
                feats["body_len_ratio"] = np.where(dia_b > 0, dia_a / dia_b, np.nan)
                valid = np.isfinite(Ax) & np.isfinite(Ay)
                nval = valid.sum(1)
                inside = ((Ax >= bxmin[:, None]) & (Ax <= bxmax[:, None]) &
                          (Ay >= bymin[:, None]) & (Ay <= bymax[:, None]))
                feats["frac_kp_inside"] = np.where(
                    nval > 0, (inside & valid).sum(1) / np.maximum(nval, 1), np.nan)
                # A fraction rather than a count, so it stays comparable across labs, which have
                # between 4 and 18 keypoints.
                dmin_a = np.nanmin(D3, axis=2)
                feats["frac_kp_close"] = np.where(
                    nval > 0, ((dmin_a < 0.25) & valid).sum(1) / np.maximum(nval, 1), np.nan)

    # The other mouse's position in the agent's EGOCENTRIC frame. This separates
    # "in front of me" from "beside me", which distance alone cannot.
    def _heading(df):
        if "nose_x" in df.columns and "tail_base_x" in df.columns:
            return np.arctan2(_get(df, "nose_y") - _get(df, "tail_base_y"),
                              _get(df, "nose_x") - _get(df, "tail_base_x"))
        return None

    ha, hb = _heading(da), _heading(db)
    if ha is not None:
        dxc, dyc = cbx - cax, cby - cay
        feats["rel_fwd"] = np.cos(ha) * dxc + np.sin(ha) * dyc
        feats["rel_lat"] = -np.sin(ha) * dxc + np.cos(ha) * dyc
    # A's nose in B's frame: (+,0) is B's face and (-,0) its tail, so this
    # separates sniffface, sniffbody and sniffgenital in a plane rather than
    # collapsing them onto a single ratio.
    if hb is not None and "nose_x" in da.columns:
        ddx = _get(da, "nose_x") - cbx
        ddy = _get(da, "nose_y") - cby
        feats["nose_a_in_b_fwd"] = np.cos(hb) * ddx + np.sin(hb) * ddy
        feats["nose_a_in_b_lat"] = -np.sin(hb) * ddx + np.cos(hb) * ddy

    # Closing speed, positive when approaching, plus absolute speeds. Without
    # these, approach, disengage, chase and escape are not separable: the
    # symmetric aggregates give the same value whether the mice close or part.
    feats["closing_speed"] = -feats["approach_rate"]
    feats["speed_a"] = np.sqrt(_finite_diff(cax) ** 2 + _finite_diff(cay) ** 2)
    feats["speed_b"] = np.sqrt(_finite_diff(cbx) ** 2 + _finite_diff(cby) ** 2)

    return pd.DataFrame(feats, index=common_frames)


# ---------------------------------------------------------------------------
# Step 5: building the sliding windows
# ---------------------------------------------------------------------------

def _make_label_array(
    annotations_df: pd.DataFrame,
    frames: np.ndarray,
    mouse_id: int,
    background_label: str = "background",
    target_id: Optional[int] = None,
) -> np.ndarray:
    """Assign a behaviour label to each frame.

    Annotations overlap, so a frame can fall inside several segments. The longest
    action covering it wins, which keeps the dominant behaviour rather than
    whichever annotation happened to be processed last.

    Parameters
    ----------
    annotations_df : pd.DataFrame
        Video annotations, with agent_id, action, start_frame and stop_frame.
    frames : np.ndarray
        Sorted frame indices.
    mouse_id : int
        The agent mouse.
    background_label : str
        Label for frames with no annotation.

    Returns
    -------
    np.ndarray of str, shape (len(frames),)
    """
    labels = np.full(len(frames), background_label, dtype=object)
    if annotations_df is None or len(annotations_df) == 0:
        return labels

    ann = annotations_df[annotations_df["agent_id"] == mouse_id]
    if target_id is not None and "target_id" in ann.columns:
        # With 3 or more mice the target has to be separated out: a mouse1->mouse3
        # annotation must not also label the mouse1->mouse2 row. Self-directed
        # actions are encoded as agent == target and belong to the agent rather than
        # to a specific pair, so they are kept across all of its pairs. With two mice
        # this reproduces the previous behaviour exactly.
        ann = ann[(ann["target_id"] == target_id) | (ann["target_id"] == mouse_id)]
    ann = ann.copy()
    # Sort by descending duration so the longest action takes precedence
    if "duration_frames" not in ann.columns:
        ann["duration_frames"] = ann["stop_frame"] - ann["start_frame"]
    ann = ann.sort_values("duration_frames", ascending=False)

    frame_to_idx = {f: i for i, f in enumerate(frames)}

    for _, row in ann.iterrows():
        for fr in range(int(row["start_frame"]), int(row["stop_frame"])):
            if fr in frame_to_idx:
                labels[frame_to_idx[fr]] = str(row["action"])

    return labels


def build_windows(
    feature_df: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    min_fill: float = MIN_WINDOW_FILL,
    pad_mode: str = "forward",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build sliding windows from a per-frame feature frame.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Per-frame features, indexed by video_frame and sorted.
    labels : np.ndarray, optional
        Per-frame string labels, the same length as feature_df.
    window_size : int
        Frames per window.
    stride : int
        Step between consecutive windows. Below window_size the windows overlap,
        which is the leakage source discussed in the notebooks.
    min_fill : float
        Minimum fraction of non-NaN frames for a window to be kept.
    pad_mode : {"forward", "zero", "mean"}
        How to fill short or partly-NaN windows:
        - "forward": repeat the last valid value
        - "zero": fill with 0
        - "mean": the window mean

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size, n_features)
    y_windows : np.ndarray | None, shape (n_windows,), the majority label
    """
    arr = feature_df.values.astype(np.float32)  # (T, F)
    T, F = arr.shape

    windows_X = []
    windows_y = []

    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        chunk = arr[start:end].copy()

        # Check the minimum fill
        valid_rows = np.isfinite(chunk).all(axis=1)
        if valid_rows.mean() < min_fill:
            continue

        # Rellenar NaN / inf
        if pad_mode == "forward":
            chunk = _forward_fill(chunk)
        elif pad_mode == "zero":
            chunk = np.nan_to_num(chunk, nan=0.0)
        elif pad_mode == "mean":
            col_means = np.nanmean(chunk, axis=0)
            nan_mask = ~np.isfinite(chunk)
            chunk[nan_mask] = np.take(col_means, nan_mask.nonzero()[1])

        windows_X.append(chunk)

        if labels is not None:
            lbl_chunk = labels[start:end]
            # Majority label, ignoring background when any other class is present
            unique, counts = np.unique(lbl_chunk, return_counts=True)
            if len(unique) > 1:
                non_bg = [(u, c) for u, c in zip(unique, counts) if u != "background"]
                if non_bg:
                    windows_y.append(max(non_bg, key=lambda x: x[1])[0])
                else:
                    windows_y.append(unique[counts.argmax()])
            else:
                windows_y.append(unique[0])

    X = np.stack(windows_X, axis=0) if windows_X else np.empty((0, window_size, F), dtype=np.float32)
    y = np.array(windows_y, dtype=object) if windows_y else None
    return X, y


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Column-wise forward fill, vectorised with numpy."""
    out = arr.copy()
    T, F = out.shape
    for f in range(F):
        col = out[:, f]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        # Forward-propagated indices: at invalid positions
        # carry forward the index of the last valid row
        idx = np.where(valid, np.arange(T), 0)
        np.maximum.accumulate(idx, out=idx)
        out[:, f] = col[idx]
    return out


# ---------------------------------------------------------------------------
# Main entry point: full extraction for one video
# ---------------------------------------------------------------------------

def extract_features(
    tracking_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    annotations_df: Optional[pd.DataFrame] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    include_dz: bool = True,
    include_relational: bool = True,
    background_label: str = "background",
    min_fill: float = MIN_WINDOW_FILL,
    center: str = "centroid",
    pairs: bool = False,
) -> Dict:
    """Extract every feature for a whole video.

    Runs the per-mouse pipeline and then merges the relational block, which is
    computed per ordered (agent, target) pair rather than once per video.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Long-format tracking, already corrected by MouseSkeleton, that is the
        output of ``smooth_video``.
    skeleton : MouseSkeleton
        Skeleton fitted for this laboratory or video.
    annotations_df : pd.DataFrame, optional
        Video annotations. With None, no labels are produced.
    window_size : int
        Window size in frames.
    stride : int
        Step between windows.
    include_dz : bool
        Add the Δz features from the pose-lifting module.
    include_relational : bool
        Add the inter-mouse relational features.
    background_label : str
        Label for frames with no annotated behaviour.

    Returns
    -------
    dict[int, dict], keyed by mouse_id:
        {
          "features_df":   pd.DataFrame  (T, n_features), per-frame features
          "X":             np.ndarray    (n_windows, W, F), the windows
          "y":             np.ndarray    (n_windows,), string labels
          "feature_names": list[str]
          "frames":        np.ndarray    (T,), frame indices
        }
    """
    kp_index = {kp: i for i, kp in enumerate(skeleton.keypoints)}
    l_body_px = skeleton.L_body_median_px

    # ── Pivote a formato ancho ────────────────────────────────────────────
    wide_all = _to_wide(tracking_df)

    # ── Per-mouse normalisation, using the group centroid to remove translation
    mice_ids = sorted(tracking_df["mouse_id"].unique().tolist())

    per_mouse_wide: Dict[int, pd.DataFrame] = {}
    for mid in mice_ids:
        sub = wide_all[wide_all["mouse_id"] == mid].copy()
        sub = _filter_zero_frames(sub)
        sub = normalize_coordinates(sub, l_body_px, center=center)
        sub = sub.set_index("video_frame").sort_index()
        per_mouse_wide[mid] = sub

    # ── Δz ────────────────────────────────────────────────────────────────
    dz_dfs: Dict[int, pd.DataFrame] = {}
    if include_dz:
        for mid in mice_ids:
            try:
                dz_df = compute_dz_features(tracking_df, skeleton, kp_index, mid)
                dz_dfs[mid] = dz_df
            except Exception:
                pass  # pose lifting can fail for one mouse; carry on with the rest

    # ── Relational features, ONE VIEW PER AGENT ─────────────────────────
    # The block is asymmetric: `facing_b`, `sniff_site_ratio` and nose-of-A-in-B's-
    # frame all describe the agent looking at the other mouse. Computing it once and
    # sharing it left half the windows with the wrong perspective.
    # `pairs=True` enumerates ALL ordered (agent, target) pairs and returns one entry
    # per pair. With two mice the only possible pair for each agent is the one
    # already used, so the result is identical to `pairs=False`, verified bit for bit
    # in `scripts/_selftest_pairs.py`. It matters for the 3-to-5-mouse labs
    # (AdaptableSnail, DeliriousFly, ReflectiveManatee), where keeping only the first
    # other mouse meant pairs like mouse1->mouse3 were never predicted.
    if pairs:
        jobs = [((a, b), a, b) for a in mice_ids for b in mice_ids if a != b]
        if not jobs:                       # single mouse: no relational block
            jobs = [((m, None), m, None) for m in mice_ids]
    else:
        jobs = [(m, m, next((o for o in mice_ids if o != m), None)) for m in mice_ids]

    relational_by_pair: Dict[tuple, pd.DataFrame] = {}
    if include_relational and len(mice_ids) >= 2:
        for _key, _a, _b in jobs:
            if _b is None or (_a, _b) in relational_by_pair:
                continue
            relational_by_pair[(_a, _b)] = compute_relational_features(
                per_mouse_wide, skeleton.keypoints, dz_dfs if include_dz else None,
                agent=_a, target=_b,
            )

    # Pairs from the same video can end up with different relational columns:
    # `compute_relational_features` adds them conditionally, so a mouse whose
    # keypoints are all NaN yields fewer. That would give a different feature width
    # per pair within the SAME video. Columns are unioned in a stable order and the
    # gaps filled with NaN. Only in `pairs` mode, so the default path is untouched
    # and therefore bit-for-bit identical.
    if pairs and relational_by_pair:
        _cols = []
        for _df in relational_by_pair.values():
            for _c in _df.columns:
                if _c not in _cols:
                    _cols.append(_c)
        for _k, _df in list(relational_by_pair.items()):
            if list(_df.columns) != _cols:
                relational_by_pair[_k] = _df.reindex(columns=_cols)

    # ── Kinematics and angles: agent-only, so they are computed once and reused ──
    base_by_mouse: Dict[int, tuple] = {}
    for mid in mice_ids:
        sub_wide = per_mouse_wide[mid]
        base_by_mouse[mid] = (compute_kinematics(sub_wide.reset_index(), skeleton.keypoints),
                              compute_segment_angles(sub_wide.reset_index(), skeleton))

    # First pass: assemble each job's feature frame.
    _built = []
    for _key, mid, _tgt in jobs:

        kin_df, ang_df = base_by_mouse[mid]

        parts = [kin_df, ang_df]

        if include_dz and mid in dz_dfs:
            parts.append(dz_dfs[mid].reindex(kin_df.index))

        relational_df = relational_by_pair.get((mid, _tgt))
        if relational_df is not None:
            parts.append(relational_df.reindex(kin_df.index))

        feat_df = pd.concat(parts, axis=1)
        feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]
        _built.append((_key, mid, _tgt, feat_df))

    # Same column alignment, for the dz_ block. `compute_dz_features` sits inside a
    # per-mouse try/except, so when it fails for one mouse that mouse's pairs lose
    # the whole dz_ block and the feature width changes within one video. Measured on
    # AdaptableSnail: 74 columns for agent 1 and 70 for agent 2. Union in stable
    # order, NaN for the gaps.
    if pairs and len(_built) > 1:
        _cols = []
        for _, _, _, _df in _built:
            for _c in _df.columns:
                if _c not in _cols:
                    _cols.append(_c)
        _built = [(_k, _m, _t, _df if list(_df.columns) == _cols
                   else _df.reindex(columns=_cols))
                  for _k, _m, _t, _df in _built]

    results: Dict = {}
    for _key, mid, _tgt, feat_df in _built:

        # Per-frame labels
        labels_arr: Optional[np.ndarray] = None
        if annotations_df is not None:
            labels_arr = _make_label_array(
                annotations_df, feat_df.index.values, mid, background_label,
                target_id=_tgt if pairs else None,
            )

        # Windows
        X, y = build_windows(feat_df, labels_arr, window_size, stride, min_fill)

        results[_key] = {
            "features_df":     feat_df,
            "agent":           mid,
            "target":          _tgt,
            "X":               X,
            "y":               y,
            "feature_names":   feat_df.columns.tolist(),
            "frames":          feat_df.index.values,
            # Per-frame labels (aligned to `frames`) so downstream re-windowing
            # steps (imputation, circular encoding) can rebuild `y` in lock-step
            # with a re-windowed `X`. None when the video is unlabelled.
            "labels_per_frame": labels_arr,
            # Windowing params actually used here. Downstream re-windowing must
            # reuse them: falling back to the module defaults silently changes
            # the window count (e.g. cfg stride 16 vs default 32 halves it) and
            # desynchronises X from y and from an already-computed split.
            "window_size":     window_size,
            "stride":          stride,
        }

    return results


# ---------------------------------------------------------------------------
# Window statistics: the flattened representation the shallow models consume
# ---------------------------------------------------------------------------

WINDOW_STATS = ("mean", "std", "min", "max", "p25", "p75", "iqr", "skew", "kurt")


def window_statistics(
    X: np.ndarray,
    feature_names: List[str],
    stats: Tuple[str, ...] = WINDOW_STATS,
) -> Tuple[np.ndarray, List[str]]:
    """Collapse each window into a vector of per-feature statistics.

    Classical models such as SVM and random forests need a fixed-width input, so
    each window is collapsed to a vector of statistics per feature. This is why the
    shallow baselines see 387 or 576 columns while the sequence models see 64.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, W, F)
    feature_names : list[str]
    stats : tuple[str]
        A subset of {"mean","std","min","max","p25","p75","iqr","skew","kurt"}.

    Returns
    -------
    X_flat : np.ndarray, shape (n_windows, F * len(stats))
    flat_names : list[str]
    """
    def _skew(x):
        # x: (n_windows, W, F) → (n_windows, F)
        return np.array([_sp_skew(x[i], axis=0, nan_policy="omit") for i in range(x.shape[0])])

    def _kurt(x):
        return np.array([_sp_kurt(x[i], axis=0, nan_policy="omit") for i in range(x.shape[0])])

    stat_funcs = {
        "mean": lambda x: np.nanmean(x, axis=1),
        "std":  lambda x: np.nanstd(x, axis=1),
        "min":  lambda x: np.nanmin(x, axis=1),
        "max":  lambda x: np.nanmax(x, axis=1),
        "p25":  lambda x: np.nanpercentile(x, 25, axis=1),
        "p75":  lambda x: np.nanpercentile(x, 75, axis=1),
        "iqr":  lambda x: np.nanpercentile(x, 75, axis=1) - np.nanpercentile(x, 25, axis=1),
        "skew": _skew,
        "kurt": _kurt,
    }

    parts = []
    names = []
    for stat in stats:
        if stat not in stat_funcs:
            continue
        parts.append(stat_funcs[stat](X))     # (n_windows, F)
        names.extend([f"{fn}_{stat}" for fn in feature_names])

    X_flat = np.concatenate(parts, axis=1)
    return X_flat, names
