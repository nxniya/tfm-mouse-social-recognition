"""
src/skeleton/keypoint_smoother.py
===================================
Kinematic outlier detection and keypoint correction (MouseSkeleton).

Pipeline for a whole video:

  1. Reshape the tracking from long to wide format, per mouse.
  2. Compute frame-to-frame speed per keypoint.
  3. Flag outliers: speed above the robust threshold, or an edge length outside
     its tolerance band.
  4. For each frame with at least one outlier, call ``correct_frame()`` from
     kinematic_constraints.
  5. Return the corrected frame in long format.

Only the flagged frames are corrected. Running the optimiser on every frame would
cost hours per video and would also rewrite frames that have nothing wrong with
them, which is the failure mode the detector calibration in notebook 02d addresses.

An important caveat for anyone reusing this module: correcting the skeleton improves
its geometry but *degrades* downstream classification, because relocating keypoints
injects jitter into exactly the fine motion the classifier reads. That result is
reproduced six ways in notebooks 04 and 05. Use the raw tracking for classification;
use this for geometry and visualisation.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # tqdm is optional — degrade gracefully
    def _tqdm(it, **kw):  # type: ignore[misc]
        return it
try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # tqdm is optional
    def _tqdm(it, **kw):  # type: ignore[misc]
        return it

from scipy.interpolate import CubicSpline, PchipInterpolator

from src.skeleton.mouse_skeleton import (
    MouseSkeleton,
    _to_wide,
    _segment_length,
    build_skeleton,
    fit_skeleton,
)
from src.skeleton.kinematic_constraints import (
    correct_frame, correct_window, compute_edge_lambdas, build_angle_triplets,
    build_edge_cache,
    DEFAULT_LAMBDA_L, DEFAULT_LAMBDA_ACC, DEFAULT_LAMBDA_ANGLE, DEFAULT_LAMBDA_JERK,
    DEFAULT_LAMBDA_T,
    compute_adaptive_edge_thresholds,
)


# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------

SPEED_SIGMA_THRESHOLD = 3.0   # speed outlier threshold, in sigmas
LENGTH_VAR_THRESHOLD  = 0.20  # length violation threshold, as a fraction
SAVGOL_WINDOW         = 5     # Savitzky-Golay window, in frames
SAVGOL_POLYORDER      = 2     # Savitzky-Golay polynomial order

# Window optimization constants
_WINDOW_HALF = 12  # context frames before/after each outlier group (~25-frame windows, v4)
_MERGE_GAP   = 4   # merge two outlier groups if gap ≤ this many frames

# Overlap / foreshortening constants
_OVERLAP_RADIUS_SCALE  = 0.55   # overlap_radius = this × l_body_px
_FORESHORTEN_EDGE_SCALE = 0.45  # edge_lambdas multiplier during overlap
                                # ≈ cos²(57°): equivalent to ~57° body tilt
_IDENTITY_SIZE_THRESH  = 0.15   # 15% body-length change triggers swap check
_IDENTITY_WIN_FRAMES   = 12     # pre/post window for size proxy computation
_IDENTITY_MAX_SWAP_LEN = 60     # max frames in a swap block per huddle event


# ---------------------------------------------------------------------------
# Overlap detection (foreshortening / virtual height)
# ---------------------------------------------------------------------------

def _build_overlap_masks(
    tracking_df: pd.DataFrame,
    all_frames: List[int],
    l_body_px: float,
) -> Dict:
    """Compute per-mouse overlap masks for a multi-animal video.

    Two mice "overlap" when their frame centroid (mean of all keypoints) is
    within ``_OVERLAP_RADIUS_SCALE × l_body_px`` pixels of each other.  During
    those frames the skeleton can appear foreshortened (one animal on top of
    another), so the length-constraint weight should be relaxed.

    Args:
        tracking_df : long-format tracking DataFrame for the whole video.
        all_frames  : sorted list of every frame index in the video.
        l_body_px   : estimated body length (pixels) for the video.

    Returns:
        {mouse_id: np.ndarray(T, dtype=bool)} — True where mouse is in an
        overlap event with any other mouse.  Returns {} for single-mouse videos.
    """
    mice = sorted(tracking_df["mouse_id"].unique())
    if len(mice) < 2:
        return {}

    fi_map = {f: i for i, f in enumerate(all_frames)}
    T = len(all_frames)

    # Build per-mouse centroid arrays (T, 2)
    centroids: Dict[str, np.ndarray] = {}
    for mouse_id, mdf in tracking_df.groupby("mouse_id"):
        c = np.full((T, 2), np.nan)
        grouped = mdf.groupby("video_frame")[["x", "y"]].mean()
        for f, row in grouped.iterrows():
            if f in fi_map:
                c[fi_map[f]] = [row["x"], row["y"]]
        centroids[mouse_id] = c

    overlap_radius = _OVERLAP_RADIUS_SCALE * max(l_body_px, 1.0)

    result: Dict[str, np.ndarray] = {}
    for mid in mice:
        mask = np.zeros(T, dtype=bool)
        for other_mid, other_c in centroids.items():
            if other_mid == mid:
                continue
            diff = centroids[mid] - other_c           # (T, 2)
            dist = np.sqrt(np.nansum(diff ** 2, axis=1))  # (T,)
            mask |= dist < overlap_radius
        result[mid] = mask

    return result


# ---------------------------------------------------------------------------
# Post-optimization identity verification
# ---------------------------------------------------------------------------

def _post_correction_identity_check(
    corrected_df: pd.DataFrame,
    kp_index: Dict[str, int],
) -> Tuple[pd.DataFrame, int]:
    """Detect and correct persistent identity swaps after kinematic optimization.

    Algorithm
    ---------
    Uses body length (nose → tail_base distance) as a size proxy.  For each
    pair of mice, the function:

    1. Detects "huddle periods" — consecutive frames where centroid distance is
       below half the typical body length.
    2. At each huddle *exit*, computes the median size proxy in the
       ``_IDENTITY_WIN_FRAMES`` frames immediately before and after.
    3. If the size-rank between the two mice inverts by more than
       ``_IDENTITY_SIZE_THRESH`` (15%), the identity is considered swapped.
    4. Applies a body-swap (keypoints exchanged) for the post-huddle segment,
       capped at ``_IDENTITY_MAX_SWAP_LEN`` frames to avoid over-correction.

    This handles the common CalMS21 failure mode where the kinematic optimizer
    correctly fixes the geometry of a swap frame but leaves the identity label
    wrong for an extended segment after a huddle event.

    Args:
        corrected_df : post-optimization tracking DataFrame.
        kp_index     : keypoint name → index mapping.

    Returns:
        (corrected_df_out, n_block_swaps_applied)
    """
    mice = sorted(corrected_df["mouse_id"].unique())
    if len(mice) != 2:
        return corrected_df, 0

    m0, m1 = mice[0], mice[1]
    nose_i = kp_index.get("nose")
    tail_i = kp_index.get("tail_base")
    if nose_i is None or tail_i is None:
        return corrected_df, 0

    frames = sorted(corrected_df["video_frame"].unique())
    T = len(frames)
    fi_map = {f: i for i, f in enumerate(frames)}

    def _get_poses(mouse_id: str) -> np.ndarray:
        n_kp = len(kp_index)
        P = np.full((T, n_kp, 2), np.nan)
        mdf = corrected_df[corrected_df["mouse_id"] == mouse_id].copy()
        mdf["_fi"] = mdf["video_frame"].map(fi_map)
        mdf["_ki"] = mdf["bodypart"].map(kp_index)
        vld = mdf["_fi"].notna() & mdf["_ki"].notna()
        sv = mdf[vld]
        if not sv.empty:
            P[sv["_fi"].values.astype(np.intp),
              sv["_ki"].values.astype(np.intp), 0] = sv["x"].values
            P[sv["_fi"].values.astype(np.intp),
              sv["_ki"].values.astype(np.intp), 1] = sv["y"].values
        return P

    P0 = _get_poses(m0)
    P1 = _get_poses(m1)

    def _body_len(P: np.ndarray) -> np.ndarray:
        """Nose → tail_base distance for each frame; NaN when either is missing."""
        return np.sqrt(np.nansum(
            (P[:, nose_i] - P[:, tail_i]) ** 2, axis=1
        ))

    def _centroid(P: np.ndarray) -> np.ndarray:
        """Mean of finite keypoints, (T, 2)."""
        with np.errstate(all="ignore"):
            return np.nanmean(P, axis=1)

    len0 = _body_len(P0)   # (T,)
    len1 = _body_len(P1)   # (T,)
    c0   = _centroid(P0)   # (T, 2)
    c1   = _centroid(P1)   # (T, 2)

    centroid_dist = np.sqrt(np.nansum((c0 - c1) ** 2, axis=1))  # (T,)

    both = np.concatenate([len0[np.isfinite(len0)], len1[np.isfinite(len1)]])
    l_body_ref = float(np.nanmedian(both)) if both.size else 1.0
    huddle_mask = centroid_dist < (0.5 * l_body_ref)  # (T,) bool

    corrected_df = corrected_df.copy()
    n_swaps = 0
    W = _IDENTITY_WIN_FRAMES

    i = 0
    while i < T:
        if not huddle_mask[i]:
            i += 1
            continue

        # Found huddle start; scan to end
        j = i
        while j < T and huddle_mask[j]:
            j += 1
        # Huddle occupies frames[i:j].  j is first non-huddle frame after exit.

        pre_s,  pre_e  = max(0, i - W), i
        post_s, post_e = j, min(T, j + W)

        if pre_e > pre_s and post_e > post_s:
            pre_l0  = float(np.nanmedian(len0[pre_s:pre_e]))
            pre_l1  = float(np.nanmedian(len1[pre_s:pre_e]))
            post_l0 = float(np.nanmedian(len0[post_s:post_e]))
            post_l1 = float(np.nanmedian(len1[post_s:post_e]))

            if all(np.isfinite(v) and v > 1.0
                   for v in [pre_l0, pre_l1, post_l0, post_l1]):
                pre_ratio  = pre_l0  / (pre_l1  + 1e-6)
                post_ratio = post_l0 / (post_l1 + 1e-6)

                # Rank inversion: one mouse was bigger before, now it's smaller
                thresh = _IDENTITY_SIZE_THRESH
                inversion = (
                    (pre_ratio > 1 + thresh and post_ratio < 1 - thresh)
                    or
                    (pre_ratio < 1 - thresh and post_ratio > 1 + thresh)
                )

                if inversion:
                    # Apply swap for [j, min(j + _IDENTITY_MAX_SWAP_LEN, T))
                    swap_end   = min(T, j + _IDENTITY_MAX_SWAP_LEN)
                    swap_frames = set(frames[j:swap_end])

                    df0_swap = corrected_df[
                        (corrected_df["mouse_id"] == m0)
                        & (corrected_df["video_frame"].isin(swap_frames))
                    ].copy()
                    df1_swap = corrected_df[
                        (corrected_df["mouse_id"] == m1)
                        & (corrected_df["video_frame"].isin(swap_frames))
                    ].copy()

                    df0_swap["mouse_id"] = m1
                    df1_swap["mouse_id"] = m0

                    keep = ~(
                        corrected_df["mouse_id"].isin([m0, m1])
                        & corrected_df["video_frame"].isin(swap_frames)
                    )
                    corrected_df = pd.concat(
                        [corrected_df[keep], df0_swap, df1_swap],
                        ignore_index=True,
                    )
                    n_swaps += 1

                    # Refresh size series for subsequent events
                    P0 = _get_poses(m0)
                    P1 = _get_poses(m1)
                    len0 = _body_len(P0)
                    len1 = _body_len(P1)

        i = j  # advance past huddle

    return corrected_df, n_swaps


# ---------------------------------------------------------------------------
# Per-video L_ref re-estimation
# ---------------------------------------------------------------------------

def _reestimate_lref_per_video(
    skeleton: MouseSkeleton,
    poses: np.ndarray,
    kp_index: Dict[str, int],
    l_body_px: float,
    zero_frame_mask: np.ndarray,
    clean_percentile: float = 50.0,
) -> MouseSkeleton:
    """Return a skeleton copy with L_ref **and L_std** re-estimated from this video.

    Replaces global training medians with per-video robust medians computed
    from the cleanest 50% of frames.  Also updates ``L_std`` from the
    inter-quartile spread of clean-frame lengths so that ``edge_lambdas``
    (which are ∝ 1/L_std²) reflect the actual per-video bone stiffness instead
    of the fixed global 5% prior.  Rigid edges with small observed variance get
    higher λ; flexible edges with large observed variance get lower λ.

    Args:
        skeleton         : globally fitted MouseSkeleton (read-only).
        poses            : shape (T, K, 2) — raw or pre-smoothed poses.
        kp_index         : keypoint name → index.
        l_body_px        : per-video body length estimate (pixels).
        zero_frame_mask  : shape (T,) bool — frames to exclude.
        clean_percentile : only frames below this percentile of relative error
                           from the global prior are used as clean reference.
                           Default 50 = lower-half of the distribution.

    Returns:
        Deep copy of skeleton with updated ``L_ref`` **and** ``L_std`` values.
    """
    video_skeleton = copy.deepcopy(skeleton)
    active = ~zero_frame_mask
    active_poses = poses[active]
    if active_poses.shape[0] < 10 or l_body_px <= 0:
        return video_skeleton

    for edge in video_skeleton.edges:
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        diffs = active_poses[:, i] - active_poses[:, j]
        seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
        seg_len = seg_len[seg_len > 0]
        if len(seg_len) < 10:
            continue

        # Select "clean" frames as those in the lower half of relative deviation
        # from the global prior.  This avoids corrupted frames biasing the
        # per-video estimate.
        if edge.L_ref > 0:
            global_lref_px = edge.L_ref * l_body_px
            r_e = np.abs(seg_len - global_lref_px) / max(global_lref_px, 1e-6)
            clean_thr = np.percentile(r_e, clean_percentile)
            clean_lens = seg_len[r_e <= clean_thr]
        else:
            clean_lens = seg_len

        if len(clean_lens) < 5:
            clean_lens = seg_len  # fallback: use all frames

        new_lref = float(np.median(clean_lens)) / l_body_px
        if new_lref > 1e-4:
            edge.L_ref = new_lref

        # Update L_std from per-video IQR of clean lengths.
        # Use IQR/1.35 as a robust σ estimate (Gaussian equivalent),
        # normalized by l_body_px so it stays in the same fractional units.
        # Clamp to [0.02, 0.20] to prevent degenerate lambdas.
        if len(clean_lens) >= 10:
            _q25, _q75 = np.percentile(clean_lens, [25, 75])
            _iqr_sigma = (_q75 - _q25) / 1.35 / max(l_body_px, 1.0)
            edge.L_std = float(np.clip(_iqr_sigma, 0.02, 0.20))

    return video_skeleton
LENGTH_VAR_THRESHOLD  = 0.20  # length violation threshold, as a fraction
SAVGOL_WINDOW         = 5     # Savitzky-Golay window, in frames
SAVGOL_POLYORDER      = 2     # Savitzky-Golay polynomial order


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _init_spline_dropouts(
    poses: np.ndarray,
    severity: np.ndarray,
    zero_frame_mask: np.ndarray,
    severity_threshold: float = 0.8,
) -> np.ndarray:
    """Initialize severe-outlier keypoints using cubic spline interpolation.

    For each keypoint/frame where ``severity > severity_threshold`` (dropout class),
    replaces the value with a cubic spline through the clean frames.  This gives
    the L-BFGS-B optimizer a much better starting point than copying ``prev_frame``,
    which is critical when consecutive dropout frames exist.

    Args:
        poses            : shape (T, K, 2) — original poses array.
        severity         : shape (T, K) — severity scores from ``_detect_outliers``.
        zero_frame_mask  : shape (T,) — frames to exclude (all-zero tracking).
        severity_threshold: frames with severity > this are filled by spline.

    Returns:
        Filled copy of poses (same shape); only severe frames are changed.
    """
    T, n_kp, _ = poses.shape
    filled = poses.copy()
    t_all = np.arange(T, dtype=float)

    for kp_idx in range(n_kp):
        for dim in range(2):
            series = poses[:, kp_idx, dim].copy().astype(float)
            # Treat severe outliers and zero frames as unknown
            severe_mask = (severity[:, kp_idx] > severity_threshold) | zero_frame_mask
            series[severe_mask] = np.nan

            valid = ~np.isnan(series)
            n_valid = int(valid.sum())
            if n_valid < 4:
                continue

            t_valid = t_all[valid]
            try:
                # PchipInterpolator is monotone-preserving (no cubic overshoot)
                # and more robust than CubicSpline for non-uniform gaps between
                # clean frames.  Falls back to linear interpolation if < 4 points.
                if n_valid >= 4:
                    cs = PchipInterpolator(t_valid, series[valid], extrapolate=True)
                else:
                    cs = CubicSpline(t_valid, series[valid], extrapolate=True)
                interp_vals = cs(t_all)
            except Exception:
                interp_vals = np.interp(t_all, t_valid, series[valid])

            # Fill only the severe non-zero frames
            fill_mask = (severity[:, kp_idx] > severity_threshold) & ~zero_frame_mask
            filled[fill_mask, kp_idx, dim] = np.clip(
                interp_vals[fill_mask],
                -1e5, 1e5,   # guard against wild cubic extrapolation
            )

    return filled


def _fix_symmetric_swaps_single_mouse(
    poses: np.ndarray,
    kp_index: Dict[str, int],
    zero_frame_mask: np.ndarray,
    velocity_improvement_ratio: float = 0.70,
    lookback: int = 3,
) -> Tuple[np.ndarray, int]:
    """Detect and fix intra-mouse symmetric keypoint swaps with multi-frame lookback.

    For each pair of bilaterally symmetric keypoints (e.g., ear_left / ear_right,
    hip_left / hip_right), checks whether swapping the pair in the current frame
    gives better temporal velocity continuity with the *average of the last
    ``lookback`` clean frames*.  Using a multi-frame reference suppresses false
    positives caused by single-frame noise spikes.

    Args:
        poses                    : shape (T, K, 2) — poses array (modified in-place).
        kp_index                 : keypoint name → index.
        zero_frame_mask          : shape (T,) bool — frames to skip.
        velocity_improvement_ratio: apply swap if vel_swapped < vel_current × this.
        lookback                 : number of previous clean frames to average as
                                   reference velocity anchor (default 3).

    Returns:
        (poses, n_swaps_applied)
    """
    sym_pairs = [
        ("ear_left",       "ear_right"),
        ("hip_left",       "hip_right"),
        ("paw_front_left", "paw_front_right"),
        ("paw_back_left",  "paw_back_right"),
    ]
    T = poses.shape[0]
    n_swaps = 0

    for (kp_a, kp_b) in sym_pairs:
        ia = kp_index.get(kp_a)
        ib = kp_index.get(kp_b)
        if ia is None or ib is None:
            continue

        for t in range(1, T):
            if zero_frame_mask[t]:
                continue
            a_curr = poses[t, ia]
            b_curr = poses[t, ib]
            if np.any(np.isnan(a_curr)) or np.any(np.isnan(b_curr)):
                continue

            # Collect up to `lookback` previous clean frames as anchor
            prev_a_list, prev_b_list = [], []
            for lag in range(1, lookback + 1):
                tp = t - lag
                if tp < 0 or zero_frame_mask[tp]:
                    break
                pa, pb = poses[tp, ia], poses[tp, ib]
                if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                    break
                prev_a_list.append(pa)
                prev_b_list.append(pb)

            if not prev_a_list:
                continue

            # Multi-frame average anchor (weighted by recency: weight = 1/lag)
            weights = np.array([1.0 / i for i in range(1, len(prev_a_list) + 1)])
            weights /= weights.sum()
            a_ref = np.average(prev_a_list, axis=0, weights=weights)
            b_ref = np.average(prev_b_list, axis=0, weights=weights)

            vel_cur = (np.sum((a_curr - a_ref) ** 2)
                       + np.sum((b_curr - b_ref) ** 2))
            vel_swp = (np.sum((b_curr - a_ref) ** 2)
                       + np.sum((a_curr - b_ref) ** 2))

            if vel_swp < vel_cur * velocity_improvement_ratio:
                poses[t, ia], poses[t, ib] = b_curr.copy(), a_curr.copy()
                n_swaps += 1

    return poses, n_swaps


# ---------------------------------------------------------------------------
# Hard clipping of teleportation spikes (pre-optimisation)
# ---------------------------------------------------------------------------

def _hard_clip_speed_spikes(
    poses: np.ndarray,
    zero_frame_mask: np.ndarray,
    speed_mad_k: float = 3.0,
) -> Tuple[np.ndarray, int]:
    """Replace extreme-velocity keypoints with NaN before the optimiser runs.

    When a keypoint "teleports" (e.g., appears at (0,0) before recovering, or
    jumps 500 px in one frame), the L-BFGS-B optimiser tries to pull it back by
    generating a severely stretched skeleton.  Treating the teleported observation
    as missing (NaN) instead lets the spline initialiser interpolate a physically
    plausible position, giving the optimiser a much better starting point.

    Threshold: ``median_velocity + speed_mad_k × MAD / 0.6745``
    (MAD / 0.6745 converts median absolute deviation to a σ-equivalent estimate).

    Args:
        poses         : shape (T, n_kp, 2) — raw pose array (not modified in-place).
        zero_frame_mask: shape (T,) bool — frames to skip.
        speed_mad_k   : threshold multiplier (default 3.0).  Lower values clip
                        more aggressively.  Recommended range: 2.0–4.0.

    Returns:
        (clipped_poses, n_clipped)  — copy of ``poses`` with spike frames set to
        NaN and count of clipped keypoint-frames.
    """
    T, n_kp, _ = poses.shape
    if T < 3:
        return poses.copy(), 0

    clipped = poses.copy()
    n_clipped = 0

    # Frame-to-frame displacements (T-1, n_kp)
    raw_speeds = np.sqrt(np.sum(np.diff(poses, axis=0) ** 2, axis=2))
    valid_trans = ~zero_frame_mask[1:] & ~zero_frame_mask[:-1]         # (T-1,)
    speeds = np.where(valid_trans[:, None], raw_speeds, np.nan)        # (T-1, n_kp)

    for kp_idx in range(n_kp):
        s = speeds[:, kp_idx]
        valid = ~np.isnan(s)
        if valid.sum() < 5:
            continue
        valid_s = s[valid]
        med = np.median(valid_s)
        mad = np.median(np.abs(valid_s - med))
        mad_sigma = max(mad / 0.6745, 1e-3)   # robust σ-equivalent
        threshold = med + speed_mad_k * mad_sigma

        # speeds[t] = displacement from frame t to frame t+1
        # → a spike at speeds[t] means frame t+1 is the outlier "arrival"
        for t in range(T - 1):
            if np.isnan(s[t]) or zero_frame_mask[t + 1]:
                continue
            if s[t] > threshold:
                clipped[t + 1, kp_idx] = np.nan
                n_clipped += 1

    return clipped, n_clipped


# ---------------------------------------------------------------------------
# Window grouping utility
# ---------------------------------------------------------------------------

def _group_into_windows(
    outlier_t_list: List[int],
    T: int,
    half_win: int = _WINDOW_HALF,
    merge_gap: int = _MERGE_GAP,
) -> List[Tuple[int, int, List[int]]]:
    """Group outlier frame indices into non-overlapping temporal windows.

    Args:
        outlier_t_list : sorted list of frame indices that need optimization.
        T              : total number of frames in the video.
        half_win       : context frames added before the first and after the
                         last outlier in each group.
        merge_gap      : if two consecutive outlier frames are ≤ this many
                         frames apart they are merged into one window.

    Returns:
        List of (t_start, t_end, outlier_indices).
        [t_start, t_end) is the full window slice (outliers + context).
        outlier_indices are the frames whose corrected values will be written
        back to the main ``corrected`` array.
    """
    if not outlier_t_list:
        return []

    windows: List[Tuple[int, int, List[int]]] = []
    i = 0
    while i < len(outlier_t_list):
        w_first    = outlier_t_list[i]
        w_last     = outlier_t_list[i]
        w_outliers = [outlier_t_list[i]]

        # Greedily absorb nearby outlier frames into the same window
        while (i + 1 < len(outlier_t_list)
               and outlier_t_list[i + 1] <= w_last + merge_gap):
            i += 1
            w_last = outlier_t_list[i]
            w_outliers.append(outlier_t_list[i])

        t_start = max(0, w_first - half_win)
        t_end   = min(T, w_last  + half_win + 1)
        windows.append((t_start, t_end, w_outliers))
        i += 1

    return windows


def detect_and_fix_swaps(
    tracking_df: pd.DataFrame,
    kp_index: Dict[str, int],
    skeleton: MouseSkeleton,
    velocity_improvement_ratio: float = 0.60,
    orientation_weight: float = 0.50,
) -> Tuple[pd.DataFrame, int]:
    """Detect and correct identity swaps using predict → hypotheses → reassign.

    Architecture (three explicit stages):

    **Stage 1 — Predict**: for each mouse, predict its expected position at each
    frame using an exponential moving average (EMA) of recent velocities.  This
    is more robust than a simple lookback average for mice with directional motion.

    **Stage 2 — Evaluate hypotheses**: at each frame, compute the assignment cost
    for the current assignment (C₀₀ + C₁₁) and the swapped assignment (C₀₁ + C₁₀)
    where Cᵢⱼ = ‖observed_j − predicted_i‖². Score orientation consistency and
    lookahead temporal continuity for both hypotheses.

    **Stage 3 — Reassign**: instead of greedy frame-by-frame swapping (which gave
    swap_recovery_rate=0% on combined corruption), we accumulate swap votes over a
    sliding window and apply a swap block only when a consistent majority votes for it.
    This resolves ambiguity when both mice are simultaneously corrupted.

    Args:
        tracking_df              : long-format tracking DataFrame.
        kp_index                 : keypoint name → index mapping.
        skeleton                 : fitted MouseSkeleton (used for nose/tail indices).
        velocity_improvement_ratio: apply swap if swapped_cost < current_cost × this.
                                   Lowered to 0.60 (was 0.70) to be less conservative.
        orientation_weight       : relative weight of body-orientation consistency.

    Returns:
        (corrected_df, n_swaps_applied)
    """
    mice = sorted(tracking_df["mouse_id"].unique())
    if len(mice) < 2:
        return tracking_df, 0

    m0, m1 = mice[0], mice[1]

    def _to_poses_array(df: pd.DataFrame, mouse_id) -> Tuple[np.ndarray, List]:
        sub = df[df["mouse_id"] == mouse_id]
        frames = sorted(sub["video_frame"].unique())
        n_f  = len(frames)
        n_kp = len(kp_index)
        P    = np.full((n_f, n_kp, 2), np.nan)
        fi_arr = {f: i for i, f in enumerate(frames)}
        sub_copy = sub.copy()
        sub_copy["_fi"] = sub_copy["video_frame"].map(fi_arr)
        sub_copy["_ki"] = sub_copy["bodypart"].map(kp_index)
        valid = sub_copy["_fi"].notna() & sub_copy["_ki"].notna()
        sv = sub_copy[valid]
        if not sv.empty:
            fi_idx = sv["_fi"].values.astype(np.intp)
            ki_idx = sv["_ki"].values.astype(np.intp)
            P[fi_idx, ki_idx, 0] = sv["x"].values
            P[fi_idx, ki_idx, 1] = sv["y"].values
        return P, frames

    P0, frames0 = _to_poses_array(tracking_df, m0)
    P1, frames1 = _to_poses_array(tracking_df, m1)

    fi0 = {f: i for i, f in enumerate(frames0)}
    fi1 = {f: i for i, f in enumerate(frames1)}
    common = sorted(set(frames0) & set(frames1))
    if not common:
        return tracking_df, 0

    nose_i = kp_index.get("nose")
    tail_i = kp_index.get("tail_base")
    N = len(common)

    # ── Stage 1: Predict using EMA velocity ──────────────────────────────────
    # Exponential moving average of velocity gives more stable predictions than
    # a weighted lookback average, especially during fast motion segments.
    EMA_ALPHA = 0.4   # EMA decay: higher = faster adaptation to recent motion

    def _build_ema_predictions(P: np.ndarray, fi_map: Dict) -> np.ndarray:
        """Build EMA velocity prediction for all common frames.

        Returns pred[N, K, 2]: predicted pose at common[i] using history before it.
        """
        n_kp = P.shape[1]
        pred = np.full((N, n_kp, 2), np.nan)
        ema_vel = np.zeros((n_kp, 2))  # running EMA velocity
        last_valid: Optional[np.ndarray] = None
        last_valid_ci: int = -1

        for ci, frame in enumerate(common):
            if frame not in fi_map:
                continue
            t = fi_map[frame]
            p = P[t]
            is_valid = np.nanmean(np.abs(p)) >= 2.0

            if last_valid is not None and is_valid:
                # Compute instantaneous velocity and update EMA
                inst_vel = p - last_valid          # (K, 2)
                inst_vel[np.any(np.isnan(p), axis=1)] = 0.0
                inst_vel[np.any(np.isnan(last_valid), axis=1)] = 0.0
                dt = max(ci - last_valid_ci, 1)
                ema_vel = EMA_ALPHA * (inst_vel / dt) + (1 - EMA_ALPHA) * ema_vel
                # Prediction = last_valid + EMA_velocity × dt
                pred[ci] = last_valid + ema_vel * dt
                pred[ci][np.any(np.isnan(last_valid), axis=1)] = np.nan

            if is_valid:
                last_valid = p.copy()
                last_valid_ci = ci

        return pred

    pred0 = _build_ema_predictions(P0, fi0)
    pred1 = _build_ema_predictions(P1, fi1)

    # ── Stage 2: Evaluate hypotheses per-frame ───────────────────────────────
    # hypothesis_score[ci] = log(cost_swap / cost_cur):
    #   positive → swap is worse (keep current assignment)
    #   negative → swap is better (vote for swap)
    hypothesis_scores: List[float] = []
    WINDOW_SIZE = 7   # sliding window for majority vote

    for ci in range(1, N):
        frame = common[ci]
        if frame not in fi0 or frame not in fi1:
            hypothesis_scores.append(0.0)
            continue

        t0c = fi0[frame]
        t1c = fi1[frame]
        p0c = P0[t0c]
        p1c = P1[t1c]

        # Skip all-zero frames
        if (np.nanmean(np.abs(p0c)) < 2.0) or (np.nanmean(np.abs(p1c)) < 2.0):
            hypothesis_scores.append(0.0)
            continue

        pr0 = pred0[ci]
        pr1 = pred1[ci]
        if np.all(np.isnan(pr0)) or np.all(np.isnan(pr1)):
            hypothesis_scores.append(0.0)
            continue

        valid = (
            ~np.any(np.isnan(p0c), axis=1)
            & ~np.any(np.isnan(p1c), axis=1)
            & ~np.any(np.isnan(pr0), axis=1)
            & ~np.any(np.isnan(pr1), axis=1)
        )
        if valid.sum() < 3:
            hypothesis_scores.append(0.0)
            continue

        # Assignment costs under two hypotheses
        c00 = float(np.sum((p0c[valid] - pr0[valid]) ** 2))
        c11 = float(np.sum((p1c[valid] - pr1[valid]) ** 2))
        c01 = float(np.sum((p0c[valid] - pr1[valid]) ** 2))
        c10 = float(np.sum((p1c[valid] - pr0[valid]) ** 2))
        cost_cur = c00 + c11
        cost_swp = c01 + c10

        # Cross-animal neck distance check (swap structural signal):
        # If M0-nose is closer to M1-neck than to M0-neck, the nose most
        # likely belongs to M1's body → vote for swap.
        neck_i = kp_index.get("neck")
        cross_animal_bonus = 0.0
        if nose_i is not None and neck_i is not None:
            n0  = p0c[nose_i]
            n1  = p1c[nose_i]
            nk0 = p0c[neck_i]
            nk1 = p1c[neck_i]
            if not any(np.any(np.isnan(v)) for v in [n0, n1, nk0, nk1]):
                d_own   = np.linalg.norm(n0 - nk0) + np.linalg.norm(n1 - nk1)
                d_cross = np.linalg.norm(n0 - nk1) + np.linalg.norm(n1 - nk0)
                # Negative when cross assignment is tighter → votes for swap
                cross_animal_bonus = -np.log(
                    max(d_cross, 1e-6) / max(d_own, 1e-6)
                ) * 0.40

        # Body-orientation consistency bonus
        orient_delta = 0.0
        if nose_i is not None and tail_i is not None:
            def _body_dir(P: np.ndarray, t: int) -> Optional[np.ndarray]:
                n, tb = P[t, nose_i], P[t, tail_i]
                if np.any(np.isnan(n)) or np.any(np.isnan(tb)):
                    return None
                d = n - tb
                nm = np.linalg.norm(d)
                return d / max(nm, 1e-6)

            # Historical body direction for mouse0 (averaged over lookback)
            hist_dirs = [
                _body_dir(P0, fi0[common[ci - lag]])
                for lag in range(1, min(5, ci) + 1)
                if common[ci - lag] in fi0
            ]
            hist_dirs = [d for d in hist_dirs if d is not None]
            if hist_dirs:
                d_ref = np.mean(hist_dirs, axis=0)
                d_ref /= max(np.linalg.norm(d_ref), 1e-6)
                d_cur = _body_dir(P0, t0c)
                d_swp = _body_dir(P1, t1c)  # would become mouse0's orientation
                if d_cur is not None and d_swp is not None:
                    dot_cur = float(np.dot(d_ref, d_cur))
                    dot_swp = float(np.dot(d_ref, d_swp))
                    # Positive when current is more consistent
                    orient_delta = orientation_weight * (dot_cur - dot_swp)

        # Composite score: positive = keep, negative = swap
        # Use log-ratio to make it scale-invariant
        eps = 1.0
        score = (np.log((cost_swp + eps) / (cost_cur + eps))
                 + orient_delta
                 + cross_animal_bonus)
        hypothesis_scores.append(float(score))

    # ── Stage 3: Sliding-window majority vote → apply swap blocks ────────────
    # Instead of greedy frame-by-frame, accumulate votes and apply swap only
    # when a sliding window has consistent majority.  This resolves ambiguity
    # in combined-corruption scenarios (swap_recovery_rate was 0%).
    hypothesis_scores_arr = np.array(hypothesis_scores)  # length N-1 (ci=1..N-1)
    swap_flags = np.zeros(N, dtype=bool)

    HALF_WIN = WINDOW_SIZE // 2
    for ci in range(1, N):
        lo = max(0,   ci - 1 - HALF_WIN)
        hi = min(N - 2, ci - 1 + HALF_WIN)
        window = hypothesis_scores_arr[lo: hi + 1]
        if len(window) == 0:
            continue
        # Majority vote: if mean score < -threshold, swap is favoured.
        # Use nanmean to tolerate any NaN entries in the score window.
        mean_window = float(np.nanmean(window))
        if np.isnan(mean_window):
            continue
        # Majority vote: if mean score < -threshold, swap is favoured
        swap_threshold = -np.log(1.0 / velocity_improvement_ratio + 1e-6)
        if mean_window < swap_threshold:
            swap_flags[ci] = True

    # Resolve swap blocks: track cumulative identity assignment
    # Consecutive swap_flags toggle the active assignment
    identity_flipped = False
    n_swaps = 0
    for ci in range(1, N):
        if swap_flags[ci]:
            identity_flipped = not identity_flipped
            n_swaps += 1

        if identity_flipped:
            frame = common[ci]
            if frame not in fi0 or frame not in fi1:
                continue
            t0c = fi0[frame]
            t1c = fi1[frame]
            tmp = P0[t0c].copy()
            P0[t0c] = P1[t1c].copy()
            P1[t1c] = tmp

    if n_swaps == 0:
        return tracking_df, 0

    # Reconstruct corrected DataFrame
    def _poses_to_df(P, frames, mouse_id):
        n_f, n_k = len(frames), len(kp_index)
        kp_names = list(kp_index.keys())
        kp_idxs  = [kp_index[k] for k in kp_names]
        return pd.DataFrame({
            "video_frame": np.repeat(frames, n_k),
            "mouse_id":    np.full(n_f * n_k, mouse_id),
            "bodypart":    np.tile(kp_names, n_f),
            "x": P[:, kp_idxs, 0].ravel().astype(float),
            "y": P[:, kp_idxs, 1].ravel().astype(float),
        })

    parts = [_poses_to_df(P0, frames0, m0), _poses_to_df(P1, frames1, m1)]
    for om in mice[2:]:
        parts.append(tracking_df[tracking_df["mouse_id"] == om].copy())

    corrected = pd.concat(parts, ignore_index=True)

    # Re-attach extra columns from original (e.g., likelihood)
    extra_cols = [c for c in tracking_df.columns
                  if c not in {"x", "y", "video_frame", "mouse_id", "bodypart"}]
    if extra_cols:
        keys = ["video_frame", "mouse_id", "bodypart"]
        orig_extra = (
            tracking_df[keys + extra_cols]
            .drop_duplicates(subset=keys)
        )
        corrected = corrected.merge(orig_extra, on=keys, how="left")

    return corrected, n_swaps


# ---------------------------------------------------------------------------
# Hampel identifier: distribution-preserving spike correction
# ---------------------------------------------------------------------------

def _hampel_1d(
    x: np.ndarray,
    k: int = 5,
    t0: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hampel identifier on a 1D signal with NaN support.

    For each index ``i`` in ``[k, len(x)-k)``, computes the local median and
    MAD over a ``(2k+1)``-frame window.  If the point's robust z-score exceeds
    ``t0``, it is replaced with the window median.  NaN values are excluded
    from window statistics and left unchanged.

    Parameters
    ----------
    x : (T,) array, may contain NaN
    k : half-window (window width = 2k+1 frames)
    t0 : detection threshold in units of σ_MAD (1.4826×MAD ≈ 1σ Gaussian)

    Returns
    -------
    x_out : corrected copy of x
    changed : boolean mask — True where the value was replaced
    """
    x_out = x.copy()
    changed = np.zeros(len(x), dtype=bool)
    for i in range(k, len(x) - k):
        if np.isnan(x[i]):
            continue
        window = x[i - k: i + k + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 3:
            continue
        med = np.median(valid)
        mad = np.median(np.abs(valid - med))
        if mad == 0.0:
            continue
        if np.abs(x[i] - med) / (1.4826 * mad) > t0:
            x_out[i] = med
            changed[i] = True
    return x_out, changed


def hampel_correct_video(
    tracking_df: pd.DataFrame,
    lab_id: str = "",
    skeleton: Optional[MouseSkeleton] = None,
    hampel_k: int = 5,
    hampel_t0: float = 3.0,
    confidence_threshold: float = 0.0,
) -> Tuple[pd.DataFrame, Dict]:
    """Fix kinematic spikes with the Hampel identifier.

    The conservative alternative to the full optimiser: only frames detected as
    spikes, that is, robust outliers within their local window, are modified.
    Everything else passes through untouched, so the speed distribution, the
    autocorrelation and the power spectrum are preserved by construction.

    That property is why this exists. The global correction shifts the speed
    distribution of `nose` enough to fail a Kolmogorov-Smirnov test, and this
    isolates how much of the downstream damage comes from smoothing everything
    rather than from fixing the genuine errors.

    Per mouse and keypoint:

    1. **Confidence gating**, optional: frames whose tracker confidence is below
       ``confidence_threshold`` are forward-filled before Hampel runs.
    2. **Hampel on x**: window ``2k+1``, threshold ``t0`` in robust MAD units.
    3. **Hampel on y**: the same, independently of x.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Formato largo — columnas requeridas: ``video_frame``, ``mouse_id``,
        ``bodypart``, ``x``, ``y``.
    lab_id : str
        Laboratory identifier, used only when the skeleton has to be built
        esqueleto internamente).
    skeleton : MouseSkeleton, opcional
        Instancia ya ajustada.  Si None, se construye y ajusta internamente.
    hampel_k : int
        Half-window of the identifier, so the full window is ``2k+1`` frames. At
        30 fps, ``k=5`` is about 0.37 s. Default 5.
    hampel_t0 : float
        Detection threshold in robust MAD units. Default 3.0.
    confidence_threshold : float
        Above 0, and when a confidence column exists (``likelihood``,
        ``confidence``, ``score`` or ``prob``), frames below this confidence are
        forward-filled before Hampel runs. 0.0 disables the gating.

    Returns
    -------
    corrected_df : pd.DataFrame
        Long-format frame with the spikes replaced.
    report : dict
        ``n_spikes_corrected``, ``spike_pct``, ``n_conf_gated``,
        ``n_total_frames``, ``hampel_k``, ``hampel_t0``.
    """
    if skeleton is None:
        skeleton = build_skeleton(lab_id)
        skeleton = fit_skeleton(skeleton, tracking_df)

    report: Dict = {
        "n_spikes_corrected": 0,
        "n_conf_gated": 0,
        "n_total_frames": 0,
        "hampel_k": hampel_k,
        "hampel_t0": hampel_t0,
    }

    result_frames: List[pd.DataFrame] = []

    for mouse_id, mouse_df in tracking_df.groupby("mouse_id"):
        mouse_df = mouse_df.sort_values("video_frame").copy()
        report["n_total_frames"] += mouse_df["video_frame"].nunique()

        # Detect confidence column once per mouse
        _lik_col = next(
            (c for c in mouse_df.columns
             if c.lower() in {"likelihood", "confidence", "score", "prob"}),
            None,
        )

        corrected_parts: List[pd.DataFrame] = []

        for bodypart, bp_df in mouse_df.groupby("bodypart"):
            bp_df = bp_df.sort_values("video_frame").copy()
            x_arr = bp_df["x"].values.astype(float)
            y_arr = bp_df["y"].values.astype(float)

            # ── Confidence gating (forward-fill low-confidence frames) ────────
            if confidence_threshold > 0 and _lik_col is not None:
                lik = bp_df[_lik_col].values.astype(float)
                low_conf = lik < confidence_threshold
                n_gated = int(low_conf.sum())
                report["n_conf_gated"] += n_gated
                if n_gated > 0:
                    last_x, last_y = x_arr[0], y_arr[0]
                    for i in range(len(x_arr)):
                        if low_conf[i]:
                            x_arr[i] = last_x
                            y_arr[i] = last_y
                        else:
                            last_x, last_y = x_arr[i], y_arr[i]

            # ── Hampel on x and y independently ──────────────────────────────
            x_corr, chg_x = _hampel_1d(x_arr, k=hampel_k, t0=hampel_t0)
            y_corr, chg_y = _hampel_1d(y_arr, k=hampel_k, t0=hampel_t0)
            report["n_spikes_corrected"] += int((chg_x | chg_y).sum())

            bp_df = bp_df.copy()
            bp_df["x"] = x_corr
            bp_df["y"] = y_corr
            corrected_parts.append(bp_df)

        result_frames.append(pd.concat(corrected_parts, ignore_index=True))

    corrected_df = pd.concat(result_frames, ignore_index=True)
    n_total_kp_frames = max(report["n_total_frames"] * tracking_df["bodypart"].nunique(), 1)
    report["spike_pct"] = round(
        report["n_spikes_corrected"] / n_total_kp_frames * 100, 2
    )
    return corrected_df, report


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------

def smooth_video(
    tracking_df: pd.DataFrame,
    lab_id: str = "",
    skeleton: Optional[MouseSkeleton] = None,
    speed_sigma: float = SPEED_SIGMA_THRESHOLD,
    length_threshold: float = LENGTH_VAR_THRESHOLD,
    lambda_l: float = DEFAULT_LAMBDA_L,
    lambda_t: float = DEFAULT_LAMBDA_T,
    lambda_acc: float = DEFAULT_LAMBDA_ACC,
    lambda_angle: float = DEFAULT_LAMBDA_ANGLE,
    lambda_jerk: float = DEFAULT_LAMBDA_JERK,
    lambda_angle_cont: float = 5.0,
    auto_edge_lambdas: bool = True,
    pre_smooth: bool = True,
    pre_smooth_window: int = 0,
    outlier_adhesion: float = 0.05,
    detection_n_sigma: float = 3.0,
    n_passes: int = 5,
    swap_detection: bool = True,
    confidence_sigma: float = 0.35,
    adaptive_thresholds_k: float = 2.5,
    speed_mad_k: float = 3.0,
    trunk_lambda_boost: float = 2.5,
    max_displacement_factor: float = 0.0,
    reference_df: Optional[pd.DataFrame] = None,
    non_outlier_adhesion_scale: float = 1.0,
    keypoint_adhesion_floor: Optional[Dict[str, float]] = None,
    per_mouse_skeletons: Optional[Dict[int, MouseSkeleton]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """Correct the keypoints of a whole video.

    Parameters
    ----------
    tracking_df : pd.DataFrame
        Long-format tracking for a single video.
    lab_id : str
        Laboratory identifier.
    skeleton : MouseSkeleton, opcional
        An already-fitted instance. When None, one is built and fitted here.
    speed_sigma : float
        Speed outlier threshold, as a multiple of the robust sigma.
    length_threshold : float
        Length variation threshold used for *reporting*, as a fraction, e.g. 0.20.
        Note this is deliberately not the same cutoff that triggers a correction.
    lambda_l : float
        Base weight of the length constraint. With ``auto_edge_lambdas=True`` this
        is the maximum, given to the stiffest edge; the rest scale by variance.
    lambda_t : float
        Weight of the first-order temporal smoothing.
    lambda_acc : float
        Weight of the acceleration regulariser, the second difference. Default 2.0.
        Valores recomendados: 1.0–5.0. Suprime jitter y mejora coherencia temporal.
    lambda_angle : float
        Weight of the body-collinearity angular constraints. Default 5.0. The
        ablation in notebook 02a found this term is the one that hurts: removing it
        lowers both the violation rate and the jerk energy.
        Recommended range 2.0 to 20.0. Penalises nose-neck-tail misalignment.
    lambda_jerk : float
        Weight of the jerk penalty, the third difference. Default 0.5. Suppresses
        residual micro-jitter left by the correction.
    lambda_angle_cont : float
        Weight of angular continuity between frames. Default 2.0. Penalises sudden
        reversals of the body axis, which usually mean an identity swap rather than
        a real turn.
    auto_edge_lambdas : bool
        When True, recommended, lambda is set per edge by inverse variance: stiff
        edges get more weight, flexible segments less. When False, ``lambda_l`` is
        applied uniformly, which lets the optimiser damage the flexible edges while
        chasing the rigid ones.
    pre_smooth : bool
        Apply a Savitzky-Golay filter before detection.
    outlier_adhesion : float
        Minimum adhesion for the most severely flagged keypoints, in 0 to 1.
        Default 0.6.
        Valores bajos (0.1–0.3) permiten correcciones muy agresivas;
        High values, 0.7 to 0.9, keep the keypoint close to what was observed.
        At 0.6, restoring valid geometry is prioritised over fidelity to the
        observations.
    detection_n_sigma : float
        Length detection threshold, as a multiple of the segment's natural sigma.
    n_passes : int
        Optimisation passes per frame. Default 3.
        Pass 0 is geometry only, pass 1 ramps in the temporal terms, and pass 2
        onwards runs the full objective.
        With staged=True the three stages fix the loss-increase-between-passes
        problem that the two-pass version had.
    swap_detection : bool
        When True, recommended, detect and repair identity swaps before the
        frame-by-frame optimisation. Only active with two or more mice.
    confidence_sigma : float
        The sigma in the confidence formula alpha_t = exp(-r_t / sigma), which
        controls how fast trust in the observation decays as the residual grows.
        Default 0.35, giving alpha about 0.05 at maximum severity, that is, a
        dropout.
    adaptive_thresholds_k : float
        The k in the per-video adaptive threshold tau_e = mu_e + k * sigma_e, which
        calibrates detection to this video's own variability rather than to a
        constant that suits one recording.
        Typical values run from 2.0 to 3.5. Defaults to 2.5.
    max_displacement_factor : float
        Above 0, caps each correction at max_displacement_factor times the 95th
        percentile of that keypoint's natural displacement. Without it the length
        optimiser inflates the apparent speed of keypoints such as tail_base while
        satisfying the geometry, which is precisely the jitter that damages
        downstream classification. 0.0 disables the cap.
        (kept for backward compatibility). Recommended value: 2.0.
    reference_df : pd.DataFrame | None
        The original raw tracking, before any correction, in long format. When
        given, the displacement guard measures the offset from the **original**
        positions rather than from the previous pass. This matters in the multi-pass
        pipeline (v1 to v2 to v3): without it, drift accumulated by earlier passes
        eats the budget and the guard stops preventing the speed spikes it exists
        to prevent.
    non_outlier_adhesion_scale : float
        Adhesion multiplier for non-outlier keypoints in frames that contain
        outliers.  1.0 = comportamiento original; valores altos (300–600)
        stop innocent keypoints being dragged along by the length constraints when
        their neighbours are corrected.
    keypoint_adhesion_floor : dict[str, float] | None
        Per-keypoint adhesion floor, applied even when the keypoint is flagged as
        an outlier. See the ``_smooth_mouse`` docstring.
        ``n_outlier_frames``, ``n_total_frames``, ``edge_lambdas`` (array),
        ``n_pre_swaps_corrected``, ``mean_displacement_px``,
        ``p95_displacement_px``, ``convergence_loss_history``.
    """
    if skeleton is None:
        skeleton = build_skeleton(lab_id)
        skeleton = fit_skeleton(skeleton, tracking_df)

    kp_index = {kp: i for i, kp in enumerate(skeleton.keypoints)}

    # Compute the per-edge lambdas once
    if auto_edge_lambdas:
        edge_lambdas = compute_edge_lambdas(skeleton, base_lambda_l=lambda_l)
    else:
        edge_lambdas = np.full(len(skeleton.edges), lambda_l)

    report: Dict = {"edge_lambdas": edge_lambdas}

    # ── Pre-processing: detect and repair identity swaps ─────────────────────
    # Runs before the frame-by-frame optimisation, and only with multiple mice.
    # A swap left in place makes every downstream geometric constraint wrong.
    if swap_detection and tracking_df["mouse_id"].nunique() >= 2:
        tracking_df, n_pre_swaps = detect_and_fix_swaps(
            tracking_df, kp_index=kp_index, skeleton=skeleton
        )
        report["n_pre_swaps_corrected"] = n_pre_swaps
    else:
        report["n_pre_swaps_corrected"] = 0

    result_frames = []

    # ── Overlap mask (foreshortening / virtual height) ────────────────────────
    # Detect per-frame inter-mouse proximity (huddle / mounting) before running
    # the per-mouse optimiser.  During overlap frames the visible skeleton is
    # foreshortened (one animal elevated over the other), so the length-constraint
    # weight is relaxed inside _smooth_mouse to avoid over-correcting geometry.
    all_frames_global = sorted(tracking_df["video_frame"].unique())
    _global_fi_map = {f: i for i, f in enumerate(all_frames_global)}
    _l_body_for_overlap = skeleton.L_body_median_px
    overlap_masks = _build_overlap_masks(
        tracking_df, all_frames_global, _l_body_for_overlap
    ) if tracking_df["mouse_id"].nunique() >= 2 else {}

    for mouse_id, mouse_df in tracking_df.groupby("mouse_id"):
        # Pre-align overlap mask to this mouse's local frame order so that
        # _smooth_mouse receives a (T_local,) array ready for direct indexing.
        _mouse_frames = sorted(mouse_df["video_frame"].unique())
        if mouse_id in overlap_masks:
            _raw_mask = overlap_masks[mouse_id]
            _mouse_overlap: Optional[np.ndarray] = np.array(
                [bool(_raw_mask[_global_fi_map[f]])
                 if f in _global_fi_map else False
                 for f in _mouse_frames],
                dtype=bool,
            )
        else:
            _mouse_overlap = None

        # ── Build reference_poses array aligned to this mouse's local frames ─
        # When reference_df is provided (e.g. df_sample for a v3 pass that starts
        # from df_corrected_v2), extract and align its poses so the displacement
        # guard measures from the original raw observations, not from the v2 output.
        _ref_poses: Optional[np.ndarray] = None
        if reference_df is not None and max_displacement_factor > 0:
            _ref_mouse = reference_df[reference_df["mouse_id"] == mouse_id]
            if not _ref_mouse.empty:
                _ref_fi = {f: i for i, f in enumerate(_mouse_frames)}
                _ref_poses = np.full((len(_mouse_frames), len(skeleton.keypoints), 2), np.nan)
                _ref_mc = _ref_mouse.copy()
                _ref_mc["_fi"] = _ref_mc["video_frame"].map(_ref_fi)
                _ref_mc["_ki"] = _ref_mc["bodypart"].map(kp_index)
                _valid_ref = _ref_mc["_fi"].notna() & _ref_mc["_ki"].notna()
                _sv_ref = _ref_mc[_valid_ref]
                if not _sv_ref.empty:
                    _fi_r = _sv_ref["_fi"].values.astype(np.intp)
                    _ki_r = _sv_ref["_ki"].values.astype(np.intp)
                    _ref_poses[_fi_r, _ki_r, 0] = _sv_ref["x"].values
                    _ref_poses[_fi_r, _ki_r, 1] = _sv_ref["y"].values

        # ── Per-mouse skeleton, for the per-mouse reference ablation ─────────
        # With per-mouse skeletons, each individual is corrected against its own
        # bone-length reference (L_ref, L_std) instead of the pooled one. The
        # topology is identical, so the inter-mouse steps (swap, overlap) still use
        # `skeleton`; only the per-mouse geometric constraints and edge weights
        # change. The ablation found this does not rescue the correction: the
        # damage comes from relocation jitter, not from the shared reference.
        if per_mouse_skeletons is not None and int(mouse_id) in per_mouse_skeletons:
            _sk_m = per_mouse_skeletons[int(mouse_id)]
            _edge_lambdas_m = (
                compute_edge_lambdas(_sk_m, base_lambda_l=lambda_l)
                if auto_edge_lambdas else np.full(len(_sk_m.edges), lambda_l)
            )
        else:
            _sk_m = skeleton
            _edge_lambdas_m = edge_lambdas

        corrected_mouse = _smooth_mouse(
            mouse_df=mouse_df.sort_values("video_frame").copy(),
            skeleton=_sk_m,
            kp_index=kp_index,
            speed_sigma=speed_sigma,
            length_threshold=length_threshold,
            edge_lambdas=_edge_lambdas_m,
            lambda_l=lambda_l,
            lambda_t=lambda_t,
            lambda_acc=lambda_acc,
            lambda_angle=lambda_angle,
            lambda_jerk=lambda_jerk,
            lambda_angle_cont=lambda_angle_cont,
            pre_smooth=pre_smooth,
            pre_smooth_window=pre_smooth_window,
            outlier_adhesion=outlier_adhesion,
            detection_n_sigma=detection_n_sigma,
            n_passes=n_passes,
            report=report,
            confidence_sigma=confidence_sigma,
            adaptive_thresholds_k=adaptive_thresholds_k,
            overlap_mask=_mouse_overlap,
            speed_mad_k=speed_mad_k,
            trunk_lambda_boost=trunk_lambda_boost,
            max_displacement_factor=max_displacement_factor,
            reference_poses=_ref_poses,
            non_outlier_adhesion_scale=non_outlier_adhesion_scale,
            keypoint_adhesion_floor=keypoint_adhesion_floor,
        )
        result_frames.append(corrected_mouse)

    corrected_df = pd.concat(result_frames, ignore_index=True)

    # ── Post-optimization identity verification ───────────────────────────────
    # After kinematic correction, check whether persistent identity swaps remain
    # by comparing body-length size proxies around huddle events.  A rank inversion
    # (one mouse suddenly becomes larger while the other shrinks) signals that the
    # optimizer fixed the geometry but left the identity label wrong.
    if swap_detection and tracking_df["mouse_id"].nunique() >= 2:
        corrected_df, n_post_swaps = _post_correction_identity_check(
            corrected_df, kp_index
        )
        report["n_post_identity_swaps"] = n_post_swaps
    else:
        report["n_post_identity_swaps"] = 0

    # ── Aggregate the displacement metrics ───────────────────────────────────
    # Collapse each mouse's running totals into global scalars
    _all_disp = report.pop("_displacement_list", [])
    if _all_disp:
        _disp_arr = np.concatenate(_all_disp)
        report["mean_displacement_px"]  = float(np.nanmean(_disp_arr))
        report["p95_displacement_px"]   = float(np.nanpercentile(_disp_arr, 95))
        report["max_displacement_px"]   = float(np.nanmax(_disp_arr))
    else:
        report["mean_displacement_px"] = 0.0
        report["p95_displacement_px"]  = 0.0
        report["max_displacement_px"]  = 0.0

    # Consolidate the convergence history, averaging the loss per pass
    _conv_lists = report.pop("_convergence_lists", [])
    if _conv_lists:
        max_passes = max(len(h) for h in _conv_lists)
        conv_avg = []
        for p in range(max_passes):
            vals = [h[p] for h in _conv_lists if p < len(h)]
            conv_avg.append(float(np.mean(vals)) if vals else np.nan)
        report["convergence_loss_history"] = conv_avg
    else:
        report["convergence_loss_history"] = []

    # ── Convergence diagnostics ───────────────────────────────────────────────
    _iters = report.pop("_n_opt_iters_list", [])
    if _iters:
        report["mean_opt_iters"] = float(np.mean(_iters))
        report["max_opt_iters"]  = int(np.max(_iters))
    else:
        report["mean_opt_iters"] = 0.0
        report["max_opt_iters"]  = 0
    report["n_failed_frames"] = report.pop("_n_failed_frames", 0)
    n_opt = report.get("n_outlier_frames", 0)
    if n_opt > 0:
        report["convergence_rate_pct"] = round(
            (1 - report["n_failed_frames"] / n_opt) * 100, 1
        )
    else:
        report["convergence_rate_pct"] = 100.0

    return corrected_df, report


# ---------------------------------------------------------------------------
# Per-mouse processing
# ---------------------------------------------------------------------------

def _smooth_mouse(
    mouse_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    speed_sigma: float,
    length_threshold: float,
    edge_lambdas: np.ndarray,
    lambda_l: float,
    lambda_t: float,
    lambda_acc: float,
    lambda_angle: float,
    lambda_jerk: float,
    lambda_angle_cont: float,
    pre_smooth: bool,
    outlier_adhesion: float,
    detection_n_sigma: float,
    n_passes: int,
    report: Dict,
    pre_smooth_window: int = 0,
    auto_edge_lambdas: bool = True,
    confidence_sigma: float = 0.35,
    adaptive_thresholds_k: float = 2.5,
    overlap_mask: Optional[np.ndarray] = None,
    speed_mad_k: float = 3.0,
    trunk_lambda_boost: float = 2.5,
    max_displacement_factor: float = 0.0,
    reference_poses: Optional[np.ndarray] = None,
    non_outlier_adhesion_scale: float = 1.0,
    keypoint_adhesion_floor: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Detect and correct outliers for a single mouse.

    reference_poses : np.ndarray | None
        When given, shape ``(T, n_kp, 2)``, the displacement guard measures from
        these positions instead of from ``poses_original``, the smoother's own
        input. Passing the raw tracking guarantees the cap is measured from the
        original observation, so successive passes cannot accumulate drift.
    non_outlier_adhesion_scale : float
        Adhesion weight for NON-outlier keypoints in frames that contain outliers,
        normally 1.0. Raise it, say to 300 or 600, when the optimiser drags innocent
        keypoints such as tail_base while correcting their neighbours hip_left and
        hip_right. That dragging inflates apparent speed and is what makes the
        Kolmogorov-Smirnov test fail. At a high value the non-outlier keypoints stay
        essentially where they were observed, even when the length constraint pulls
        at them.
    keypoint_adhesion_floor : dict[str, float] | None
        Per-keypoint adhesion FLOOR, applied even when the keypoint is flagged as
        an outlier. Useful for stable keypoints such as tail_base, which are often
        flagged as collateral outliers because an edge like hip_left to tail_base is
        violated, when the real error is at the hip. With a high floor, say 200, the
        optimiser moves the other end of the edge instead, removing speed spikes
        that would otherwise fail the Kolmogorov-Smirnov test.
        Ejemplo: ``{'tail_base': 200.0}``.
    """
    wide = _to_wide(mouse_df)
    frames = sorted(wide["video_frame"].unique())
    n_kp = len(skeleton.keypoints)

    T = len(frames)
    frame_idx = {f: i for i, f in enumerate(frames)}
    poses = np.full((T, n_kp, 2), np.nan)

    # overlap_mask, when provided, is already aligned to this mouse's local
    # frame list (pre-aligned by smooth_video before calling here), so its
    # length equals T.  If it somehow doesn't match, disable to avoid errors.
    if overlap_mask is not None and len(overlap_mask) != T:
        overlap_mask = None

    # ── Vectorized pose loading (replaces iterrows inner loop) ────────────────
    mouse_df_copy = mouse_df.copy()
    mouse_df_copy["_fi"] = mouse_df_copy["video_frame"].map(frame_idx)
    mouse_df_copy["_ki"] = mouse_df_copy["bodypart"].map(kp_index)
    _valid = mouse_df_copy["_fi"].notna() & mouse_df_copy["_ki"].notna()
    _sv = mouse_df_copy[_valid]
    if not _sv.empty:
        _fi_idx = _sv["_fi"].values.astype(np.intp)
        _ki_idx = _sv["_ki"].values.astype(np.intp)
        poses[_fi_idx, _ki_idx, 0] = _sv["x"].values
        poses[_fi_idx, _ki_idx, 1] = _sv["y"].values

    # ── Likelihood array (T, n_kp) from tracker confidence column ────────────
    # If the tracking data carries a per-keypoint confidence (e.g. DLC likelihood),
    # build a (T, n_kp) array in [0, 1] to modulate observation adhesion weights.
    # Missing values default to 1.0 (full trust — no tracker signal).
    _lik_col = next(
        (c for c in mouse_df.columns
         if c.lower() in {"likelihood", "confidence", "score", "prob"}),
        None,
    )
    likelihoods: Optional[np.ndarray] = None
    if _lik_col is not None and not _sv.empty:
        likelihoods = np.ones((T, n_kp), dtype=float)
        _lik_vals = _sv[_lik_col].values.astype(float) if _lik_col in _sv.columns else None
        if _lik_vals is not None:
            np.clip(_lik_vals, 0.0, 1.0, out=_lik_vals)
            likelihoods[_fi_idx, _ki_idx] = _lik_vals

    # ── Mask of zero frames, which carry no real tracking ────────────────────
    # Frames where EVERY keypoint is (0,0) are placeholders, not data. Correcting
    # them would invent a pose out of nothing.
    zero_frame_mask = np.all(
        (poses[:, :, 0] == 0) & (poses[:, :, 1] == 0), axis=1
    )  # shape (T,)

    # ── Per-video L_body estimation (P5: adaptive cross-video normalisation) ──
    # Re-compute l_body_px from THIS video's data instead of the global training
    # median.  This handles videos where mice differ in size from training data.
    _nose_idx_lbody = kp_index.get("nose")
    _tail_idx_lbody = kp_index.get("tail_base")
    if _nose_idx_lbody is not None and _tail_idx_lbody is not None:
        _nt_dists = np.sqrt(np.sum(
            (poses[:, _nose_idx_lbody] - poses[:, _tail_idx_lbody]) ** 2, axis=1
        ))
        _valid_dists = _nt_dists[
            (_nt_dists > skeleton.L_body_median_px * 0.25) & ~zero_frame_mask
        ]
        l_body_px = float(np.median(_valid_dists)) if len(_valid_dists) > 10 \
            else skeleton.L_body_median_px
    else:
        l_body_px = skeleton.L_body_median_px

    # ── Savitzky-Golay pre-smoothing, on non-zero frames only ────────────────
    # pre_smooth_window overrides the global default when the caller needs a
    # wider kernel (e.g. under high Gaussian noise, window=9 vs default 5).
    _sg_window = pre_smooth_window if pre_smooth_window > 0 else SAVGOL_WINDOW
    if pre_smooth and T > _sg_window:
        # Process all (n_kp × 2) coordinate series in a single loop pass.
        # Reshaping avoids the previous double loop (n_kp iterations × 2 dims).
        _sg_data = poses.reshape(T, n_kp * 2).copy()  # (T, n_kp*2)
        _sg_data[zero_frame_mask] = np.nan
        for _col in range(n_kp * 2):
            _ser = _sg_data[:, _col]
            _val = ~np.isnan(_ser)
            if _val.sum() > _sg_window:
                _ser[_val] = savgol_filter(
                    _ser[_val], _sg_window, SAVGOL_POLYORDER, mode="nearest"
                )
        # Write smoothed values back, preserving zero-frame originals
        _sg_poses = _sg_data.reshape(T, n_kp, 2)
        poses = np.where(zero_frame_mask[:, None, None], poses, _sg_poses)

    # ── Hard clipping: replace teleportation spikes with NaN ─────────────────
    # Keypoints that move more than speed_mad_k × MAD-sigma from the median
    # velocity are treated as missing values.  The downstream spline initialiser
    # will interpolate a plausible position, preventing the optimiser from
    # creating "stretched" skeletons to satisfy an absurdly far observation.
    poses, _n_hard_clipped = _hard_clip_speed_spikes(
        poses, zero_frame_mask, speed_mad_k=speed_mad_k
    )
    report["n_hard_clipped"] = report.get("n_hard_clipped", 0) + _n_hard_clipped

    # ── Trunk edge lambda boosting ────────────────────────────────────────────
    # The main body axis (nose-neck, neck-hip, hip-tail_base) is the most
    # anatomically rigid part of the mouse.  Multiply their λ_L by
    # trunk_lambda_boost to lock down the trunk while keeping ears/tail flexible.
    _TRUNK_EDGES = {
        ("nose",      "neck"),
        ("neck",      "hip_left"),
        ("neck",      "hip_right"),
        ("hip_left",  "tail_base"),
        ("hip_right", "tail_base"),
        ("neck",      "body_center"),
        ("body_center", "tail_base"),
    }
    _boosted_edge_lambdas = edge_lambdas.copy()
    for _ei, _edge in enumerate(skeleton.edges):
        if (_edge.src, _edge.dst) in _TRUNK_EDGES or (_edge.dst, _edge.src) in _TRUNK_EDGES:
            _boosted_edge_lambdas[_ei] = min(
                _boosted_edge_lambdas[_ei] * trunk_lambda_boost,
                edge_lambdas.max() * trunk_lambda_boost,
            )
    edge_lambdas = _boosted_edge_lambdas

    # ── Adaptive λ_edge softening under high-frequency noise ─────────────────
    # When the input signal has high-frequency velocity variance (e.g., Gaussian
    # noise σ≥5 px), rigid edge constraints push the optimizer to over-correct
    # and can create new violations instead of fixing existing ones.
    # Heuristic: if the mean variance of keypoint velocity exceeds the threshold
    # (≈5 px σ noise on a typical 30 fps recording), scale down edge lambdas
    # to let the pre-filter absorb the noise rather than the optimizer.
    _HIGH_NOISE_VAR_THRESHOLD = 25.0   # px² ≈ σ=5px Gaussian noise
    _NOISE_LAMBDA_SOFTENING = 0.6      # scale factor when noise detected
    if T > 1:
        _vel2 = np.diff(poses, axis=0) ** 2  # (T-1, n_kp, 2)
        _vel_var = float(np.nanmean(_vel2))
        if _vel_var > _HIGH_NOISE_VAR_THRESHOLD:
            edge_lambdas = edge_lambdas * _NOISE_LAMBDA_SOFTENING

    # ── Per-video L_ref re-estimation ────────────────────────────────────────
    # Uses this video's clean frames to calibrate each edge's reference length,
    # replacing global training priors.  Addresses the σ > threshold problem:
    # when global L_ref differs from the video's actual proportions, almost
    # every frame registers as a violation even with perfectly valid motion.
    skeleton = _reestimate_lref_per_video(
        skeleton, poses, kp_index, l_body_px, zero_frame_mask
    )

    # ── Recompute edge_lambdas from data-driven L_std (CRITICAL #1) ──────────
    # _reestimate_lref_per_video now updates both L_ref and L_std.  Recompute
    # edge_lambdas (∝ 1/L_std²) so per-video bone stiffness priors replace
    # the fixed 5% global default, then re-apply trunk boost and noise softening.
    if auto_edge_lambdas:
        edge_lambdas = compute_edge_lambdas(skeleton, base_lambda_l=lambda_l)
        # Re-apply trunk boost with updated lambdas
        _boosted_redo = edge_lambdas.copy()
        for _ei, _edge in enumerate(skeleton.edges):
            if ((_edge.src, _edge.dst) in _TRUNK_EDGES
                    or (_edge.dst, _edge.src) in _TRUNK_EDGES):
                _boosted_redo[_ei] = min(
                    _boosted_redo[_ei] * trunk_lambda_boost,
                    edge_lambdas.max() * trunk_lambda_boost,
                )
        edge_lambdas = _boosted_redo
        # Re-apply noise softening if high-noise regime detected earlier
        if T > 1 and float(np.nanmean(np.diff(poses, axis=0) ** 2)) > _HIGH_NOISE_VAR_THRESHOLD:
            edge_lambdas = edge_lambdas * _NOISE_LAMBDA_SOFTENING

    # ── Per-frame mean speed (for velocity-adaptive lambda_l) ────────────────
    # Computes the mean keypoint speed for every frame.  Used later in the
    # per-frame correction loop to reduce geometric rigidity during fast motion:
    # when a frame exceeds the video's 75th-percentile speed the length-constraint
    # weight is scaled down, allowing the optimizer to track rapid movements
    # without generating implausible anatomical corrections.
    _frame_mean_speed = np.zeros(T, dtype=float)
    if T > 1:
        _raw_spd = np.sqrt(np.sum(np.diff(poses, axis=0) ** 2, axis=2))  # (T-1, n_kp)
        _valid_trans = ~zero_frame_mask[1:] & ~zero_frame_mask[:-1]       # (T-1,)
        _raw_spd_masked = np.where(_valid_trans[:, None], _raw_spd, np.nan)
        # Use nansum/count instead of nanmean to avoid RuntimeWarning on fully
        # masked rows (transitions where both frames are zero-frame placeholders).
        _kp_count = np.sum(np.isfinite(_raw_spd_masked), axis=1)           # (T-1,)
        _kp_sum   = np.nansum(_raw_spd_masked, axis=1)                     # (T-1,)
        # Avoid dividing for rows where _kp_count==0 or _kp_sum is NaN/inf.
        # np.where evaluates both branches eagerly, so any NaN/zero in the
        # division operands triggers RuntimeWarning even when that row is
        # masked out.  Index-assignment only executes the division where safe.
        _mean_spd = np.zeros(len(_kp_count), dtype=float)
        _valid_spd = (_kp_count > 0) & np.isfinite(_kp_sum)
        _mean_spd[_valid_spd] = _kp_sum[_valid_spd] / _kp_count[_valid_spd]
        _frame_mean_speed[1:] = _mean_spd
    _nonzero_speeds = _frame_mean_speed[_frame_mean_speed > 0]
    _speed_p75 = (float(np.percentile(_nonzero_speeds, 75))
                  if len(_nonzero_speeds) > 10 else np.inf)

    # ── Outlier detection ────────────────────────────────────────────────────
    # Compute per-video adaptive detection thresholds BEFORE outlier detection.
    # τ_e = μ_e + k·σ_e calibrates each edge to this video's natural variability,
    # preventing over-detection (84% violations from 5px noise at fixed thresholds).
    adaptive_thresholds = compute_adaptive_edge_thresholds(
        poses=poses,
        skeleton=skeleton,
        kp_index=kp_index,
        l_body_px=l_body_px,
        k=adaptive_thresholds_k,
        zero_frame_mask=zero_frame_mask,
    )
    report.setdefault("adaptive_thresholds", {}).update(adaptive_thresholds)

    outlier_frames_mask, severity_scores = _detect_outliers(
        poses=poses,
        skeleton=skeleton,
        kp_index=kp_index,
        speed_sigma=speed_sigma,
        length_threshold=length_threshold,
        l_body_px=l_body_px,
        zero_frame_mask=zero_frame_mask,
        detection_n_sigma=detection_n_sigma,
        adaptive_thresholds=adaptive_thresholds,
    )   # shape (T, n_kp) bool, float

    n_outlier_frames = int(np.any(outlier_frames_mask, axis=1).sum())
    report.setdefault("n_outlier_frames", 0)
    report.setdefault("n_total_frames", 0)
    report["n_outlier_frames"] += n_outlier_frames
    report["n_total_frames"] += T - int(zero_frame_mask.sum())  # excluir zeros

    # ── Angular triplets, computed once per mouse ────────────────────────────
    angle_triplets = (
        build_angle_triplets(skeleton, kp_index)
        if (lambda_angle > 0.0 or lambda_angle_cont > 0.0) else None
    )

    # ── Pre-compute the body-axis cross product for angular continuity ───────
    _nose_idx  = kp_index.get("nose")
    _neck_idx  = kp_index.get("neck")
    _tail_idx  = kp_index.get("tail_base")
    angle_cross_series: Optional[np.ndarray] = None
    if (lambda_angle_cont > 0.0 and _nose_idx is not None
            and _neck_idx is not None and _tail_idx is not None):
        angle_cross_series = np.full(T, np.nan)
        _l2 = max(l_body_px, 1.0) ** 2
        # Vectorized cross-product computation
        _n_pts  = poses[:, _nose_idx]
        _c_pts  = poses[:, _neck_idx]
        _tb_pts = poses[:, _tail_idx]
        _u_vecs = _n_pts - _c_pts    # (T, 2)
        _v_vecs = _tb_pts - _c_pts   # (T, 2)
        _cross  = _u_vecs[:, 0] * _v_vecs[:, 1] - _u_vecs[:, 1] * _v_vecs[:, 0]
        _any_nan = (
            np.any(np.isnan(_n_pts),  axis=1)
            | np.any(np.isnan(_c_pts),  axis=1)
            | np.any(np.isnan(_tb_pts), axis=1)
            | zero_frame_mask
        )
        angle_cross_series = np.where(_any_nan, np.nan, _cross / _l2)

    # ── Spline initialisation for dropouts ───────────────────────────────────
    # For high-severity frames, that is, dropouts, replace the (0,0) starting point
    # with a cubic interpolation through the surrounding clean frames. This gives
    # the optimiser a far better x0 than copying the previous frame, which is what
    # it would otherwise start from and which biases the result towards standing
    # still.
    poses_init = _init_spline_dropouts(poses, severity_scores, zero_frame_mask)

    # ── Intra-mouse symmetric swap detection ─────────────────────────────────
    # Fix bilaterally symmetric keypoint swaps (e.g., ear_left ↔ ear_right) that
    # occur when the detector assigns the wrong identity to a pair.  This runs on
    # the spline-initialized poses so that clean temporal continuity guides detection.
    poses_init, n_sym_swaps = _fix_symmetric_swaps_single_mouse(
        poses_init, kp_index, zero_frame_mask
    )
    report.setdefault("n_symmetric_swaps_corrected", 0)
    report["n_symmetric_swaps_corrected"] += n_sym_swaps

    # ── Keep the original poses for the displacement diagnostic ──────────────
    poses_original = poses.copy()

    # ── Natural p95 displacement per keypoint (used by displacement guard) ───
    _nat_p95_per_kp = np.full(n_kp, l_body_px * 0.4)  # fallback: 40% of body
    if max_displacement_factor > 0:
        for _ki in range(n_kp):
            _xy = poses_original[:, _ki, :]
            _vf = ~np.isnan(_xy[:, 0]) & ~zero_frame_mask
            _xy_v = _xy[_vf]
            if len(_xy_v) > 10:
                _dx = np.diff(_xy_v[:, 0])
                _dy = np.diff(_xy_v[:, 1])
                _disps = np.sqrt(_dx ** 2 + _dy ** 2)
                _clean = _disps[_disps > 0]
                if len(_clean) > 5:
                    _nat_p95_per_kp[_ki] = max(float(np.percentile(_clean, 95)), 1.0)

    # ── Correction: dropouts by spline, then the temporal window optimiser ───
    # Separate outlier frames into two classes:
    #   dropout frames  (all outlier kp have severity ≥ 0.95) → spline directly
    #   window  frames  (everything else)                      → windowed optimizer
    # The windowed optimizer processes groups of nearby outlier frames jointly
    # with _WINDOW_HALF clean context frames on each side acting as temporal
    # anchors.  This makes swaps, drift, and speed spikes detectable because
    # a swap that looks geometrically valid in one frame becomes inconsistent
    # over the trajectory spanned by the full window.
    corrected = poses.copy()
    frame_loss_histories: List[List[float]] = []

    _adh_floor = max(outlier_adhesion * 0.05, 0.02)
    _l2 = max(l_body_px, 1.0) ** 2  # normalization (also used in angle_cross update)

    _dropout_ts: List[int] = []
    _window_ts:  List[int] = []
    for t in range(T):
        if zero_frame_mask[t] or not np.any(outlier_frames_mask[t]):
            continue
        outlier_kp_indices = np.where(outlier_frames_mask[t])[0]
        if (len(outlier_kp_indices) > 0
                and np.all(severity_scores[t, outlier_kp_indices] >= 0.95)):
            _dropout_ts.append(t)
        else:
            _window_ts.append(t)

    # ── Phase 1: pure dropout frames → spline (no optimizer) ─────────────────
    for t in _dropout_ts:
        corrected[t] = poses_init[t]
        frame_loss_histories.append([0.0])
        if angle_cross_series is not None and _nose_idx is not None:
            _n  = corrected[t, _nose_idx]
            _c  = corrected[t, _neck_idx]
            _tb = corrected[t, _tail_idx]
            if not (np.any(np.isnan(_n)) or np.any(np.isnan(_c))
                    or np.any(np.isnan(_tb))):
                _u = _n - _c; _v = _tb - _c
                angle_cross_series[t] = (_u[0] * _v[1] - _u[1] * _v[0]) / _l2

    # ── Phase 2: windowed joint optimization ─────────────────────────────────
    # Build edge cache once per mouse (reused across all windows).
    _win_cache = build_edge_cache(skeleton.edges, kp_index, angle_triplets)

    # Warm-start buffer: after each window is corrected, its output is stored
    # here and re-used as the initial point for the next adjacent window.
    # This reduces L-BFGS-B iterations substantially when consecutive windows
    # overlap (they share context frames whose corrected values are already
    # near the optimum).  Initialized to spline-interpolated poses so that
    # the very first window also benefits from the spline pre-processing.
    _warm_buffer = poses_init.copy()  # shape (T, n_kp, 2)

    _windows_list = list(_group_into_windows(_window_ts, T))
    _mid_label = str(mouse_df['mouse_id'].iloc[0]) if not mouse_df.empty else '?'
    for (t_start, t_end, outlier_ts) in _tqdm(
        _windows_list,
        desc=f'  mouse {_mid_label}',
        leave=False,
        unit='win',
    ):
        T_win = t_end - t_start

        # Observed values: spline-initialized so dropout kps have good x0
        window_observed = poses_init[t_start:t_end].copy()   # (T_win, n_kp, 2)
        window_valid    = ~zero_frame_mask[t_start:t_end]    # (T_win,) bool

        # Adhesion weights:
        #   clean frames (not in outlier_ts) → 1.0 (strong anchor)
        #   outlier frames                   → confidence from severity
        # non_outlier_adhesion_scale > 1 boosts the adhesion of keypoints that
        # are NOT flagged as outliers in an outlier frame.  This prevents the
        # optimizer from dragging "innocent" keypoints (e.g. tail_base) as
        # collateral damage when correcting their neighbours (hip_left/right),
        # which would inflate apparent velocity and fail the KS fidelity test.
        #
        # keypoint_adhesion_floor (if set) raises the minimum adhesion for
        # specific keypoints even when they ARE flagged as outliers.  This is
        # needed when a stable keypoint (tail_base) is flagged as an outlier
        # because it lies on a violated edge (hip→tail) even though the actual
        # error is in the neighbour (hip_left/right).  With a high floor, the
        # optimizer fixes the edge by moving the neighbour, not tail_base.
        window_adhesion = np.ones((T_win, n_kp), dtype=float)
        for t in range(t_start, t_end):
            ti = t - t_start
            if not window_valid[ti]:
                window_adhesion[ti] = 0.0
                continue
            if np.any(outlier_frames_mask[t]):
                _sev = np.where(np.isnan(severity_scores[t]), 0.5,
                                severity_scores[t])
                kp_conf = compute_observation_confidence(
                    _sev, sigma=confidence_sigma, floor=_adh_floor
                )
                window_adhesion[ti] = np.where(outlier_frames_mask[t],
                                               kp_conf,
                                               non_outlier_adhesion_scale)
                if likelihoods is not None:
                    _lik_t = np.clip(likelihoods[t], 0.01, 1.0)
                    # Likelihood gate applies to ALL keypoints in outlier frames.
                    # For outlier kps: soft-weighted correction.
                    # For non-outlier kps: also penalise low tracker confidence so
                    # that clean-frame keypoints with poor DLC confidence don't
                    # receive full adhesion and can be gently adjusted.
                    window_adhesion[ti] = np.clip(
                        window_adhesion[ti] * _lik_t, _adh_floor, non_outlier_adhesion_scale
                    )
            # else: clean frame → apply likelihood scaling if available
            elif likelihoods is not None:
                _lik_t = np.clip(likelihoods[t], 0.01, 1.0)
                # Scale clean-frame adhesion by tracker confidence (per-kp).
                # High confidence → adhesion ≈ 1.0 (unchanged).
                # Low confidence → adhesion reduced, letting the optimizer
                # correct mildly unreliable but geometrically un-flagged detections.
                window_adhesion[ti] = np.clip(
                    non_outlier_adhesion_scale * _lik_t, _adh_floor, non_outlier_adhesion_scale
                )

        # Apply keypoint_adhesion_floor: raise the floor for specific keypoints
        # regardless of their outlier status.  This ensures that stable keypoints
        # (e.g. tail_base) are not displaced even when flagged as outliers.
        if keypoint_adhesion_floor:
            _inv_kp = {v: k for k, v in kp_index.items()}
            for _kp_i in range(n_kp):
                _fl = keypoint_adhesion_floor.get(_inv_kp.get(_kp_i, ''), 0.0)
                if _fl > 0:
                    window_adhesion[:, _kp_i] = np.maximum(
                        window_adhesion[:, _kp_i], _fl
                    )

        _win_edge_lambdas = edge_lambdas

        # Foreshortening (virtual height) scale: when the mouse overlaps with
        # another animal during this window, the 2-D skeleton appears compressed
        # (L_observed ≈ L_real × cos θ).  Reduce edge_lambdas so the optimizer
        # stops fighting the apparent foreshortening.
        if overlap_mask is not None:
            # t_start:t_end are LOCAL frame indices within this mouse's poses
            # array; overlap_mask is also indexed by local frame order.
            _local_ts = np.arange(t_start, t_end)
            _valid_idx = _local_ts[_local_ts < len(overlap_mask)]
            _win_in_overlap = overlap_mask[_valid_idx] & window_valid[:len(_valid_idx)]
            if np.any(_win_in_overlap):
                _win_edge_lambdas = _win_edge_lambdas * _FORESHORTEN_EDGE_SCALE

        # ── Confidence-aware λ scheduling ────────────────────────────────────
        # Compute average confidence of *outlier* keypoints in this window.
        # Clean-frame keypoints (adhesion = 1.0) are excluded so that windows
        # dominated by clean context frames don't inflate the confidence estimate.
        # When _win_conf is high (good detections): geometry is enforced strongly,
        # smoothing is lighter.  When low (uncertain/missing): geometry relaxes,
        # temporal smoothing and jerk damping increase to stabilise the trajectory.
        _outlier_adh = window_adhesion[window_adhesion < 1.0 - 1e-6]
        _win_conf = float(np.clip(
            np.mean(_outlier_adh) if _outlier_adh.size > 0 else 1.0,
            0.05, 1.0,
        ))
        # Confidence schedule (v4):
        #   lambda_length  ∝  conf           – trust geometry when detections are good
        #   lambda_smooth  ∝  2*(1-conf+0.1) – more smoothing when uncertain
        #   lambda_acc     ∝  conf           – tighter acceleration when confident
        #   lambda_angle   ∝  conf           – enforce pose rigidity only when confident
        #   lambda_jerk    ∝  (1-conf+0.2)/0.6 – damp jitter more when uncertain
        _win_edge_lambdas = _win_edge_lambdas * _win_conf
        _win_lambda_t     = lambda_t     * 2.0 * (1.0 - _win_conf + 0.1)
        _win_lambda_acc   = lambda_acc   * _win_conf
        _win_lambda_angle = lambda_angle * _win_conf
        _win_lambda_jerk  = lambda_jerk  * (1.0 - _win_conf + 0.2) / 0.6

        # ── Velocity-adaptive temporal regularisation (CRITICAL #3) ──────────
        # During fast-motion windows (locomotion bursts, grooming transitions)
        # a global λ_T suppresses real behavioral dynamics.  Scale down λ_T
        # and λ_acc proportionally when the window mean speed exceeds the
        # video's 75th-percentile speed.  λ_jerk is increased during fast
        # motion to damp spurious high-frequency jitter caused by rapid poses.
        # Only reduces λ_T (never amplifies), preserving smoothing during rest.
        _win_mean_speed = float(np.mean(
            _frame_mean_speed[t_start:t_end][window_valid]
        )) if window_valid.any() else 0.0
        if _speed_p75 > 0 and _win_mean_speed > _speed_p75:
            _vel_ratio = _win_mean_speed / _speed_p75   # > 1 during fast motion
            _vel_scale = 1.0 / _vel_ratio               # < 1 → relax temporal penalty
            _win_lambda_t   = _win_lambda_t   * _vel_scale
            _win_lambda_acc = _win_lambda_acc * _vel_scale

        corrected_win, final_loss, conv_meta = correct_window(
            window_observed=window_observed,
            window_adhesion=window_adhesion,
            window_valid=window_valid,
            skeleton=skeleton,
            kp_index=kp_index,
            l_body_px=l_body_px,
            edge_lambdas=_win_edge_lambdas,
            lambda_t=_win_lambda_t,
            lambda_acc=_win_lambda_acc,
            lambda_angle=_win_lambda_angle,
            lambda_jerk=_win_lambda_jerk,
            angle_triplets=angle_triplets,
            edge_cache=_win_cache,
            return_convergence=True,
            x0_warm=_warm_buffer[t_start:t_end],
        )

        # Update warm-start buffer with the full corrected window output
        # (including both outlier frames and clean context frames).
        # The next adjacent window will start from this as its initial point.
        _warm_buffer[t_start:t_end] = corrected_win

        # Write back only the outlier frames (context frames keep their values)
        for t in outlier_ts:
            ti = t - t_start
            corrected[t] = corrected_win[ti]
            frame_loss_histories.append([final_loss])

        # Aggregate convergence diagnostics (one entry per corrected frame)
        n_out_in_win = len(outlier_ts)
        report.setdefault("_n_opt_iters_list", []).extend(
            [conv_meta.get("n_opt_iters_last", 0)] * n_out_in_win
        )
        if not conv_meta.get("converged", True):
            report["_n_failed_frames"] = (
                report.get("_n_failed_frames", 0) + n_out_in_win
            )

        # Update angle_cross_series for corrected frames
        if angle_cross_series is not None and _nose_idx is not None:
            for t in outlier_ts:
                _n  = corrected[t, _nose_idx]
                _c  = corrected[t, _neck_idx]
                _tb = corrected[t, _tail_idx]
                if not (np.any(np.isnan(_n)) or np.any(np.isnan(_c))
                        or np.any(np.isnan(_tb))):
                    _u = _n - _c; _v = _tb - _c
                    angle_cross_series[t] = (_u[0]*_v[1] - _u[1]*_v[0]) / _l2

    # ── Displacement guard: clamp window corrections for kinematic fidelity ───
    # For non-dropout outlier frames (_window_ts), limit each keypoint's
    # displacement from its original valid position to at most
    # max_displacement_factor × natural_p95_displacement.  This prevents the
    # bone-length optimizer from creating apparent velocity spikes when solving
    # geometry at the cost of temporal coherence (tail_base inflation problem).
    # When reference_poses is provided (e.g. raw df_sample passed through
    # smooth_video), the guard anchors to the original raw observation so that
    # multi-pass pipelines (v1→v2→v3) do not accumulate drift beyond the limit.
    _guard_poses = reference_poses if (reference_poses is not None
                                        and reference_poses.shape == poses_original.shape
                                        ) else poses_original
    if max_displacement_factor > 0 and _window_ts:
        _max_disp_arr = _nat_p95_per_kp * max_displacement_factor  # (n_kp,)
        for t in _window_ts:
            for _ki in range(n_kp):
                if np.isnan(_guard_poses[t, _ki, 0]):
                    continue  # reference was missing — skip
                _delta = corrected[t, _ki] - _guard_poses[t, _ki]
                _d2 = float(np.dot(_delta, _delta))
                _lim = _max_disp_arr[_ki]
                if _d2 > _lim ** 2:
                    corrected[t, _ki] = (
                        _guard_poses[t, _ki]
                        + (_lim / np.sqrt(_d2)) * _delta
                    )

    # ── Displacement diagnostic ──────────────────────────────────────────────
    # How far the keypoints moved from the original observation. All-zero frames are
    # excluded, since they had no observation to move away from.
    active_mask = ~zero_frame_mask
    if active_mask.any():
        displacements = np.sqrt(
            np.sum((corrected[active_mask] - poses_original[active_mask]) ** 2, axis=2)
        )  # shape (n_active, n_kp)
        disp_flat = displacements.ravel()
        disp_flat = disp_flat[~np.isnan(disp_flat)]
        report.setdefault("_displacement_list", []).append(disp_flat)

    # ── Convergence history ──────────────────────────────────────────────────
    if frame_loss_histories:
        report.setdefault("_convergence_lists", []).extend(frame_loss_histories)

    # ── Reconstruir formato largo (vectorizado) ──────────────────────────────
    mouse_id_val = mouse_df["mouse_id"].iloc[0]
    n_frames_out = len(frames)
    n_kp_out = len(kp_index)
    # Preallocate the arrays; appending row by row is far slower
    kp_names = list(kp_index.keys())
    frame_rep  = np.repeat(frames, n_kp_out)
    mid_rep    = np.full(n_frames_out * n_kp_out, mouse_id_val)
    bp_rep     = np.tile(kp_names, n_frames_out)
    kp_indices = [kp_index[k] for k in kp_names]
    x_vals = corrected[:, kp_indices, 0].ravel()
    y_vals = corrected[:, kp_indices, 1].ravel()
    # Type conversion only; the NaNs are already NaN
    return pd.DataFrame({
        "video_frame": frame_rep,
        "mouse_id":    mid_rep,
        "bodypart":    bp_rep,
        "x": x_vals.astype(float),
        "y": y_vals.astype(float),
    })


# ---------------------------------------------------------------------------
# Observation confidence: α_t = exp(-r_t / σ)
# ---------------------------------------------------------------------------

def compute_observation_confidence(
    severity_scores: np.ndarray,
    sigma: float = 0.35,
    floor: float = 0.02,
) -> np.ndarray:
    """Compute per-keypoint observation confidence α_t = exp(-r_t / σ).

    Maps composite residuals (severity_scores) in [0, 1] to a confidence
    weight in (0, 1] that controls how strongly the optimizer is anchored
    to the observed position:

      α = 1.0  → full trust (clean keypoint, r ≈ 0)
      α ≈ 0.37 → partial trust (r = σ: speed spike or mild length violation)
      α ≈ 0.05 → near-zero trust (r >> σ: dropout or severe corruption)

    This replaces the previous piecewise linear/exponential formula with a
    single principled formula derived from Gaussian noise assumptions.

    Args:
        severity_scores : shape (n_kp,) composite residual in [0, 1].
                          0 = borderline outlier, 1 = severe (dropout/spike).
        sigma           : noise scale that controls trust decay rate.
                          Smaller σ → faster decay (harsher penalisation).
                          Default 0.35 gives α ≈ 0.05 at severity = 1.0.
        floor           : minimum confidence to maintain a weak observation
                          anchor and prevent unconstrained drift.

    Returns:
        confidence : shape (n_kp,) float32 in [floor, 1.0].
    """
    alpha = np.exp(-np.asarray(severity_scores, dtype=float) / max(sigma, 1e-6))
    return np.clip(alpha, floor, 1.0)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def _detect_outliers(
    poses: np.ndarray,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    speed_sigma: float,
    length_threshold: float,
    l_body_px: float,
    zero_frame_mask: Optional[np.ndarray] = None,
    detection_n_sigma: float = 0.0,
    adaptive_thresholds: Optional[Dict[str, float]] = None,
    return_breakdown: bool = False,
):
    """Return an outlier mask, (T, n_kp) bool, and a severity, (T, n_kp) in [0,1].

    Criterio OR:
      (a) speed above median + speed_sigma * IQR/1.349, a robust estimator, or
      (b) an adjacent segment violating its length constraint.

    With ``detection_n_sigma > 0`` a per-edge threshold applies:
        detect_thr_e = max(length_threshold, detection_n_sigma × L_std_e)
    This stops variation that is within a segment's natural biological range from
    being flagged. Without it the detector marks most of the video, which is the
    calibration problem notebook 02d had to fix.

    When ``adaptive_thresholds`` is supplied, a dict of ``tau_e = mu_e + k * sigma_e``
    from ``compute_adaptive_edge_thresholds``, it overrides the per-edge threshold.
    That makes detection robust across videos with different movement regimes,
    rather than tuned to one recording.

    Severity in [0,1] measures how far past its threshold a keypoint is: 0 is
    exactly at the threshold, 1 is twice it. It feeds straight into
    ``compute_observation_confidence`` for the anchoring alpha_t = exp(-r_t/sigma),
    so detection and correction strength are continuous rather than a hard switch.
    """
    T, n_kp, _ = poses.shape
    if zero_frame_mask is None:
        zero_frame_mask = np.zeros(T, dtype=bool)

    outlier  = np.zeros((T, n_kp), dtype=bool)
    severity = np.zeros((T, n_kp), dtype=float)

    # Per-criterion FRAME-level accumulators (used only when return_breakdown).
    # Each is True for a frame where that criterion flagged ≥1 keypoint.
    _f_nan      = np.zeros(T, dtype=bool)
    _f_speed    = np.zeros(T, dtype=bool)
    _f_length   = np.zeros(T, dtype=bool)
    _f_nearzero = np.zeros(T, dtype=bool)
    _f_accel    = np.zeros(T, dtype=bool)
    _f_swap     = np.zeros(T, dtype=bool)

    # An explicit NaN
    nan_mask = np.any(np.isnan(poses), axis=2)
    outlier |= nan_mask
    severity[nan_mask] = 1.0
    _f_nan |= np.any(nan_mask, axis=1)

    # ── (a) Robust speed, median and IQR ─────────────────────────────────────
    # NaN = no valid speed (adjacent to zero frame or first frame).
    # Using NaN (not 0) prevents conflating "no data" with "stationary mouse"
    # which would distort IQR statistics and poison threshold computation.
    speeds = np.full((T, n_kp), np.nan)
    if T > 1:
        raw_speeds = np.sqrt(np.sum(np.diff(poses, axis=0) ** 2, axis=2))  # (T-1, n_kp)
        # Vectorized: valid transitions where neither endpoint is a zero frame
        valid_trans = ~zero_frame_mask[1:] & ~zero_frame_mask[:-1]  # (T-1,)
        speeds[1:] = np.where(valid_trans[:, None], raw_speeds, np.nan)

    for kp_idx in range(n_kp):
        s = speeds[:, kp_idx]
        valid_mask = ~np.isnan(s) & ~zero_frame_mask
        valid_speeds = s[valid_mask]
        if len(valid_speeds) < 10:
            continue
        q25, q75 = np.percentile(valid_speeds, [25, 75])
        iqr_sigma = max((q75 - q25) / 1.349, 1e-3)  # avoid IQR=0 for stationary mice
        med = np.median(valid_speeds)
        threshold_speed = med + speed_sigma * iqr_sigma
        above = valid_mask & (s > threshold_speed)
        outlier[:, kp_idx] |= above
        _f_speed |= above
        # Severity: 0 at threshold, 1 at threshold + speed_sigma×iqr_sigma above
        spread = speed_sigma * iqr_sigma
        sev = np.where(valid_mask,
                       np.clip((s - threshold_speed) / max(spread, 1e-6), 0.0, 1.0),
                       0.0)
        severity[:, kp_idx] = np.maximum(severity[:, kp_idx], sev)

    # ── (b) Segment length violations, vectorised ────────────────────────────
    # Pre-build index arrays so the T-dimension is handled by NumPy (not Python
    # loops).  Per-edge threshold lookup stays in a short O(n_edges) loop.
    _edge_rows = []
    for edge in skeleton.edges:
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        L_ref_px = edge.L_ref * l_body_px
        if L_ref_px <= 0:
            continue
        edge_key = f"{edge.src}→{edge.dst}"
        if adaptive_thresholds and edge_key in adaptive_thresholds:
            thr = adaptive_thresholds[edge_key]
        elif detection_n_sigma > 0:
            thr = max(length_threshold, detection_n_sigma * edge.L_std)
        else:
            thr = length_threshold
        _edge_rows.append((i, j, L_ref_px, max(thr, 1e-6)))

    if _edge_rows:
        _e_src  = np.array([r[0] for r in _edge_rows], dtype=np.intp)
        _e_dst  = np.array([r[1] for r in _edge_rows], dtype=np.intp)
        _e_lref = np.array([r[2] for r in _edge_rows])   # (n_edges,)
        _e_thr  = np.array([r[3] for r in _edge_rows])   # (n_edges,)

        # (T, n_edges) distance computation — no Python loop over T
        _deltas   = poses[:, _e_src] - poses[:, _e_dst]           # (T, n_edges, 2)
        _seg_lens = np.sqrt(np.sum(_deltas ** 2, axis=2))         # (T, n_edges)
        _rel_err  = (np.abs(_seg_lens - _e_lref[None, :])
                     / np.maximum(_e_lref[None, :], 1e-6))        # (T, n_edges)
        _viol = (_rel_err > _e_thr[None, :]) & ~zero_frame_mask[:, None]
        _sev_e = np.clip(_rel_err / _e_thr[None, :] - 1.0, 0.0, 1.0)
        _sev_e[zero_frame_mask] = 0.0

        # Scatter (T, n_edges) → (T, n_kp); short loop over edges only
        for ei in range(len(_e_src)):
            si, di = int(_e_src[ei]), int(_e_dst[ei])
            outlier[:, si] |= _viol[:, ei]
            outlier[:, di] |= _viol[:, ei]
            severity[:, si] = np.maximum(severity[:, si], _sev_e[:, ei])
            severity[:, di] = np.maximum(severity[:, di], _sev_e[:, ei])
        _f_length |= np.any(_viol, axis=1)

    # ── (c) Individual dropouts: x and y near 0 but the frame is not all-zero
    # One keypoint at (0,0) while the others are valid is a detector dropout.
    # Vectorized over all keypoints simultaneously.
    _near_zero = (
        (np.abs(poses[:, :, 0]) < 2.0)
        & (np.abs(poses[:, :, 1]) < 2.0)
        & ~zero_frame_mask[:, None]
    )  # (T, n_kp)
    outlier[_near_zero] = True
    severity[_near_zero] = 1.0
    _f_nearzero |= np.any(_near_zero, axis=1)

    # ── (d) Acceleration spikes, where the second difference is far above normal
    # A sudden acceleration spike means a corrupted observation: a real mouse
    # cannot change velocity that fast. Same robust estimator as for speed.
    if T > 2:
        # Vectorized second-order difference; NaN for zero-adjacent frames
        _diff2 = poses[2:] - 2.0 * poses[1:-1] + poses[:-2]  # (T-2, n_kp, 2)
        _accel_vals = np.sqrt(np.sum(_diff2 ** 2, axis=2))    # (T-2, n_kp)
        _valid_acc_trans = (
            ~zero_frame_mask[2:] & ~zero_frame_mask[1:-1] & ~zero_frame_mask[:-2]
        )  # (T-2,)
        raw_accel = np.full((T, n_kp), np.nan)
        raw_accel[2:] = np.where(_valid_acc_trans[:, None], _accel_vals, np.nan)

        for kp_idx in range(n_kp):
            a = raw_accel[:, kp_idx]
            valid_mask_a = ~np.isnan(a) & ~zero_frame_mask
            valid_a = a[valid_mask_a]
            if len(valid_a) < 10:
                continue
            q25, q75 = np.percentile(valid_a, [25, 75])
            iqr_s = max((q75 - q25) / 1.349, 1e-3)
            threshold_acc = np.median(valid_a) + speed_sigma * iqr_s
            above_acc = valid_mask_a & (a > threshold_acc)
            outlier[:, kp_idx] |= above_acc
            _f_accel |= above_acc
            spread_acc = speed_sigma * iqr_s
            sev_acc = np.where(valid_mask_a,
                               np.clip((a - threshold_acc) / max(spread_acc, 1e-6), 0.0, 1.0),
                               0.0) * 0.75
            severity[:, kp_idx] = np.maximum(severity[:, kp_idx], sev_acc)

    # ── (e) Identity swaps, from a sudden reversal of body orientation ───────
    # A swap flips the nose-to-tail_base direction from one frame to the next. It is
    # flagged when the cross product changes sign AND the rotation is large enough:
    # 90 degrees or more, which means |cross| above sin(90) of about 0.5. Requiring
    # both avoids flagging an ordinary sharp turn.
    _nose_i = kp_index.get("nose")
    _neck_i = kp_index.get("neck")
    _tail_i = kp_index.get("tail_base")
    if _nose_i is not None and _neck_i is not None and _tail_i is not None:
        # Vectorized cross-product over all frames at once
        _u = poses[:, _nose_i] - poses[:, _neck_i]   # (T, 2)
        _v = poses[:, _tail_i] - poses[:, _neck_i]   # (T, 2)
        _cross_raw = _u[:, 0] * _v[:, 1] - _u[:, 1] * _v[:, 0]  # (T,)
        _norms = (np.sqrt(np.sum(_u ** 2, axis=1))
                  * np.sqrt(np.sum(_v ** 2, axis=1)))  # (T,)
        _nan_mask_cv = (
            np.any(np.isnan(poses[:, _nose_i]), axis=1)
            | np.any(np.isnan(poses[:, _neck_i]), axis=1)
            | np.any(np.isnan(poses[:, _tail_i]), axis=1)
            | zero_frame_mask
        )
        cross_vals = np.where(
            _nan_mask_cv | (_norms < 1e-6),
            np.nan,
            _cross_raw / np.maximum(_norms, 1e-6),
        )  # normalised sin(θ)

        if T > 1:
            cross_diff = np.abs(np.diff(cross_vals, prepend=cross_vals[0]))
            # Sudden large change in body orientation → potential swap
            swap_frames = (cross_diff > 1.0) & ~zero_frame_mask & ~np.isnan(cross_diff)
            if swap_frames.any():
                _f_swap |= swap_frames
                for kp_i in range(n_kp):
                    outlier[swap_frames, kp_i] = True
                    severity[swap_frames, kp_i] = np.maximum(
                        severity[swap_frames, kp_i],
                        np.clip(cross_diff[swap_frames] - 1.0, 0.0, 1.0),
                    )

    # Make sure zero frames are never flagged
    outlier[zero_frame_mask] = False
    severity[zero_frame_mask] = 0.0

    if not return_breakdown:
        return outlier, severity

    # ── Per-criterion frame-level breakdown (calibration / diagnostics) ───────
    # Fractions are computed over NON-zero frames (the same denominator that
    # smooth_video uses for n_total_frames), so they are directly comparable to
    # the headline "% of frames with outliers" figure.
    _valid_frames = ~zero_frame_mask
    _denom = int(_valid_frames.sum()) or 1
    for _m in (_f_nan, _f_speed, _f_length, _f_nearzero, _f_accel, _f_swap):
        _m[zero_frame_mask] = False
    _frame_any = np.any(outlier, axis=1) & _valid_frames
    breakdown = {
        "n_total_frames":   int(_denom),
        "n_flagged_frames": int(_frame_any.sum()),
        "frac_flagged":     float(_frame_any.sum() / _denom),
        "frac_speed":       float(_f_speed.sum() / _denom),
        "frac_length":      float(_f_length.sum() / _denom),
        "frac_near_zero":   float(_f_nearzero.sum() / _denom),
        "frac_accel":       float(_f_accel.sum() / _denom),
        "frac_swap":        float(_f_swap.sum() / _denom),
        "frac_nan":         float(_f_nan.sum() / _denom),
    }
    return outlier, severity, breakdown


def detect_outliers_for_video(
    tracking_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    speed_sigma: float = SPEED_SIGMA_THRESHOLD,
    length_threshold: float = LENGTH_VAR_THRESHOLD,
    detection_n_sigma: float = 3.0,
    adaptive_thresholds_k: float = 2.5,
    speed_mad_k: float = 3.0,
    pre_smooth: bool = True,
    use_adaptive: bool = True,
) -> Dict[str, float]:
    """Run **only** the outlier-detection stage of the pipeline for a video.

    Reproduces the detection-relevant preprocessing performed inside
    ``_smooth_mouse`` (zero-frame masking, per-video ``L_body``/``L_ref``
    re-estimation, optional Savitzky-Golay pre-smoothing, hard speed-spike
    clipping and per-video adaptive edge thresholds) and then calls
    ``_detect_outliers`` with ``return_breakdown=True``.  The expensive per-frame
    L-BFGS-B optimisation is skipped, so this is fast enough to sweep a grid of
    detection parameters for calibration (see scripts/calibrate_detector.py).

    The fractions use the same denominator as ``smooth_video``'s
    ``n_total_frames`` (non-zero frames), so ``frac_flagged`` is directly
    comparable to the headline "% of frames with outliers" figure.

    Returns a dict with frame-weighted (across mice) fractions ``frac_flagged``,
    ``frac_speed``, ``frac_length``, ``frac_near_zero``, ``frac_accel``,
    ``frac_swap``, ``frac_nan`` plus counts ``n_total_frames`` /
    ``n_flagged_frames``.
    """
    kp_index = {kp: i for i, kp in enumerate(skeleton.keypoints)}
    n_kp = len(skeleton.keypoints)

    agg_keys = ["frac_speed", "frac_length", "frac_near_zero",
                "frac_accel", "frac_swap", "frac_nan"]
    agg = {k: 0.0 for k in agg_keys}
    n_flagged_total = 0
    n_frames_total = 0

    for _mouse_id, mouse_df in tracking_df.groupby("mouse_id"):
        wide = _to_wide(mouse_df.sort_values("video_frame").copy())
        frames = sorted(wide["video_frame"].unique())
        T = len(frames)
        if T < 3:
            continue
        frame_idx = {f: i for i, f in enumerate(frames)}
        poses = np.full((T, n_kp, 2), np.nan)

        mdf = mouse_df.copy()
        mdf["_fi"] = mdf["video_frame"].map(frame_idx)
        mdf["_ki"] = mdf["bodypart"].map(kp_index)
        _v = mdf["_fi"].notna() & mdf["_ki"].notna()
        _sv = mdf[_v]
        if not _sv.empty:
            _fi = _sv["_fi"].values.astype(np.intp)
            _ki = _sv["_ki"].values.astype(np.intp)
            poses[_fi, _ki, 0] = _sv["x"].values
            poses[_fi, _ki, 1] = _sv["y"].values

        zero_frame_mask = np.all(
            (poses[:, :, 0] == 0) & (poses[:, :, 1] == 0), axis=1
        )

        _ni = kp_index.get("nose")
        _ti = kp_index.get("tail_base")
        if _ni is not None and _ti is not None:
            _nt = np.sqrt(np.sum((poses[:, _ni] - poses[:, _ti]) ** 2, axis=1))
            _valid_nt = _nt[(_nt > skeleton.L_body_median_px * 0.25) & ~zero_frame_mask]
            l_body_px = (float(np.median(_valid_nt)) if len(_valid_nt) > 10
                         else skeleton.L_body_median_px)
        else:
            l_body_px = skeleton.L_body_median_px

        if pre_smooth and T > SAVGOL_WINDOW:
            _sg = poses.reshape(T, n_kp * 2).copy()
            _sg[zero_frame_mask] = np.nan
            for _c in range(n_kp * 2):
                _ser = _sg[:, _c]
                _val = ~np.isnan(_ser)
                if _val.sum() > SAVGOL_WINDOW:
                    _ser[_val] = savgol_filter(
                        _ser[_val], SAVGOL_WINDOW, SAVGOL_POLYORDER, mode="nearest"
                    )
            poses = np.where(zero_frame_mask[:, None, None], poses,
                             _sg.reshape(T, n_kp, 2))

        poses, _ = _hard_clip_speed_spikes(
            poses, zero_frame_mask, speed_mad_k=speed_mad_k
        )

        _sk = _reestimate_lref_per_video(
            skeleton, poses, kp_index, l_body_px, zero_frame_mask
        )

        adaptive = (
            compute_adaptive_edge_thresholds(
                poses=poses, skeleton=_sk, kp_index=kp_index,
                l_body_px=l_body_px, k=adaptive_thresholds_k,
                zero_frame_mask=zero_frame_mask,
            ) if use_adaptive else None
        )

        _, _, bd = _detect_outliers(
            poses=poses, skeleton=_sk, kp_index=kp_index,
            speed_sigma=speed_sigma, length_threshold=length_threshold,
            l_body_px=l_body_px, zero_frame_mask=zero_frame_mask,
            detection_n_sigma=detection_n_sigma, adaptive_thresholds=adaptive,
            return_breakdown=True,
        )

        nf = bd["n_total_frames"]
        n_frames_total += nf
        n_flagged_total += bd["n_flagged_frames"]
        for k in agg_keys:
            agg[k] += bd[k] * nf

    if n_frames_total == 0:
        return {"frac_flagged": 0.0, "n_total_frames": 0, "n_flagged_frames": 0,
                **{k: 0.0 for k in agg_keys}}

    out = {k: agg[k] / n_frames_total for k in agg_keys}
    out["frac_flagged"]     = n_flagged_total / n_frames_total
    out["n_total_frames"]   = n_frames_total
    out["n_flagged_frames"] = n_flagged_total
    return out


# ---------------------------------------------------------------------------
# Helper: violation rate before and after
# ---------------------------------------------------------------------------

def violation_rate(
    poses: np.ndarray,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    threshold: float = LENGTH_VAR_THRESHOLD,
) -> float:
    """Fraction of frames where any segment violates its length constraint."""
    l_body = skeleton.L_body_median_px
    any_viol = np.zeros(len(poses), dtype=bool)
    for edge in skeleton.edges:
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        L_ref_px = edge.L_ref * l_body
        if L_ref_px <= 0:
            continue
        seg_lens = np.sqrt(np.sum((poses[:, i] - poses[:, j]) ** 2, axis=1))
        rel_err = np.abs(seg_lens - L_ref_px) / L_ref_px
        any_viol |= (rel_err > threshold)
    return float(any_viol.mean())
