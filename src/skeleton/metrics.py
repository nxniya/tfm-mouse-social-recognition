"""Skeleton correction metrics and benchmarking utilities.

Public API
----------
Primary continuous metrics (motion quality — use as headline metrics):
  normalized_rmse_per_keypoint(df_before, df_after, mouse_id, l_body_px, ...)
  acceleration_smoothness(df, mouse_id, l_body_px, keypoints)
  jerk_energy(df, mouse_id, l_body_px, keypoints)
  spectral_energy_ratio(df_before, df_after, mouse_id, fps, keypoints)
  velocity_kl_divergence(df_before, df_after, mouse_id, fps, arena_diag, ...)
  turning_angle_distribution(df_before, df_after, mouse_id, keypoints)
  motion_preservation_report(df_before, df_after, mouse_id, fps, l_body_px, ...)

Violation metrics (secondary / supplementary):
  per_keypoint_violations(viol_edge_df)
  violation_summary(df_long, skeleton, mouse_id, threshold, label)
  violation_magnitude_summary(df_long, skeleton, mouse_id, threshold)
  angular_alignment_score(df_long, skeleton, mouse_id, triplet)

Temporal fidelity:
  temporal_fidelity_report(df_before, df_after, mouse_id, fps, arena_diag, ...)
  ks_velocity_test(df_before, df_after, mouse_id, fps, arena_diag, bodypart)
  psd_comparison_plot(df_before, df_after, mouse_id, fps, keypoints, ...)

Corruption injection (idealized):
  inject_corruption(df_clean, noise_std, dropout_rate, swap_prob, ...)
  benchmark_by_corruption_type(df_clean, skeleton, mouse_id, smooth_fn, ...)

Realistic corruption injection (hard benchmarks):
  inject_occlusion_sequence(df_clean, keypoint_groups, min_len, max_len, ...)
  inject_fast_motion_burst(df_clean, event_type, n_events, duration_frames, ...)
  inject_correlated_corruption(df_clean, keypoint_groups, corruption_prob, ...)
  benchmark_realistic_corruptions(df_clean, skeleton, mouse_id, smooth_fn, ...)

Statistical validation:
  paired_wilcoxon(before_series, after_series)
  bootstrap_ci(values, n_bootstrap, ci, statistic)
  compare_variants_statistically(results_dict, metric, baseline_key)

Diagnostic metrics (Phase E):
  bone_length_variance(df, skeleton, mouse_id)
  temporal_curvature_score(df, mouse_id, keypoints, fps)
  psd_smoothness_score(df, mouse_id, fps, keypoints, high_freq_cutoff)
  tracking_recovery_time(corrected_df, before_df, outlier_frames, mouse_id, ...)

Utilities:
  reconstruction_rmse(df_clean, df_recovered, mouse_id, bodypart, arena_diag)
  compute_displacement_stats(df_before, df_after, mouse_id, arena_diag)
  plot_convergence_history(convergence_data, axes, title)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import ks_2samp


# ── Violation aggregation ─────────────────────────────────────────────────

def per_keypoint_violations(viol_edge_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-edge violation rates to per-keypoint (max over adjacent edges).

    Args:
        viol_edge_df: DataFrame with columns
                      ['edge', 'violation_rate', 'mean_error_rel'].
                      Edges must be formatted as 'src→dst'.

    Returns:
        DataFrame with columns ['keypoint', 'max_viol_rate', 'max_err_rel'],
        sorted descending by max_viol_rate.
    """
    kp_data: dict[str, tuple[float, float]] = {}
    for _, row in viol_edge_df.iterrows():
        parts = row["edge"].split("→")
        if len(parts) != 2:
            continue
        src, dst = parts
        for kp in (src, dst):
            prev = kp_data.get(kp, (-1.0, 0.0))  # sentinel ensures 0.0 rates are stored
            vr = float(row["violation_rate"])
            er = float(row.get("mean_error_rel", 0.0))
            if vr > prev[0]:
                kp_data[kp] = (vr, er)
    rows = [
        {"keypoint": kp, "max_viol_rate": v[0], "max_err_rel": v[1]}
        for kp, v in kp_data.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["keypoint", "max_viol_rate", "max_err_rel"])
    return (
        pd.DataFrame(rows)
        .sort_values("max_viol_rate", ascending=False)
        .reset_index(drop=True)
    )


def violation_summary(
    df_long: pd.DataFrame,
    skeleton,
    mouse_id: int,
    threshold: float = 0.20,
    label: str = "violation_rate",
) -> pd.DataFrame:
    """Compute constraint violation summary for one mouse from long-format tracking.

    Internally calls _to_wide → _segment_length → compute_constraint_violations.
    Returns a DataFrame with the violation_rate column renamed to `label`.
    """
    from src.skeleton.mouse_skeleton import _to_wide, _segment_length
    from src.skeleton import compute_constraint_violations

    wide = _to_wide(df_long[df_long["mouse_id"] == mouse_id])
    l_body = _segment_length(wide, "nose", "tail_base")
    viol = compute_constraint_violations(wide, skeleton, {}, l_body, threshold=threshold)
    return viol.rename(
        columns={
            "violation_rate": label,
            "mean_error_rel": f"err_{label}",
        }
    )


# ── Temporal fidelity ─────────────────────────────────────────────────────

def normalized_speeds(
    df_long: pd.DataFrame,
    mouse_id: int,
    bodypart: str,
    fps: float,
    arena_diag: float = 1.0,
) -> np.ndarray:
    """Compute normalized speed series (fraction_of_arena / s) for one bodypart.

    Returns an array aligned with df rows (first element is NaN due to diff).
    """
    sub = (
        df_long[
            (df_long["bodypart"] == bodypart) & (df_long["mouse_id"] == mouse_id)
        ]
        .sort_values("video_frame")
    )
    dx = sub["x"].diff().values
    dy = sub["y"].diff().values
    return np.sqrt(dx ** 2 + dy ** 2) * fps / max(arena_diag, 1e-6)


def temporal_fidelity_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    arena_diag: float = 1.0,
    label: str = "after",
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """Compare kinematic properties before and after skeleton correction.

    Metrics computed (all normalized by arena diagonal and FPS):
      - speed_median  : median of |Δpos|·fps / arena_diag
      - speed_p99     : 99th-percentile speed
      - accel_mean    : mean |Δspeed|
      - jerk_rms      : RMS of |Δ²speed|

    Args:
        df_before, df_after : long-format tracking DataFrames.
        mouse_id            : mouse to evaluate.
        fps                 : video frame rate (Hz).
        arena_diag          : arena diagonal in pixels for normalisation.
        label               : column label for the "after" values.
        keypoints           : list of keypoints to evaluate; defaults to
                              ['nose', 'tail_base'].

    Returns:
        DataFrame with columns [keypoint, metric, before, <label>, rel_change_%].
    """
    if keypoints is None:
        keypoints = ["nose", "tail_base"]

    rows = []
    for kp in keypoints:
        bp_before = df_before["bodypart"].unique()
        bp_after  = df_after["bodypart"].unique()
        if kp not in bp_before or kp not in bp_after:
            continue

        # ── align both series to frames valid in df_before ──────────────────
        # Corrector-filled dropouts would inflate "after" speed stats if we
        # compared the full series independently (before: NaN→excluded,
        # after: interpolated→included).  Only compare at originally valid frames.
        sub_b = (
            df_before[(df_before["bodypart"] == kp) & (df_before["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        sub_a = (
            df_after[(df_after["bodypart"] == kp) & (df_after["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        _valid_b = set(sub_b.dropna(subset=["x", "y"])["video_frame"].values)
        _common  = sorted(_valid_b & set(sub_a.dropna(subset=["x", "y"])["video_frame"].values))
        sub_b = sub_b[sub_b["video_frame"].isin(_common)].copy()
        sub_a = sub_a[sub_a["video_frame"].isin(_common)].copy()

        # Frame-gap-corrected speed: divide by actual Δframe (not assuming 1).
        # Avoids inflating speed when consecutive rows span multiple frames.
        def _gap_speed(sub: pd.DataFrame) -> np.ndarray:
            dx = sub["x"].diff().values
            dy = sub["y"].diff().values
            dt = sub["video_frame"].diff().values
            dt = np.where(np.isnan(dt) | (dt <= 0), 1.0, dt)
            return np.sqrt(dx ** 2 + dy ** 2) * fps / dt / max(arena_diag, 1e-6)

        v_b = _gap_speed(sub_b)
        v_a = _gap_speed(sub_a)

        v_b_clean = v_b[~np.isnan(v_b)]
        v_a_clean = v_a[~np.isnan(v_a)]

        a_b = np.abs(np.diff(v_b_clean)) if len(v_b_clean) > 1 else np.array([np.nan])
        a_a = np.abs(np.diff(v_a_clean)) if len(v_a_clean) > 1 else np.array([np.nan])
        j_b = np.abs(np.diff(a_b)) if len(a_b) > 1 else np.array([np.nan])
        j_a = np.abs(np.diff(a_a)) if len(a_a) > 1 else np.array([np.nan])

        # Frequency-domain preservation score: PSD correlation in low-freq band
        # (0–2 Hz) where locomotion signals live.
        # sub_b / sub_a are already aligned to common valid frames (computed above).
        freq_pres = np.nan
        try:
            x_b = sub_b["x"].interpolate().values
            x_a = sub_a["x"].interpolate().values
            min_len = min(len(x_b), len(x_a))
            if min_len > 16:
                nperseg = min(64, max(4, min_len // 4))
                f_b, psd_b = signal.welch(x_b[:min_len], fs=fps, nperseg=nperseg)
                f_a, psd_a = signal.welch(x_a[:min_len], fs=fps, nperseg=nperseg)
                low_mask = f_b <= min(2.0, fps / 4.0)
                if low_mask.sum() > 1:
                    freq_pres = float(np.corrcoef(psd_b[low_mask], psd_a[low_mask])[0, 1])
        except Exception:
            freq_pres = np.nan

        # Phase lag: mean absolute phase difference at dominant frequency
        # sub_b / sub_a are already aligned to common valid frames (computed above).
        phase_lag_deg = np.nan
        try:
            x_b = sub_b["x"].interpolate().values
            x_a = sub_a["x"].interpolate().values
            min_len = min(len(x_b), len(x_a))
            if min_len > 16:
                # Cross-correlation based phase estimate
                xcorr = np.correlate(x_b[:min_len] - x_b[:min_len].mean(),
                                     x_a[:min_len] - x_a[:min_len].mean(), mode="full")
                lag_frames = int(xcorr.argmax()) - (min_len - 1)
                phase_lag_deg = float(abs(lag_frames) / max(min_len, 1) * 360.0)
        except Exception:
            phase_lag_deg = np.nan

        metrics = {
            "speed_median":    (np.nanmedian(v_b_clean), np.nanmedian(v_a_clean)),
            "speed_p99":       (np.nanpercentile(v_b_clean, 99), np.nanpercentile(v_a_clean, 99)),
            "accel_mean":      (np.nanmean(a_b), np.nanmean(a_a)),
            "jerk_rms":        (
                float(np.sqrt(np.nanmean(j_b ** 2))),
                float(np.sqrt(np.nanmean(j_a ** 2))),
            ),
            "freq_preservation": (1.0, freq_pres),   # 1.0 = perfect (reference)
            "phase_lag_deg":     (0.0, phase_lag_deg),
        }

        for metric, (val_b, val_a) in metrics.items():
            rel = (
                (val_a - val_b) / max(abs(val_b), 1e-12) * 100
                if not (np.isnan(val_b) or val_b == 0)
                else np.nan
            )
            rows.append(
                {
                    "keypoint": kp,
                    "metric": metric,
                    "before": round(float(val_b), 6) if not np.isnan(val_b) else np.nan,
                    label:    round(float(val_a), 6) if not np.isnan(val_a) else np.nan,
                    "rel_change_%": round(rel, 2) if not np.isnan(rel) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def ks_velocity_test(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    arena_diag: float = 1.0,
    bodypart: str = "nose",
) -> tuple[float, float]:
    """Two-sample KS test on normalized speed distributions before vs. after.

    Returns:
        (statistic, p_value) — large D or small p means the correction
        significantly altered the speed distribution.
    """
    # Align to frames valid in df_before before comparing speed distributions.
    _b = (
        df_before[(df_before["bodypart"] == bodypart) & (df_before["mouse_id"] == mouse_id)]
        .sort_values("video_frame").dropna(subset=["x", "y"])
    )
    _a = (
        df_after[(df_after["bodypart"] == bodypart) & (df_after["mouse_id"] == mouse_id)]
        .sort_values("video_frame").dropna(subset=["x", "y"])
    )
    _common = sorted(set(_b["video_frame"].values) & set(_a["video_frame"].values))
    if len(_common) < 5:
        return np.nan, np.nan
    _b = _b[_b["video_frame"].isin(_common)]
    _a = _a[_a["video_frame"].isin(_common)]
    _dt_b = np.maximum(_b["video_frame"].diff().values, 1.0)
    _dt_a = np.maximum(_a["video_frame"].diff().values, 1.0)
    v_b = (np.sqrt(_b["x"].diff().values ** 2 + _b["y"].diff().values ** 2)
           * fps / _dt_b / max(arena_diag, 1e-6))
    v_a = (np.sqrt(_a["x"].diff().values ** 2 + _a["y"].diff().values ** 2)
           * fps / _dt_a / max(arena_diag, 1e-6))
    mask = ~np.isnan(v_b) & ~np.isnan(v_a)
    if mask.sum() < 4:
        return np.nan, np.nan
    return ks_2samp(v_b[mask], v_a[mask], method='asymp')


def psd_comparison_plot(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    keypoints: list[str],
    after_label: str = "after",
    axes=None,
):
    """Plot Welch PSD comparison of the x-coordinate for one or more keypoints.

    If `axes` is None a new figure is created. Returns the axes array.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, len(keypoints), figsize=(7 * len(keypoints), 4))
    if len(keypoints) == 1:
        axes = [axes]

    for ax, kp in zip(axes, keypoints):
        for df_src, lbl, ls in [
            (df_before, "Before", "-"),
            (df_after,  f"After ({after_label})", "--"),
        ]:
            sub = (
                df_src[
                    (df_src["bodypart"] == kp) & (df_src["mouse_id"] == mouse_id)
                ]
                .sort_values("video_frame")
            )
            x_vals = sub["x"].interpolate().values
            if len(x_vals) > 16:
                nperseg = min(64, max(4, len(x_vals) // 4))
                freqs, psd = signal.welch(x_vals, fs=fps, nperseg=nperseg)
                ax.semilogy(freqs, psd, linestyle=ls, lw=1.5, label=lbl)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title(f"PSD — {kp}")
        ax.legend(fontsize=9)

    return axes


# ── Synthetic corruption benchmark ───────────────────────────────────────

def inject_corruption(
    df_clean: pd.DataFrame,
    noise_std: float = 5.0,
    dropout_rate: float = 0.05,
    swap_prob: float = 0.02,
    speed_spikes_prob: float = 0.0,
    drift_amplitude: float = 0.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject synthetic corruption into a clean tracking DataFrame.

    Corruption types:
      - Gaussian noise   : additive N(0, noise_std) to x and y.
      - Dropouts         : set x=y=0.0 for ``dropout_rate`` fraction of rows,
                           mimicking detector failure frames.
      - Identity swaps   : exchange (x, y) of mouse[0] and mouse[1] for
                           ``swap_prob`` fraction of frames.
      - Speed spikes     : single-frame jumps of 20–80 px for
                           ``speed_spikes_prob`` fraction of frames.
      - Positional drift : slow sinusoidal low-frequency drift of amplitude
                           ``drift_amplitude`` pixels over the full recording.

    Args:
        df_clean          : long-format tracking DataFrame.
        noise_std         : std-dev of Gaussian noise in pixels.
        dropout_rate      : fraction of rows to zero-out.
        swap_prob         : fraction of frames to swap identity.
        speed_spikes_prob : fraction of frames to inject a sharp velocity spike.
        drift_amplitude   : peak amplitude of low-frequency positional drift (px).
        rng               : numpy Generator; created from seed 42 if None.

    Returns:
        Corrupted copy of df_clean.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_clean.copy()
    n = len(df)

    # 1. Gaussian noise
    noise = rng.normal(0.0, noise_std, size=(n, 2))
    df["x"] = df["x"] + noise[:, 0]
    df["y"] = df["y"] + noise[:, 1]

    # 2. Dropouts
    dropout_mask = rng.random(n) < dropout_rate
    df.loc[dropout_mask, ["x", "y"]] = 0.0

    # 3. Identity swaps
    mice = sorted(df["mouse_id"].unique())
    if len(mice) >= 2:
        frames = df["video_frame"].unique()
        n_swaps = max(1, int(len(frames) * swap_prob))
        swap_frames = rng.choice(frames, size=n_swaps, replace=False)
        for fr in swap_frames:
            for bp in df["bodypart"].unique():
                idx0 = df[
                    (df["video_frame"] == fr)
                    & (df["bodypart"] == bp)
                    & (df["mouse_id"] == mice[0])
                ].index
                idx1 = df[
                    (df["video_frame"] == fr)
                    & (df["bodypart"] == bp)
                    & (df["mouse_id"] == mice[1])
                ].index
                if len(idx0) and len(idx1):
                    tmp = df.loc[idx0, ["x", "y"]].values.copy()
                    df.loc[idx0, ["x", "y"]] = df.loc[idx1, ["x", "y"]].values
                    df.loc[idx1, ["x", "y"]] = tmp

    # 4. Speed spikes — single-frame large displacement
    if speed_spikes_prob > 0:
        frames_all = df["video_frame"].unique()
        n_spike = max(0, int(len(frames_all) * speed_spikes_prob))
        if n_spike > 0:
            spike_frames = rng.choice(frames_all, size=n_spike, replace=False)
            for fr in spike_frames:
                mask = df["video_frame"] == fr
                mag = rng.uniform(20.0, 80.0)
                angle = rng.uniform(0, 2 * np.pi)
                df.loc[mask, "x"] += mag * np.cos(angle)
                df.loc[mask, "y"] += mag * np.sin(angle)

    # 5. Positional drift — slow sinusoidal low-frequency displacement
    if drift_amplitude > 0:
        frames_sorted = sorted(df["video_frame"].unique())
        T = len(frames_sorted)
        t_norm = np.linspace(0, 1, T)
        drift_x = drift_amplitude * np.sin(2 * np.pi * t_norm)
        drift_y = drift_amplitude * 0.5 * np.cos(4 * np.pi * t_norm)
        frame_to_drift = {
            f: (float(dx), float(dy))
            for f, dx, dy in zip(frames_sorted, drift_x, drift_y)
        }
        for fr, (dx, dy) in frame_to_drift.items():
            mask = df["video_frame"] == fr
            df.loc[mask, "x"] += dx
            df.loc[mask, "y"] += dy

    return df


def reconstruction_rmse(
    df_clean: pd.DataFrame,
    df_recovered: pd.DataFrame,
    mouse_id: int,
    bodypart: str = "nose",
    arena_diag: float = 1.0,
) -> float:
    """RMSE between recovered and clean ground-truth, normalized by arena diagonal.

    Returns:
        float in [0, ∞) — 0 means perfect reconstruction.
    """
    def _sel(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[(df["bodypart"] == bodypart) & (df["mouse_id"] == mouse_id)]
            .sort_values("video_frame")[["video_frame", "x", "y"]]
        )

    merged = _sel(df_clean).merge(
        _sel(df_recovered), on="video_frame", suffixes=("_c", "_r")
    )
    sq_err = (
        (merged["x_r"] - merged["x_c"]) ** 2
        + (merged["y_r"] - merged["y_c"]) ** 2
    )
    return float(np.sqrt(sq_err.mean())) / max(arena_diag, 1e-6)


# ── Violation magnitude (continuous, not binary) ─────────────────────────

def violation_magnitude_summary(
    df_long: pd.DataFrame,
    skeleton,
    mouse_id: int,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Compute continuous deviation stats per edge (not just binary violation).

    In addition to the binary violation_rate, computes:
      - mean_abs_dev_px  : mean absolute deviation from L_ref in pixels.
      - max_abs_dev_px   : worst-case deviation in pixels.
      - p95_abs_dev_px   : 95th-percentile deviation (robust extreme).
      - mean_error_rel   : mean relative error (fraction of L_ref).

    This allows distinguishing between frames that barely cross the threshold
    vs. frames with catastrophic skeleton deformation.
    """
    from src.skeleton.mouse_skeleton import _to_wide, _segment_length
    from src.skeleton import compute_constraint_violations

    wide = _to_wide(df_long[df_long["mouse_id"] == mouse_id])
    l_body = _segment_length(wide, "nose", "tail_base")
    viol = compute_constraint_violations(wide, skeleton, {}, l_body, threshold=threshold)
    return viol  # already includes mean_abs_dev_px etc. from updated kinematic_constraints


def angular_alignment_score(
    df_long: pd.DataFrame,
    skeleton,
    mouse_id: int,
    triplet: tuple[str, str, str] = ("nose", "neck", "tail_base"),
) -> dict:
    """Measure body-axis alignment via the cross-product of the main axis triplet.

    For triplet (A, B, C) = (nose, neck, tail_base):
      cross = (A-B) × (C-B)  [2-D scalar]
      normalised_cross = cross / (|A-B| * |C-B|) = sin(angle_at_B)

    Returns:
        dict with keys:
          - mean_abs_cross  : mean |normalized_cross| (0=perfectly straight)
          - p95_abs_cross   : 95th-percentile |normalized_cross|
          - pct_bent        : fraction of frames where |norm_cross| > 0.5
                              (angle > 30° from straight)
    """
    from src.skeleton.mouse_skeleton import _to_wide

    wide = _to_wide(df_long[df_long["mouse_id"] == mouse_id])
    a_name, b_name, c_name = triplet
    cols = [f"{n}_{ax}" for n in [a_name, b_name, c_name] for ax in ["x", "y"]]
    missing = [c for c in cols if c not in wide.columns]
    if missing:
        return {"mean_abs_cross": np.nan, "p95_abs_cross": np.nan, "pct_bent": np.nan}

    ax = wide[f"{a_name}_x"].values; ay = wide[f"{a_name}_y"].values
    bx = wide[f"{b_name}_x"].values; by = wide[f"{b_name}_y"].values
    cx = wide[f"{c_name}_x"].values; cy = wide[f"{c_name}_y"].values

    ux, uy = ax - bx, ay - by   # A - B
    vx, vy = cx - bx, cy - by   # C - B
    cross = ux * vy - uy * vx
    norm = np.sqrt(ux**2 + uy**2) * np.sqrt(vx**2 + vy**2)
    safe_norm = np.where(norm > 1e-6, norm, 1.0)
    norm_cross = np.where(norm > 1e-6, cross / safe_norm, 0.0)
    abs_cross = np.abs(norm_cross)

    return {
        "mean_abs_cross": float(np.nanmean(abs_cross)),
        "p95_abs_cross":  float(np.nanpercentile(abs_cross, 95)),
        "pct_bent":       float(np.nanmean(abs_cross > 0.5)),
    }


def benchmark_by_corruption_type(
    df_clean: pd.DataFrame,
    skeleton,
    mouse_id: int,
    smooth_fn,
    arena_diag: float = 1.0,
    noise_std: float = 5.0,
    dropout_rate: float = 0.05,
    swap_prob: float = 0.02,
    speed_spikes_prob: float = 0.03,
    drift_amplitude: float = 15.0,
    threshold: float = 0.20,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Run separate recovery benchmarks for each corruption type.

    Evaluates: Gaussian noise, dropouts, identity swaps, speed spikes,
    positional drift, and all combined.

    Args:
        df_clean           : ground-truth clean tracking DataFrame.
        skeleton           : fitted MouseSkeleton instance.
        mouse_id           : mouse to evaluate.
        smooth_fn          : callable(df_corrupted) → (df_recovered, report).
        arena_diag         : arena diagonal in pixels for RMSE normalization.
        noise_std          : std-dev of Gaussian noise (px).
        dropout_rate       : fraction of rows to zero-out.
        swap_prob          : fraction of frames to swap identity.
        speed_spikes_prob  : fraction of frames to inject a sharp velocity spike.
        drift_amplitude    : peak amplitude of sinusoidal drift (px).
        threshold          : violation threshold (fraction).
        rng_seed           : reproducibility seed.

    Returns:
        DataFrame with one row per corruption type.
    """
    from src.skeleton.mouse_skeleton import _to_wide, _segment_length
    from src.skeleton import compute_constraint_violations

    def _viol_rate(df_l: pd.DataFrame) -> float:
        w = _to_wide(df_l[df_l["mouse_id"] == mouse_id])
        l = _segment_length(w, "nose", "tail_base")
        v = compute_constraint_violations(w, skeleton, {}, l, threshold=threshold)
        return float(v["violation_rate"].mean())

    def _male(df_l: pd.DataFrame) -> float:
        """Mean Absolute Length Error (MALE) — mean relative error across all edges."""
        w = _to_wide(df_l[df_l["mouse_id"] == mouse_id])
        l = _segment_length(w, "nose", "tail_base")
        v = compute_constraint_violations(w, skeleton, {}, l, threshold=threshold)
        return float(v["mean_error_rel"].mean()) if "mean_error_rel" in v.columns else np.nan

    def _rmse(df_recovered: pd.DataFrame, bp: str) -> float:
        return reconstruction_rmse(df_clean, df_recovered, mouse_id, bp, arena_diag)

    def _swap_recovery_rate(
        df_corr: pd.DataFrame, df_rec: pd.DataFrame, swap_frames_set: set
    ) -> float:
        """Fraction of injected swap frames where identity orientation was recovered.

        A swap frame is considered recovered if the body-axis cross-product in the
        recovered output is closer to the clean reference than in the corrupted input.

        Fully vectorized: precomputes per-frame body-axis cross product via pivot
        tables (O(N) instead of O(N·T) per-frame loops).
        """
        if not swap_frames_set:
            return np.nan
        mice = sorted(df_clean["mouse_id"].unique())
        if len(mice) < 2:
            # Identity swaps require ≥2 mice; with 1 mouse no swap was injected.
            return np.nan
        mid = mice[0]

        def _vectorized_crosses(df: pd.DataFrame) -> dict:
            """Return {frame: normalised_cross_product} for all frames at once."""
            sub = df[df["mouse_id"] == mid][["video_frame", "bodypart", "x", "y"]]
            needed = {"nose", "neck", "tail_base"}
            sub = sub[sub["bodypart"].isin(needed)]
            if sub.empty:
                return {}
            piv = sub.pivot_table(
                index="video_frame", columns="bodypart",
                values=["x", "y"], aggfunc="first",
            )
            # Flatten column multi-index → "nose_x", "nose_y", etc.
            piv.columns = [f"{bp}_{coord}" for coord, bp in piv.columns]
            req = ["nose_x", "nose_y", "neck_x", "neck_y", "tail_base_x", "tail_base_y"]
            if any(c not in piv.columns for c in req):
                return {}
            nx, ny   = piv["nose_x"].values,      piv["nose_y"].values
            cx, cy   = piv["neck_x"].values,      piv["neck_y"].values
            tx, ty   = piv["tail_base_x"].values, piv["tail_base_y"].values
            ux, uy   = nx - cx, ny - cy
            vx_, vy_ = tx - cx, ty - cy
            cross    = ux * vy_ - uy * vx_
            norm     = np.sqrt(ux ** 2 + uy ** 2) * np.sqrt(vx_ ** 2 + vy_ ** 2)
            safe_norm = np.where(norm > 1e-6, norm, 1.0)
            norm_cross = np.where(norm > 1e-6, cross / safe_norm, np.nan)
            return dict(zip(piv.index.tolist(), norm_cross.tolist()))

        c_clean_map = _vectorized_crosses(df_clean)
        c_corr_map  = _vectorized_crosses(df_corr)
        c_rec_map   = _vectorized_crosses(df_rec)
        if not c_clean_map:
            return np.nan

        recovered = 0
        valid_total = 0
        for fr in swap_frames_set:
            cc = c_clean_map.get(fr, np.nan)
            co = c_corr_map.get(fr,  np.nan)
            cr = c_rec_map.get(fr,   np.nan)
            if any(isinstance(v, float) and np.isnan(v) for v in (cc, co, cr)):
                continue
            valid_total += 1
            # Recovered: corrected sign matches clean AND corrupted sign didn't
            if np.sign(cr) == np.sign(cc) and np.sign(co) != np.sign(cc):
                recovered += 1
        return recovered / max(valid_total, 1) if valid_total > 0 else np.nan

    def _temporal_inconsistency(df_l: pd.DataFrame, bp: str = "nose") -> float:
        """Mean absolute jerk (3rd derivative of position), proxy for temporal consistency."""
        sub = (
            df_l[(df_l["bodypart"] == bp) & (df_l["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 4:
            return np.nan
        jerk_x = np.abs(np.diff(x, n=3))
        jerk_y = np.abs(np.diff(y, n=3))
        return float(np.nanmean(np.sqrt(jerk_x ** 2 + jerk_y ** 2))) / max(arena_diag, 1e-6)

    scenarios = [
        ("gaussian_noise", dict(noise_std=noise_std,  dropout_rate=0.0,          swap_prob=0.0,
                                speed_spikes_prob=0.0, drift_amplitude=0.0)),
        ("dropouts",       dict(noise_std=0.0,         dropout_rate=dropout_rate, swap_prob=0.0,
                                speed_spikes_prob=0.0, drift_amplitude=0.0)),
        ("swaps",          dict(noise_std=0.0,         dropout_rate=0.0,          swap_prob=swap_prob,
                                speed_spikes_prob=0.0, drift_amplitude=0.0)),
        ("speed_spikes",   dict(noise_std=0.0,         dropout_rate=0.0,          swap_prob=0.0,
                                speed_spikes_prob=speed_spikes_prob, drift_amplitude=0.0)),
        ("drift",          dict(noise_std=0.0,         dropout_rate=0.0,          swap_prob=0.0,
                                speed_spikes_prob=0.0, drift_amplitude=drift_amplitude)),
        ("all_combined",   dict(noise_std=noise_std,  dropout_rate=dropout_rate, swap_prob=swap_prob,
                                speed_spikes_prob=speed_spikes_prob, drift_amplitude=drift_amplitude)),
    ]

    rate_clean = _viol_rate(df_clean)
    ti_clean   = _temporal_inconsistency(df_clean)
    rows = []
    for name, params in scenarios:
        rng = np.random.default_rng(rng_seed)
        # For swap scenarios, track which frames were swapped for recovery scoring
        if params.get("swap_prob", 0) > 0:
            rng2 = np.random.default_rng(rng_seed)
            df_corr = inject_corruption(df_clean, rng=rng2, **params)
            # Re-derive swap frames by re-running the rng deterministically
            rng3 = np.random.default_rng(rng_seed)
            _n = len(df_clean)
            if params.get("noise_std", 0) > 0:
                rng3.normal(0.0, params["noise_std"], size=(_n, 2))
            if params.get("dropout_rate", 0) > 0:
                rng3.random(_n)
            frames_all = df_clean["video_frame"].unique()
            n_swaps = max(1, int(len(frames_all) * params["swap_prob"]))
            swap_frames_set = set(rng3.choice(frames_all, size=n_swaps, replace=False))
        else:
            df_corr = inject_corruption(df_clean, rng=rng, **params)
            swap_frames_set = set()

        rate_corrupt = _viol_rate(df_corr)
        male_corrupt = _male(df_corr)
        ti_corrupt   = _temporal_inconsistency(df_corr)
        try:
            df_rec, _ = smooth_fn(df_corr)
            rate_rec   = _viol_rate(df_rec)
            male_rec   = _male(df_rec)
            rmse_nose  = _rmse(df_rec, "nose")
            rmse_tail  = _rmse(df_rec, "tail_base")
            reduction  = (1 - rate_rec / max(rate_corrupt, 1e-9)) * 100
            male_reduction = (1 - male_rec / max(male_corrupt, 1e-9)) * 100
            ti_rec     = _temporal_inconsistency(df_rec)
            swap_rec   = _swap_recovery_rate(df_corr, df_rec, swap_frames_set)
        except Exception:
            rate_rec = np.nan; male_rec = np.nan; rmse_nose = np.nan; rmse_tail = np.nan
            reduction = np.nan; male_reduction = np.nan; ti_rec = np.nan; swap_rec = np.nan
        rows.append({
            "corruption_type":         name,
            "rate_clean":              round(rate_clean, 4),
            "viol_rate_corrupted":     round(rate_corrupt, 4),
            "viol_rate_recovered":     round(rate_rec, 4)  if not np.isnan(rate_rec)  else np.nan,
            "violation_reduction_pct": round(reduction, 1) if not np.isnan(reduction) else np.nan,
            "male_corrupted":          round(male_corrupt, 4),
            "male_recovered":          round(male_rec, 4)  if not np.isnan(male_rec)  else np.nan,
            "male_reduction_pct":      round(male_reduction, 1) if not np.isnan(male_reduction) else np.nan,
            "rmse_nose_norm":          round(rmse_nose, 5) if not np.isnan(rmse_nose) else np.nan,
            "rmse_tail_norm":          round(rmse_tail, 5) if not np.isnan(rmse_tail) else np.nan,
            "temporal_inconsistency_corrupted": round(ti_corrupt, 6) if not np.isnan(ti_corrupt) else np.nan,
            "temporal_inconsistency_recovered": round(ti_rec,     6) if not np.isnan(ti_rec)     else np.nan,
            "swap_recovery_rate":      round(swap_rec, 4) if (not isinstance(swap_rec, float) or not np.isnan(swap_rec)) else np.nan,
            "jerk_mean_norm":          round(ti_rec, 6) if not np.isnan(ti_rec) else np.nan,
        })
    return pd.DataFrame(rows)


# ── Temporal inconsistency score ──────────────────────────────────────────

def temporal_inconsistency_score(
    df_long: pd.DataFrame,
    mouse_id: int,
    arena_diag: float = 1.0,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """Compute temporal inconsistency (mean-jerk) per keypoint.

    Jerk = |Δ³pos| normalised by arena diagonal.  High values indicate
    abrupt, non-biological motion (typical of uncorrected swaps or dropouts).

    Returns:
        DataFrame with columns [keypoint, jerk_mean_norm, jerk_p95_norm].
    """
    if keypoints is None:
        keypoints = df_long["bodypart"].unique().tolist()
    rows = []
    for kp in keypoints:
        sub = (
            df_long[(df_long["bodypart"] == kp) & (df_long["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 4:
            rows.append({"keypoint": kp, "jerk_mean_norm": np.nan, "jerk_p95_norm": np.nan})
            continue
        jerk = np.sqrt(np.diff(x, n=3) ** 2 + np.diff(y, n=3) ** 2) / max(arena_diag, 1e-6)
        rows.append({
            "keypoint":      kp,
            "jerk_mean_norm": round(float(np.nanmean(jerk)), 6),
            "jerk_p95_norm":  round(float(np.nanpercentile(jerk, 95)), 6),
        })
    return pd.DataFrame(rows).sort_values("jerk_mean_norm", ascending=False).reset_index(drop=True)


# ── Displacement statistics ───────────────────────────────────────────────

def compute_displacement_stats(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    arena_diag: float = 1.0,
) -> pd.DataFrame:
    """Per-keypoint displacement statistics between before and after correction.

    Quantifies how much the optimizer moved each keypoint.  Useful for
    distinguishing under-correction (tiny displacements despite violations)
    from over-correction (large displacements everywhere).

    Metrics:
      - mean_disp_px    : mean Euclidean displacement per keypoint (pixels).
      - p95_disp_px     : 95th-percentile displacement (robust extreme).
      - max_disp_px     : worst-case displacement.
      - mean_disp_norm  : mean_disp_px / arena_diag (normalised).
      - pct_moved_gt5px : fraction of frames where displacement > 5 px.

    Args:
        df_before, df_after : long-format DataFrames before/after correction.
        mouse_id            : mouse to evaluate.
        arena_diag          : arena diagonal for normalisation.

    Returns:
        DataFrame sorted descending by mean_disp_px.
    """
    from src.skeleton.mouse_skeleton import _to_wide

    wide_b = _to_wide(df_before[df_before["mouse_id"] == mouse_id])
    wide_a = _to_wide(df_after[df_after["mouse_id"] == mouse_id])
    merged = wide_b.merge(wide_a, on="video_frame", suffixes=("_b", "_a"))

    rows = []
    kps = {
        col.replace("_x_b", "")
        for col in merged.columns
        if col.endswith("_x_b")
    }
    for kp in kps:
        xb = merged.get(f"{kp}_x_b")
        yb = merged.get(f"{kp}_y_b")
        xa = merged.get(f"{kp}_x_a")
        ya = merged.get(f"{kp}_y_a")
        if xb is None or xa is None:
            continue
        disp = np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
        valid = disp.notna().values
        if not valid.any():
            continue
        dv = disp.values[valid]
        rows.append({
            "keypoint":       kp,
            "mean_disp_px":   float(np.mean(dv)),
            "p95_disp_px":    float(np.percentile(dv, 95)),
            "max_disp_px":    float(np.max(dv)),
            "mean_disp_norm": float(np.mean(dv)) / max(arena_diag, 1e-6),
            "pct_moved_gt5px": float(np.mean(dv > 5.0) * 100),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean_disp_px", ascending=False)
        .reset_index(drop=True)
    )


# ── Convergence plot ──────────────────────────────────────────────────────

def plot_convergence_history(
    convergence_data,
    axes=None,
    title: str = "Optimiser convergence, per pass",
):
    """Plot per-pass objective loss from smooth_video convergence tracking.

    Args:
        convergence_data : list[float] (one value per pass) or
                           dict {label: list[float]} for multiple series.
        axes             : matplotlib Axes; creates a new figure if None.
        title            : plot title.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(figsize=(8, 4))

    if isinstance(convergence_data, (list, np.ndarray)):
        convergence_data = {"pipeline": list(convergence_data)}

    for label, losses in convergence_data.items():
        passes = list(range(1, len(losses) + 1))
        axes.plot(passes, losses, "o-", label=label, linewidth=2, markersize=5)

    axes.set_xlabel("Optimiser pass")
    axes.set_ylabel("Objective value (mean)")
    axes.set_title(title)
    axes.legend(fontsize=9)
    axes.grid(True, alpha=0.3)
    return axes


# ── Primary continuous metrics (Phase 1) ─────────────────────────────────

def normalized_rmse_per_keypoint(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    l_body_px: float,
    keypoints: list[str] | None = None,
    df_ground_truth: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Normalized RMSE per keypoint.

    If ``df_ground_truth`` is provided, computes RMSE of before/after vs. clean
    ground truth (for synthetic benchmark use).  Otherwise computes displacement
    (how far the correction moved each keypoint from the raw observation).

    Normalization: divided by ``l_body_px`` so values are body-length fractions.

    Returns:
        DataFrame with columns [keypoint, rmse_before_norm, rmse_after_norm,
        rmse_reduction_pct].
    """

    def _sel(df: pd.DataFrame, kp: str) -> pd.DataFrame:
        return (
            df[(df["bodypart"] == kp) & (df["mouse_id"] == mouse_id)]
            .sort_values("video_frame")[["video_frame", "x", "y"]]
        )

    if keypoints is None:
        keypoints = df_before["bodypart"].unique().tolist()

    rows = []
    for kp in keypoints:
        try:
            s_b = _sel(df_before, kp)
            s_a = _sel(df_after, kp)
            if df_ground_truth is not None:
                s_gt = _sel(df_ground_truth, kp)
                mb = s_gt.merge(s_b, on="video_frame", suffixes=("_gt", "_b"))
                ma = s_gt.merge(s_a, on="video_frame", suffixes=("_gt", "_a"))
                rmse_b = float(np.sqrt(np.mean(
                    (mb["x_gt"] - mb["x_b"]) ** 2 + (mb["y_gt"] - mb["y_b"]) ** 2
                ))) / max(l_body_px, 1e-6)
                rmse_a = float(np.sqrt(np.mean(
                    (ma["x_gt"] - ma["x_a"]) ** 2 + (ma["y_gt"] - ma["y_a"]) ** 2
                ))) / max(l_body_px, 1e-6)
            else:
                merged = s_b.merge(s_a, on="video_frame", suffixes=("_b", "_a"))
                disp = np.sqrt(
                    (merged["x_a"] - merged["x_b"]) ** 2
                    + (merged["y_a"] - merged["y_b"]) ** 2
                )
                rmse_b = np.nan
                rmse_a = float(np.sqrt(np.mean(disp ** 2))) / max(l_body_px, 1e-6)
            reduction = (
                (rmse_b - rmse_a) / max(rmse_b, 1e-12) * 100
                if not np.isnan(rmse_b)
                else np.nan
            )
            rows.append({
                "keypoint":          kp,
                "rmse_before_norm":  round(rmse_b, 6) if not np.isnan(rmse_b) else np.nan,
                "rmse_after_norm":   round(rmse_a, 6),
                "rmse_reduction_pct": round(reduction, 1) if not np.isnan(reduction) else np.nan,
            })
        except Exception:
            rows.append({
                "keypoint": kp,
                "rmse_before_norm": np.nan,
                "rmse_after_norm": np.nan,
                "rmse_reduction_pct": np.nan,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("rmse_after_norm", ascending=False)
        .reset_index(drop=True)
    )


def acceleration_smoothness(
    df: pd.DataFrame,
    mouse_id: int,
    l_body_px: float,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """RMS of 2nd-order finite differences per keypoint.

    Lower values indicate smoother motion.  Normalized by ``l_body_px``
    so values are interpretable as body-length fractions per frame².

    Returns:
        DataFrame with columns [keypoint, accel_rms_norm].
    """
    if keypoints is None:
        keypoints = df["bodypart"].unique().tolist()

    rows = []
    for kp in keypoints:
        sub = (
            df[(df["bodypart"] == kp) & (df["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 3:
            rows.append({"keypoint": kp, "accel_rms_norm": np.nan})
            continue
        ax = np.diff(x, n=2)
        ay = np.diff(y, n=2)
        accel_mag = np.sqrt(ax ** 2 + ay ** 2)
        rows.append({
            "keypoint":       kp,
            "accel_rms_norm": round(
                float(np.sqrt(np.nanmean(accel_mag ** 2))) / max(l_body_px, 1e-6),
                6,
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("accel_rms_norm", ascending=False)
        .reset_index(drop=True)
    )


def jerk_energy(
    df: pd.DataFrame,
    mouse_id: int,
    l_body_px: float,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """Sum of squared 3rd-order finite differences (jerk energy) per keypoint.

    Jerk energy = Σ_t ‖Δ³pos_t‖² / l_body_px².  Lower = smoother motion.
    Captures high-frequency jitter that acceleration smoothness may miss.

    Returns:
        DataFrame with columns [keypoint, jerk_energy_norm].
    """
    if keypoints is None:
        keypoints = df["bodypart"].unique().tolist()

    rows = []
    for kp in keypoints:
        sub = (
            df[(df["bodypart"] == kp) & (df["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 4:
            rows.append({"keypoint": kp, "jerk_energy_norm": np.nan})
            continue
        jx = np.diff(x, n=3)
        jy = np.diff(y, n=3)
        energy = float(np.nansum(jx ** 2 + jy ** 2)) / max(l_body_px, 1e-6) ** 2
        rows.append({"keypoint": kp, "jerk_energy_norm": round(energy, 4)})

    return (
        pd.DataFrame(rows)
        .sort_values("jerk_energy_norm", ascending=False)
        .reset_index(drop=True)
    )


def spectral_energy_ratio(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """Ratio of total PSD (spectral energy) before vs. after correction, per keypoint.

    A ratio near 1.0 means the correction preserved the motion energy spectrum.
    Values < 1.0 signal over-smoothing.  Values > 1.0 signal added energy.
    Computed on the x-coordinate via Welch's estimator, summed over all frequencies.

    Returns:
        DataFrame with columns [keypoint, energy_before, energy_after, energy_ratio].
    """
    if keypoints is None:
        keypoints = df_before["bodypart"].unique().tolist()

    rows = []
    for kp in keypoints:
        try:
            def _psd_total(df_src: pd.DataFrame) -> float:
                sub = (
                    df_src[(df_src["bodypart"] == kp) & (df_src["mouse_id"] == mouse_id)]
                    .sort_values("video_frame")
                )
                x = sub["x"].interpolate().values
                if len(x) <= 16:
                    return np.nan
                nperseg = min(64, max(4, len(x) // 4))
                _, psd = signal.welch(x, fs=fps, nperseg=nperseg)
                return float(np.sum(psd))

            e_b = _psd_total(df_before)
            e_a = _psd_total(df_after)
            ratio = (
                e_a / max(e_b, 1e-12)
                if not (np.isnan(e_b) or np.isnan(e_a))
                else np.nan
            )
            rows.append({
                "keypoint":     kp,
                "energy_before": round(e_b, 4) if not np.isnan(e_b) else np.nan,
                "energy_after":  round(e_a, 4) if not np.isnan(e_a) else np.nan,
                "energy_ratio":  round(ratio, 4) if not np.isnan(ratio) else np.nan,
            })
        except Exception:
            rows.append({
                "keypoint": kp,
                "energy_before": np.nan,
                "energy_after": np.nan,
                "energy_ratio": np.nan,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("energy_ratio", ascending=True)
        .reset_index(drop=True)
    )


def velocity_kl_divergence(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    arena_diag: float = 1.0,
    keypoints: list[str] | None = None,
    n_bins: int = 50,
) -> pd.DataFrame:
    """KL divergence between speed distributions before and after correction.

    KL(before ‖ after) — lower means correction preserved the speed distribution.
    Values > 0.1 signal that the correction significantly altered motion statistics.

    Returns:
        DataFrame with columns [keypoint, kl_div, ks_stat, ks_pval].
    """
    from scipy.stats import entropy as _scipy_entropy

    if keypoints is None:
        keypoints = df_before["bodypart"].unique().tolist()

    rows = []
    for kp in keypoints:
        try:
            v_b = normalized_speeds(df_before, mouse_id, kp, fps, arena_diag)
            v_a = normalized_speeds(df_after,  mouse_id, kp, fps, arena_diag)
            mask = ~np.isnan(v_b) & ~np.isnan(v_a)
            if mask.sum() < 10:
                rows.append({"keypoint": kp, "kl_div": np.nan,
                             "ks_stat": np.nan, "ks_pval": np.nan})
                continue
            vb_c = v_b[mask]
            va_c = v_a[mask]
            lo = min(vb_c.min(), va_c.min())
            hi = max(vb_c.max(), va_c.max()) + 1e-8
            bins = np.linspace(lo, hi, n_bins + 1)
            p, _ = np.histogram(vb_c, bins=bins, density=True)
            q, _ = np.histogram(va_c, bins=bins, density=True)
            eps = 1e-10
            p = (p + eps); p /= p.sum()
            q = (q + eps); q /= q.sum()
            kl = float(_scipy_entropy(p, q))
            ks_stat, ks_pval = ks_2samp(vb_c, va_c, method='asymp')
            rows.append({
                "keypoint": kp,
                "kl_div":  round(kl, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_pval": round(float(ks_pval), 4),
            })
        except Exception:
            rows.append({"keypoint": kp, "kl_div": np.nan,
                         "ks_stat": np.nan, "ks_pval": np.nan})

    return (
        pd.DataFrame(rows)
        .sort_values("kl_div", ascending=False)
        .reset_index(drop=True)
    )


def turning_angle_distribution(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """KS test comparing turning angle (heading change) distributions before/after.

    Turning angle = angle between consecutive velocity vectors.  Large KS
    statistics indicate the correction altered the animal's turning behavior.

    Returns:
        DataFrame with columns [keypoint, ks_stat, ks_pval,
        mean_turn_before_deg, mean_turn_after_deg].
    """
    if keypoints is None:
        keypoints = df_before["bodypart"].unique().tolist()

    def _turning_angles(df: pd.DataFrame, kp: str) -> np.ndarray:
        sub = (
            df[(df["bodypart"] == kp) & (df["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 3:
            return np.array([])
        dx = np.diff(x)
        dy = np.diff(y)
        angles = np.arctan2(dy, dx)
        d_angles = np.diff(angles)
        # Wrap to [−π, π]
        d_angles = (d_angles + np.pi) % (2 * np.pi) - np.pi
        speed = np.sqrt(dx[:-1] ** 2 + dy[:-1] ** 2)
        return d_angles[speed > 0.5]

    rows = []
    for kp in keypoints:
        ta_b = _turning_angles(df_before, kp)
        ta_a = _turning_angles(df_after, kp)
        if len(ta_b) < 5 or len(ta_a) < 5:
            rows.append({
                "keypoint": kp, "ks_stat": np.nan, "ks_pval": np.nan,
                "mean_turn_before_deg": np.nan, "mean_turn_after_deg": np.nan,
            })
            continue
        ks_stat, ks_pval = ks_2samp(ta_b, ta_a, method='asymp')
        rows.append({
            "keypoint":              kp,
            "ks_stat":               round(float(ks_stat), 4),
            "ks_pval":               round(float(ks_pval), 4),
            "mean_turn_before_deg":  round(float(np.degrees(np.nanmean(np.abs(ta_b)))), 2),
            "mean_turn_after_deg":   round(float(np.degrees(np.nanmean(np.abs(ta_a)))), 2),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("ks_stat", ascending=False)
        .reset_index(drop=True)
    )


def motion_preservation_report(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: int,
    fps: float,
    l_body_px: float,
    arena_diag: float = 1.0,
    keypoints: list[str] | None = None,
) -> pd.DataFrame:
    """Comprehensive motion preservation report.

    Aggregates primary continuous metrics into a single DataFrame.
    Each row is one (keypoint, metric) pair.

    Metrics:
      - accel_rms_norm     : acceleration smoothness (lower after = smoother)
      - jerk_energy_norm   : jerk energy (lower after = less jitter)
      - spectral_energy_ratio: PSD energy ratio (near 1.0 = preserved)
      - velocity_kl_div    : speed distribution KL divergence (near 0 = preserved)
      - turn_ks_stat       : turning angle KS statistic (near 0 = preserved)

    Returns:
        DataFrame with columns [keypoint, metric, value_before, value_after,
        delta, interpretation].
    """
    if keypoints is None:
        available = set(df_before["bodypart"].unique()) & set(df_after["bodypart"].unique())
        headline = [kp for kp in ["nose", "neck", "tail_base"] if kp in available]
        keypoints = headline if headline else list(available)

    rows = []

    # Acceleration smoothness
    accel_b = acceleration_smoothness(df_before, mouse_id, l_body_px, keypoints)
    accel_a = acceleration_smoothness(df_after,  mouse_id, l_body_px, keypoints)
    for _, rb in accel_b.iterrows():
        kp = rb["keypoint"]
        ra = accel_a[accel_a["keypoint"] == kp]
        if ra.empty:
            continue
        vb = rb["accel_rms_norm"]
        va = ra.iloc[0]["accel_rms_norm"]
        delta = round(va - vb, 6) if not (np.isnan(vb) or np.isnan(va)) else np.nan
        rows.append({"keypoint": kp, "metric": "accel_rms_norm",
                     "value_before": vb, "value_after": va,
                     "delta": delta, "interpretation": "lower_is_better"})

    # Jerk energy
    jerk_b = jerk_energy(df_before, mouse_id, l_body_px, keypoints)
    jerk_a = jerk_energy(df_after,  mouse_id, l_body_px, keypoints)
    for _, rb in jerk_b.iterrows():
        kp = rb["keypoint"]
        ra = jerk_a[jerk_a["keypoint"] == kp]
        if ra.empty:
            continue
        vb = rb["jerk_energy_norm"]
        va = ra.iloc[0]["jerk_energy_norm"]
        delta = round(va - vb, 6) if not (np.isnan(vb) or np.isnan(va)) else np.nan
        rows.append({"keypoint": kp, "metric": "jerk_energy_norm",
                     "value_before": vb, "value_after": va,
                     "delta": delta, "interpretation": "lower_is_better"})

    # Spectral energy ratio
    spec = spectral_energy_ratio(df_before, df_after, mouse_id, fps, keypoints)
    for _, rs in spec.iterrows():
        ratio = rs["energy_ratio"]
        delta = round(ratio - 1.0, 6) if not np.isnan(ratio) else np.nan
        rows.append({"keypoint": rs["keypoint"], "metric": "spectral_energy_ratio",
                     "value_before": 1.0, "value_after": ratio,
                     "delta": delta, "interpretation": "near_1.0_is_best"})

    # KL divergence
    kl = velocity_kl_divergence(df_before, df_after, mouse_id, fps, arena_diag, keypoints)
    for _, rk in kl.iterrows():
        kl_val = rk["kl_div"]
        delta = round(kl_val, 6) if not np.isnan(kl_val) else np.nan
        rows.append({"keypoint": rk["keypoint"], "metric": "velocity_kl_divergence",
                     "value_before": 0.0, "value_after": kl_val,
                     "delta": delta, "interpretation": "lower_is_better"})

    # Turning angle KS
    ta = turning_angle_distribution(df_before, df_after, mouse_id, keypoints)
    for _, rt in ta.iterrows():
        ks = rt["ks_stat"]
        delta = round(ks, 6) if not np.isnan(ks) else np.nan
        rows.append({"keypoint": rt["keypoint"], "metric": "turn_ks_stat",
                     "value_before": 0.0, "value_after": ks,
                     "delta": delta, "interpretation": "lower_is_better"})

    return pd.DataFrame(rows)


# ── Realistic corruption injection (Phase 3) ─────────────────────────────

def inject_occlusion_sequence(
    df_clean: pd.DataFrame,
    keypoint_groups: dict | None = None,
    min_len: int = 10,
    max_len: int = 50,
    n_segments: int = 3,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject contiguous occlusion sequences (hard benchmark corruption).

    Unlike random dropouts, occlusions are contiguous segments where tracking
    is lost for an entire keypoint group for 10–50 consecutive frames.  This
    is far harder than random dropout because:
      - missing segments span more frames than any smoothing window
      - the optimizer cannot anchor on nearby group keypoints

    Args:
        df_clean        : clean tracking DataFrame.
        keypoint_groups : dict group_name → list[keypoint].  Defaults to
                          {"head": ["nose","ear_left","ear_right"],
                           "pelvis": ["hip_left","hip_right","tail_base"]}.
        min_len, max_len: range of contiguous occlusion lengths in frames.
        n_segments      : occlusion events per group.
        rng             : numpy Generator (seed 42 if None).

    Returns:
        Corrupted copy with occlusion frames set to (0, 0).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if keypoint_groups is None:
        keypoint_groups = {
            "head":   ["nose", "ear_left", "ear_right"],
            "pelvis": ["hip_left", "hip_right", "tail_base"],
        }

    df = df_clean.copy()
    frames = sorted(df["video_frame"].unique())
    T = len(frames)

    for _group_name, kps in keypoint_groups.items():
        kps_present = [k for k in kps if k in df["bodypart"].unique()]
        if not kps_present:
            continue
        for _ in range(n_segments):
            seg_len = int(rng.integers(min_len, max_len + 1))
            if T - seg_len <= 0:
                continue
            start_idx = int(rng.integers(0, T - seg_len))
            occlusion_frames = set(frames[start_idx: start_idx + seg_len])
            mask = (
                df["video_frame"].isin(occlusion_frames)
                & df["bodypart"].isin(kps_present)
            )
            df.loc[mask, ["x", "y"]] = 0.0

    return df


def inject_fast_motion_burst(
    df_clean: pd.DataFrame,
    event_type: str = "grooming",
    n_events: int = 5,
    duration_frames: int = 15,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject synthetic fast-motion bursts to stress-test temporal smoothing.

    Models rapid behaviors that break naive smoothness priors:
      - grooming  : rapid head oscillation (±20–40 px at high frequency).
      - jump      : whole-body single-frame large displacement then return.
      - collision : correlated displacement of a random subset of keypoints.

    Args:
        df_clean       : clean tracking DataFrame.
        event_type     : "grooming", "jump", or "collision".
        n_events       : number of burst events.
        duration_frames: frames per burst (for grooming; jump uses 1 frame).
        rng            : numpy Generator (seed 42 if None).

    Returns:
        Corrupted copy of df_clean.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    df = df_clean.copy()
    frames = sorted(df["video_frame"].unique())
    T = len(frames)

    for _ in range(n_events):
        if event_type == "grooming":
            seg_len = duration_frames
            if T - seg_len <= 0:
                continue
            start_idx = int(rng.integers(0, T - seg_len))
            burst_frames = frames[start_idx: start_idx + seg_len]
            head_kps = [k for k in ["nose", "ear_left", "ear_right"]
                        if k in df["bodypart"].unique()]
            amp = float(rng.uniform(20.0, 40.0))
            freq = float(rng.uniform(3.0, 8.0))
            for i, fr in enumerate(burst_frames):
                phase = 2 * np.pi * freq * i / max(seg_len, 1)
                dx = amp * np.sin(phase)
                dy = amp * 0.3 * np.cos(phase * 1.3)
                mask = (df["video_frame"] == fr) & df["bodypart"].isin(head_kps)
                df.loc[mask, "x"] += dx
                df.loc[mask, "y"] += dy

        elif event_type == "jump":
            if T < 3:
                continue
            fr_idx = int(rng.integers(1, T - 1))
            fr = frames[fr_idx]
            mag = float(rng.uniform(40.0, 80.0))
            angle = float(rng.uniform(0, 2 * np.pi))
            mask = df["video_frame"] == fr
            df.loc[mask, "x"] += mag * np.cos(angle)
            df.loc[mask, "y"] += mag * np.sin(angle)

        elif event_type == "collision":
            if T < 2:
                continue
            fr_idx = int(rng.integers(0, T))
            fr = frames[fr_idx]
            all_kps = df["bodypart"].unique().tolist()
            n_affected = max(2, int(len(all_kps) * 0.5))
            affected_kps = rng.choice(all_kps, size=n_affected, replace=False).tolist()
            mag = float(rng.uniform(20.0, 50.0))
            angle = float(rng.uniform(0, 2 * np.pi))
            mask = (df["video_frame"] == fr) & df["bodypart"].isin(affected_kps)
            df.loc[mask, "x"] += mag * np.cos(angle)
            df.loc[mask, "y"] += mag * np.sin(angle)

    return df


def inject_correlated_corruption(
    df_clean: pd.DataFrame,
    keypoint_groups: dict | None = None,
    corruption_prob: float = 0.05,
    noise_std: float = 10.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject realistic group-correlated multi-keypoint corruption.

    In real DLC failures, keypoints rarely fail independently: when tracking
    the nose fails, the ear detections usually degrade too.  Here, corruption
    is sampled per-group: when a group's frame is selected, ALL keypoints in
    that group receive the same displacement, making recovery harder because
    skeletal constraints can't use co-located group keypoints as anchors.

    Args:
        df_clean        : clean tracking DataFrame.
        keypoint_groups : dict group_name → list[keypoint].  Defaults to
                          {"head", "neck_torso", "pelvis"}.
        corruption_prob : probability a given frame has group-level corruption.
        noise_std       : std-dev of the correlated Gaussian noise applied.
        rng             : numpy Generator (seed 42 if None).

    Returns:
        Corrupted copy of df_clean.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if keypoint_groups is None:
        keypoint_groups = {
            "head":       ["nose", "ear_left", "ear_right"],
            "neck_torso": ["neck"],
            "pelvis":     ["hip_left", "hip_right", "tail_base"],
        }

    df = df_clean.copy()
    frames = sorted(df["video_frame"].unique())

    for _group_name, kps in keypoint_groups.items():
        kps_present = [k for k in kps if k in df["bodypart"].unique()]
        if not kps_present:
            continue
        corrupt_mask = rng.random(len(frames)) < corruption_prob
        for fr, is_corrupt in zip(frames, corrupt_mask):
            if not is_corrupt:
                continue
            # All keypoints in the group get the SAME noise (correlated failure)
            dx = float(rng.normal(0.0, noise_std))
            dy = float(rng.normal(0.0, noise_std))
            mask = (df["video_frame"] == fr) & df["bodypart"].isin(kps_present)
            df.loc[mask, "x"] += dx
            df.loc[mask, "y"] += dy

    return df


def benchmark_realistic_corruptions(
    df_clean: pd.DataFrame,
    skeleton,
    mouse_id: int,
    smooth_fn,
    arena_diag: float = 1.0,
    rng_seed: int = 42,
    threshold: float = 0.20,
) -> pd.DataFrame:
    """Recovery benchmarks for realistic (ecologically valid) corruption types.

    Complements ``benchmark_by_corruption_type`` (idealized independent
    corruptions) with harder failure modes:
      - occlusion_head    : 10–50 frame contiguous head-group occlusion
      - occlusion_pelvis  : 10–50 frame contiguous pelvis-group occlusion
      - fast_grooming     : high-frequency head oscillation bursts
      - fast_jump         : whole-body teleportation events
      - correlated_noise  : group-level correlated Gaussian noise

    Returns:
        DataFrame with one row per corruption type.
    """
    from src.skeleton.mouse_skeleton import _to_wide, _segment_length
    from src.skeleton import compute_constraint_violations

    def _viol_rate(df_l: pd.DataFrame) -> float:
        w = _to_wide(df_l[df_l["mouse_id"] == mouse_id])
        l = _segment_length(w, "nose", "tail_base")
        v = compute_constraint_violations(w, skeleton, {}, l, threshold=threshold)
        return float(v["violation_rate"].mean())

    def _male(df_l: pd.DataFrame) -> float:
        w = _to_wide(df_l[df_l["mouse_id"] == mouse_id])
        l = _segment_length(w, "nose", "tail_base")
        v = compute_constraint_violations(w, skeleton, {}, l, threshold=threshold)
        return float(v["mean_error_rel"].mean()) if "mean_error_rel" in v.columns else np.nan

    def _rmse(df_rec: pd.DataFrame, bp: str) -> float:
        return reconstruction_rmse(df_clean, df_rec, mouse_id, bp, arena_diag)

    def _ti(df_l: pd.DataFrame, bp: str = "nose") -> float:
        sub = (
            df_l[(df_l["bodypart"] == bp) & (df_l["mouse_id"] == mouse_id)]
            .sort_values("video_frame")
        )
        x = sub["x"].interpolate().values
        y = sub["y"].interpolate().values
        if len(x) < 4:
            return np.nan
        jx = np.abs(np.diff(x, n=3))
        jy = np.abs(np.diff(y, n=3))
        return float(np.nanmean(np.sqrt(jx ** 2 + jy ** 2))) / max(arena_diag, 1e-6)

    kp_groups = {
        "head":   ["nose", "ear_left", "ear_right"],
        "pelvis": ["hip_left", "hip_right", "tail_base"],
    }

    scenarios = [
        ("occlusion_head", lambda: inject_occlusion_sequence(
            df_clean,
            keypoint_groups={"head": kp_groups["head"]},
            min_len=10, max_len=50, n_segments=3,
            rng=np.random.default_rng(rng_seed),
        )),
        ("occlusion_pelvis", lambda: inject_occlusion_sequence(
            df_clean,
            keypoint_groups={"pelvis": kp_groups["pelvis"]},
            min_len=10, max_len=50, n_segments=3,
            rng=np.random.default_rng(rng_seed),
        )),
        ("fast_grooming", lambda: inject_fast_motion_burst(
            df_clean, event_type="grooming", n_events=5, duration_frames=15,
            rng=np.random.default_rng(rng_seed),
        )),
        ("fast_jump", lambda: inject_fast_motion_burst(
            df_clean, event_type="jump", n_events=10,
            rng=np.random.default_rng(rng_seed),
        )),
        ("correlated_noise", lambda: inject_correlated_corruption(
            df_clean, keypoint_groups=kp_groups,
            corruption_prob=0.05, noise_std=15.0,
            rng=np.random.default_rng(rng_seed),
        )),
    ]

    rows = []
    rate_clean = _viol_rate(df_clean)
    for name, corrupt_fn in scenarios:
        df_corr = corrupt_fn()
        rate_c = _viol_rate(df_corr)
        male_c = _male(df_corr)
        ti_c   = _ti(df_corr)
        try:
            df_rec, _ = smooth_fn(df_corr)
            rate_r    = _viol_rate(df_rec)
            male_r    = _male(df_rec)
            rmse_n    = _rmse(df_rec, "nose")
            rmse_t    = _rmse(df_rec, "tail_base")
            reduction = (1 - rate_r / max(rate_c, 1e-9)) * 100
            male_red  = (1 - male_r / max(male_c, 1e-9)) * 100
            ti_r      = _ti(df_rec)
        except Exception:
            rate_r = np.nan; male_r = np.nan; rmse_n = np.nan; rmse_t = np.nan
            reduction = np.nan; male_red = np.nan; ti_r = np.nan

        rows.append({
            "corruption_type":                   name,
            "rate_clean":                        round(rate_clean, 4),
            "viol_rate_corrupted":               round(rate_c, 4),
            "viol_rate_recovered":               round(rate_r, 4) if not np.isnan(rate_r) else np.nan,
            "violation_reduction_pct":           round(reduction, 1) if not np.isnan(reduction) else np.nan,
            "male_corrupted":                    round(male_c, 4),
            "male_recovered":                    round(male_r, 4) if not np.isnan(male_r) else np.nan,
            "male_reduction_pct":                round(male_red, 1) if not np.isnan(male_red) else np.nan,
            "rmse_nose_norm":                    round(rmse_n, 5) if not np.isnan(rmse_n) else np.nan,
            "rmse_tail_norm":                    round(rmse_t, 5) if not np.isnan(rmse_t) else np.nan,
            "temporal_inconsistency_corrupted":  round(ti_c, 6) if not np.isnan(ti_c) else np.nan,
            "temporal_inconsistency_recovered":  round(ti_r, 6) if not np.isnan(ti_r) else np.nan,
        })

    return pd.DataFrame(rows)


# ── Statistical validation (Phase 6) ─────────────────────────────────────

def paired_wilcoxon(
    before_series,
    after_series,
) -> dict:
    """Paired Wilcoxon signed-rank test between two matched series.

    Tests whether the difference ``before − after`` differs significantly from
    zero.  Non-parametric alternative to the paired t-test; appropriate for
    non-Gaussian trajectory metrics.

    Returns:
        dict with keys: statistic, p_value, effect_size (rank-biserial r),
        n_pairs, significant (bool, p < 0.05).
    """
    from scipy.stats import wilcoxon as _wilcoxon

    a = np.asarray(before_series, dtype=float)
    b = np.asarray(after_series, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    n = int(valid.sum())
    if n < 5:
        return {"statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "n_pairs": n, "significant": False}

    diffs = a[valid] - b[valid]
    nonzero = diffs != 0
    if nonzero.sum() < 5:
        return {"statistic": np.nan, "p_value": 1.0,
                "effect_size": 0.0, "n_pairs": n, "significant": False}

    stat, pval = _wilcoxon(diffs[nonzero], alternative="two-sided")
    n_nz = int(nonzero.sum())
    # Rank-biserial correlation r = 1 − 2W / (n*(n+1))
    r = 1.0 - (2.0 * float(stat)) / max(n_nz * (n_nz + 1), 1)
    return {
        "statistic":   round(float(stat), 4),
        "p_value":     round(float(pval), 4),
        "effect_size": round(float(r), 4),
        "n_pairs":     n,
        "significant": bool(pval < 0.05),
    }


def bootstrap_ci(
    values,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    statistic: str = "mean",
    rng_seed: int = 0,
) -> tuple:
    """Bootstrap confidence interval for a scalar statistic.

    Args:
        values      : 1-D array of observations.
        n_bootstrap : number of bootstrap resamples.
        ci          : confidence level (e.g. 0.95 for 95% CI).
        statistic   : "mean", "median", or "std".
        rng_seed    : reproducibility seed.

    Returns:
        (lower, upper) — confidence interval bounds.
    """
    rng = np.random.default_rng(rng_seed)
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        mu = float(np.nanmean(v)) if len(v) > 0 else np.nan
        return (mu, mu)
    stat_fn = {"mean": np.mean, "median": np.median, "std": np.std}.get(
        statistic, np.mean
    )
    boot_stats = np.array([
        stat_fn(rng.choice(v, size=len(v), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot_stats, alpha * 100))
    hi = float(np.percentile(boot_stats, (1 - alpha) * 100))
    return (round(lo, 6), round(hi, 6))


def compare_variants_statistically(
    results_dict: dict,
    metric: str,
    baseline_key: str | None = None,
) -> pd.DataFrame:
    """Pairwise statistical comparison of variants using paired Wilcoxon test.

    Args:
        results_dict : {variant_label: list/array of per-frame metric values}.
                       All arrays must have the same length (aligned observations).
        metric       : metric name (for display in the output).
        baseline_key : baseline variant key.  If None, the first key is used.

    Returns:
        DataFrame with columns [variant_a, variant_b, metric, statistic,
        p_value, effect_size, n_pairs, significant].
    """
    keys = list(results_dict.keys())
    if baseline_key is None:
        baseline_key = keys[0]
    comparisons = [(baseline_key, k) for k in keys if k != baseline_key]

    rows = []
    for ka, kb in comparisons:
        result = paired_wilcoxon(results_dict[ka], results_dict[kb])
        rows.append({"variant_a": ka, "variant_b": kb, "metric": metric, **result})

    return pd.DataFrame(rows)


# ── Phase-E diagnostic metrics ────────────────────────────────────────────────

def bone_length_variance(
    df: pd.DataFrame,
    skeleton,
    mouse_id: str,
) -> pd.DataFrame:
    """Per-edge bone-length statistics: mean, std, and coefficient of variation.

    Parameters
    ----------
    df:
        Long-format tracking DataFrame with columns
        [``seq_id``, ``mouse_id``, ``bodypart``, ``frame``, ``x``, ``y``].
    skeleton:
        MouseSkeleton instance; its ``edges`` attribute is a list of
        ``(bp_a, bp_b)`` string tuples.
    mouse_id:
        The mouse identifier to analyse.

    Returns
    -------
    DataFrame with columns [edge, mean_px, std_px, cv].
    """
    sub = df[df["mouse_id"] == mouse_id]
    pivot = sub.pivot_table(index=["seq_id", "frame"],
                            columns="bodypart",
                            values=["x", "y"])
    pivot.columns = [f"{v}_{c}" for v, c in pivot.columns]
    pivot = pivot.reset_index(drop=True)

    rows = []
    for bp_a, bp_b in skeleton.edges:
        xa, ya = f"x_{bp_a}", f"y_{bp_a}"
        xb, yb = f"x_{bp_b}", f"y_{bp_b}"
        if xa not in pivot or xb not in pivot:
            continue
        lengths = np.sqrt(
            (pivot[xa] - pivot[xb]) ** 2 + (pivot[ya] - pivot[yb]) ** 2
        ).dropna()
        if len(lengths) == 0:
            continue
        m, s = float(lengths.mean()), float(lengths.std(ddof=1))
        rows.append({
            "edge": f"{bp_a}—{bp_b}",
            "mean_px": m,
            "std_px": s,
            "cv": s / m if m > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def temporal_curvature_score(
    df: pd.DataFrame,
    mouse_id: str,
    keypoints: list[str] | None = None,
    fps: float = 30.0,
) -> float:
    """Mean normalised second-order temporal derivative (|d²x/dt²| / l_body).

    Lower values indicate smoother trajectories.

    Parameters
    ----------
    df:
        Long-format tracking DataFrame.
    mouse_id:
        Mouse identifier.
    keypoints:
        Keypoints to average over.  Defaults to all available.
    fps:
        Frames per second (used to convert frames to seconds).

    Returns
    -------
    Mean |acceleration| across all keypoints and frames, in px / s².
    """
    sub = df[df["mouse_id"] == mouse_id].copy()
    if keypoints is None:
        keypoints = list(sub["bodypart"].unique())

    pivot = sub.pivot_table(index=["seq_id", "frame"],
                            columns="bodypart",
                            values=["x", "y"])
    pivot.columns = [f"{v}_{c}" for v, c in pivot.columns]
    pivot = pivot.sort_index()

    accel_norms = []
    for kp in keypoints:
        xc, yc = f"x_{kp}", f"y_{kp}"
        if xc not in pivot:
            continue
        xy = pivot[[xc, yc]].values.astype(float)  # (T, 2)
        # second-order difference along the time axis
        acc = xy[2:] - 2.0 * xy[1:-1] + xy[:-2]   # (T-2, 2)
        norms = np.sqrt(np.sum(acc ** 2, axis=1)) * (fps ** 2)
        accel_norms.append(np.nanmean(norms))

    return float(np.mean(accel_norms)) if accel_norms else np.nan


def psd_smoothness_score(
    df: pd.DataFrame,
    mouse_id: str,
    fps: float = 30.0,
    keypoints: list[str] | None = None,
    high_freq_cutoff: float = 4.0,
) -> float:
    """Ratio of high-frequency (> *high_freq_cutoff* Hz) to total PSD power.

    Lower values indicate smoother, more low-frequency motion.

    Parameters
    ----------
    df:
        Long-format tracking DataFrame.
    mouse_id:
        Mouse identifier.
    fps:
        Frames per second.
    keypoints:
        Keypoints to average over.  Defaults to all available.
    high_freq_cutoff:
        Frequency (Hz) above which power is considered high-frequency noise.

    Returns
    -------
    Mean high-freq / total power ratio across keypoints.
    """
    sub = df[df["mouse_id"] == mouse_id].copy()
    if keypoints is None:
        keypoints = list(sub["bodypart"].unique())

    pivot = sub.pivot_table(index=["seq_id", "frame"],
                            columns="bodypart",
                            values=["x", "y"])
    pivot.columns = [f"{v}_{c}" for v, c in pivot.columns]
    pivot = pivot.sort_index()

    ratios = []
    for kp in keypoints:
        for coord in ["x", "y"]:
            col = f"{coord}_{kp}"
            if col not in pivot:
                continue
            series = pivot[col].dropna().values.astype(float)
            if len(series) < 16:
                continue
            freqs, psd = signal.welch(series, fs=fps, nperseg=min(256, len(series)))
            total_power = np.sum(psd)
            if total_power == 0:
                continue
            high_power = np.sum(psd[freqs > high_freq_cutoff])
            ratios.append(high_power / total_power)

    return float(np.mean(ratios)) if ratios else np.nan


def tracking_recovery_time(
    corrected_df: pd.DataFrame,
    before_df: pd.DataFrame,
    outlier_frames: np.ndarray,
    mouse_id: str,
    keypoint: str = "nose",
    threshold_px: float = 5.0,
) -> float:
    """Median frames needed to re-converge to a stable trajectory after outliers.

    For each known outlier frame, the function looks ahead in the corrected
    trajectory and finds the first frame where the position difference from
    the clean reference has dropped below *threshold_px*.

    Parameters
    ----------
    corrected_df:
        Long-format DataFrame after skeleton correction.
    before_df:
        Long-format DataFrame before correction (treated as reference).
    outlier_frames:
        1-D array of frame indices that were flagged as outliers.
    mouse_id:
        Mouse identifier.
    keypoint:
        Keypoint to measure recovery on.
    threshold_px:
        Position error (px) below which the trajectory is considered recovered.

    Returns
    -------
    Median recovery time in frames (NaN if no valid measurements).
    """
    def _xy(source: pd.DataFrame) -> dict[int, np.ndarray]:
        sub = source[(source["mouse_id"] == mouse_id) &
                     (source["bodypart"] == keypoint)]
        return {int(r.frame): np.array([r.x, r.y], dtype=float)
                for _, r in sub.iterrows()}

    ref_xy  = _xy(before_df)
    cor_xy  = _xy(corrected_df)
    all_frames = sorted(cor_xy)
    max_frame  = max(all_frames) if all_frames else 0

    recovery_times = []
    for of in outlier_frames:
        of = int(of)
        for lag in range(0, 50):
            f = of + lag
            if f > max_frame:
                break
            if f not in cor_xy or f not in ref_xy:
                continue
            err = float(np.linalg.norm(cor_xy[f] - ref_xy[f]))
            if err < threshold_px:
                recovery_times.append(lag)
                break

    return float(np.median(recovery_times)) if recovery_times else np.nan


# ── NEW: Identity tracking metrics (Step 2a) ─────────────────────────────────

def identity_tracking_metrics(
    df: pd.DataFrame,
    fps: float = 30.0,
) -> dict:
    """Quantify inter-mouse identity persistence based on body-axis orientation.

    Detects identity swaps as sign-flips in the cross product of the body-axis
    vectors of each mouse pair.  A sustained flip (≥ 2 consecutive frames) is
    counted as a swap *event*; its duration is the run length.

    Uses ``video_frame`` / ``mouse_id`` / ``bodypart`` / ``x`` / ``y`` schema.

    Parameters
    ----------
    df:
        Long-format tracking DataFrame (either raw or corrected).
    fps:
        Recording frame rate (Hz).  Used to express durations in seconds.

    Returns
    -------
    dict with keys:
        swap_count               – total number of swap events detected
        mean_swap_duration_s     – mean swap event duration in seconds
        id_persistence_score     – fraction of frames without a swap (0–1)
        fragmentation_count      – number of orientation-sign run-length segments
    """
    mice = sorted(df["mouse_id"].unique())
    if len(mice) < 2:
        return {
            "swap_count": 0,
            "mean_swap_duration_s": 0.0,
            "id_persistence_score": 1.0,
            "fragmentation_count": 0,
        }

    m0, m1 = mice[0], mice[1]

    def _body_axis(sub: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
        """Return (N,2) nose→tail_base vectors per frame; NaN when missing."""
        # Prefer nose→tail_base; fall back to whichever head/tail kps exist
        _head_kps = ["nose", "neck"]
        _tail_kps = ["tail_base", "tail"]
        fr_map = {f: i for i, f in enumerate(frames)}
        out = np.full((len(frames), 2), np.nan)
        for h_kp in _head_kps:
            h = sub[sub["bodypart"] == h_kp].copy()
            if h.empty:
                continue
            for t_kp in _tail_kps:
                t = sub[sub["bodypart"] == t_kp].copy()
                if t.empty:
                    continue
                hm = h.set_index("video_frame")[["x", "y"]]
                tm = t.set_index("video_frame")[["x", "y"]]
                for f in frames:
                    if f in hm.index and f in tm.index:
                        hi = fr_map[f]
                        out[hi] = (tm.loc[f, ["x", "y"]].values
                                   - hm.loc[f, ["x", "y"]].values)
                return out
        return out

    frames = np.array(sorted(set(df["video_frame"].unique())))
    sub0 = df[df["mouse_id"] == m0]
    sub1 = df[df["mouse_id"] == m1]
    ax0 = _body_axis(sub0, frames)
    ax1 = _body_axis(sub1, frames)

    # Cross product z-component: ax0 × ax1  (positive = ax1 is CCW of ax0)
    cross = ax0[:, 0] * ax1[:, 1] - ax0[:, 1] * ax1[:, 0]
    valid = np.isfinite(cross) & (cross != 0)
    sign_series = np.sign(cross[valid])

    if len(sign_series) < 2:
        return {
            "swap_count": 0,
            "mean_swap_duration_s": 0.0,
            "id_persistence_score": 1.0,
            "fragmentation_count": 0,
        }

    # Detect sign-flip runs of length >= 2
    flips = np.where(np.diff(sign_series) != 0)[0] + 1  # positions where sign changes
    run_starts = np.concatenate([[0], flips])
    run_ends   = np.concatenate([flips, [len(sign_series)]])
    run_lengths = run_ends - run_starts

    # A swap event: sign flip followed by a run of ≥ 2 frames with the new sign
    swap_durations = []
    for i in range(1, len(run_lengths)):  # skip first run (baseline orientation)
        if run_lengths[i] >= 2:
            swap_durations.append(int(run_lengths[i]))

    n_valid = int(valid.sum())
    swap_frame_count = sum(swap_durations)

    return {
        "swap_count": len(swap_durations),
        "mean_swap_duration_s": float(np.mean(swap_durations) / fps) if swap_durations else 0.0,
        "id_persistence_score": float(1.0 - swap_frame_count / max(n_valid, 1)),
        "fragmentation_count": len(run_starts),
    }


# ── NEW: Spectral diagnostics (Step 2b) ──────────────────────────────────────

def spectral_entropy(
    df: pd.DataFrame,
    mouse_id: str,
    fps: float = 30.0,
    keypoints: list[str] | None = None,
) -> float:
    """Shannon entropy of the normalised Welch PSD (averaged over keypoints/axes).

    Low spectral entropy → energy concentrated at few frequencies (smooth motion).
    High spectral entropy → energy spread broadly (noisy / erratic motion).

    Uses ``video_frame`` / ``mouse_id`` / ``bodypart`` / ``x`` / ``y`` schema.

    Parameters
    ----------
    df        : Long-format tracking DataFrame.
    mouse_id  : Identity label to filter on.
    fps       : Recording frame rate (Hz).
    keypoints : Keypoints to include (default: all available).

    Returns
    -------
    Mean spectral entropy (nats) across all keypoint-axes.  NaN if insufficient data.
    """
    sub = df[df["mouse_id"] == mouse_id].sort_values("video_frame")
    if keypoints is None:
        keypoints = list(sub["bodypart"].unique())

    entropies = []
    for kp in keypoints:
        kp_df = sub[sub["bodypart"] == kp].sort_values("video_frame")
        for col in ["x", "y"]:
            series = kp_df[col].values.astype(float)
            valid = np.isfinite(series)
            if valid.sum() < 16:
                continue
            series = series[valid]
            freqs, psd = signal.welch(series, fs=fps,
                                      nperseg=min(256, len(series)))
            total = psd.sum()
            if total <= 0:
                continue
            p = psd / total
            # Avoid log(0): mask zero bins
            p_nz = p[p > 0]
            entropies.append(float(-np.sum(p_nz * np.log(p_nz))))

    return float(np.mean(entropies)) if entropies else np.nan


def motion_bandwidth(
    df: pd.DataFrame,
    mouse_id: str,
    fps: float = 30.0,
    keypoints: list[str] | None = None,
    power_fraction: float = 0.90,
) -> float:
    """Frequency below which ``power_fraction`` of PSD power is contained (Hz).

    Lower bandwidth → smoother, slower-varying trajectory.
    Higher bandwidth → more high-frequency content (fast motion or noise).

    Uses ``video_frame`` / ``mouse_id`` / ``bodypart`` / ``x`` / ``y`` schema.

    Parameters
    ----------
    df             : Long-format tracking DataFrame.
    mouse_id       : Identity label to filter on.
    fps            : Recording frame rate (Hz).
    keypoints      : Keypoints to include (default: all available).
    power_fraction : Cumulative power threshold (default 0.90 = 90% bandwidth).

    Returns
    -------
    Mean 90% bandwidth (Hz) across keypoint-axes.  NaN if insufficient data.
    """
    sub = df[df["mouse_id"] == mouse_id].sort_values("video_frame")
    if keypoints is None:
        keypoints = list(sub["bodypart"].unique())

    bandwidths = []
    for kp in keypoints:
        kp_df = sub[sub["bodypart"] == kp].sort_values("video_frame")
        for col in ["x", "y"]:
            series = kp_df[col].values.astype(float)
            valid = np.isfinite(series)
            if valid.sum() < 16:
                continue
            series = series[valid]
            freqs, psd = signal.welch(series, fs=fps,
                                      nperseg=min(256, len(series)))
            total = psd.sum()
            if total <= 0:
                continue
            cum_power = np.cumsum(psd) / total
            idx = int(np.searchsorted(cum_power, power_fraction))
            idx = min(idx, len(freqs) - 1)
            bandwidths.append(float(freqs[idx]))

    return float(np.mean(bandwidths)) if bandwidths else np.nan


# ── NEW: Trajectory curvature metrics (Step 2c) ──────────────────────────────

def trajectory_curvature_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    mouse_id: str,
    fps: float = 30.0,
    keypoints: list[str] | None = None,
    l_body_px: float = 1.0,
) -> pd.DataFrame:
    """Per-keypoint trajectory curvature before vs. after correction.

    Curvature is defined as the mean second-order finite difference magnitude
    divided by body length, giving a normalised bending measure that is
    independent of animal size:

        curvature = mean(‖Δ²pos[t]‖) / l_body_px

    where Δ²pos[t] = pos[t+1] − 2·pos[t] + pos[t−1].

    Curvature preservation ratio < 1 means the corrected trajectory is smoother
    (desirable for noise removal); values >> 1 may indicate over-smoothing.

    Uses ``video_frame`` / ``mouse_id`` / ``bodypart`` / ``x`` / ``y`` schema.

    Parameters
    ----------
    df_before  : Long-format tracking DataFrame (raw / before correction).
    df_after   : Long-format tracking DataFrame (corrected / after correction).
    mouse_id   : Identity label to filter on.
    fps        : Recording frame rate (Hz) — reserved for future per-frame weighting.
    keypoints  : Keypoints to include (default: all available in df_before).
    l_body_px  : Body length in pixels for normalisation (default 1.0 = raw px).

    Returns
    -------
    DataFrame with columns:
        keypoint, curvature_before, curvature_after, curvature_preservation_ratio
    sorted by curvature_before descending.
    """
    sub_b = df_before[df_before["mouse_id"] == mouse_id].sort_values("video_frame")
    sub_a = df_after[df_after["mouse_id"] == mouse_id].sort_values("video_frame")

    if keypoints is None:
        keypoints = list(sub_b["bodypart"].unique())

    l_ref = max(float(l_body_px), 1.0)
    rows = []
    for kp in keypoints:
        def _curvature(sub: pd.DataFrame) -> float:
            kp_df = sub[sub["bodypart"] == kp].sort_values("video_frame")
            if len(kp_df) < 3:
                return np.nan
            pos = kp_df[["x", "y"]].values.astype(float)
            # Second-order central differences
            d2 = pos[2:] - 2 * pos[1:-1] + pos[:-2]
            mag = np.sqrt((d2 ** 2).sum(axis=1))
            valid = np.isfinite(mag)
            if not valid.any():
                return np.nan
            return float(np.mean(mag[valid])) / l_ref

        c_before = _curvature(sub_b)
        c_after  = _curvature(sub_a)
        if np.isfinite(c_before) and c_before > 0:
            ratio = c_after / c_before
        else:
            ratio = np.nan
        rows.append({
            "keypoint":                     kp,
            "curvature_before":             c_before,
            "curvature_after":              c_after,
            "curvature_preservation_ratio": ratio,
        })

    return (pd.DataFrame(rows)
            .sort_values("curvature_before", ascending=False)
            .reset_index(drop=True))

