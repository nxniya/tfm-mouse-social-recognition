"""
src/data/feature_pipeline.py
==============================
Orchestration of the feature-engineering pipeline.

Keeps the high-level logic (split, imputation, scaling, diagnostics,
visualisation, export) apart from the pure extraction in features.py. Notebook
03_features.ipynb calls these functions; features.py only extracts.

Sections:
  1. Config        – carga Hydra en contexto notebook (compose API)
  2. Load/Correct  – tracking plus the skeletal correction
  3. Extract       – wraps extract_features
  4. Split         – train/val/test, before any diagnostic runs
  5. Imputation    – KNN / interpolate / ffill on features_df, then re-window
  6. Encoding      – circular encoding of angles (sin/cos)
  7. Scaling       – RobustScaler / StandardScaler / QuantileTransformer
  8. Flat features – window_statistics, computed per split
  9. Validation    – NaN%, assertions, isfinite
 10. Redundancy    – Pearson + MI + VIF + dcor opcional
 11. Temporal      – autocorr, smoothness, drift, transition entropy
 12. Embeddings    – PCA(50) → UMAP / t-SNE
 13. Importance    – RF + permutation + SHAP
 14. Stability     – intra-video KS + inter-animal KS
 15. Window DS     – MouseBehaviorDataset + train_val_test_split
 16. Export HDF5   – with full versioning metadata
 17. Parallel      – extract_batch_parallel (joblib)
 18. Plot helpers  – sin boilerplate repetido
"""

from __future__ import annotations

import logging
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Config: the Hydra compose API, which works inside notebooks
# ---------------------------------------------------------------------------

def load_config(config_dir: str = "../configs", config_name: str = "default"):
    """Load the Hydra configuration from a relative directory.

    Uses Hydra's compose API rather than @hydra.main, which requires a script
    entrypoint and therefore cannot run inside a Jupyter notebook.

    Parameters
    ----------
    config_dir : str
        Path to the config directory, relative or absolute.
    config_name : str
        YAML file name, without the extension.

    Returns
    -------
    omegaconf.DictConfig
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    abs_cfg_dir = str(Path(config_dir).resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=abs_cfg_dir, version_base=None):
        cfg = compose(config_name=config_name)
    logger.info("Config loaded from %s/%s.yaml", abs_cfg_dir, config_name)
    return cfg


# ---------------------------------------------------------------------------
# 2. Load / Correct
# ---------------------------------------------------------------------------

def load_and_correct(cfg) -> Tuple[pd.DataFrame, Any, Optional[pd.DataFrame]]:
    """Load tracking and annotations, then apply the skeletal correction.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``video`` and ``skeleton`` sections.

    Returns
    -------
    df_corrected : pd.DataFrame
        Corrected tracking, in long format.
    skeleton : MouseSkeleton
        Skeleton fitted to this video.
    ann_df : pd.DataFrame | None
        Annotations, or None when the video has none.
    """
    from src.data.loader import load_tracking, load_annotations
    from src.skeleton import build_skeleton, fit_skeleton, smooth_video

    lab      = cfg.video.lab
    video_id = cfg.video.id
    sk_cfg   = cfg.skeleton

    logger.info("Cargando tracking: lab=%s  video=%s", lab, video_id)
    df_raw = load_tracking(video_id, lab)
    ann_df = load_annotations(video_id, lab)

    logger.info("Construyendo y ajustando esqueleto…")
    skeleton = build_skeleton(lab)
    skeleton = fit_skeleton(skeleton, df_raw)
    logger.info("L_body_median_px = %.1f px", skeleton.L_body_median_px)

    logger.info("Applying the skeletal correction (smooth_video)…")
    # Correction params come from the single source of truth (configs/skeleton.yaml),
    # NOT default.yaml's skeleton: block — see src/skeleton/config.py (A1 unification).
    from src.skeleton.config import skeleton_smooth_kwargs
    smooth_kwargs = skeleton_smooth_kwargs()

    # C2: cache corrected tracking to disk (deterministic in its inputs) so
    # repeated runs / multi-video extraction don't re-pay the L-BFGS-B cost.
    use_cache = bool(sk_cfg.get("use_correction_cache", True))
    if use_cache:
        from src.data.correction_cache import load_or_correct
        df_corrected, meta = load_or_correct(
            df_raw, lab, str(video_id), skeleton, smooth_kwargs
        )
        logger.info("Correction %s (hash=%s, %d frames)",
                    "from cache" if meta["from_cache"] else "computed and cached",
                    meta["hash"], meta["n_frames"])
    else:
        df_corrected, report = smooth_video(df_raw, lab, skeleton=skeleton, **smooth_kwargs)
        n_out   = report.get("n_outlier_frames", 0)
        n_total = report.get("n_total_frames", 1)
        logger.info("Frames with outliers: %d / %d (%.1f%%)", n_out, n_total,
                    100 * n_out / n_total)

    return df_corrected, skeleton, ann_df


# ---------------------------------------------------------------------------
# 3. Extract
# ---------------------------------------------------------------------------

def extract_and_prepare(
    df_corrected: pd.DataFrame,
    skeleton: Any,
    ann_df: Optional[pd.DataFrame],
    cfg,
) -> Dict[int, Dict]:
    """Extract the full feature set for the video.

    Thin wrapper over ``extract_features``, taking its parameters from cfg.

    Returns
    -------
    dict[int, dict]  — same structure as extract_features.
    """
    from src.data.features import extract_features

    window_size = int(cfg.data.window_size)
    stride      = int(cfg.data.stride)

    logger.info("Extrayendo features  W=%d  stride=%d …", window_size, stride)
    results = extract_features(
        df_corrected, skeleton, ann_df,
        window_size=window_size,
        stride=stride,
        include_dz=True,
        include_relational=True,
    )

    for mid, res in results.items():
        logger.info("Mouse %d: features_df=%s  X=%s  y=%s",
                    mid, res["features_df"].shape, res["X"].shape,
                    res["y"].shape if res["y"] is not None else None)
    return results


# ---------------------------------------------------------------------------
# 4. Split: MUST run before any diagnostic
# ---------------------------------------------------------------------------

def split_results(results: Dict[int, Dict], cfg) -> Dict[str, Any]:
    """Split the window indices into train / val / test.

    Stratifies by label when there are enough classes. **Every diagnostic that
    follows must use the train split only**: running redundancy, drift or
    importance analyses on the full set leaks test information into the feature
    selection, which is the mistake the leakage review of notebook 03 found.

    Parameters
    ----------
    results : dict
        Output of extract_and_prepare.
    cfg : DictConfig

    Returns
    -------
    split : dict with keys:
        mid0, X_all, y_all, idx_train, idx_val, idx_test,
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    from sklearn.model_selection import train_test_split

    val_frac  = float(cfg.data.val_frac)
    test_frac = float(cfg.data.test_frac)
    seed      = int(cfg.data.seed)

    mid0   = sorted(results.keys())[0]
    X_all  = results[mid0]["X"]     # (N, W, F)
    y_all  = results[mid0]["y"]     # (N,) | None

    N = len(X_all)
    assert N > 0, "No windows produced; check the video and the parameters."

    idx = np.arange(N)

    if y_all is not None and len(np.unique(y_all)) > 1:
        stratify = y_all
    else:
        stratify = None
        logger.warning("Unstratified split: not enough distinct labels.")

    # Test fraction, relative to the whole set
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_frac, random_state=seed, stratify=stratify
    )
    # Validation fraction, relative to the trainval remainder
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    strat_tv = y_all[idx_trainval] if stratify is not None else None
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac_adjusted,
        random_state=seed, stratify=strat_tv
    )

    split = dict(
        mid0=mid0,
        X_all=X_all,
        y_all=y_all,
        idx_train=idx_train, idx_val=idx_val, idx_test=idx_test,
        X_train=X_all[idx_train], X_val=X_all[idx_val],   X_test=X_all[idx_test],
        y_train=y_all[idx_train] if y_all is not None else None,
        y_val=y_all[idx_val]   if y_all is not None else None,
        y_test=y_all[idx_test] if y_all is not None else None,
    )

    logger.info("Split: train=%d  val=%d  test=%d  (N=%d)", len(idx_train),
                len(idx_val), len(idx_test), N)

    assert len(idx_train) + len(idx_val) + len(idx_test) == N, \
        "The split does not cover every window."

    return split


# ---------------------------------------------------------------------------
# 5. Imputation on features_df, followed by re-windowing
# ---------------------------------------------------------------------------

_ANGLE_COLS = frozenset([
    "body_angle", "body_angle_vel",
    "neck_spine_angle", "hip_tail_angle_left", "hip_tail_angle_right",
    "angle_relative",
])

def impute_features(results: Dict[int, Dict], cfg) -> Dict[int, Dict]:
    """Impute the NaNs in features_df with KNN, interpolation or forward-fill.

    The strategy depends on the kind of column:
    - Temporal angles and kinematics: forward-fill then backward-fill, which
      preserves the temporal structure that KNN destroys on long series
    - Spatial and kinematic: KNNImputer
    - Relational, which are NaN by design in single-mouse videos: 0.0

    After imputing, X and y are regenerated with build_windows.
    """
    strategy = cfg.features.imputation.strategy
    knn_k    = int(cfg.features.imputation.knn_k)

    logger.info("Imputing NaNs, strategy=%s …", strategy)

    from src.data.features import build_windows, _make_label_array  # noqa: F401

    # Relational columns, always NaN in single-mouse videos
    _REL_COLS = {
        "dist_centroid", "dist_nose_nose", "dist_nose_a_tail_b",
        "dist_nose_b_tail_a", "dist_nose_a_neck_b", "dist_nose_a_ear_b",
        "dist_nose_a_hip_b", "sniff_site_ratio", "angle_relative",
        "speed_relative", "approach_rate", "facing_b", "dz_centroid_diff",
    }

    for mid, res in results.items():
        fdf = res["features_df"].copy()

        # ── 1. Relational: fill with 0, since they do not exist here ─────
        rel_present = [c for c in fdf.columns if c in _REL_COLS]
        if rel_present:
            fdf[rel_present] = fdf[rel_present].fillna(0.0)

        # ── 2. Angles: ffill then bfill, preserving the temporal structure ─
        ang_present = [c for c in fdf.columns if c in _ANGLE_COLS]
        if ang_present:
            fdf[ang_present] = (
                fdf[ang_present].ffill().bfill()
            )

        # ── 3. Everything else, according to strategy ─────────────────────
        remaining_nan = fdf.columns[fdf.isna().any()].tolist()
        if remaining_nan:
            if strategy == "knn":
                from sklearn.impute import KNNImputer
                imp = KNNImputer(n_neighbors=knn_k)
                fdf[remaining_nan] = imp.fit_transform(fdf[remaining_nan])
            elif strategy == "interpolate":
                fdf[remaining_nan] = (
                    fdf[remaining_nan]
                    .interpolate(method="linear", limit_direction="both")
                    .bfill()
                    .ffill()
                )
            elif strategy == "ffill":
                fdf[remaining_nan] = fdf[remaining_nan].ffill().bfill()
            else:  # mean
                fdf[remaining_nan] = fdf[remaining_nan].fillna(
                    fdf[remaining_nan].mean()
                )

        # ── 4. Re-window from the imputed features_df ────────────────────
        # Imputation changes values but not the frame index, so the per-frame
        # labels remain valid. X and y are re-windowed together so they stay
        # aligned; windowing them separately is what produced the
        # X.shape[0] != len(y) mismatch.
        labels_arr = res.get("labels_per_frame")
        window_size = res.get("window_size") or (
            res["X"].shape[1] if res["X"].ndim == 3 else 64)
        # Reuse the original stride: the module default (32) does not match
        # cfg.data.stride (16) and would halve the number of windows.
        stride = res.get("stride", int(cfg.data.stride))
        X_new, y_new = build_windows(fdf, labels_arr, window_size, stride)

        res = dict(res)
        res["features_df"] = fdf
        if X_new.shape[0] > 0:
            res["X"] = X_new
            # y_new is paired with X_new, from the same build_windows pass. It
            # is only None when the video has no labels, in which case res["y"]
            # was already None.
            res["y"] = y_new if labels_arr is not None else res.get("y")
        results[mid] = res

    nan_after = sum(
        results[m]["features_df"].isna().sum().sum()
        for m in results
    )
    logger.info("NaNs remaining after imputation: %d", nan_after)
    assert nan_after == 0, \
        f"{nan_after} NaNs remain after imputation; review the strategy."

    return results


# ---------------------------------------------------------------------------
# 6. Circular angle encoding
# ---------------------------------------------------------------------------

def encode_circular_angles(
    features_df: pd.DataFrame,
    angle_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Replace angular columns with (sin, cos) pairs.

    Avoids the discontinuity at plus or minus pi, where two nearly identical
    orientations sit at opposite ends of the range and every Euclidean distance
    between them is wrong.

    Parameters
    ----------
    features_df : pd.DataFrame
    angle_cols : list[str] | None
        When None, angular columns are detected by name.

    Returns
    -------
    pd.DataFrame with each original column replaced by {name}_sin, {name}_cos.
    """
    if angle_cols is None:
        # Detect by name; these are angles in radians
        angle_cols = [
            c for c in features_df.columns
            if ("angle" in c or c.startswith("facing_"))
            and c not in ("body_angle_vel",)   # angular velocity is not circular
        ]

    df = features_df.copy()
    for col in angle_cols:
        if col not in df.columns:
            continue
        vals = df[col].values
        df[f"{col}_sin"] = np.sin(vals)
        df[f"{col}_cos"] = np.cos(vals)
        df.drop(columns=[col], inplace=True)

    logger.info("Circular encoding: %d angular columns to sin/cos pairs", len(angle_cols))
    return df


def apply_circular_encoding_to_results(
    results: Dict[int, Dict],
    angle_cols: Optional[List[str]] = None,
    cfg=None,
) -> Dict[int, Dict]:
    """Apply the circular encoding and re-window, for every mouse."""
    from src.data.features import STRIDE, build_windows

    for mid, res in results.items():
        fdf_enc = encode_circular_angles(res["features_df"], angle_cols)
        window_size = res.get("window_size") or (
            res["X"].shape[1] if res["X"].ndim == 3 else 64)
        # Re-window X and y together from the per-frame labels, reusing the
        # original stride: the circular encoding changes columns, not rows.
        labels_arr = res.get("labels_per_frame")
        stride = res.get("stride")
        if stride is None:
            stride = int(cfg.data.stride) if cfg is not None else STRIDE
        X_new, y_new = build_windows(fdf_enc, labels_arr, window_size=window_size,
                                     stride=stride)
        res = dict(res)
        res["features_df"]   = fdf_enc
        if X_new.shape[0] > 0:
            res["X"] = X_new
            res["y"] = y_new if labels_arr is not None else res.get("y")
        res["feature_names"] = fdf_enc.columns.tolist()
        results[mid] = res
    return results


# ---------------------------------------------------------------------------
# 7. Scaling
# ---------------------------------------------------------------------------

def build_scaling_pipeline(cfg, include_pca: bool = False):
    """Build an sklearn pipeline with a scaler, and optionally PCA.

    Parameters
    ----------
    cfg : DictConfig
    include_pca : bool
        When True, append PCA(n_components) to the pipeline. Use True only for
        the embeddings (UMAP); use False for classification, where the PCA
        would discard information the classifier can still use.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        QuantileTransformer, RobustScaler, StandardScaler,
    )
    from sklearn.decomposition import PCA

    method = cfg.features.scaling.method.lower()
    if method == "robust":
        scaler = RobustScaler()
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="normal",
                                     random_state=int(cfg.data.seed))
    else:
        scaler = StandardScaler()

    steps = [("scaler", scaler)]

    if include_pca:
        n_pca = int(cfg.features.embedding.pca_components)
        steps.append(("pca", PCA(n_components=n_pca, random_state=int(cfg.data.seed))))

    return Pipeline(steps)


# ---------------------------------------------------------------------------
# 8. Flat features
# ---------------------------------------------------------------------------

def build_flat_features(
    results: Dict[int, Dict],
    split: Dict[str, Any],
    cfg,
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Build X_flat_{train,val,test} with window_statistics.

    The scaler is fitted on train ONLY; val and test are transformed with it.

    Returns
    -------
    X_flat_train, X_flat_val, X_flat_test : np.ndarray
    flat_names : list[str]
    scaler_cls : fitted Pipeline, without PCA, for classification
    """
    from src.data.features import window_statistics

    mid0   = split["mid0"]
    X_all  = results[mid0]["X"]
    fnames = results[mid0]["feature_names"]

    X_flat_all, flat_names = window_statistics(X_all, fnames)

    idx_tr  = split["idx_train"]
    idx_val = split["idx_val"]
    idx_te  = split["idx_test"]

    # Clip to a common size: re-windowing can change N
    n_flat = X_flat_all.shape[0]
    idx_tr  = idx_tr[idx_tr < n_flat]
    idx_val = idx_val[idx_val < n_flat]
    idx_te  = idx_te[idx_te < n_flat]

    X_flat_train = X_flat_all[idx_tr]
    X_flat_val   = X_flat_all[idx_val]
    X_flat_test  = X_flat_all[idx_te]

    logger.info("X_flat shapes: train=%s  val=%s  test=%s",
                X_flat_train.shape, X_flat_val.shape, X_flat_test.shape)

    return X_flat_train, X_flat_val, X_flat_test, flat_names


def fit_transform_split(
    pipeline,
    X_flat_train: np.ndarray,
    X_flat_val: np.ndarray,
    X_flat_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the pipeline on train, then transform train, val and test.

    Any residual NaNs are imputed with the column mean before scaling.
    """
    def _impute(X: np.ndarray) -> np.ndarray:
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        mask = ~np.isfinite(X)
        X = X.copy()
        X[mask] = np.take(col_means, mask.nonzero()[1])
        return X

    X_tr  = _impute(X_flat_train)
    X_val = _impute(X_flat_val)
    X_te  = _impute(X_flat_test)

    X_tr_sc  = pipeline.fit_transform(X_tr)
    X_val_sc = pipeline.transform(X_val)
    X_te_sc  = pipeline.transform(X_te)

    assert not np.isnan(X_tr_sc).any(),  "NaN en X_train escalado."
    assert not np.isnan(X_val_sc).any(), "NaN en X_val escalado."
    assert not np.isnan(X_te_sc).any(),  "NaN en X_test escalado."

    return X_tr_sc, X_val_sc, X_te_sc


# ---------------------------------------------------------------------------
# 9. Validation
# ---------------------------------------------------------------------------

def validate_features(
    results: Dict[int, Dict],
    split: Dict[str, Any],
) -> Dict[str, Any]:
    """Check NaN percentage, shapes and finiteness on the train split.

    Returns
    -------
    dict with keys: nan_pct, high_nan_features, n_windows_train
    """
    mid0  = split["mid0"]
    fdf   = results[mid0]["features_df"]
    X     = results[mid0]["X"]
    y     = results[mid0]["y"]

    # Shape
    assert X.ndim == 3, f"X must be 3D (n_windows, W, F); shape={X.shape}"
    if y is not None:
        assert X.shape[0] == len(y), \
            f"X.shape[0]={X.shape[0]} != len(y)={len(y)}"

    # NaN percentage in features_df, at frame level
    nan_pct = fdf.isna().mean() * 100
    high_nan = nan_pct[nan_pct > 5].sort_values(ascending=False)

    logger.info("Features with more than 5%% NaN: %d / %d",
                len(high_nan), len(nan_pct))
    if len(high_nan):
        logger.warning("High-NaN features: %s", high_nan.to_dict())

    # Finiteness of X_train
    idx_tr = split["idx_train"]
    idx_tr = idx_tr[idx_tr < len(X)]
    X_train = X[idx_tr]
    finite_pct = np.isfinite(X_train).mean() * 100
    logger.info("Finite fraction of X_train: %.2f%%", finite_pct)

    return dict(
        nan_pct=nan_pct,
        high_nan_features=high_nan,
        n_windows_train=len(idx_tr),
    )


# ---------------------------------------------------------------------------
# 10. Redundancy
# ---------------------------------------------------------------------------

def analyze_redundancy(
    X_flat_train: np.ndarray,
    flat_names: List[str],
    y_train: Optional[np.ndarray],
    cfg,
) -> Dict[str, Any]:
    """Compute redundancy metrics between features.

    Metrics:
    - Pearson correlation matrix
    - Mutual information (sklearn)
    - Variance Inflation Factor (statsmodels)
    - Distance correlation (dcor), optional and slow

    Returns
    -------
    dict with keys: corr_matrix, high_pairs, mi_scores, vif_df, dcor_matrix
    """
    from sklearn.feature_selection import mutual_info_classif

    rthresh = float(cfg.features.redundancy.pearson_threshold)

    df_flat = pd.DataFrame(X_flat_train, columns=flat_names)
    df_flat_clean = df_flat.dropna()

    output: Dict[str, Any] = {}

    # ── Pearson ───────────────────────────────────────────────────────────
    logger.info("Computing Pearson correlation…")
    corr = df_flat_clean.corr(method="pearson")
    output["corr_matrix"] = corr

    high_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > rthresh:
                high_pairs.append((cols[i], cols[j], float(r)))
    high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    output["high_pairs"] = high_pairs
    logger.info("Pairs with |r| > %.2f: %d", rthresh, len(high_pairs))

    # ── Mutual information ────────────────────────────────────────────────
    if cfg.features.redundancy.compute_mi and y_train is not None:
        logger.info("Computing mutual information…")
        from sklearn.preprocessing import LabelEncoder as LE
        le = LE()
        y_enc = le.fit_transform(y_train)
        # Subsample when there are many samples: MI is O(N log N)
        n_samp = min(len(df_flat_clean), 5000)
        rng = np.random.default_rng(42)
        idx_mi = rng.choice(len(df_flat_clean), n_samp, replace=False)
        mi = mutual_info_classif(
            df_flat_clean.iloc[idx_mi].values,
            y_enc[idx_mi],
            random_state=42,
        )
        output["mi_scores"] = pd.Series(mi, index=flat_names).sort_values(ascending=False)
    else:
        output["mi_scores"] = None

    # ── VIF ───────────────────────────────────────────────────────────────
    if cfg.features.redundancy.compute_vif:
        logger.info("Calculando VIF…")
        try:
            from statsmodels.stats.outliers_influence import (
                variance_inflation_factor,
            )
            X_vif = df_flat_clean.values
            # VIF requiere columna constante (intercepto)
            X_vif_c = np.column_stack([np.ones(len(X_vif)), X_vif])
            vif_vals = [
                variance_inflation_factor(X_vif_c, i + 1)
                for i in range(X_vif.shape[1])
            ]
            output["vif_df"] = pd.DataFrame(
                {"feature": flat_names, "VIF": vif_vals}
            ).sort_values("VIF", ascending=False)
            logger.info("Features with VIF > 10: %d",
                        (output["vif_df"]["VIF"] > 10).sum())
        except Exception as exc:
            logger.warning("VIF failed: %s", exc)
            output["vif_df"] = None
    else:
        output["vif_df"] = None

    # ── Distance correlation, optional and potentially slow ──────────────
    if cfg.features.redundancy.compute_dcor:
        logger.info("Computing distance correlation; this can take a while…")
        try:
            import dcor
            n_dc = min(len(df_flat_clean), 500)
            rng  = np.random.default_rng(42)
            idx_dc = rng.choice(len(df_flat_clean), n_dc, replace=False)
            X_dc = df_flat_clean.iloc[idx_dc].values
            dcor_mat = np.zeros((len(flat_names), len(flat_names)))
            for i in range(len(flat_names)):
                for j in range(i, len(flat_names)):
                    v = float(dcor.distance_correlation(X_dc[:, i], X_dc[:, j]))
                    dcor_mat[i, j] = dcor_mat[j, i] = v
            output["dcor_matrix"] = pd.DataFrame(
                dcor_mat, index=flat_names, columns=flat_names
            )
        except Exception as exc:
            logger.warning("dcor failed: %s", exc)
            output["dcor_matrix"] = None
    else:
        output["dcor_matrix"] = None

    return output


# ---------------------------------------------------------------------------
# 11. Temporal validation
# ---------------------------------------------------------------------------

def validate_temporal(
    results: Dict[int, Dict],
    split: Dict[str, Any],
) -> Dict[str, Any]:
    """Inspect the temporal properties of the features.

    Computed on the training split, at frame level:
    - autocorr  : ACF at lags [1, 5, 10, 20], per feature
    - smoothness: var(dx) / var(x); lower is smoother
    - drift     : KS statistic, first 20% of the video against the last 20%
    - transition_entropy : Shannon entropy of the label sequences

    Returns
    -------
    dict with keys: autocorr_df, smoothness_df, drift_df, transition_entropy
    """
    from scipy.stats import ks_2samp, entropy as scipy_entropy

    mid0 = split["mid0"]
    fdf  = results[mid0]["features_df"]
    arr  = fdf.values.astype(np.float64)
    cols = fdf.columns.tolist()
    T    = len(arr)

    # ── Autocorrelation ──────────────────────────────────────────────────
    lags = [1, 5, 10, 20]
    acf_rows = []
    for lag in lags:
        lag_corrs = []
        for f in range(arr.shape[1]):
            col = arr[:, f]
            valid = np.isfinite(col)
            if valid.sum() > lag + 10:
                x1 = col[valid][:-lag]
                x2 = col[valid][lag:]
                if x1.std() > 1e-9 and x2.std() > 1e-9:
                    r = float(np.corrcoef(x1, x2)[0, 1])
                else:
                    r = 0.0
            else:
                r = np.nan
            lag_corrs.append(r)
        acf_rows.append(lag_corrs)
    autocorr_df = pd.DataFrame(
        acf_rows, index=[f"lag_{l}" for l in lags], columns=cols
    ).T

    # ── Smoothness = var(Δx) / var(x) ─────────────────────────────────────
    smoothness = []
    for f in range(arr.shape[1]):
        col = arr[:, f]
        valid = col[np.isfinite(col)]
        if len(valid) > 2:
            v_sig  = float(np.var(valid))
            v_diff = float(np.var(np.diff(valid)))
            sm = v_diff / (v_sig + 1e-12)
        else:
            sm = np.nan
        smoothness.append(sm)
    smoothness_df = pd.Series(smoothness, index=cols, name="smoothness").sort_values()

    # ── Drift: KS of the first 20% against the last 20% ──────────────────
    cut = max(1, T // 5)
    drift_rows = []
    for f in range(arr.shape[1]):
        col = arr[:, f]
        a = col[:cut][np.isfinite(col[:cut])]
        b = col[-cut:][np.isfinite(col[-cut:])]
        if len(a) > 5 and len(b) > 5:
            stat, pval = ks_2samp(a, b)
        else:
            stat, pval = np.nan, np.nan
        drift_rows.append({"feature": cols[f], "ks_stat": stat, "p_value": pval})
    drift_df = (
        pd.DataFrame(drift_rows)
        .set_index("feature")
        .sort_values("ks_stat", ascending=False)
    )
    n_drift = int((drift_df["p_value"] < 0.05).sum())
    logger.info("Features with significant drift (p<0.05): %d / %d",
                n_drift, len(cols))

    # ── Transition entropy ────────────────────────────────────────────────
    y_all = results[mid0].get("y")
    if y_all is not None:
        unique_lbl, counts = np.unique(y_all, return_counts=True)
        probs = counts / counts.sum()
        t_entropy = float(scipy_entropy(probs, base=2))
        logger.info("Transition entropy: %.3f bits  (clases: %d)", t_entropy,
                    len(unique_lbl))
    else:
        t_entropy = None

    return dict(
        autocorr_df=autocorr_df,
        smoothness_df=smoothness_df,
        drift_df=drift_df,
        transition_entropy=t_entropy,
    )


# ---------------------------------------------------------------------------
# 12. Embeddings — PCA(50) → UMAP
# ---------------------------------------------------------------------------

def compute_embeddings(
    X_scaled_train: np.ndarray,
    y_train: Optional[np.ndarray],
    cfg,
    X_full: Optional[np.ndarray] = None,
    y_full: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Reduce dimensionality with PCA followed by UMAP, or t-SNE as a fallback.

    The PCA is fitted on X_scaled_train only, so no test information leaks into
    the projection. When X_full is given, those points are projected too.

    Returns
    -------
    dict: method, emb_train, emb_full (si se pide), sil_pca, sil_emb
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    emb_cfg = cfg.features.embedding
    n_pca   = int(emb_cfg.pca_components)
    seed    = int(cfg.data.seed)

    logger.info("PCA(%d), original dimensionality: %d …", n_pca, X_scaled_train.shape[1])
    n_pca_actual = min(n_pca, X_scaled_train.shape[1], X_scaled_train.shape[0] - 1)
    pca = PCA(n_components=n_pca_actual, random_state=seed)
    X_pca_train = pca.fit_transform(X_scaled_train)
    explained = pca.explained_variance_ratio_.sum()
    logger.info("PCA varianza explicada (%.0f comp): %.1f%%",
                n_pca_actual, 100 * explained)

    # Silhouette sobre PCA
    sil_pca = None
    if y_train is not None and len(np.unique(y_train)) > 1:
        n_sil = min(len(X_pca_train), 2000)
        rng = np.random.default_rng(seed)
        idx_sil = rng.choice(len(X_pca_train), n_sil, replace=False)
        sil_pca = float(silhouette_score(X_pca_train[idx_sil],
                                         y_train[idx_sil]))
        logger.info("Silhouette PCA: %.3f", sil_pca)

    # ── UMAP ─────────────────────────────────────────────────────────────
    try:
        import umap
        method = "UMAP"
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(emb_cfg.umap_n_neighbors),
            min_dist=float(emb_cfg.umap_min_dist),
            random_state=int(emb_cfg.umap_random_state),
        )
        emb_train = reducer.fit_transform(X_pca_train)
        logger.info("UMAP finalizado.")
    except ImportError:
        from sklearn.manifold import TSNE
        method = "t-SNE"
        perp = min(30, len(X_pca_train) - 1)
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perp)
        emb_train = reducer.fit_transform(X_pca_train)
        logger.info("t-SNE finalizado (umap-learn no instalado).")

    # Silhouette sobre embedding
    sil_emb = None
    if y_train is not None and len(np.unique(y_train)) > 1:
        n_sil = min(len(emb_train), 2000)
        rng = np.random.default_rng(seed)
        idx_sil = rng.choice(len(emb_train), n_sil, replace=False)
        sil_emb = float(silhouette_score(emb_train[idx_sil], y_train[idx_sil]))
        logger.info("Silhouette %s: %.3f", method, sil_emb)

    # Proyectar conjunto completo si se pide
    emb_full = None
    if X_full is not None:
        X_pca_full = pca.transform(X_full)
        if method == "UMAP":
            emb_full = reducer.transform(X_pca_full)
        else:
            emb_full = TSNE(  # t-SNE has no transform, so refit
                n_components=2, random_state=seed,
                perplexity=min(30, len(X_pca_full) - 1)
            ).fit_transform(X_pca_full)

    return dict(
        method=method,
        emb_train=emb_train,
        emb_full=emb_full,
        sil_pca=sil_pca,
        sil_emb=sil_emb,
        pca=pca,
        reducer=reducer,
    )


# ---------------------------------------------------------------------------
# 13. Feature importance
# ---------------------------------------------------------------------------

def compute_feature_importance(
    X_flat_train: np.ndarray,
    y_train: np.ndarray,
    flat_names: List[str],
    X_flat_val: np.ndarray,
    y_val: np.ndarray,
    cfg,
) -> Dict[str, Any]:
    """Feature importance: random forest, permutation and SHAP.

    Parameters
    ----------
    X_flat_train, X_flat_val : np.ndarray — static features, without PCA
    y_train, y_val : np.ndarray
    flat_names : list[str]
    cfg : DictConfig

    Returns
    -------
    dict: rf_importance, perm_importance, shap_values, shap_feature_names
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import LabelEncoder as LE

    imp_cfg = cfg.features.importance
    n_est   = int(imp_cfg.rf_n_estimators)
    seed    = int(cfg.data.seed)

    le      = LE()
    y_tr_e  = le.fit_transform(y_train)
    y_val_e = le.transform(y_val)

    logger.info("Training the random forest for feature importance…")
    rf = RandomForestClassifier(
        n_estimators=n_est,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X_flat_train, y_tr_e)
    logger.info("RF accuracy (val): %.3f", rf.score(X_flat_val, y_val_e))

    rf_imp = pd.Series(rf.feature_importances_, index=flat_names).sort_values(
        ascending=False
    )

    # ── Permutation importance ───────────────────────────────────────────
    logger.info("Calculando permutation importance (n_repeats=10)…")
    perm = permutation_importance(
        rf, X_flat_val, y_val_e, n_repeats=10, random_state=seed, n_jobs=-1
    )
    perm_imp = pd.DataFrame(
        {
            "mean":  perm.importances_mean,
            "std":   perm.importances_std,
        },
        index=flat_names,
    ).sort_values("mean", ascending=False)

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_values = None
    if imp_cfg.compute_shap:
        try:
            import shap
            n_shap = min(int(imp_cfg.shap_sample_size), len(X_flat_train))
            rng    = np.random.default_rng(seed)
            idx_shap = rng.choice(len(X_flat_train), n_shap, replace=False)
            logger.info("Calculando SHAP (n=%d)…", n_shap)
            explainer   = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_flat_train[idx_shap])
            logger.info("SHAP finalizado.")
        except Exception as exc:
            logger.warning("SHAP failed: %s", exc)

    return dict(
        rf=rf,
        rf_importance=rf_imp,
        perm_importance=perm_imp,
        shap_values=shap_values,
        label_encoder=le,
    )


# ---------------------------------------------------------------------------
# 14. Feature stability
# ---------------------------------------------------------------------------

def analyze_stability(
    results: Dict[int, Dict],
    split: Dict[str, Any],
) -> Dict[str, Any]:
    """Check stability within a video and across animals.

    Intra-video : KS test between the first and second temporal halves.
    Inter-animal: KS test between the per-mouse_id distributions, when there
                  are at least two mice.

    Returns
    -------
    dict: intra_video_df, inter_animal_df, unstable_features
    """
    from scipy.stats import ks_2samp

    mid0 = split["mid0"]
    fdf  = results[mid0]["features_df"]
    arr  = fdf.values.astype(np.float64)
    cols = fdf.columns.tolist()
    T    = len(arr)

    # ── Intra-video ──────────────────────────────────────────────────────
    half = T // 2
    intra_rows = []
    for f, col in enumerate(cols):
        a = arr[:half, f]
        b = arr[half:, f]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) > 5 and len(b) > 5:
            stat, pval = ks_2samp(a, b)
        else:
            stat, pval = np.nan, np.nan
        intra_rows.append({"feature": col, "ks_stat": stat, "p_value": pval})

    intra_df = (
        pd.DataFrame(intra_rows)
        .set_index("feature")
        .sort_values("ks_stat", ascending=False)
    )
    n_unstable = int((intra_df["p_value"] < 0.05).sum())
    logger.info("Intra-video unstable features (p<0.05): %d / %d",
                n_unstable, len(cols))

    # ── Inter-animal ──────────────────────────────────────────────────────
    inter_df = None
    mice = sorted(results.keys())
    if len(mice) >= 2:
        inter_rows = []
        fdf_a = results[mice[0]]["features_df"]
        fdf_b = results[mice[1]]["features_df"]
        common_cols = [c for c in fdf_a.columns if c in fdf_b.columns]
        for col in common_cols:
            a = fdf_a[col].dropna().values
            b = fdf_b[col].dropna().values
            if len(a) > 5 and len(b) > 5:
                stat, pval = ks_2samp(a, b)
            else:
                stat, pval = np.nan, np.nan
            inter_rows.append({"feature": col, "ks_stat": stat, "p_value": pval})
        inter_df = (
            pd.DataFrame(inter_rows)
            .set_index("feature")
            .sort_values("ks_stat", ascending=False)
        )
        n_inter = int((inter_df["p_value"] < 0.05).sum())
        logger.info("Features inter-animal distintas (p<0.05): %d / %d",
                    n_inter, len(common_cols))

    unstable = intra_df[intra_df["p_value"] < 0.05].index.tolist()

    return dict(
        intra_video_df=intra_df,
        inter_animal_df=inter_df,
        unstable_features=unstable,
    )


# ---------------------------------------------------------------------------
# 15. Window dataset (PyTorch)
# ---------------------------------------------------------------------------

def build_window_dataset(
    results: Dict[int, Dict],
    split: Dict[str, Any],
    cfg,
):
    """Build a MouseBehaviorDataset for train, val and test.

    Returns
    -------
    train_ds, val_ds, test_ds : MouseBehaviorDataset
    enc : LabelEncoder
    """
    from src.data.dataset import MouseBehaviorDataset, LabelEncoder

    mid0   = split["mid0"]
    X_all  = results[mid0]["X"]
    y_all  = results[mid0]["y"]
    fnames = results[mid0]["feature_names"]

    enc    = LabelEncoder()
    y_enc  = enc.encode(y_all) if y_all is not None else None

    idx_tr  = split["idx_train"]
    idx_val = split["idx_val"]
    idx_te  = split["idx_test"]
    n_flat  = X_all.shape[0]
    idx_tr  = idx_tr[idx_tr < n_flat]
    idx_val = idx_val[idx_val < n_flat]
    idx_te  = idx_te[idx_te < n_flat]

    def _make_ds(idx):
        X_sub = X_all[idx]
        y_sub = y_enc[idx] if y_enc is not None else None
        return MouseBehaviorDataset(
            X_sub, y_sub,
            feature_names=fnames,
            include_background=bool(cfg.data.include_background),
        )

    train_ds = _make_ds(idx_tr)
    val_ds   = _make_ds(idx_val)
    test_ds  = _make_ds(idx_te)

    logger.info("Datasets: train=%d  val=%d  test=%d",
                len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds, enc


# ---------------------------------------------------------------------------
# 16. HDF5 export, with full versioning metadata
# ---------------------------------------------------------------------------

def export_hdf5(
    results: Dict[int, Dict],
    split: Dict[str, Any],
    scaler_pipeline,
    cfg,
    out_dir: Optional[str] = None,
) -> Path:
    """Write the features and the versioning metadata to HDF5.

    Metadatos guardados:
        feature_version, git_commit, normalization, fps,
        sklearn_version, numpy_version, extraction_timestamp,
        scaler_type, angle_encoding, window_size, stride,
        lab, video_id, n_features
    """
    import sklearn
    from omegaconf import OmegaConf

    out_base = Path(out_dir or "dataset/features")
    out_base.mkdir(parents=True, exist_ok=True)

    lab_name = cfg.video.lab
    vid_id   = cfg.video.id
    out_path = out_base / f"{lab_name}_{vid_id}_features.h5"

    # ── Git commit ────────────────────────────────────────────────────────
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        git_commit = "unknown"

    ts = datetime.now(timezone.utc).isoformat()

    with h5py.File(out_path, "w") as hf:
        # ── Root metadata ─────────────────────────────────────────────────
        hf.attrs["lab"]                  = lab_name
        hf.attrs["video_id"]             = str(vid_id)
        hf.attrs["feature_version"]      = cfg.features.versioning.feature_version
        hf.attrs["git_commit"]           = git_commit
        hf.attrs["normalization"]        = cfg.features.versioning.normalization
        hf.attrs["fps"]                  = int(cfg.video.fps)
        hf.attrs["sklearn_version"]      = sklearn.__version__
        hf.attrs["numpy_version"]        = np.__version__
        hf.attrs["extraction_timestamp"] = ts
        hf.attrs["scaler_type"]          = cfg.features.scaling.method
        hf.attrs["angle_encoding"]       = cfg.features.scaling.angle_encoding
        hf.attrs["window_size"]          = int(cfg.data.window_size)
        hf.attrs["stride"]               = int(cfg.data.stride)
        hf.attrs["n_features"]           = len(
            results[split["mid0"]]["feature_names"]
        )

        # ── Split indices ─────────────────────────────────────────────────
        sp_grp = hf.create_group("split")
        sp_grp.create_dataset("train", data=split["idx_train"])
        sp_grp.create_dataset("val",   data=split["idx_val"])
        sp_grp.create_dataset("test",  data=split["idx_test"])

        # ── Per mouse ────────────────────────────────────────────────────
        for mid, res in results.items():
            grp = hf.create_group(f"mouse_{mid}")
            if res["X"].size > 0:
                grp.create_dataset("X", data=res["X"], compression="gzip")
            if res["y"] is not None:
                grp.create_dataset("y", data=res["y"].astype("S"),
                                   compression="gzip")
            grp.create_dataset("frames", data=res["frames"])
            grp.attrs["feature_names"] = "|".join(res["feature_names"])

    size_kb = out_path.stat().st_size / 1024
    logger.info("HDF5 guardado: %s  (%.1f KB)  git=%s", out_path, size_kb, git_commit)
    return out_path


# ---------------------------------------------------------------------------
# 17. Parallel extraction
# ---------------------------------------------------------------------------

def extract_batch_parallel(
    video_ids: List[str],
    lab: str,
    cfg,
    n_jobs: Optional[int] = None,
) -> Dict[str, Dict]:
    """Extract features for several videos in parallel, using joblib.

    Parameters
    ----------
    video_ids : list[str]
    lab : str
    cfg : DictConfig  — video.id is overwritten for each job
    n_jobs : int | None — when None, cfg.features.parallel.n_jobs is used

    Returns
    -------
    dict[video_id → results_dict]
    """
    from joblib import Parallel, delayed
    from omegaconf import OmegaConf

    if n_jobs is None:
        n_jobs = int(cfg.features.parallel.n_jobs)

    def _extract_one(vid_id: str) -> Tuple[str, Dict]:
        # Clonar cfg y sobreescribir video.id
        cfg_copy = OmegaConf.merge(cfg, {"video": {"id": vid_id, "lab": lab}})
        try:
            df_corr, skeleton, ann_df = load_and_correct(cfg_copy)
            res = extract_and_prepare(df_corr, skeleton, ann_df, cfg_copy)
            return vid_id, res
        except Exception as exc:
            logger.error("Error extracting video %s: %s", vid_id, exc)
            return vid_id, {}

    logger.info("Parallel extraction: %d videos, n_jobs=%d", len(video_ids), n_jobs)
    out_pairs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_extract_one)(vid) for vid in video_ids
    )
    return dict(out_pairs)


# ---------------------------------------------------------------------------
# 18. Plot helpers
# ---------------------------------------------------------------------------

def _make_figure(nrows: int, ncols: int, figsize=None, **kwargs):
    import matplotlib.pyplot as plt
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def plot_kinematics(
    results: Dict[int, Dict],
    split: Dict[str, Any],
    cfg=None,
) -> None:
    """Histograms of centroid speed, body angle and dorsal curvature."""
    import matplotlib.pyplot as plt

    fig, axes = _make_figure(1, 3, figsize=(15, 4))
    for mid, res in results.items():
        fdf = res["features_df"]
        # Centroid speed
        if "centroid_speed" in fdf.columns:
            axes[0].hist(fdf["centroid_speed"].dropna(), bins=60, alpha=0.6,
                         label=f"Mouse {mid}")
        # Body angle: raw, or _sin/_cos once the circular encoding has run
        if "body_angle" in fdf.columns:
            axes[1].hist(np.degrees(fdf["body_angle"].dropna()), bins=72,
                         alpha=0.6, label=f"Mouse {mid}")
        elif "body_angle_sin" in fdf.columns:
            ang = np.degrees(np.arctan2(fdf["body_angle_sin"].dropna(),
                                        fdf["body_angle_cos"].dropna()))
            axes[1].hist(ang, bins=72, alpha=0.6, label=f"Mouse {mid}")
        # Dorsal curvature
        neck_col = next((c for c in ("neck_spine_angle",
                                      "neck_spine_angle_sin") if c in fdf.columns), None)
        if neck_col:
            vals = fdf[neck_col].dropna()
            if neck_col.endswith("_sin"):
                cos_col = neck_col.replace("_sin", "_cos")
                vals = np.degrees(np.arctan2(vals, fdf[cos_col].dropna()))
            else:
                vals = np.degrees(vals)
            axes[2].hist(vals, bins=50, alpha=0.6, label=f"Mouse {mid}")

    axes[0].set(xlabel="Centroid speed (L_body/frame)", title="Speed")
    axes[1].set(xlabel="Body angle (deg)", title="Body orientation")
    axes[2].set(xlabel="neck_spine_angle (°)", title="Curvatura dorsal")
    for ax in axes:
        ax.legend()
    fig.suptitle("Kinematics, train split", fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_dz_by_behavior(
    results: Dict[int, Dict],
    split: Dict[str, Any],
) -> None:
    """Horizontal bars of the median dz_sum, per behaviour."""
    import matplotlib.pyplot as plt

    mid0 = split["mid0"]
    fdf  = results[mid0]["features_df"].copy()
    y    = results[mid0].get("y")

    if "dz_sum" not in fdf.columns or y is None:
        logger.info("No dz features or no labels available.")
        return

    # Map labels down to frame level, approximating with the window label
    idx_tr = split["idx_train"]
    n = len(fdf)
    frame_labels = np.full(n, "background", dtype=object)
    ws = fdf.shape[0]
    window_size = results[mid0]["X"].shape[1] if results[mid0]["X"].ndim == 3 else 64
    stride = max(1, ws // max(len(y), 1))
    for i, lbl in enumerate(y):
        start = i * stride
        end   = min(start + window_size, n)
        frame_labels[start:end] = lbl

    fdf["behavior"] = frame_labels
    dz_by_beh = fdf.groupby("behavior")["dz_sum"].median().sort_values()

    fig, ax = plt.subplots(figsize=(8, max(3, len(dz_by_beh) * 0.55)))
    dz_by_beh.plot(kind="barh", ax=ax, color="darkorange")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set(xlabel="median dz_sum", title="dz by behaviour (higher = more upright)")
    plt.tight_layout()
    plt.show()


def plot_relational_features(
    results: Dict[int, Dict],
    split: Dict[str, Any],
) -> None:
    """Histograms of dist_nose_nose and angle_relative."""
    import matplotlib.pyplot as plt

    mid0 = split["mid0"]
    fdf  = results[mid0]["features_df"]

    # Check for relational features; they may be raw or circularly encoded
    has_dist = "dist_nose_nose" in fdf.columns
    has_ang  = any(c in fdf.columns for c in ("angle_relative",
                                               "angle_relative_sin"))

    if not has_dist and not has_ang:
        logger.info("No relational features; this is a single-mouse video.")
        return

    fig, axes = _make_figure(1, 2, figsize=(12, 4))
    for mid, res in results.items():
        fdf_m = res["features_df"]
        if "dist_nose_nose" in fdf_m.columns:
            axes[0].hist(fdf_m["dist_nose_nose"].dropna(), bins=60,
                         alpha=0.7, label=f"Mouse {mid}")
        ang_col = next((c for c in ("angle_relative", "angle_relative_sin")
                         if c in fdf_m.columns), None)
        if ang_col:
            vals = fdf_m[ang_col].dropna()
            if ang_col.endswith("_sin"):
                cos_col = ang_col.replace("_sin", "_cos")
                vals = np.degrees(np.arctan2(vals, fdf_m[cos_col].dropna()))
            else:
                vals = np.degrees(vals)
            axes[1].hist(vals, bins=72, alpha=0.7, label=f"Mouse {mid}")

    axes[0].set(xlabel="dist_nose_nose (L_body)", title="Nose-to-nose distance")
    axes[1].set(xlabel="Relative angle (deg)", title="Relative orientation")
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_redundancy_heatmap(
    corr_matrix: pd.DataFrame,
    high_pairs: List[Tuple],
    threshold: float = 0.90,
    max_features: int = 60,
) -> None:
    """Correlation heatmap of the features in redundant pairs, plus a bar chart."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ── Subset: only features that appear in at least one redundant pair ──────
    involved: set = set()
    for c1, c2, _ in high_pairs:
        involved.add(c1)
        involved.add(c2)

    if involved:
        # Keep at most max_features; prefer those with the highest max |r|
        if len(involved) > max_features:
            _max_r = {f: corr_matrix.loc[f, list(involved)].abs().max()
                      for f in involved if f in corr_matrix.index}
            involved = set(sorted(_max_r, key=_max_r.get, reverse=True)[:max_features])

        cols = [c for c in corr_matrix.columns if c in involved]
        sub = corr_matrix.loc[cols, cols]

        # Hierarchical clustering order for visual grouping
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform
            dist = 1 - sub.abs().values
            np.fill_diagonal(dist, 0.0)
            dist = np.clip(dist, 0, None)
            order = leaves_list(linkage(squareform(dist), method="average"))
            sub = sub.iloc[order, order]
        except Exception:
            pass

        n = len(sub)
        tick_fs = max(4, min(8, 200 // n))
        fig_side = max(8, min(18, n * 0.28))
        fig, axes = plt.subplots(1, 2, figsize=(fig_side + 12, fig_side),
                                 gridspec_kw={"width_ratios": [1, 1.1]})

        mask = np.triu(np.ones(n, dtype=bool))
        sns.heatmap(
            sub, mask=mask, ax=axes[0],
            cmap="RdBu_r", vmin=-1, vmax=1,
            linewidths=0.3, square=True,
            cbar_kws={"shrink": 0.6, "label": "Pearson r"},
            xticklabels=sub.columns,
            yticklabels=sub.columns,
        )
        axes[0].tick_params(axis="x", rotation=90, labelsize=tick_fs)
        axes[0].tick_params(axis="y", rotation=0,  labelsize=tick_fs)
        axes[0].set_title(
            f"Pearson r — features en pares redundantes (n={n})",
            fontweight="bold",
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        axes[0].text(0.5, 0.5, "Sin pares redundantes", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=12)
        axes[0].set_title("Pearson r", fontweight="bold")

    # ── Bar chart: top redundant pairs ────────────────────────────────────────
    if high_pairs:
        top = high_pairs[:20]
        n_top = len(top)
        y_pos = np.arange(n_top)
        colors = ["#d62728" if abs(r) > 0.95 else "#ff7f0e" for _, _, r in top]
        axes[1].barh(y_pos, [abs(r) for _, _, r in top], color=colors)
        axes[1].set_yticks(y_pos)
        # Full names — truncate only if very long
        labels = []
        for c1, c2, r in top:
            lbl = f"{c1} ↔ {c2}  r={r:+.2f}"
            if len(lbl) > 80:
                lbl = f"{c1[:30]}… ↔ {c2[:30]}…  r={r:+.2f}"
            labels.append(lbl)
        axes[1].set_yticklabels(labels, fontsize=7)
        axes[1].set_xlim(0, 1.05)
        axes[1].axvline(threshold, color="k", linestyle="--", linewidth=0.8,
                        label=f"umbral {threshold}")
        axes[1].set(xlabel="|Pearson r|",
                    title=f"Top pares redundantes (|r| > {threshold})")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, f"No pairs with |r| > {threshold}",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    plt.show()


def plot_mi_barplot(
    mi_scores: pd.Series,
    top_n: int = 30,
) -> None:
    """Horizontal bar plot of mutual information, per feature."""
    import matplotlib.pyplot as plt

    if mi_scores is None:
        logger.info("Mutual information was not computed.")
        return

    top = mi_scores.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(np.arange(len(top)), top.values, color="steelblue")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set(xlabel="Mutual information", title=f"Top {top_n} features by MI")
    plt.tight_layout()
    plt.show()


def plot_embeddings(
    emb: np.ndarray,
    y: Optional[np.ndarray],
    method: str = "UMAP",
    sil_pca: Optional[float] = None,
    sil_emb: Optional[float] = None,
    cfg=None,
) -> None:
    """2D scatter, coloured by behaviour label."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    unique_labels = list(np.unique(y)) if y is not None else ["unknown"]
    palette = sns.color_palette("tab10", len(unique_labels))
    cmap = {lbl: c for lbl, c in zip(unique_labels, palette)}

    fig, ax = plt.subplots(figsize=(9, 7))
    if y is not None:
        for lbl in unique_labels:
            mask_l = y == lbl
            ax.scatter(emb[mask_l, 0], emb[mask_l, 1],
                       c=[cmap[lbl]], label=lbl, s=15, alpha=0.7)
    else:
        ax.scatter(emb[:, 0], emb[:, 1], s=15, alpha=0.5)

    title = f"{method}: feature space, train split"
    if sil_pca is not None and sil_emb is not None:
        title += f"\nSilhouette PCA={sil_pca:.3f}  {method}={sil_emb:.3f}"
    ax.set_title(title, fontweight="bold")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=8, markerscale=2)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    importance_dict: Dict[str, Any],
    top_n: int = 20,
) -> None:
    """Barplot horizontal: RF importance + permutation CI + SHAP summary.

    Each panel is rendered in its own figure so label widths and SHAP's
    own figure management don't interfere with each other.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rf_imp   = importance_dict.get("rf_importance")
    perm_imp = importance_dict.get("perm_importance")
    shap_val = importance_dict.get("shap_values")

    row_h  = 0.42           # inches per feature row
    lbl_w  = 3.6            # inches reserved for y-axis labels
    bar_w  = 4.0            # inches for the bar area

    def _barh_panel(ax, values, labels, xlabel, title, color,
                    xerr=None, color_list=None):
        n = len(values)
        y = np.arange(n)
        c = color_list if color_list is not None else color
        ax.barh(y, values, color=c, xerr=xerr,
                capsize=3 if xerr is not None else 0,
                error_kw={"elinewidth": 0.8, "ecolor": "gray"})
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()          # rank 1 at top
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
        ax.tick_params(axis="x", labelsize=8)
        ax.axvline(0, color="k", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    # ── RF MDI ────────────────────────────────────────────────────────────
    if rf_imp is not None:
        top  = rf_imp.head(top_n)
        n    = len(top)
        fig, ax = plt.subplots(figsize=(lbl_w + bar_w, max(4, n * row_h)))
        _barh_panel(ax, top.values, top.index,
                    xlabel="RF importance (MDI)",
                    title=f"Top {n} — RandomForest MDI",
                    color="steelblue")
        fig.subplots_adjust(left=lbl_w / (lbl_w + bar_w))
        plt.tight_layout()
        plt.show()

    # ── Permutation importance ────────────────────────────────────────────
    if perm_imp is not None:
        top_p = perm_imp.head(top_n)
        n     = len(top_p)
        # color negative means by mean < 0 (no signal)
        colors = ["#d62728" if v < 0 else "darkorange"
                  for v in top_p["mean"].values]
        fig, ax = plt.subplots(figsize=(lbl_w + bar_w, max(4, n * row_h)))
        _barh_panel(ax, top_p["mean"].values, top_p.index,
                    xlabel="Permutation importance (±std)",
                    title=f"Top {n} — Permutation (val)",
                    color="darkorange",
                    xerr=top_p["std"].values,
                    color_list=colors)
        fig.subplots_adjust(left=lbl_w / (lbl_w + bar_w))
        plt.tight_layout()
        plt.show()

    # ── SHAP ─────────────────────────────────────────────────────────────
    if shap_val is not None:
        try:
            import shap
            # shap.summary_plot manages its own figure; render it standalone
            shap.summary_plot(
                shap_val,
                feature_names=rf_imp.index.tolist() if rf_imp is not None else None,
                max_display=top_n,
                show=False,
                plot_type="bar",
            )
            fig_shap = plt.gcf()
            fig_shap.set_size_inches(lbl_w + bar_w + 1.5, max(5, top_n * row_h))
            fig_shap.subplots_adjust(left=(lbl_w + 1.0) / (lbl_w + bar_w + 1.5))
            ax_shap = fig_shap.axes[0]
            ax_shap.set_title("SHAP feature importance", fontweight="bold",
                              fontsize=10)
            ax_shap.tick_params(axis="y", labelsize=8)
            ax_shap.tick_params(axis="x", labelsize=8)
            ax_shap.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4g"))
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            logger.warning("SHAP plot failed: %s", exc)


def plot_temporal_validation(temporal_dict: Dict[str, Any]) -> None:
    """Four panels: mean autocorrelation, smoothness, drift and entropy."""
    import matplotlib.pyplot as plt

    fig, axes = _make_figure(1, 3, figsize=(16, 5))

    # ── Lag-1 autocorrelation ────────────────────────────────────────────
    acf_df = temporal_dict.get("autocorr_df")
    if acf_df is not None and "lag_1" in acf_df.columns:
        lag1 = acf_df["lag_1"].sort_values(ascending=False)
        top  = lag1.head(20)
        axes[0].barh(np.arange(len(top)), top.values, color="steelblue")
        axes[0].set_yticks(np.arange(len(top)))
        axes[0].set_yticklabels(top.index, fontsize=7)
        axes[0].set(xlabel="ACF lag-1",
                    title="Top 20 most autocorrelated features (lag-1)")

    # ── Smoothness ────────────────────────────────────────────────────────
    sm = temporal_dict.get("smoothness_df")
    if sm is not None:
        top_sm = sm.head(20)
        axes[1].barh(np.arange(len(top_sm)), top_sm.values, color="forestgreen")
        axes[1].set_yticks(np.arange(len(top_sm)))
        axes[1].set_yticklabels(top_sm.index, fontsize=7)
        axes[1].set(xlabel="var(dx)/var(x)  (lower = smoother)",
                    title="Top 20 smoothest features")

    # ── Drift ─────────────────────────────────────────────────────────────
    drift = temporal_dict.get("drift_df")
    if drift is not None:
        top_d = drift.head(20)
        colors = ["#d62728" if p < 0.05 else "#1f77b4"
                  for p in top_d["p_value"].values]
        axes[2].barh(np.arange(len(top_d)), top_d["ks_stat"].values,
                     color=colors)
        axes[2].set_yticks(np.arange(len(top_d)))
        axes[2].set_yticklabels(top_d.index, fontsize=7)
        axes[2].set(xlabel="KS stat (first half vs second)",
                    title="Temporal drift (red = p<0.05)")

    t_ent = temporal_dict.get("transition_entropy")
    if t_ent is not None:
        fig.suptitle(f"Temporal validation, transition entropy: {t_ent:.3f} bits",
                     fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_stability(stability_dict: Dict[str, Any]) -> None:
    """Two panels: intra-video KS and inter-animal KS."""
    import matplotlib.pyplot as plt

    intra = stability_dict.get("intra_video_df")
    inter = stability_dict.get("inter_animal_df")
    n_plots = 1 + (inter is not None)
    fig, axes = _make_figure(1, n_plots, figsize=(8 * n_plots, 5))
    axes = np.atleast_1d(axes)

    if intra is not None:
        top = intra.head(20)
        colors = ["#d62728" if p < 0.05 else "#1f77b4"
                  for p in top["p_value"].values]
        axes[0].barh(np.arange(len(top)), top["ks_stat"].values, color=colors)
        axes[0].set_yticks(np.arange(len(top)))
        axes[0].set_yticklabels(top.index, fontsize=7)
        axes[0].set(xlabel="KS stat", title="Intra-video stability (red = p<0.05)")

    if inter is not None and n_plots > 1:
        top_i = inter.head(20)
        colors = ["#d62728" if p < 0.05 else "#1f77b4"
                  for p in top_i["p_value"].values]
        axes[1].barh(np.arange(len(top_i)), top_i["ks_stat"].values,
                     color=colors)
        axes[1].set_yticks(np.arange(len(top_i)))
        axes[1].set_yticklabels(top_i.index, fontsize=7)
        axes[1].set(xlabel="KS stat",
                    title="Inter-animal variability (red = p<0.05)")

    unstable = stability_dict.get("unstable_features", [])
    fig.suptitle(f"Feature stability: {len(unstable)} unstable (intra-video)",
                 fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_normalization_validation(results: Dict[int, Dict]) -> None:
    """Histogram of body_length, to check the normalisation lands near 1.0."""
    import matplotlib.pyplot as plt

    mid0 = sorted(results.keys())[0]
    fdf  = results[mid0]["features_df"]

    col = "body_length"
    if col not in fdf.columns:
        logger.info("body_length not found in features_df.")
        return

    bl = fdf[col].dropna()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(bl, bins=50, color="steelblue", alpha=0.8)
    ax.axvline(1.0, color="red", linestyle="--", label="Expected ~ 1")
    ax.set(xlabel="body_length (normalised)", ylabel="Frequency",
           title="Distribution of normalised body length")
    ax.legend()
    mu, sigma = float(bl.mean()), float(bl.std())
    ax.text(0.98, 0.95, f"μ={mu:.3f}  σ={sigma:.3f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9)
    plt.tight_layout()
    plt.show()
