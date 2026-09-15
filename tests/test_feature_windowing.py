"""Re-windowing must preserve the original windowing params and X/y alignment.

Regression for the 03_features failure `X.shape[0]=179 != len(y)=357`: the
imputation / circular-encoding steps re-window `X` from the modified
features_df, but omitted the stride, so they silently fell back to the module
default (32) instead of the configured one (16) — halving the window count and
desynchronising X from y (and from an already-computed split).
"""
import numpy as np
import pandas as pd

from src.data.features import build_windows, STRIDE
from src.data.feature_pipeline import apply_circular_encoding_to_results

T, F, W, S = 5760, 6, 64, 16          # S=16 mirrors configs/default.yaml


def _results():
    idx = np.arange(T)
    cols = [f"body_angle_{i}" if i < 2 else f"f{i}" for i in range(F)]
    fdf = pd.DataFrame(np.random.RandomState(0).randn(T, F), columns=cols, index=idx)
    labels = np.array(["background"] * T, dtype=object)
    labels[100:400] = "sniff"
    X, y = build_windows(fdf, labels, W, S)
    res = {"features_df": fdf, "X": X, "y": y, "feature_names": cols,
           "frames": idx, "labels_per_frame": labels,
           "window_size": W, "stride": S}
    return {1: res}, X, y


def test_configured_stride_differs_from_module_default():
    # Guards the premise: if these ever coincide the regression is untestable.
    assert S != STRIDE


def test_circular_encoding_preserves_window_count_and_alignment():
    results, X0, y0 = _results()
    out = apply_circular_encoding_to_results(results)[1]
    assert out["X"].shape[0] == len(out["y"])        # aligned
    assert out["X"].shape[0] == X0.shape[0]          # count unchanged (stride kept)


def test_default_stride_would_have_halved_the_windows():
    # Documents the original bug: the module default yields ~half the windows.
    _, X0, _ = _results()
    idx = np.arange(T)
    cols = [f"body_angle_{i}" if i < 2 else f"f{i}" for i in range(F)]
    fdf = pd.DataFrame(np.random.RandomState(0).randn(T, F), columns=cols, index=idx)
    X_default, _ = build_windows(fdf, None, W, STRIDE)
    assert X_default.shape[0] < X0.shape[0]
    assert X0.shape[0] == 357 and X_default.shape[0] == 179
