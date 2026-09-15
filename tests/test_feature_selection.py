"""B5 — feature-selection diagnostics."""
import numpy as np
from src.data.feature_selection import vif_scores, drift_scores


def test_vif_flags_perfect_collinearity():
    rng = np.random.RandomState(0)
    a = rng.randn(500)
    b = rng.randn(500)
    c = a + b                                   # exact linear combination
    Z = np.column_stack([a, b, c])
    Z = (Z - Z.mean(0)) / Z.std(0)
    vif = vif_scores(Z)
    assert vif.max() > 100                      # the dependent column blows up


def test_vif_low_for_independent_features():
    rng = np.random.RandomState(1)
    Z = rng.randn(500, 4)
    Z = (Z - Z.mean(0)) / Z.std(0)
    assert vif_scores(Z).max() < 5              # independent → ~1


def test_drift_high_when_feature_shifts_across_videos():
    # feature 0 has a per-video offset (drift); feature 1 is iid across videos
    rng = np.random.RandomState(2)
    vids = np.array(["v0"] * 100 + ["v1"] * 100 + ["v2"] * 100)
    offset = np.where(vids == "v0", 0.0, np.where(vids == "v1", 10.0, 20.0))
    f0 = offset + rng.randn(300) * 0.1
    f1 = rng.randn(300)
    X = np.column_stack([f0, f1])
    d = drift_scores(X, vids)
    assert d[0] > d[1] * 10                      # drifting feature scores far higher
