"""Label encoding + temporal split (no-leakage) — high-risk pure functions."""
import numpy as np
from src.data.dataset import (
    LabelEncoder, MouseBehaviorDataset, train_val_test_split_temporal,
)


def test_label_encoder_roundtrip():
    enc = LabelEncoder(classes=["sniff", "mount", "approach"])
    idx = enc.encode(["mount", "approach", "sniff", "nope", "background"])
    assert list(idx) == [1, 2, 0, -1, -1]            # unknown + background → -1
    assert enc.decode([0, 1, 2]) == ["sniff", "mount", "approach"]


def test_temporal_split_purge_removes_boundary_overlap():
    N, W, F = 100, 8, 4
    X = np.random.RandomState(0).randn(N, W, F).astype(np.float32)
    y = np.zeros(N, dtype=np.int64)
    ds = MouseBehaviorDataset(X, y, include_background=True)
    tr, va, te = train_val_test_split_temporal(ds, val_frac=0.2, test_frac=0.2, purge=5)
    # purge shrinks the sets vs no-purge and keeps them non-empty
    tr0, va0, te0 = train_val_test_split_temporal(ds, val_frac=0.2, test_frac=0.2, purge=0)
    assert len(tr) < len(tr0)
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0
    # total with purge is strictly less than N (windows were dropped at the seams)
    assert len(tr) + len(va) + len(te) < N


def test_dataset_filters_background_by_default():
    X = np.zeros((4, 2, 3), dtype=np.float32)
    y = np.array([-1, 0, 1, -1], dtype=np.int64)
    ds = MouseBehaviorDataset(X, y)                  # include_background=False
    assert len(ds) == 2
