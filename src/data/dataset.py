"""
src/data/dataset.py
====================
A PyTorch Dataset for mouse behaviour classification.

Wraps the feature windows produced by ``features.py`` so they can be fed
straight into a PyTorch training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

# Target classes for the CalMS21 subset: the behaviours with at least 200
# segments there. Rarer behaviours are left out because a class with a handful
# of segments cannot be evaluated meaningfully under LOVO.
TARGET_CLASSES: List[str] = [
    "sniff", "attack", "chase", "escape", "mount",
    "approach", "rear", "selfgroom",
]


class LabelEncoder:
    """Bidirectional encoder between a class string and its integer code.

    Parameters
    ----------
    classes : list[str]
        Ordered class list; the position in the list is the integer code.
    background_label : str
        The background class, which is given index -1 so it can be filtered out.
    """

    def __init__(
        self,
        classes: List[str] = TARGET_CLASSES,
        background_label: str = "background",
    ) -> None:
        self.classes = list(classes)
        self.background_label = background_label
        self._label2idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def encode(self, labels: Union[List[str], np.ndarray]) -> np.ndarray:
        """Encode string labels as integers.

        Unknown classes and background both map to -1.
        """
        return np.array(
            [self._label2idx.get(str(lbl), -1) for lbl in labels],
            dtype=np.int64,
        )

    def decode(self, indices: Union[List[int], np.ndarray]) -> List[str]:
        """Decode integers back into string labels."""
        return [
            self.classes[i] if 0 <= i < len(self.classes) else self.background_label
            for i in indices
        ]


# ---------------------------------------------------------------------------
# The main dataset
# ---------------------------------------------------------------------------

class MouseBehaviorDataset(Dataset):
    """A dataset of behaviour windows, for classification.

    Each example is a window of ``window_size`` frames carrying ``n_features``
    kinematic and relational features per frame, plus one class label.

    Parameters
    ----------
    X : np.ndarray, shape (N, W, F)
        Feature windows. N is the number of windows, W the window size and F the
        number of features.
    y : np.ndarray, shape (N,)
        Integer class labels; -1 means background, that is, no class.
    feature_names : list[str], optional
        Names of the F features, for interpretability.
    include_background : bool
        When False, the default, examples labelled -1 are filtered out.
    transform : callable, optional
        Applied to each ``(x, label)`` sample before it is returned. Used for
        augmentation.

    Examples
    --------
    >>> encoder = LabelEncoder()
    >>> y_enc = encoder.encode(y_str)
    >>> ds = MouseBehaviorDataset(X, y_enc, include_background=False)
    >>> loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        include_background: bool = False,
        transform=None,
    ) -> None:
        self.feature_names = feature_names or []
        self.transform = transform

        if not include_background:
            mask = y >= 0
            X = X[mask]
            y = y[mask]

        # Replace NaN and infinities once, at construction time, rather than on
        # every batch of the training loop.
        X_clean = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.X = torch.from_numpy(X_clean)                # (N, W, F)
        self.y = torch.from_numpy(y.astype(np.int64))     # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        label = self.y[idx]
        if self.transform is not None:
            x, label = self.transform(x, label)
        return x, label

    @property
    def n_features(self) -> int:
        return self.X.shape[-1]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.y.max().item()) + 1 if len(self.y) > 0 else 0

    def class_counts(self) -> Dict[int, int]:
        """How many examples the dataset holds of each class."""
        unique, counts = torch.unique(self.y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def class_weights(self) -> torch.Tensor:
        """Weights inversely proportional to each class's frequency.

        Intended for ``torch.nn.CrossEntropyLoss(weight=...)``.

        Returns
        -------
        torch.Tensor, shape (n_classes,) — weights normalised to mean 1.
        """
        counts = self.class_counts()
        max_cls = max(counts.keys()) + 1
        weights = torch.zeros(max_cls)
        for cls, cnt in counts.items():
            weights[cls] = 1.0 / cnt
        weights = weights / weights[weights > 0].mean()
        return weights


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def train_val_test_split(
    dataset: MouseBehaviorDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[MouseBehaviorDataset, MouseBehaviorDataset, MouseBehaviorDataset]:
    """Stratified random split of the dataset into train/val/test.

    Keeps each class in the same proportion across the three splits.

    Parameters
    ----------
    dataset : MouseBehaviorDataset
    val_frac, test_frac : float
        Fraction of examples going to validation and to test.
    seed : int

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    rng = np.random.default_rng(seed)
    N = len(dataset)
    labels = dataset.y.numpy()
    classes = np.unique(labels)

    train_idx, val_idx, test_idx = [], [], []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test_idx.extend(cls_idx[:n_test].tolist())
        val_idx.extend(cls_idx[n_test:n_test + n_val].tolist())
        train_idx.extend(cls_idx[n_test + n_val:].tolist())

    def _subset(idx_list: List[int]) -> MouseBehaviorDataset:
        idx = torch.tensor(idx_list, dtype=torch.long)
        ds = MouseBehaviorDataset.__new__(MouseBehaviorDataset)
        ds.X = dataset.X[idx]
        ds.y = dataset.y[idx]
        ds.feature_names = dataset.feature_names
        ds.transform = dataset.transform
        return ds

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def train_val_test_split_temporal(
    dataset: MouseBehaviorDataset,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    purge: int = 0,
) -> Tuple[MouseBehaviorDataset, MouseBehaviorDataset, MouseBehaviorDataset]:
    """Temporal split of the dataset into train/val/test, with a purge margin.

    Windows keep their original index order, which is the temporal order they
    were extracted in at a fixed stride. Test is the last ``test_frac`` of the
    sequence, val the ``val_frac`` before it, and train everything earlier.

    ``purge`` drops the ``purge`` windows nearest each boundary. This matters:
    at stride 16 with a window of 64, consecutive windows overlap by 75%, so a
    plain cut leaves training frames sitting inside validation and test windows.
    A purge of WINDOW_SIZE (64 windows) removes that overlap leakage entirely.

    Parameters
    ----------
    dataset : MouseBehaviorDataset
    val_frac, test_frac : float
        Fraction of windows going to validation and to test.
    purge : int
        Windows to drop on each side of the train/val and val/test boundaries.
        0 disables the purge, reproducing the earlier, leaky behaviour.

    Returns
    -------
    (train_ds, val_ds, test_ds)
    """
    N = len(dataset)
    n_test = max(1, int(N * test_frac))
    n_val  = max(1, int(N * val_frac))
    n_train = N - n_val - n_test

    # Nominal boundaries
    b1 = n_train          # train | val
    b2 = n_train + n_val  # val   | test

    # Indices, with the purge applied around each boundary
    train_idx = list(range(0,          max(0, b1 - purge)))
    val_idx   = list(range(min(N, b1 + purge), max(0, b2 - purge)))
    test_idx  = list(range(min(N, b2 + purge), N))

    def _subset(idx_list: List[int]) -> MouseBehaviorDataset:
        idx = torch.tensor(idx_list, dtype=torch.long)
        ds = MouseBehaviorDataset.__new__(MouseBehaviorDataset)
        ds.X = dataset.X[idx]
        ds.y = dataset.y[idx]
        ds.feature_names = dataset.feature_names
        ds.transform = dataset.transform
        return ds

    return _subset(train_idx), _subset(val_idx), _subset(test_idx)


def make_weighted_sampler(
    y: np.ndarray,
) -> "torch.utils.data.WeightedRandomSampler":
    """Build a ``WeightedRandomSampler`` inversely weighted by class frequency.

    Each sample gets a weight of ``1 / count(class)``, so in expectation the
    minority classes are oversampled up to the majority class. This is balanced
    oversampling without duplicating any example in memory.

    Parameters
    ----------
    y : np.ndarray, shape (N,)
        Integer class labels, all >= 0; -1 background must not be present.

    Returns
    -------
    WeightedRandomSampler
        ``replacement=True``, ``num_samples=len(y)``.
    """
    from torch.utils.data import WeightedRandomSampler

    classes, counts = np.unique(y, return_counts=True)
    class_weight = {int(c): 1.0 / int(cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([class_weight[int(label)] for label in y], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(y),
        replacement=True,
    )


def collate_padded(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A collate function that pads windows of differing length.

    In practice every window in the Dataset is ``window_size`` long, but this
    handles the general case.
    """
    xs, ys = zip(*batch)
    max_len = max(x.shape[0] for x in xs)
    F = xs[0].shape[-1]
    padded = torch.full((len(xs), max_len, F), pad_value)
    for i, x in enumerate(xs):
        padded[i, :x.shape[0]] = x
    return padded, torch.stack(ys)
