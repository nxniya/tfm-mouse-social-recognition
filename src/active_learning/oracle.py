"""
src/active_learning/oracle.py
==============================
A simulated expert annotator for the active-learning loop.

The oracle holds the true training labels. When the active learner queries a
sample, the oracle returns its true label, which simulates manual annotation
without needing a real human annotator in the loop.

The module also manages the pool of unlabelled samples and provides helpers for
initialising an AL experiment from the precomputed features of the reference
video, CalMS21_task1_13638642.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import h5py


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FEATURES_H5 = _REPO_ROOT / "dataset" / "features" / "CalMS21_task1_13638642_features.h5"


# ---------------------------------------------------------------------------
# The simulated oracle
# ---------------------------------------------------------------------------

class SimulatedOracle:
    """A simulated expert annotator for active learning.

    Tracks the state of the unlabelled pool and the labelled set, and exposes a
    ``query()`` method that reveals the true labels.

    Parameters
    ----------
    pool_X : np.ndarray, shape (N, T, F)
        Feature windows making up the unlabelled pool.
    pool_y : np.ndarray, shape (N,)
        True pool labels. The model must never see these during AL.
    feature_names : list[str], optional
        Per-frame feature names.
    pool_meta : dict, optional
        Extra per-window metadata, for instance ``sniff_site_ratio``.
    random_state : int
        Seed, for reproducibility.
    """

    def __init__(
        self,
        pool_X: np.ndarray,
        pool_y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        pool_meta: Optional[Dict[str, np.ndarray]] = None,
        random_state: int = 42,
    ) -> None:
        if len(pool_X) != len(pool_y):
            raise ValueError(
                f"pool_X ({len(pool_X)}) and pool_y ({len(pool_y)}) "
                "must hold the same number of samples."
            )
        self._pool_X = pool_X.copy()
        self._pool_y = pool_y.copy()
        self.feature_names = feature_names or []
        self._pool_meta = {k: v.copy() for k, v in (pool_meta or {}).items()}
        self._rng = np.random.default_rng(random_state)

        # Original pool indices, kept for traceability
        self._original_indices = np.arange(len(pool_X))

        # State: which of the current pool's indices are still available
        self._available_mask = np.ones(len(pool_X), dtype=bool)

        # Query history, as (round, original_index, label)
        self.query_history: List[Tuple[int, int, int]] = []
        self._round = 0

    # ------------------------------------------------------------------
    # Properties describing the pool's current state
    # ------------------------------------------------------------------

    @property
    def n_available(self) -> int:
        """Number of samples still available to query."""
        return int(self._available_mask.sum())

    @property
    def n_queried(self) -> int:
        """Number of samples that have already been queried."""
        return len(self.query_history)

    @property
    def pool_X(self) -> np.ndarray:
        """The available pool windows, that is, the ones not yet queried."""
        return self._pool_X[self._available_mask]

    @property
    def pool_y_hidden(self) -> np.ndarray:
        """Pool labels. For internal evaluation only; never use these in a real
        AL loop, since reading them is exactly the leak the oracle exists to
        prevent."""
        return self._pool_y[self._available_mask]

    @property
    def pool_meta(self) -> Dict[str, np.ndarray]:
        """Metadata for the available pool."""
        return {k: v[self._available_mask] for k, v in self._pool_meta.items()}

    @property
    def available_indices(self) -> np.ndarray:
        """Original indices of the available samples."""
        return self._original_indices[self._available_mask]

    # ------------------------------------------------------------------
    # Querying the oracle
    # ------------------------------------------------------------------

    def query(
        self,
        pool_indices: Union[List[int], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reveal the labels of the queried samples.

        The indices are relative to the **current** pool, not the original one.
        Once queried, the samples are marked unavailable for later rounds.

        Parameters
        ----------
        pool_indices : array-like of int
            Indices into ``self.pool_X`` to annotate.

        Returns
        -------
        (X_queried, y_queried) : (np.ndarray, np.ndarray)
            Features and labels of the queried samples.
        """
        pool_indices = np.asarray(pool_indices, dtype=int)
        available_orig = self.available_indices  # original indices still available

        if pool_indices.max() >= len(available_orig):
            raise IndexError(
                f"Index {pool_indices.max()} is outside the current pool "
                f"(size {len(available_orig)})."
            )

        orig_idx = available_orig[pool_indices]

        # Record in the history
        for oi, pi in zip(orig_idx, pool_indices):
            label = int(self._pool_y[oi])
            self.query_history.append((self._round, int(oi), label))

        # Mark as unavailable
        self._available_mask[orig_idx] = False
        self._round += 1

        return (
            self._pool_X[orig_idx].copy(),
            self._pool_y[orig_idx].copy(),
        )

    # ------------------------------------------------------------------
    # Random initial seed
    # ------------------------------------------------------------------

    def get_initial_seed(
        self,
        n: int,
        stratified: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw an initial labelled seed of ``n`` samples.

        Parameters
        ----------
        n : int
            Seed size.
        stratified : bool
            When True, take at least one sample per class. Without this, a rare
            class can be missing from the seed entirely, and the first model has
            no way to score samples of a class it has never seen.

        Returns
        -------
        (X_seed, y_seed)
        """
        y_avail = self._pool_y[self._available_mask]
        avail_orig = self.available_indices

        if stratified:
            classes = np.unique(y_avail)
            per_class = max(1, n // len(classes))
            seed_local: List[int] = []
            for c in classes:
                class_idx = np.where(y_avail == c)[0]
                take = min(per_class, len(class_idx))
                chosen = self._rng.choice(class_idx, size=take, replace=False)
                seed_local.extend(chosen.tolist())
            # If n exceeds the per-class picks, fill the remainder at random
            remaining_local = np.setdiff1d(
                np.arange(len(y_avail)), seed_local, assume_unique=True
            )
            extra = n - len(seed_local)
            if extra > 0 and len(remaining_local) > 0:
                take = min(extra, len(remaining_local))
                extra_idx = self._rng.choice(remaining_local, size=take, replace=False)
                seed_local.extend(extra_idx.tolist())
            seed_local_arr = np.array(seed_local[:n])
        else:
            n_take = min(n, len(y_avail))
            seed_local_arr = self._rng.choice(
                len(y_avail), size=n_take, replace=False
            )

        return self.query(seed_local_arr)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def label_distribution(self) -> Dict[int, int]:
        """Label distribution over the original pool."""
        labels, counts = np.unique(self._pool_y, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    def queried_label_distribution(self) -> Dict[int, int]:
        """Label distribution over the samples queried so far."""
        if not self.query_history:
            return {}
        queried_labels = [entry[2] for entry in self.query_history]
        labels, counts = np.unique(queried_labels, return_counts=True)
        return dict(zip(labels.tolist(), counts.tolist()))

    def summary(self) -> str:
        """A textual summary of the oracle's state."""
        lines = [
            f"SimulatedOracle - original pool: {len(self._pool_X)} samples",
            f"  Available    : {self.n_available}",
            f"  Queried      : {self.n_queried}",
            f"  Current round: {self._round}",
            "  Original pool distribution:",
        ]
        for c, cnt in self.label_distribution().items():
            lines.append(f"    class {c}: {cnt}")
        if self.query_history:
            lines.append("  Queried distribution:")
            for c, cnt in self.queried_label_distribution().items():
                lines.append(f"    class {c}: {cnt}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading the pool from the precomputed HDF5
# ---------------------------------------------------------------------------

def load_pool_from_h5(
    h5_path: Union[str, Path] = _FEATURES_H5,
    labeled_split: str = "train",
    include_unlabeled: bool = True,
) -> Tuple[
    np.ndarray,  # X_labeled
    np.ndarray,  # y_labeled
    np.ndarray,  # X_pool, the unlabelled pool
    np.ndarray,  # y_pool_hidden, the pool's labels, held back for the simulation
    List[str],   # feature_names
    Dict[str, np.ndarray],  # pool_meta
]:
    """Load the data from the precomputed feature HDF5.

    Returns:
    - ``X_labeled / y_labeled``   — labelled windows, train and val
    - ``X_pool / y_pool_hidden``  — unlabelled windows forming the AL pool
    - ``feature_names``           — feature names
    - ``pool_meta``               — pool metadata, such as sniff_site_ratio

    The H5 file must have the layout written by ``notebooks/03_features.ipynb``.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {h5_path}\n"
            "Run notebooks/03_features.ipynb first"
        )

    with h5py.File(h5_path, "r") as f:
        # Try the standard split names first
        if "X_train" in f and "y_train" in f:
            X_labeled = f["X_train"][:]
            y_labeled = f["y_train"][:]
        elif "X" in f and "y" in f:
            X_labeled = f["X"][:]
            y_labeled = f["y"][:]
        else:
            raise KeyError(
                "The HDF5 file does not contain the expected keys. "
                "Available keys: " + str(list(f.keys()))
            )

        # Feature names
        if "feature_names" in f:
            raw = f["feature_names"][:]
            feature_names = [
                n.decode() if isinstance(n, bytes) else str(n) for n in raw
            ]
        else:
            n_feat = X_labeled.shape[-1] if X_labeled.ndim == 3 else X_labeled.shape[-1]
            feature_names = [f"feat_{i}" for i in range(n_feat)]

        # Unlabelled pool
        X_pool = np.empty((0,) + X_labeled.shape[1:], dtype=X_labeled.dtype)
        y_pool_hidden = np.empty(0, dtype=y_labeled.dtype)
        pool_meta: Dict[str, np.ndarray] = {}

        if include_unlabeled and "X_unlabeled" in f:
            X_pool = f["X_unlabeled"][:]
            y_pool_hidden = (
                f["y_unlabeled"][:] if "y_unlabeled" in f
                else np.full(len(X_pool), -1, dtype=np.int64)
            )
            # Optional pool metadata
            if "pool_meta" in f:
                for key in f["pool_meta"]:
                    pool_meta[key] = f["pool_meta"][key][:]

    return X_labeled, y_labeled, X_pool, y_pool_hidden, feature_names, pool_meta


def build_oracle_from_h5(
    h5_path: Union[str, Path] = _FEATURES_H5,
    random_state: int = 42,
) -> Tuple["SimulatedOracle", np.ndarray, np.ndarray, List[str]]:
    """Build a ready-to-use SimulatedOracle from the feature H5 file.

    Returns
    -------
    oracle : SimulatedOracle
        The unlabelled pool, managed by the oracle.
    X_labeled : np.ndarray
        Features of the labelled set, used to initialise the model.
    y_labeled : np.ndarray
        Labels of the labelled set.
    feature_names : list[str]
    """
    X_lab, y_lab, X_pool, y_pool, feat_names, pool_meta = load_pool_from_h5(h5_path)

    oracle = SimulatedOracle(
        pool_X=X_pool,
        pool_y=y_pool,
        feature_names=feat_names,
        pool_meta=pool_meta,
        random_state=random_state,
    )

    return oracle, X_lab, y_lab, feat_names
