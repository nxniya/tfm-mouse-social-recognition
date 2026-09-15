"""
src/models/baseline.py
======================
Classical scikit-learn classifiers.

They operate on the flat vector of 300 window statistics produced by
``src.data.features.window_statistics()``.

Three feature configurations are compared:
- B1: raw keypoints, the MARS-style baseline
- B2: corrected keypoints, isolating the effect of MouseSkeleton
- B3: implicit-3D features, isolating the effect of dz
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class _BaseClassifier:
    """The interface shared by every classical baseline."""

    pipeline: Pipeline

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseClassifier":
        """Fit the pipeline.

        Parameters
        ----------
        X : array, shape (N, n_features)
            Flat vector of window statistics.
        y : array, shape (N,)
            Class labels, as integers >= 0.
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities. The inner estimator must expose
        ``predict_proba``, as RF and GB do, or be calibrated, as the SVM is.
        """
        return self.pipeline.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Plain accuracy."""
        return self.pipeline.score(X, y)

    def f1_macro(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average="macro", zero_division=0)

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Feature importances; available for RF and GB only."""
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
        return None

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Stratified cross-validation.

        Returns
        -------
        dict with keys ``fit_time``, ``test_f1_macro``, ``test_accuracy``.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scoring = {
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "accuracy": "accuracy",
        }
        results = cross_validate(
            self.pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
        )
        return {
            "fit_time": results["fit_time"],
            "test_f1_macro": results["test_f1_macro"],
            "test_accuracy": results["test_accuracy"],
        }

    def cross_validate_temporal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """Stratified cross-validation with shuffling (StratifiedKFold).

        StratifiedKFold keeps the class distribution intact in every fold.
        TimeSeriesSplit is not usable here: the behaviours are ordered in time,
        so its early folds contain a single class and sklearn 1.8 and later
        return NaN for them, error_score being nan by default.

        Note this does NOT control the overlap leakage between windows; that is
        handled in the hold-out split by the ``purge`` parameter. Scores from
        this method are therefore optimistic, which is exactly the leak the LOVO
        protocol was introduced to replace.

        Returns
        -------
        dict with keys ``fit_time``, ``test_f1_macro``, ``test_accuracy``.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scoring = {
            "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
            "accuracy": "accuracy",
        }
        results = cross_validate(
            self.pipeline, X, y, cv=cv, scoring=scoring,
            return_train_score=False, error_score="raise",
        )
        return {
            "fit_time": results["fit_time"],
            "test_f1_macro": results["test_f1_macro"],
            "test_accuracy": results["test_accuracy"],
        }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestBaseline(_BaseClassifier):
    """Random forest over the 300 window statistics.

    Parameters
    ----------
    n_estimators : int
        Number of trees.
    max_depth : int | None
        Maximum depth; None leaves it unrestricted.
    min_samples_leaf : int
        Minimum samples per leaf, which regularises against overfitting. The
        value 4 cuts the variance at no meaningful cost in bias.
    class_weight : str | dict | None
        ``"balanced"`` weights each class inversely to its frequency.
    seed : int
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 4,
        class_weight: str = "balanced",
        seed: int = 42,
    ) -> None:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=seed,
        )
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", rf)]
        )


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

class SVMBaseline(_BaseClassifier):
    """A calibrated linear SVM over the window statistics.

    Uses ``LinearSVC``, which scales well with large N, wrapped in
    ``CalibratedClassifierCV`` so that probabilities are available.

    Parameters
    ----------
    C : float
        Regularisation parameter.
    class_weight : str | None
        ``"balanced"`` weights each class inversely to its frequency.
    seed : int
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        seed: int = 42,
    ) -> None:
        svm = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
            dual="auto",
        )
        calibrated = CalibratedClassifierCV(svm, cv=5, method="isotonic")
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", calibrated),
            ]
        )


# ---------------------------------------------------------------------------
# Gradient Boosting
# ---------------------------------------------------------------------------

class GradientBoostingBaseline(_BaseClassifier):
    """HistGradientBoosting over the window statistics.

    Faster than the classic ``GradientBoostingClassifier``, and it handles NaN
    natively.

    Parameters
    ----------
    max_iter : int
        Maximum boosting iterations.
    learning_rate : float
    max_depth : int | None
    n_iter_no_change : int
        Early stopping: halt training when the validation metric has not
        improved for this many iterations.
    validation_fraction : float
        Fraction of the training set held out for early stopping.
    seed : int
    """

    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        max_depth: Optional[int] = 5,
        n_iter_no_change: int = 20,
        validation_fraction: float = 0.1,
        class_weight: Optional[str] = "balanced",
        seed: int = 42,
    ) -> None:
        from sklearn.ensemble import HistGradientBoostingClassifier

        gb = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            class_weight=class_weight,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            random_state=seed,
        )
        # HistGBC exposes predict_proba directly, so no calibration is needed
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", gb)]
        )


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LGBMBaseline(_BaseClassifier):
    """LightGBM over the window statistics.

    Requires ``lightgbm`` to be installed (``pip install lightgbm``). LightGBM
    handles NaN natively, so no imputer is used.

    Parameters
    ----------
    n_estimators : int
        Number of trees; LightGBM calls this num_iterations.
    learning_rate : float
    num_leaves : int
        Tree complexity. The default is 31; lower it on small datasets.
    min_child_samples : int
        Minimum samples per leaf, which regularises the fit.
    seed : int
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 5,
        seed: int = 42,
    ) -> None:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            ) from exc

        lgbm = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", lgbm)]
        )


# ---------------------------------------------------------------------------
# Helper: comparing feature configurations
# ---------------------------------------------------------------------------

def compare_feature_configs(
    configs: Dict[str, Tuple[np.ndarray, np.ndarray]],
    model_factory=None,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compare several feature configurations using the same classifier.

    Parameters
    ----------
    configs : dict[str, (X, y)]
        Maps a name to the ``(X_flat, y)`` pair for that configuration, for
        example ``{"B1_raw": (X_b1, y), "B2_corr": (X_b2, y), ...}``.
    model_factory : callable | None
        Zero-argument callable returning a fresh ``_BaseClassifier``. Defaults
        to ``RandomForestBaseline``.
    n_splits : int
    seed : int

    Returns
    -------
    dict[config_name, dict[metric, float_mean]]
    """
    if model_factory is None:
        model_factory = RandomForestBaseline

    results: Dict[str, Dict[str, float]] = {}
    for name, (X, y) in configs.items():
        clf = model_factory()
        cv = clf.cross_validate(X, y, n_splits=n_splits, seed=seed)
        results[name] = {
            "f1_macro_mean": float(np.mean(cv["test_f1_macro"])),
            "f1_macro_std": float(np.std(cv["test_f1_macro"])),
            "accuracy_mean": float(np.mean(cv["test_accuracy"])),
        }
    return results
