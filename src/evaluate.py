"""
src/evaluate.py
===============
Evaluation helpers for the classification models.

Provides:
- ``evaluate_model()``          — inference and metrics over a DataLoader
- ``evaluate_baseline()``       — metrics for the sklearn classifiers
- ``classification_report_df()``— a DataFrame summary: F1, precision, recall
- ``plot_confusion_matrix()``   — normalised matplotlib figure
- ``plot_training_curves()``    — loss and F1 curves per epoch
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Inference with a PyTorch model
# ---------------------------------------------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    return_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run inference and collect the predictions.

    Parameters
    ----------
    model : nn.Module
        A trained model.
    loader : DataLoader
    device : torch.device | None
    return_probs : bool
        When True, also return the softmax probabilities.

    Returns
    -------
    (y_true, y_pred, probs | None)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_prob: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            if return_probs:
                all_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    probs_arr = np.concatenate(all_prob) if return_probs else None

    return y_true, y_pred, probs_arr


def evaluate_model_tta(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    n_tta: int = 10,
    noise_std: float = 0.015,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inference with test-time augmentation (TTA).

    Runs ``n_tta`` passes with additive Gaussian noise and averages the softmax
    probabilities. This cuts the prediction variance on small datasets, where a
    minority class can have so few windows that one wrong prediction costs some
    25 percentage points of its F1.

    Parameters
    ----------
    n_tta : int
        Number of noisy passes; 10 is a reasonable compromise.
    noise_std : float
        Standard deviation of the additive noise, about 0.015 once the features
        are normalised.

    Returns
    -------
    (y_true, y_pred, avg_probs), where avg_probs is (N, n_classes)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    accumulated_probs: Optional[np.ndarray] = None
    y_true_arr: Optional[np.ndarray] = None

    for _ in range(n_tta):
        run_probs: List[np.ndarray] = []
        run_true:  List[np.ndarray] = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                x = x + noise_std * torch.randn_like(x)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                logits = model(x)
                probs  = torch.softmax(logits, dim=-1)
                run_probs.append(probs.cpu().numpy())
                run_true.append(y.numpy())

        probs_pass = np.concatenate(run_probs)
        if accumulated_probs is None:
            accumulated_probs = probs_pass
            y_true_arr = np.concatenate(run_true)
        else:
            accumulated_probs = accumulated_probs + probs_pass

    avg_probs: np.ndarray = accumulated_probs / n_tta   # type: ignore[operator]
    y_pred = avg_probs.argmax(axis=1)
    return y_true_arr, y_pred, avg_probs      # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Evaluating the sklearn baselines
# ---------------------------------------------------------------------------

def evaluate_baseline(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    return_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Evaluate an sklearn classifier.

    Parameters
    ----------
    clf : _BaseClassifier
        A fitted ``RandomForestBaseline``, ``SVMBaseline`` or similar.
    X : array (N, n_features)
    y : array (N,)
    return_probs : bool

    Returns
    -------
    (y_true, y_pred, probs | None)
    """
    y_pred = clf.predict(X)
    probs = clf.predict_proba(X) if return_probs else None
    return y, y_pred, probs


# ---------------------------------------------------------------------------
# Classification report
# ---------------------------------------------------------------------------

def classification_report_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """sklearn's classification report, as a DataFrame.

    Parameters
    ----------
    y_true, y_pred : array (N,)
    class_names : list[str] | None
        Class names, in index order.

    Returns
    -------
    pd.DataFrame with precision, recall, f1-score and support columns.
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    target_names = (
        [class_names[i] for i in labels]
        if class_names is not None
        else [str(i) for i in labels]
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    # Round for readability
    for col in ["precision", "recall", "f1-score"]:
        if col in df.columns:
            df[col] = df[col].round(4)
    return df


# ---------------------------------------------------------------------------
# Summary of the headline metrics
# ---------------------------------------------------------------------------

def metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """A compact summary of the headline metrics.

    Returns
    -------
    dict with ``f1_macro``, ``f1_weighted``, ``accuracy`` and ``f1_per_class``.
    """
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    per_class = {
        (class_names[i] if class_names else str(i)): float(f1_per[j])
        for j, i in enumerate(labels)
        if j < len(f1_per)
    }
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "f1_per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Plotting: confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 7),
):
    """A matplotlib figure of the normalised confusion matrix.

    Parameters
    ----------
    normalize : bool
        When True, normalise by row, so each cell reads as a per-class recall.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # `out=` is required, not optional: with `where=` alone the masked entries
        # are left as uninitialised memory and render as garbage. A row sums to
        # zero whenever a class appears in neither y_true nor y_pred, which happens
        # for a rare behaviour in a single LOVO fold.
        cm_plot = np.divide(cm.astype(float), row_sums,
                            out=np.zeros(cm.shape, dtype=float),
                            where=row_sums != 0)
    else:
        cm_plot = cm.astype(float)

    names = (
        [class_names[i] for i in labels]
        if class_names is not None
        else [str(i) for i in labels]
    )

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(names)),
        yticks=np.arange(len(names)),
        xticklabels=names,
        yticklabels=names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm_plot.max() / 2.0
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            ax.text(
                j, i,
                format(cm_plot[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plotting: training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (16, 4),
):
    """A figure with the loss, F1-macro and learning-rate curves per epoch.

    Parameters
    ----------
    history : dict
        Output of ``src.train.fit()``. Required keys: ``train_loss``,
        ``val_loss``, ``val_f1``. Optional: ``train_f1``, ``lr``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    has_train_f1 = "train_f1" in history and len(history["train_f1"]) > 0
    has_lr       = "lr" in history and len(history["lr"]) > 0
    n_panels = 2 + (1 if has_lr else 0)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(title)

    # Panel 1: loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"],   label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: F1-macro, train as well as val when available
    ax = axes[1]
    if has_train_f1:
        ax.plot(epochs, history["train_f1"], color="steelblue",
                linestyle="--", alpha=0.8, label="Train F1-macro")
    ax.plot(epochs, history["val_f1"], color="green", label="Val F1-macro")
    best_ep = int(np.argmax(history["val_f1"])) + 1
    best_f1 = max(history["val_f1"])
    ax.axvline(best_ep, color="green", linestyle="--", alpha=0.5,
               label=f"Best epoch {best_ep} ({best_f1:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Learning rate
    if has_lr:
        ax = axes[2]
        ax.plot(epochs, history["lr"], color="darkorange", label="LR")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Post-processing: temporal smoothing of the predictions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Calibration: expected calibration error
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[float], List[float], List[int]]:
    """Expected calibration error (ECE), plus the reliability-diagram data.

    A perfectly calibrated model has confidence equal to accuracy, so ECE = 0.
    Models trained with focal loss tend to be overconfident, and score a high
    ECE even when their F1 is good.

    Parameters
    ----------
    y_true : np.ndarray (N,)
    probs  : np.ndarray (N, C)  — softmax probabilities
    n_bins : int
        Number of confidence bins spanning [0, 1].

    Returns
    -------
    (ece, bin_mean_conf, bin_mean_acc, bin_counts)
        ece            : float — the expected calibration error
        bin_mean_conf  : List[float] — mean confidence per bin
        bin_mean_acc   : List[float] — mean accuracy per bin
        bin_counts     : List[int]   — samples per bin
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies  = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_means_conf: List[float] = []
    bin_means_acc:  List[float] = []
    bin_counts:     List[int]   = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc  = float(accuracies[mask].mean())
        ece += (n_bin / len(y_true)) * abs(avg_conf - avg_acc)
        bin_means_conf.append(avg_conf)
        bin_means_acc.append(avg_acc)
        bin_counts.append(n_bin)

    return ece, bin_means_conf, bin_means_acc, bin_counts


def temporal_smooth_predictions(
    y_pred: np.ndarray,
    window: int = 5,
) -> np.ndarray:
    """Sliding-majority vote over contiguous predictions.

    Removes spurious single-window predictions, which are classification noise
    rather than real behaviour, by taking a majority vote over a centred window
    of size ``window``. Near the edges the window is truncated, as in "same"
    mode.

    Parameters
    ----------
    y_pred : np.ndarray, shape (N,)
        The original predictions, as class integers.
    window : int
        How many contiguous predictions to vote over. Must be odd; an even
        value is incremented by one.

    Returns
    -------
    np.ndarray, shape (N,)
        The smoothed predictions.
    """
    if window % 2 == 0:
        window += 1
    half = window // 2
    n = len(y_pred)
    smoothed = np.empty_like(y_pred)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = y_pred[lo:hi]
        smoothed[i] = int(np.bincount(chunk).argmax())

    return smoothed
