"""
src/train.py
============
Training loop for the PyTorch models.

Provides:
- ``fit()``           — trains with early stopping, returns the history
- ``train_epoch()``   — one training epoch
- ``val_epoch()``     — one validation or evaluation epoch

Design notes:
- No dependencies beyond torch and sklearn, the latter only for F1
- The best model is kept in memory as best_state_dict and never written to disk
  unless ``checkpoint_path`` is given
- Works with any model that takes (N, T, F) and returns (N, C)
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: Optional[float] = 1.0,
    scheduler: Optional[object] = None,
    augment: bool = False,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    """Train the model for one full epoch.

    Parameters
    ----------
    scheduler : a per-batch lr_scheduler, such as OneCycleLR.
        When given, ``scheduler.step()`` is called after every batch.
    augment : bool
        When True, apply light augmentation to the features:
        - Gaussian noise with sigma 0.02
        - Random temporal dropout of whole frames, p=0.05 per frame
    mixup_alpha : float
        The alpha of the Beta distribution used for mixup; 0 disables it. 0.4
        works well here. Mixup convexly interpolates pairs of examples, which is
        a strong regulariser on a small, imbalanced dataset.

    Returns
    -------
    Tuple[float, float]
        (mean loss per example, F1-macro over the training split)
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Replace NaN with 0, which is neutral once the features are centred on
        # the centroid. This stops BatchNorm1d accumulating NaN into its
        # running_mean and running_var, where one bad batch would poison every
        # subsequent forward pass.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if augment:
            # Gaussian noise on the features
            x = x + 0.02 * torch.randn_like(x)
            # Temporal dropout of whole frames, p=0.05. The mask is shaped to
            # broadcast over both 3-D (B,T,F) and 4-D (B,T,V,F) inputs.
            mask_shape = (x.size(0), x.size(1)) + (1,) * (x.dim() - 2)
            mask = (torch.rand(*mask_shape, device=device) > 0.05).float()
            x = x * mask

        optimizer.zero_grad()

        if mixup_alpha > 0:
            # Mixup: a convex blend of two examples from the batch
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx = torch.randperm(x.size(0), device=device)
            x_mix = lam * x + (1.0 - lam) * x[idx]
            logits = model(x_mix)
            loss = lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y[idx])
        else:
            logits = model(x)
            loss = criterion(logits, y)

        loss.backward()

        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        # Collect predictions against the primary label y, which works whether
        # or not mixup is active
        all_preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
        all_labels.append(y.cpu().numpy())

    train_f1 = f1_score(
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        average="macro",
        zero_division=0,
    )
    return total_loss / max(n_batches, 1), train_f1


# ---------------------------------------------------------------------------
# One validation epoch
# ---------------------------------------------------------------------------

def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set.

    Returns
    -------
    (val_loss, val_f1_macro)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    val_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return total_loss / max(n_batches, 1), val_f1


# ---------------------------------------------------------------------------
# The main training loop
# ---------------------------------------------------------------------------

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: Optional[torch.device] = None,
    criterion: Optional[nn.Module] = None,
    clip_grad_norm: float = 1.0,
    checkpoint_path: Optional[str] = None,
    scheduler_type: str = "cosine",
    augment: bool = False,
    mixup_alpha: float = 0.0,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train the model, with early stopping on the validation F1-macro.

    Parameters
    ----------
    model : nn.Module
        The model to train, for example ``BehaviorLSTM``.
    train_loader, val_loader : DataLoader
    epochs : int
        Maximum number of epochs.
    lr : float
        Initial learning rate for AdamW.
    weight_decay : float
        L2 regularisation, as applied by AdamW.
    patience : int
        Epochs without improvement before training stops.
    device : torch.device | None
        When None, use CUDA if available and CPU otherwise.
    criterion : nn.Module | None
        Loss function. When None, plain CrossEntropyLoss is used.
    clip_grad_norm : float
        Gradient-norm clipping threshold; 0 disables it.
    checkpoint_path : str | None
        When given, the best model is saved to this path.
    scheduler_type : str
        ``"cosine"`` — CosineAnnealingLR, stepped per epoch; the default.
        ``"one_cycle"`` — OneCycleLR, stepped per batch, warmup plus cosine
        decay. On the CNN models this was worth roughly 5 to 10 points of F1
        over a fixed learning rate.
    augment : bool
        When True, apply data augmentation during training; see ``train_epoch``.
    mixup_alpha : float
        Mixup alpha, passed through to ``train_epoch``; 0 disables it.
    verbose : bool
        Print per-epoch progress.

    Returns
    -------
    history : dict of lists ``train_loss``, ``val_loss``, ``val_f1``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Scheduler
    batch_scheduler: Optional[object] = None
    if scheduler_type == "one_cycle":
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        epoch_scheduler = None
    else:  # "cosine"
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 1e-2
        )
        batch_scheduler = None

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_state: Optional[dict] = None
    wait = 0
    clip = clip_grad_norm if clip_grad_norm > 0 else None

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, clip,
            scheduler=batch_scheduler, augment=augment, mixup_alpha=mixup_alpha,
        )
        vl_loss, vl_f1 = val_epoch(model, val_loader, criterion, device)
        if epoch_scheduler is not None:
            epoch_scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(vl_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            wait += 1

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_loss={tr_loss:.4f}  tr_f1={tr_f1:.4f} | "
                f"val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"({elapsed:.1f}s)"
                # ASCII marker: a non-ASCII one crashes on a cp1252 console
                + (" *" if wait == 0 else "")
            )

        if wait >= patience:
            if verbose:
                print(f"  -> Early stopping at epoch {epoch} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history
