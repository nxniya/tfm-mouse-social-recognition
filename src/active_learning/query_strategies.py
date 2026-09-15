"""
src/active_learning/query_strategies.py
========================================
Query strategies for the active learning loop.

Every strategy takes a ``pool`` of unlabelled samples and returns the indices of the
``n_query`` samples worth annotating next.

Implemented strategies
----------------------
- ``LeastConfidenceStrategy``  lowest confidence in the predicted class
- ``MarginSamplingStrategy``   smallest margin between the top two classes
- ``EntropySamplingStrategy``  highest Shannon entropy
- ``BALDStrategy``             BALD with MC dropout, that is, mutual information
- ``CoresetStrategy``          greedy coreset, geometric diversity
- ``CombinedStrategy``         combined score by rank fusion
- ``BADGEStrategy``            BADGE: gradient embeddings plus k-means++

Referencia: Baseline_implicaciones_fases_siguientes.md §2.4
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.stats import rankdata


# ---------------------------------------------------------------------------
# Clase base
# ---------------------------------------------------------------------------

class QueryStrategy(ABC):
    """Common interface for every query strategy.

    Parameters
    ----------
    name : str
        Identifier used in logs and figures.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        """Select ``n_query`` pool indices to annotate.

        Parameters
        ----------
        model : nn.Module
            The trained classifier.
        pool_X : np.ndarray, shape (N, T, F)
            Unlabelled samples.
        n_query : int
            How many samples to query.
        device : torch.device | None
            Inference device.

        Returns
        -------
        np.ndarray, shape (n_query,)
            The selected indices into ``pool_X``.
        """

    # ------------------------------------------------------------------
    # Helpers internos compartidos
    # ------------------------------------------------------------------

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1].

        If all values are equal (flat signal), returns 0.5 everywhere
        instead of NaN so it acts as a neutral weight.
        """
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _get_probs(
        model: nn.Module,
        pool_X: np.ndarray,
        device: torch.device,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Inferencia softmax en batches. Devuelve (N, C) float32."""
        model.eval()
        all_probs: List[np.ndarray] = []
        n = len(pool_X)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                x = torch.tensor(
                    pool_X[start : start + batch_size], dtype=torch.float32
                ).to(device)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                logits = model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)  # (N, C)

    @staticmethod
    def _get_embeddings(
        model: nn.Module,
        pool_X: np.ndarray,
        device: torch.device,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Embeddings from the penultimate layer, before the linear head.

        Works with BehaviorLSTM and with any model exposing ``get_embedding(x)``.
        """
        model.eval()
        all_emb: List[np.ndarray] = []
        n = len(pool_X)

        # Use get_embedding when the model provides it
        if hasattr(model, "get_embedding"):
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    x = torch.tensor(
                        pool_X[start : start + batch_size], dtype=torch.float32
                    ).to(device)
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    emb = model.get_embedding(x).cpu().numpy()
                    all_emb.append(emb)
            return np.concatenate(all_emb, axis=0)

        # Fallback: use the softmax probabilities as the representation
        return QueryStrategy._get_probs(model, pool_X, device, batch_size)

    @staticmethod
    def _mc_dropout_probs(
        model: nn.Module,
        pool_X: np.ndarray,
        device: torch.device,
        n_mc: int = 30,
        batch_size: int = 256,
    ) -> np.ndarray:
        """MC dropout: ``n_mc`` forward passes with dropout left active.

        The model needs dropout layers for this to mean anything; without them every
        pass is identical and the disagreement term below is zero by construction.
        Returns an (N, n_mc, C) array of per-pass probabilities.
        """
        n = len(pool_X)
        # Turn dropout on even though this is inference
        model.train()

        # Keep BatchNorm from updating its running statistics
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()

        all_mc: List[np.ndarray] = []
        for _ in range(n_mc):
            pass_probs: List[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, n, batch_size):
                    x = torch.tensor(
                        pool_X[start : start + batch_size], dtype=torch.float32
                    ).to(device)
                    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    pass_probs.append(probs)
            all_mc.append(np.concatenate(pass_probs, axis=0))

        model.eval()
        return np.stack(all_mc, axis=1)  # (N, n_mc, C)


# ---------------------------------------------------------------------------
# Least Confidence
# ---------------------------------------------------------------------------

class LeastConfidenceStrategy(QueryStrategy):
    """Select the samples with the lowest top-1 confidence.

    .. math::
        \\text{score}(x) = 1 - \\max_c P(y=c \\mid x)
    """

    def __init__(self) -> None:
        super().__init__("least_confidence")

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        probs = self._get_probs(model, pool_X, device)  # (N, C)
        scores = 1.0 - probs.max(axis=1)               # (N,)
        return np.argsort(scores)[::-1][:n_query]


# ---------------------------------------------------------------------------
# Margin Sampling
# ---------------------------------------------------------------------------

class MarginSamplingStrategy(QueryStrategy):
    """Smallest margin between the most and second most likely class.

    .. math::
        \\text{score}(x) = P_{\\hat{y}_1}(x) - P_{\\hat{y}_2}(x)

    The samples queried are those with the smallest margin, that is, those the
    model finds most ambiguous between its top two classes.
    """

    def __init__(self) -> None:
        super().__init__("margin_sampling")

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        probs = self._get_probs(model, pool_X, device)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]  # (N,)
        return np.argsort(margins)[:n_query]  # menor margen → mayor incertidumbre


# ---------------------------------------------------------------------------
# Entropy Sampling
# ---------------------------------------------------------------------------

class EntropySamplingStrategy(QueryStrategy):
    """Highest Shannon entropy of the predictive distribution.

    .. math::
        \\text{score}(x) = -\\sum_c P(y=c \\mid x) \\log P(y=c \\mid x)
    """

    def __init__(self) -> None:
        super().__init__("entropy_sampling")

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        probs = self._get_probs(model, pool_X, device)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)  # (N,)
        return np.argsort(entropy)[::-1][:n_query]


# ---------------------------------------------------------------------------
# BALD (Bayesian Active Learning by Disagreement)
# ---------------------------------------------------------------------------

class BALDStrategy(QueryStrategy):
    """BALD: highest mutual information between predictions and parameters.

    Uses MC dropout to approximate the Bayesian predictive distribution.

    .. math::
        \\text{BALD}(x) = H[y|x] - \\mathbb{E}_{\\theta}[H[y|x,\\theta]]

    The first term is the entropy of the mean prediction, the second the mean
    entropy of the individual passes. Their difference isolates disagreement between
    dropout masks, which is uncertainty about the model rather than about the data.
    That distinction is the point of BALD: a sample can be genuinely ambiguous, and
    annotating it teaches the model nothing.

    Parameters
    ----------
    n_mc : int
        Number of MC dropout passes. 30 trades variance against cost reasonably.
    """

    def __init__(self, n_mc: int = 30) -> None:
        super().__init__("bald")
        self.n_mc = n_mc

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        # mc_probs: (N, n_mc, C)
        mc_probs = self._mc_dropout_probs(model, pool_X, device, self.n_mc)

        # Entropy of the mean prediction: H[E_θ p(y|x,θ)]
        mean_probs = mc_probs.mean(axis=1)  # (N, C)
        H_mean = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=1)

        # Mean of the individual entropies: E_θ[H[p(y|x,θ)]]
        H_each = -np.sum(mc_probs * np.log(mc_probs + 1e-12), axis=2)  # (N, n_mc)
        E_H = H_each.mean(axis=1)  # (N,)

        bald_scores = H_mean - E_H  # (N,)
        return np.argsort(bald_scores)[::-1][:n_query]


# ---------------------------------------------------------------------------
# Coreset (K-Center Greedy)
# ---------------------------------------------------------------------------

class CoresetStrategy(QueryStrategy):
    """Greedy k-centre coreset selection in embedding space.

    Maximises coverage of the feature space by picking samples that minimise the
    largest distance from any pool point to the already
    etiquetadas (K-Center / Farthest First Traversal).

    The embedding space comes from the BehaviorLSTM attention layer, or from the
    penultimate layer when attention is not available.

    Parameters
    ----------
    metric : str
        Distance metric passed to scipy.spatial.distance.cdist.
        ``"cosine"`` o ``"euclidean"``.
    """

    def __init__(self, metric: str = "cosine") -> None:
        super().__init__("coreset")
        self.metric = metric

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        labeled_X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        labeled_X : np.ndarray, shape (M, T, F), opcional
            Already-labelled samples. When given, the centre set is initialised
            from their embeddings rather than from scratch.
        """
        if device is None:
            device = next(model.parameters()).device

        pool_emb = self._get_embeddings(model, pool_X, device)  # (N, d)

        if labeled_X is not None and len(labeled_X) > 0:
            labeled_emb = self._get_embeddings(model, labeled_X, device)
            # Distance from each pool point to its nearest labelled point
            dist_to_labeled = cdist(pool_emb, labeled_emb, metric=self.metric)
            min_dist = dist_to_labeled.min(axis=1)  # (N,)
        else:
            # No reference points: start from the global centroid
            centroid = pool_emb.mean(axis=0, keepdims=True)
            min_dist = cdist(pool_emb, centroid, metric=self.metric).squeeze()

        selected: List[int] = []
        for _ in range(n_query):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            # Update the minimum distances
            new_center = pool_emb[idx : idx + 1]
            dist_to_new = cdist(pool_emb, new_center, metric=self.metric).squeeze()
            min_dist = np.minimum(min_dist, dist_to_new)
            min_dist[idx] = -np.inf  # exclude from the next round

        return np.array(selected)


# ---------------------------------------------------------------------------
# Estrategia combinada (§2.4 Baseline_implicaciones)
# ---------------------------------------------------------------------------

class CombinedStrategy(QueryStrategy):
    r"""Combined score, tailored to the CalMS21 class structure.

    .. math::

        \\text{score}_{AL}(w) =
        H(p(y|w))
        \\times r_{\\hat{y}}
        \\times (1 + \\delta_{\\text{sniff}}(w))

    where:

    - :math:`H(p(y|w))` softmax entropy, the model's own uncertainty
    - :math:`r_{\\hat{y}}` inverse-frequency weight of the predicted class
    - :math:`\\delta_{\\text{sniff}}` ambiguous-sniff indicator:
      :math:`\\mathbf{1}[|\\texttt{sniff\\_site\\_ratio}| < 0.3]`

    The score combines model uncertainty, class rarity, and a bonus for the
    sniffbody-sniffface ambiguity zone.

    Parameters
    ----------
    class_counts : array (C,)
        Labelled samples per class, for the inverse weight. With None every weight
        is 1, which reduces this to plain entropy sampling.
    sniff_site_ratio : array (N,), opcional
        sniff_site_ratio for the pool windows. With None the sniff bonus is off.
    sniff_threshold : float
        |sniff_site_ratio| threshold that activates the bonus.
    n_mc : int
        Above 0, estimate entropy with MC dropout instead of a single
        deterministic pass.
    """

    def __init__(
        self,
        class_counts: Optional[np.ndarray] = None,
        sniff_site_ratio: Optional[np.ndarray] = None,
        sniff_threshold: float = 0.3,
        n_mc: int = 0,
    ) -> None:
        super().__init__("combined")
        self.class_counts = class_counts
        self.sniff_site_ratio = sniff_site_ratio
        self.sniff_threshold = sniff_threshold
        self.n_mc = n_mc

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        class_counts: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        # class_counts can be updated each round
        current_counts = class_counts if class_counts is not None else self.class_counts

        # --- Entropy ---
        if self.n_mc > 0:
            mc_probs = self._mc_dropout_probs(model, pool_X, device, self.n_mc)
            probs = mc_probs.mean(axis=1)  # (N, C)
        else:
            probs = self._get_probs(model, pool_X, device)  # (N, C)

        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)  # (N,)

        # --- Class rarity: inverse frequency weight ---
        pred_classes = probs.argmax(axis=1)  # (N,)
        if current_counts is not None:
            # Avoid dividing by zero
            counts = np.maximum(current_counts, 1).astype(float)
            inv_freq = 1.0 / counts
            # Normalise so the mean weight is 1
            inv_freq = inv_freq / inv_freq.mean()
            r_class = inv_freq[pred_classes]  # (N,)
        else:
            r_class = np.ones(len(pool_X))

        # --- sniff-site bonus, only when the size matches the current pool ---
        N = len(pool_X)
        if self.sniff_site_ratio is not None and len(self.sniff_site_ratio) == N:
            sniff_arr = np.asarray(self.sniff_site_ratio)
            delta_sniff = (np.abs(sniff_arr) < self.sniff_threshold).astype(float)
        else:
            delta_sniff = np.zeros(N)

        # --- Rank fusion: each signal is normalised by its relative rank ---
        # Ranks are invariant to scale and distribution, which stops one component
        # dominating simply because its absolute magnitude drifts between rounds.
        rank_ent   = rankdata(-entropy)               / N  # high entropy means a high rank
        rank_rare  = rankdata(-r_class)               / N  # clase rara → rango alto
        rank_sniff = rankdata(-delta_sniff.astype(float)) / N  # sniff ambiguo → rango alto
        scores = 0.5 * rank_ent + 0.3 * rank_rare + 0.2 * rank_sniff
        return np.argsort(scores)[::-1][:n_query]


# ---------------------------------------------------------------------------
# Hybrid Coreset-Entropy
# ---------------------------------------------------------------------------

class HybridCoresetEntropyStrategy(QueryStrategy):
    r"""Combines geometric diversity (coreset) with uncertainty (entropy).

    Both signals are normalised to [0, 1] before combining, so that neither
    dominates the other through a difference of scale alone.

    .. math::

        \\text{score}(x) =
        \\alpha \\cdot H_{\\text{norm}}(x) +
        (1-\\alpha) \\cdot d_{\\text{norm}}(x)

    where :math:`d(x)` is the distance from ``x`` to the nearest already-labelled
    sample in embedding space, as in the coreset strategy.

    Parameters
    ----------
    alpha : float
        Weight on entropy. 0 is pure coreset, 1 is pure entropy, 0.5 balances them.
    metric : str
        Distance metric for ``cdist``.
    """

    def __init__(self, alpha: float = 0.5, metric: str = "cosine") -> None:
        super().__init__("hybrid_coreset_entropy")
        self.alpha = alpha
        self.metric = metric

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        labeled_X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        # --- Normalised entropy ---
        probs = self._get_probs(model, pool_X, device)           # (N, C)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1) # (N,)
        entropy_norm = self._minmax(entropy)

        # --- Normalised distance to the nearest labelled sample: diversity ---
        pool_emb = self._get_embeddings(model, pool_X, device)   # (N, d)
        if labeled_X is not None and len(labeled_X) > 0:
            labeled_emb = self._get_embeddings(model, labeled_X, device)
            dist_mat = cdist(pool_emb, labeled_emb, metric=self.metric)
            min_dist = dist_mat.min(axis=1)
        else:
            centroid = pool_emb.mean(axis=0, keepdims=True)
            min_dist = cdist(pool_emb, centroid, metric=self.metric).squeeze()
        dist_norm = self._minmax(min_dist)

        # --- Combined score ---
        scores = self.alpha * entropy_norm + (1.0 - self.alpha) * dist_norm
        return np.argsort(scores)[::-1][:n_query]


# ---------------------------------------------------------------------------
# Hybrid Schedule (BALD → Margin)
# ---------------------------------------------------------------------------

class HybridScheduleStrategy(QueryStrategy):
    """Scheduled two-phase strategy: BALD first, then margin sampling.

    - **Rounds 0 to switch_round-1**: BALD, Bayesian exploration. MC dropout
      maximises mutual information, which is what you want while the model still
      has high epistemic uncertainty and its decision boundary means little.
    - **Rounds switch_round onwards**: margin sampling, exploitation. Once the model
      is more confident, the margin locates the boundary more precisely than BALD.

    The switch is automatic: the strategy counts rounds itself, incrementing on each
    call to ``query()``.

    Parameters
    ----------
    switch_round : int
        First round that uses margin sampling. Default 5, so rounds 0 to 4 use BALD
        and rounds 5 onwards use margin.
    n_mc : int
        MC dropout passes for BALD.
    """

    def __init__(self, switch_round: int = 5, n_mc: int = 30) -> None:
        super().__init__("hybrid_schedule")
        self.switch_round = switch_round
        self._bald = BALDStrategy(n_mc=n_mc)
        self._margin = MarginSamplingStrategy()
        self._round: int = 0

    def reset(self) -> None:
        """Reset the round counter, for starting a fresh active learning cycle."""
        self._round = 0

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if self._round < self.switch_round:
            active = self._bald
            phase = "BALD"
        else:
            active = self._margin
            phase = "Margin"

        result = active.query(model, pool_X, n_query, device=device, **kwargs)
        self._round += 1
        return result

    @property
    def current_phase(self) -> str:
        return "BALD" if self._round < self.switch_round else "Margin"


# ---------------------------------------------------------------------------
# BADGE (Batch Active learning by Diverse Gradient Embeddings)
# ---------------------------------------------------------------------------

class BADGEStrategy(QueryStrategy):
    r"""BADGE: Batch Active learning by Diverse Gradient Embeddings.

    Combines uncertainty and diversity through gradient embeddings. For each pool
    sample the gradient of the cross entropy with respect to the last linear layer's
    weights is computed in closed form, then k-means++ initialisation picks samples
    that are both diverse and uncertain.

    The gradient magnitude carries the uncertainty, and the direction carries what
    the sample would change, so one embedding captures both without weighting two
    separate scores against each other.

    .. math::

        g_x = (\\text{softmax}(f(x)) - \\mathbb{1}_{\\hat{y}}) \\otimes h_x

    where :math:`h_x` is the penultimate embedding and :math:`\\hat{y}` the
    predicted class, used as a pseudo-label.

    Reference: Ash et al. (2020), "Deep Batch Active Learning by Diverse,
    Uncertain Gradient Lower Bounds"
    """

    def __init__(self) -> None:
        super().__init__("badge")

    def query(
        self,
        model: nn.Module,
        pool_X: np.ndarray,
        n_query: int,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> np.ndarray:
        if device is None:
            device = next(model.parameters()).device

        # 1. Embeddings h_x y probabilidades p_x
        embeddings = self._get_embeddings(model, pool_X, device)  # (N, d)
        probs      = self._get_probs(model, pool_X, device)        # (N, C)

        N, d = embeddings.shape
        C    = probs.shape[1]

        # 2. Pseudo-label: the argmax of the probabilities
        pseudo_labels = probs.argmax(axis=1)  # (N,)

        # 3. Gradient embedding, in closed form:
        #    g_x = (p_x - one_hot(pseudo_label)) ⊗ h_x  → shape (C*d,)
        one_hot = np.zeros_like(probs)  # (N, C)
        one_hot[np.arange(N), pseudo_labels] = 1.0
        prob_minus_onehot = probs - one_hot  # (N, C), the gradient wrt the bias

        # Producto exterior: (N, C, 1) * (N, 1, d) → (N, C, d) → (N, C*d)
        grad_emb = (
            prob_minus_onehot[:, :, None] * embeddings[:, None, :]
        ).reshape(N, C * d)

        # 4. k-means++ to pick n_query diverse centres
        selected = self._kmeans_pp_init(grad_emb, n_query)
        return np.array(selected)

    @staticmethod
    def _kmeans_pp_init(X: np.ndarray, k: int) -> List[int]:
        """k-means++ initialisation over the gradient embeddings.

        Picks k diverse samples, choosing each with probability proportional to its
        squared distance from the nearest centre already selected.

        Parameters
        ----------
        X : ndarray, shape (N, d)
            Normalised gradient embeddings.
        k : int
            Number of centres to select.

        Returns
        -------
        A list of k indices into X.
        """
        rng = np.random.default_rng(0)
        N = len(X)
        k = min(k, N)

        # Primer centro: aleatorio
        first = int(rng.integers(0, N))
        centers: List[int] = [first]

        # Squared distances to the nearest centre
        sq_dist = np.sum((X - X[first]) ** 2, axis=1)  # (N,)

        for _ in range(k - 1):
            total = sq_dist.sum()
            if total < 1e-12:
                # Degenerate case: every point is identical
                remaining = np.setdiff1d(np.arange(N), centers)
                if len(remaining) == 0:
                    break
                centers.append(int(remaining[0]))
                break

            # Sample with probability proportional to the squared distance
            prob = sq_dist / total
            idx = int(rng.choice(N, p=prob))
            centers.append(idx)

            # Update the minimum distances
            new_sq = np.sum((X - X[idx]) ** 2, axis=1)
            sq_dist = np.minimum(sq_dist, new_sq)
            sq_dist[idx] = 0.0  # no reseleccionar

        return centers[:k]


# ---------------------------------------------------------------------------
# Helpers: the active learning curve
# ---------------------------------------------------------------------------

def run_al_cycle(
    strategy: QueryStrategy,
    model_factory,
    train_X: np.ndarray,
    train_y: np.ndarray,
    pool_X: np.ndarray,
    pool_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    *,
    n_initial: int = 20,
    n_query_per_round: int = 10,
    n_rounds: int = 10,
    device: Optional[torch.device] = None,
    fit_fn=None,
    eval_fn=None,
    strategy_kwargs: Optional[dict] = None,
    random_floor: float = 0.1,
    random_state: int = 42,
    add_train_remainder_to_pool: bool = False,
    early_stopping: bool = False,
    min_delta: float = 0.005,
    es_patience: int = 3,
    verbose: bool = True,
) -> dict:
    """Run the active learning loop and return its learning curve.

    Parameters
    ----------
    strategy : QueryStrategy
        The acquisition strategy.
    model_factory : callable
        Takes no arguments and returns a freshly initialised model.
    train_X, train_y : ndarray
        Initial pool of labelled samples. The loop starts from ``n_initial`` of
        these when the seed set is larger, and uses all of them otherwise.
    pool_X, pool_y : ndarray
        Unlabelled pool. The labels exist but are only revealed when the oracle is
        queried, which is what makes the simulation honest.
    val_X, val_y : ndarray
        Validation set, fixed for the whole cycle.
    n_initial : int
        Size of the initial labelled set.
    n_query_per_round : int
        Samples to query each round.
    n_rounds : int
        Number of rounds.
    device : torch.device | None
    fit_fn : callable(model, X_tr, y_tr, X_val, y_val, device) -> model
        Training function. Defaults to plain cross-entropy training.
    eval_fn : callable(model, X, y, device) -> float
        Evaluation function returning macro-F1.
    strategy_kwargs : dict, optional
        Extra keyword arguments for ``strategy.query()``.
    random_floor : float
        Minimum fraction of random queries per round, in [0, 1). At 0.1, one query
        in ten is random. This stops the strategy collapsing onto a high-uncertainty
        region that is not representative of the pool, which is a failure mode of
        pure uncertainty sampling rather than a hypothetical.
    random_state : int
        Seed for the internal random number generator, so runs are reproducible and
        independent repeats can be seeded apart.
    add_train_remainder_to_pool : bool
        When True, the train_X samples **not** chosen as seed are prepended to
        pool_X before the cycle starts. Useful when train_X holds the full labelled
        source and only n_initial of it should be the starting point.
    early_stopping : bool
        Stop when the gain in validation F1 stays below ``min_delta`` for
        ``es_patience`` consecutive rounds.
    min_delta : float
        Minimum validation F1 gain per round to count as improvement.
    es_patience : int
        Consecutive rounds without improvement before stopping.
    verbose : bool

    Returns
    -------
    A dict of lists:
        ``"n_labeled"``      cumulative useful labels per round
        ``"val_f1"``         validation macro-F1 per round
        ``"train_f1"``       — F1-macro en entrenamiento
        ``"queried_idx"``    pool indices queried each round
        ``"queried_labels"`` labels revealed by those queries
        ``"wasted_queries"`` queries that hit background samples (label -1)
    """
    from sklearn.metrics import f1_score
    import copy as _copy

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if strategy_kwargs is None:
        strategy_kwargs = {}

    # ---- Initial seed set, stratified by class ----
    rng = np.random.default_rng(random_state)
    n_seed = min(n_initial, len(train_X))
    classes = np.unique(train_y)
    per_class = max(1, n_seed // len(classes))
    seed_idx_list: List[int] = []
    for c in classes:
        c_idx = np.where(train_y == c)[0]
        chosen = rng.choice(c_idx, size=min(per_class, len(c_idx)), replace=False)
        seed_idx_list.extend(chosen.tolist())
    # Top up to n_seed with random samples from the remainder
    remaining_budget = n_seed - len(seed_idx_list)
    if remaining_budget > 0:
        chosen_set = set(seed_idx_list)
        available = [i for i in range(len(train_X)) if i not in chosen_set]
        if available:
            extra = rng.choice(available, size=min(remaining_budget, len(available)), replace=False)
            seed_idx_list.extend(extra.tolist())
    seed_idx = np.array(seed_idx_list[:n_seed])
    labeled_X = train_X[seed_idx].copy()
    labeled_y = train_y[seed_idx].copy()

    # Candidate pool, mutated as samples are queried
    # With add_train_remainder_to_pool, the train_X samples not chosen as seed are
    # prepended to the query pool.
    if add_train_remainder_to_pool:
        not_seed_idx = np.setdiff1d(np.arange(len(train_X)), seed_idx)
        extra_X = train_X[not_seed_idx]
        extra_y = train_y[not_seed_idx]
        remaining_X = np.concatenate([extra_X, pool_X.copy()], axis=0)
        remaining_y = np.concatenate([extra_y, pool_y.copy()], axis=0)
        remaining_idx = np.arange(len(remaining_X))
    else:
        remaining_X = pool_X.copy()
        remaining_y = pool_y.copy()
        remaining_idx = np.arange(len(pool_X))

    history = {
        "n_labeled":      [],
        "val_f1":         [],
        "train_f1":       [],
        "queried_idx":    [],
        "queried_labels": [],
        "wasted_queries": [],  # queries that hit background samples (label -1)
    }

    for rnd in range(n_rounds + 1):  # round 0 is the initial baseline
        # ---- Entrenar ----
        model = model_factory()
        if fit_fn is not None:
            model = fit_fn(model, labeled_X, labeled_y, val_X, val_y, device)
        else:
            model = _default_fit(model, labeled_X, labeled_y, device)

        # ---- Evaluar val y train ----
        if eval_fn is not None:
            val_f1   = eval_fn(model, val_X,     val_y,     device)
            train_f1 = eval_fn(model, labeled_X, labeled_y, device)
        else:
            val_f1   = _default_eval(model, val_X,     val_y,     device)
            train_f1 = _default_eval(model, labeled_X, labeled_y, device)

        history["n_labeled"].append(len(labeled_X))
        history["val_f1"].append(val_f1)
        history["train_f1"].append(train_f1)

        if verbose:
            print(
                f"[AL {strategy.name}] round {rnd:2d} | "
                f"etiquetadas={len(labeled_X):3d} | "
                f"train_f1={train_f1:.4f} | "
                f"val_f1={val_f1:.4f}"
            )

        if rnd == n_rounds or len(remaining_X) == 0:
            break

        # ---- Early stopping on a validation F1 plateau ----
        if early_stopping and len(history["val_f1"]) >= es_patience + 1:
            recent_gains = [
                history["val_f1"][-i] - history["val_f1"][-i - 1]
                for i in range(1, es_patience + 1)
            ]
            if all(abs(g) < min_delta for g in recent_gains):
                if verbose:
                    print(
                        f"[AL {strategy.name}] early stop at round {rnd} "
                        f"(no gain above {min_delta} in {es_patience} rounds)"
                    )
                break

        # ---- Query, with the random floor that prevents collapse ----
        n_q = min(n_query_per_round, len(remaining_X))
        n_random = max(0, min(int(n_q * random_floor), n_q - 1))  # leave at least 1 for the strategy
        n_strategy = n_q - n_random

        # Refresh class_counts for the strategies that use it
        dynamic_kwargs = dict(strategy_kwargs)
        dynamic_kwargs["labeled_X"] = labeled_X
        if hasattr(strategy, "class_counts") and strategy.class_counts is not None:
            uniq_l, cnts_l = np.unique(labeled_y, return_counts=True)
            dyn_counts = np.zeros_like(strategy.class_counts)
            for c, cnt in zip(uniq_l, cnts_l):
                if 0 <= c < len(dyn_counts):
                    dyn_counts[c] = cnt
            dynamic_kwargs["class_counts"] = dyn_counts

        strategy_queried = strategy.query(
            model,
            remaining_X,
            n_strategy,
            device=device,
            **dynamic_kwargs,
        )
        strategy_queried = np.asarray(strategy_queried)

        # Add the random-floor queries
        if n_random > 0 and len(remaining_X) > len(strategy_queried):
            not_strategy = np.setdiff1d(
                np.arange(len(remaining_X)), strategy_queried
            )
            random_queried = rng.choice(
                not_strategy,
                size=min(n_random, len(not_strategy)),
                replace=False,
            )
            queried = np.concatenate([strategy_queried, random_queried]).astype(int)
        else:
            queried = strategy_queried.astype(int)
        history["queried_idx"].append(remaining_idx[queried].tolist())

        # ---- Split useful queries (label >= 0) from background (label -1) ----
        queried_labels = remaining_y[queried]
        history["queried_labels"].append(queried_labels.tolist())
        useful_mask = queried_labels >= 0           # annotatable samples
        n_wasted    = int((~useful_mask).sum())     # queries spent on background
        history["wasted_queries"].append(n_wasted)

        # Only samples with a valid label join the labelled set
        useful_queried = queried[useful_mask]
        if len(useful_queried) > 0:
            labeled_X = np.concatenate([labeled_X, remaining_X[useful_queried]], axis=0)
            labeled_y = np.concatenate([labeled_y, remaining_y[useful_queried]], axis=0)

        # Remove EVERY queried sample from the pool, useful or background alike:
        # a wasted query still costs the annotator, so it cannot be queried twice
        remaining_X   = np.delete(remaining_X,   queried, axis=0)
        remaining_y   = np.delete(remaining_y,   queried, axis=0)
        remaining_idx = np.delete(remaining_idx, queried, axis=0)

    return history


# ---------------------------------------------------------------------------
# Default training and evaluation functions used by run_al_cycle
# ---------------------------------------------------------------------------

def _default_fit(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Fast training without validation, for the active learning loop."""
    import torch.utils.data as tud

    model = model.to(device)
    dataset = tud.TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            xb = torch.nan_to_num(xb, nan=0.0)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def _default_eval(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Fast macro-F1 for the active learning loop."""
    from sklearn.metrics import f1_score

    model.eval()
    preds: List[int] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = torch.tensor(X[start : start + batch_size], dtype=torch.float32).to(device)
            xb = torch.nan_to_num(xb, nan=0.0)
            logits = model(xb)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return float(f1_score(y, np.array(preds), average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Random baseline + learning-curve metric (B4 — used by 06_active_learning)
# ---------------------------------------------------------------------------

class RandomStrategy(QueryStrategy):
    """Uniform-random acquisition — the baseline AL must beat to be worth it."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__("random")
        self.rng = np.random.default_rng(seed)

    def query(self, model, pool_X, n_query, device=None, **kwargs):
        n = min(n_query, len(pool_X))
        return self.rng.choice(len(pool_X), size=n, replace=False)


def aulc(n_labeled, f1) -> float:
    """Normalized area under the learning curve (trapezoid over #labels)."""
    n_labeled = np.asarray(n_labeled, float)
    f1 = np.asarray(f1, float)
    span = n_labeled[-1] - n_labeled[0]
    if span <= 0:
        return float(np.mean(f1))
    return float(np.trapezoid(f1, n_labeled) / span)
