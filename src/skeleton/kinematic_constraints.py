"""
src/skeleton/kinematic_constraints.py
=======================================
Cost function for the kinematic correction of keypoints.

The objective balances five things that pull against each other. Staying close to
what was observed, keeping bone lengths near their reference, and keeping the pose
continuous in time, acceleration and jerk. Every failure mode of this module is a
case of one term overpowering the others:

    J(k̂) = Σ_{i}     w_i · ‖k̂_i - k_i‖²              (adhesion to observations)
          + λ_L · Σ_{(i,j)∈E}  λ_e · (L_ij/L̄_ij - 1)² (per-edge length)
          + λ_T · Σ_i ‖k̂_i - k_{t-1}‖²                 (suavizado temporal)
          + λ_A · Σ_i ‖k̂_i - 2k_{t-1} + k_{t-2}‖²      (acceleration)
          + λ_θ · Σ_{(A,B,C)} (cross(A-B, C-B) / L_body²)² (orientation)

where:
  - w_i       per-keypoint adhesion weight; 1 is trusted, low means a severe outlier
  - λ_L, λ_T  base weight of the length and temporal-smoothing terms
  - λ_A       acceleration regulariser, which suppresses jitter
  - λ_θ       body-collinearity angular constraint
  - λ_e       per-edge weight, by inverse variance when auto_edge_lambdas is on

Minimised with scipy.optimize.minimize (L-BFGS-B) using an analytic gradient. The
gradient is analytic rather than numerical because the finite-difference version
made a per-frame optimisation over a whole video too slow to run at all.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from src.skeleton.mouse_skeleton import MouseSkeleton, SkeletonEdge


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_LAMBDA_L     = 6.5    # base weight of the length constraint (v4: 8.0 to 6.5)


# ---------------------------------------------------------------------------
# Vectorised cache of edges and triplets, built once per video
# ---------------------------------------------------------------------------

class EdgeCache(NamedTuple):
    """Pre-computed index arrays for vectorized edge-length computation.

    Build once per video with ``build_edge_cache()``, then pass to
    ``kinematic_cost_grad`` and ``correct_frame`` to avoid re-running
    ``kp_index.get()`` and list comprehensions on every L-BFGS-B iteration.
    """
    src_idx: np.ndarray   # shape (n_valid_edges,) int
    dst_idx: np.ndarray   # shape (n_valid_edges,) int
    L_refs:  np.ndarray   # shape (n_valid_edges,) float  (normalised, not × l_body)
    e_idx:   np.ndarray   # shape (n_valid_edges,) int – indices into edge_lambdas
    trip_arr: Optional[np.ndarray]  # shape (n_triplets, 3) int or None


def build_edge_cache(
    edges: List[SkeletonEdge],
    kp_index: Dict[str, int],
    angle_triplets: Optional[List[Tuple[int, int, int]]],
) -> EdgeCache:
    """Build an ``EdgeCache`` from skeleton edges and keypoint index.

    Only edges whose both endpoints exist in ``kp_index`` and have L_ref > 0
    are included.  The returned arrays are ready for vectorized computation.
    """
    rows = [
        (e_idx, kp_index[e.src], kp_index[e.dst], e.L_ref)
        for e_idx, e in enumerate(edges)
        if (kp_index.get(e.src) is not None
            and kp_index.get(e.dst) is not None
            and e.L_ref > 0)
    ]
    if rows:
        e_idxs, srcs, dsts, lrefs = map(np.array, zip(*rows))
        e_idxs = e_idxs.astype(np.intp)
        srcs   = srcs.astype(np.intp)
        dsts   = dsts.astype(np.intp)
        lrefs  = lrefs.astype(float)
    else:
        e_idxs = srcs = dsts = lrefs = np.empty(0, dtype=float)

    trip_arr = (np.array(angle_triplets, dtype=np.intp)
                if angle_triplets else None)

    return EdgeCache(src_idx=srcs, dst_idx=dsts,
                     L_refs=lrefs, e_idx=e_idxs, trip_arr=trip_arr)

DEFAULT_LAMBDA_T     = 1.8    # temporal smoothing (v4: 5.0 to 1.8; fidelity vs smoothing)
DEFAULT_LAMBDA_ACC   = 0.6    # acceleration (v4: 0.2 to 0.6; cuts speed spikes)
DEFAULT_LAMBDA_ANGLE = 4.0    # angular collinearity (v4: 10.0 to 4.0; avoids rigid poses)
DEFAULT_LAMBDA_JERK  = 0.35   # jerk (v4: 0.05 to 0.35; removes temporal jitter)

# Huber delta for the length constraint, in normalised units, that is, as a fraction
# of L_ref. Below the threshold the penalty is quadratic, above it linear, so a
# single badly detected frame cannot dominate the whole optimisation.
# 0.30 means switching to L1 once the violation exceeds 30% of the reference length.
_HUBER_DELTA_L = 0.30


# ---------------------------------------------------------------------------
# Per-edge weights, auto-tuned by inverse variance
# ---------------------------------------------------------------------------

def compute_edge_lambdas(
    skeleton: MouseSkeleton,
    base_lambda_l: float = DEFAULT_LAMBDA_L,
    min_factor: float = 0.1,
    length_weighted: bool = True,
) -> np.ndarray:
    """Per-edge λ_L, weighted by inverse variance and by length.

    Stiffer edges, meaning smaller L_std, are penalised harder. A uniform lambda
    would instead let the optimiser move the neck to satisfy the ear edges and damage
    the flexible nose-to-neck edge in the process, which is what the v1 baseline did.

    With ``length_weighted=True``, the default, each weight is further scaled
    inversely to the edge's reference length, so short peripheral segments such as
    neck to ear carry more relative weight than the long trunk edges.

    Formula::

        λ_e = base_lambda_l * (σ_min / max(σ_e, ε))²
              [× (L_ref_median / L_ref_e)  si length_weighted]

    where σ_e is ``edge.L_std``, the standard deviation of the normalised length.

    Parameters
    ----------
    skeleton : MouseSkeleton
        Fitted skeleton, with L_std estimated by ``fit_skeleton``.
    base_lambda_l : float
        Weight given to the stiffest edge, that is, the maximum lambda.
    min_factor : float
        Floor that keeps λ_e above zero on very flexible edges:
        λ_e >= base_lambda_l * min_factor.
    length_weighted : bool
        Multiply each λ_e by (L_ref_median / L_ref_e) to strengthen the constraint on
        short edges. The upper clip rises to three times base_lambda_l.

    Returns
    -------
    np.ndarray, shape (n_edges,)
        One λ_L per edge, in the same order as ``skeleton.edges``.
    """
    stds = np.array([max(e.L_std, 1e-6) for e in skeleton.edges])
    sigma_min = stds.min()
    lambdas = base_lambda_l * (sigma_min / stds) ** 2
    if length_weighted:
        L_refs = np.array([max(e.L_ref, 1e-6) for e in skeleton.edges])
        L_ref_median = float(np.median(L_refs))
        length_scale = L_ref_median / L_refs  # short edges → scale > 1
        lambdas = lambdas * length_scale
    max_clip = base_lambda_l * (3.0 if length_weighted else 1.0)
    lambdas = np.clip(lambdas, base_lambda_l * min_factor, max_clip)
    return lambdas


# ---------------------------------------------------------------------------
# Cost function and gradient
# ---------------------------------------------------------------------------

def kinematic_cost_grad(
    x_flat: np.ndarray,
    observed: np.ndarray,
    prev_frame: Optional[np.ndarray],
    edges: List[SkeletonEdge],
    kp_index: Dict[str, int],
    l_body_px: float,
    visible_mask: np.ndarray,
    edge_lambdas: np.ndarray,   # shape (n_edges,)
    lambda_t: float = DEFAULT_LAMBDA_T,
    outlier_adhesion: float = 0.6,
    adhesion_weights: Optional[np.ndarray] = None,
    prev_prev_frame: Optional[np.ndarray] = None,
    lambda_acc: float = DEFAULT_LAMBDA_ACC,
    angle_triplets: Optional[List[Tuple[int, int, int]]] = None,
    lambda_angle: float = DEFAULT_LAMBDA_ANGLE,
    prev_prev_prev_frame: Optional[np.ndarray] = None,
    lambda_jerk: float = DEFAULT_LAMBDA_JERK,
    prev_angle_cross: Optional[float] = None,
    lambda_angle_cont: float = 0.0,
    edge_cache: Optional["EdgeCache"] = None,
) -> Tuple[float, np.ndarray]:
    """Cost and analytic gradient for L-BFGS-B, with a per-edge λ_L.

    Parameters
    ----------
    x_flat : np.ndarray, shape (2 * n_kp,)
        Optimisation variables, laid out as [x0,y0, x1,y1, ...].
    observed : np.ndarray, shape (n_kp, 2)
        Observed coordinates. NaN marks an absent keypoint.
    prev_frame : np.ndarray or None, shape (n_kp, 2)
        The previous frame, for temporal smoothing.
    edges : list[SkeletonEdge]
        Skeleton edges.
    kp_index : dict[str, int]
        Mapping from keypoint name to its index in n_kp.
    l_body_px : float
        Reference body length, in pixels.
    visible_mask : np.ndarray, shape (n_kp,), bool
        True = keypoint no-outlier (anclaje fuerte).
    edge_lambdas : np.ndarray, shape (n_edges,)
        One λ_L per edge, in the same order as ``edges``.
    lambda_t : float
        Weight of the temporal smoothing term.
    outlier_adhesion : float
        Fallback adhesion for outlier keypoints. Only used when
        ``adhesion_weights`` es None.
    adhesion_weights : np.ndarray or None, shape (n_kp,)
        Per-keypoint adhesion weights in [0,1]. When given, this replaces the
        binary visible/outlier scheme: each joint gets its own anchoring strength
        derived from its severity, so a marginal outlier is nudged rather than
        released entirely.
    edge_cache : EdgeCache or None
        Pre-computed index arrays from ``build_edge_cache()``.  If provided,
        the edge-length and angle terms are computed with vectorized numpy
        instead of Python loops (avoids re-running kp_index lookups on every
        L-BFGS-B iteration call).

    Returns
    -------
    (cost, gradient), ready for scipy.optimize.minimize with jac=True.
    """
    n_kp = len(kp_index)
    x = x_flat.reshape(n_kp, 2)
    grad = np.zeros_like(x)

    cost = 0.0

    # Normalisation: every motion term (adhesion, temporal, acceleration, jerk) is
    # divided by l_body_px squared to make it dimensionless and comparable with the
    # geometric terms (length, angle), which are already normalised internally.
    # Without this the relative lambdas would mean something different in every lab.
    l2_norm = max(l_body_px, 1.0) ** 2

    # Term 1: weighted adhesion to the observations
    #   With adhesion_weights supplied, each joint gets its own continuous weight
    #   Si no: binario — visibles=1.0, outliers=outlier_adhesion
    valid_obs = ~np.any(np.isnan(observed), axis=1)
    if adhesion_weights is not None:
        weights = adhesion_weights.astype(float).copy()
    else:
        weights = np.where(visible_mask, 1.0, outlier_adhesion).astype(float)
    weights[~valid_obs] = 0.0

    diff_obs = np.zeros_like(x)
    diff_obs[valid_obs] = x[valid_obs] - observed[valid_obs]
    # Huber robust adhesion: L2 for small deviations, L1 beyond delta_px.
    # This resists corrupted/near-zero observations pulling the solution too far.
    _delta_px = 0.5 * max(l_body_px, 1.0)  # switch at half a body length
    _diff_norms = np.sqrt(np.sum(diff_obs ** 2, axis=1))  # (n_kp,)
    _l2_reg = _diff_norms <= _delta_px
    _huber = np.where(_l2_reg,
                      _diff_norms ** 2,
                      2.0 * _delta_px * _diff_norms - _delta_px ** 2) / l2_norm
    cost += float(np.dot(weights, _huber))
    _g_scale = np.where(_l2_reg,
                        2.0 / l2_norm,
                        2.0 * _delta_px / (np.maximum(_diff_norms, 1e-8) * l2_norm))
    grad += weights[:, None] * _g_scale[:, None] * diff_obs

    # Term 2: per-edge length constraint, each with its own lambda
    # ── Vectorized path (fast, uses pre-built index arrays) ──────────────────
    if edge_cache is not None and edge_cache.src_idx.size > 0:
        L_ref_px_arr = edge_cache.L_refs * l_body_px        # (n_valid,)
        lam_arr      = edge_lambdas[edge_cache.e_idx]        # (n_valid,)
        deltas = x[edge_cache.src_idx] - x[edge_cache.dst_idx]  # (n_valid, 2)
        dists  = np.linalg.norm(deltas, axis=1)                  # (n_valid,)
        nonzero = dists > 1e-8
        violations = np.where(nonzero, (dists - L_ref_px_arr) / L_ref_px_arr, 0.0)
        # Huber loss for length: L2 for |v| ≤ δ, L1 beyond.  Reduces sensitivity
        # to extreme violations that would otherwise dominate the gradient and
        # cause unstable corrections / temporal explosions.
        _abs_viol = np.abs(violations)
        _l2_mask  = _abs_viol <= _HUBER_DELTA_L
        _huber_l  = np.where(_l2_mask,
                             violations ** 2,
                             2.0 * _HUBER_DELTA_L * _abs_viol - _HUBER_DELTA_L ** 2)
        cost += float(np.dot(lam_arr, _huber_l))
        g_denom   = np.where(nonzero, L_ref_px_arr * dists, 1.0)
        _dL_dv    = np.where(_l2_mask,
                             2.0 * violations,
                             2.0 * _HUBER_DELTA_L * np.sign(violations))
        g_coeffs  = np.where(nonzero, lam_arr * _dL_dv / g_denom, 0.0)
        np.add.at(grad, edge_cache.src_idx, g_coeffs[:, None] * deltas)
        np.add.at(grad, edge_cache.dst_idx, -g_coeffs[:, None] * deltas)
    else:
        # ── Fallback Python loop (when no cache is available) ─────────────────
        for e_idx, edge in enumerate(edges):
            i = kp_index.get(edge.src)
            j = kp_index.get(edge.dst)
            if i is None or j is None:
                continue
            L_ref_px = edge.L_ref * l_body_px
            if L_ref_px <= 0:
                continue
            lam = float(edge_lambdas[e_idx])
            delta = x[i] - x[j]
            d = np.linalg.norm(delta)
            if d < 1e-8:
                continue
            length_violation = (d - L_ref_px) / L_ref_px
            # Huber loss (same formulation as vectorized path)
            abs_v = abs(length_violation)
            if abs_v <= _HUBER_DELTA_L:
                cost += lam * length_violation ** 2
                dL_dv = 2.0 * length_violation
            else:
                cost += lam * (2.0 * _HUBER_DELTA_L * abs_v - _HUBER_DELTA_L ** 2)
                dL_dv = 2.0 * _HUBER_DELTA_L * (1.0 if length_violation > 0 else -1.0)
            g_coeff = lam * dL_dv / (L_ref_px * d)
            grad[i] += g_coeff * delta
            grad[j] -= g_coeff * delta

    # Term 3: first-order temporal smoothing, pseudo-Huber for robustness.
    # E(d) = sqrt(||d||^2 + eps) keeps the gradient bounded for large displacements,
    # where a plain quadratic would explode.
    if prev_frame is not None:
        valid_prev = ~np.any(np.isnan(prev_frame), axis=1)
        diff_temp = np.zeros_like(x)
        diff_temp[valid_prev] = x[valid_prev] - prev_frame[valid_prev]
        _eps_t = 0.05
        _dt_norms = np.sqrt(np.sum(diff_temp[valid_prev] ** 2, axis=1) + _eps_t)  # (n_valid,)
        cost += lambda_t * np.sum(_dt_norms) / l2_norm
        grad[valid_prev] += lambda_t / l2_norm * diff_temp[valid_prev] / _dt_norms[:, None]

    # Term 4: second-order acceleration regulariser, pseudo-Huber
    # Penaliza Δ²x = x[t] - 2·x[t-1] + x[t-2], suprimiendo jitter
    if lambda_acc > 0.0 and prev_frame is not None and prev_prev_frame is not None:
        valid_prev = ~np.any(np.isnan(prev_frame), axis=1)
        valid_pp   = ~np.any(np.isnan(prev_prev_frame), axis=1)
        valid_acc  = valid_prev & valid_pp
        if valid_acc.any():
            accel = np.zeros_like(x)
            accel[valid_acc] = (x[valid_acc]
                                - 2.0 * prev_frame[valid_acc]
                                + prev_prev_frame[valid_acc])
            _eps_a = 0.02
            _acc_norms = np.sqrt(np.sum(accel[valid_acc] ** 2, axis=1) + _eps_a)  # (n_valid,)
            cost += lambda_acc * np.sum(_acc_norms) / l2_norm
            grad[valid_acc] += lambda_acc / l2_norm * accel[valid_acc] / _acc_norms[:, None]

    # Term 5: body-orientation angular constraints.
    # Penalises departures from collinearity in each (A, B, C) triplet:
    #   cross(A-B, C-B) = 0 ⟺ A, B, C colineales
    # Normalised by l_body_px squared for scale invariance.
    # Priority: explicit angle_triplets param > edge_cache.trip_arr.
    # If angle_triplets is None, the term is skipped (regardless of cache).
    if angle_triplets is not None:
        _trip = (edge_cache.trip_arr
                 if edge_cache is not None and edge_cache.trip_arr is not None
                 else np.array(angle_triplets, dtype=np.intp))
    else:
        _trip = None

    if lambda_angle > 0.0 and _trip is not None and len(_trip) > 0:
        l2 = max(l_body_px, 1.0) ** 2
        # ── Vectorized triplet computation ───────────────────────────────────
        ia, ib, ic = _trip[:, 0], _trip[:, 1], _trip[:, 2]
        u = x[ia] - x[ib]                          # (n_trips, 2)
        v = x[ic] - x[ib]                          # (n_trips, 2)
        cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]   # (n_trips,)
        sc    = cross / l2                          # scaled cross
        cost += lambda_angle * float(np.dot(sc, sc))
        coeff = (2.0 * lambda_angle / l2) * sc      # (n_trips,)
        # Analytic gradients: same formulas as before, applied in batch
        np.add.at(grad, (ia, 0),  coeff *  v[:, 1])
        np.add.at(grad, (ia, 1),  coeff * -v[:, 0])
        np.add.at(grad, (ib, 0),  coeff * (-v[:, 1] + u[:, 1]))
        np.add.at(grad, (ib, 1),  coeff * (-u[:, 0] + v[:, 0]))
        np.add.at(grad, (ic, 0),  coeff * -u[:, 1])
        np.add.at(grad, (ic, 1),  coeff *  u[:, 0])

    # Term 6: jerk, the third-order difference, pseudo-Huber. Suppresses the
    # residual micro-jitter.
    # jerk_t = x[t] - 3·x[t-1] + 3·x[t-2] - x[t-3]
    if (lambda_jerk > 0.0 and prev_frame is not None
            and prev_prev_frame is not None
            and prev_prev_prev_frame is not None):
        valid_prev  = ~np.any(np.isnan(prev_frame),       axis=1)
        valid_pp    = ~np.any(np.isnan(prev_prev_frame),  axis=1)
        valid_ppp   = ~np.any(np.isnan(prev_prev_prev_frame), axis=1)
        valid_jerk  = valid_prev & valid_pp & valid_ppp
        if valid_jerk.any():
            jerk = np.zeros_like(x)
            jerk[valid_jerk] = (
                x[valid_jerk]
                - 3.0 * prev_frame[valid_jerk]
                + 3.0 * prev_prev_frame[valid_jerk]
                - prev_prev_prev_frame[valid_jerk]
            )
            _eps_j = 0.01
            _jerk_norms = np.sqrt(np.sum(jerk[valid_jerk] ** 2, axis=1) + _eps_j)  # (n_valid,)
            cost += lambda_jerk * np.sum(_jerk_norms) / l2_norm
            grad[valid_jerk] += lambda_jerk / l2_norm * jerk[valid_jerk] / _jerk_norms[:, None]

    # Term 7: angular continuity between frames.
    # Penalises change in the body-axis orientation (nose, neck, tail_base)
    # against the previous frame. This is what catches identity swaps, which show
    # up as a sudden reversal rather than a gradual turn.
    if lambda_angle_cont > 0.0 and prev_angle_cross is not None and angle_triplets:
        l2 = max(l_body_px, 1.0) ** 2
        for (i_a, i_b, i_c) in angle_triplets[:1]:  # main body axis only
            u = x[i_a] - x[i_b]
            v = x[i_c] - x[i_b]
            cross_now = (u[0] * v[1] - u[1] * v[0]) / l2
            delta_cross = cross_now - prev_angle_cross
            cost += lambda_angle_cont * delta_cross ** 2
            coeff = 2.0 * lambda_angle_cont * delta_cross / l2
            grad[i_a, 0] += coeff * v[1]
            grad[i_a, 1] += coeff * (-v[0])
            grad[i_b, 0] += coeff * (-v[1] + u[1])
            grad[i_b, 1] += coeff * (-u[0] + v[0])
            grad[i_c, 0] += coeff * (-u[1])
            grad[i_c, 1] += coeff * u[0]

    # Gradient clipping: prevents runaway L-BFGS-B steps from injecting artifacts
    _gn = np.linalg.norm(grad)
    if _gn > 1e3:
        grad *= 1e3 / _gn

    return cost, grad.ravel()


# ---------------------------------------------------------------------------
# Diagnostic: the magnitude of each objective term
# ---------------------------------------------------------------------------

def compute_term_magnitudes(
    x_opt: np.ndarray,
    observed: np.ndarray,
    prev_frame: Optional[np.ndarray],
    edges: List[SkeletonEdge],
    kp_index: Dict[str, int],
    l_body_px: float,
    edge_lambdas: np.ndarray,
    lambda_t: float = DEFAULT_LAMBDA_T,
    outlier_adhesion: float = 0.6,
    adhesion_weights: Optional[np.ndarray] = None,
    prev_prev_frame: Optional[np.ndarray] = None,
    lambda_acc: float = DEFAULT_LAMBDA_ACC,
    angle_triplets: Optional[List[Tuple[int, int, int]]] = None,
    lambda_angle: float = DEFAULT_LAMBDA_ANGLE,
    prev_prev_prev_frame: Optional[np.ndarray] = None,
    lambda_jerk: float = DEFAULT_LAMBDA_JERK,
    prev_angle_cross: Optional[float] = None,
    lambda_angle_cont: float = 0.0,
    verbose: bool = False,
) -> Dict[str, float]:
    """Compute individual loss-term magnitudes at a given solution point.

    Use this to diagnose objective imbalance: if one term dominates, the
    optimizer ignores the others.  The ratio between the largest and smallest
    non-zero terms indicates whether re-scaling of λ parameters is needed.

    Args:
        x_opt            : current solution, shape (2*n_kp,) or (n_kp, 2).
        observed         : observed keypoints, shape (n_kp, 2).
        verbose          : if True, prints a formatted table.

    Returns:
        dict with keys 'adhesion', 'length', 'temporal', 'accel',
                        'angle', 'jerk', 'angle_cont', 'total'.
    """
    n_kp = len(kp_index)
    x = np.asarray(x_opt, dtype=float).reshape(n_kp, 2)
    terms: Dict[str, float] = {}
    l2_norm = max(l_body_px, 1.0) ** 2  # normalization factor (same as in kinematic_cost_grad)

    # ── Term 1: adhesion (normalized by l_body²) ─────────────────────────
    valid_obs = ~np.any(np.isnan(observed), axis=1)
    if adhesion_weights is not None:
        weights = adhesion_weights.astype(float).copy()
    else:
        weights = np.where(valid_obs, 1.0, outlier_adhesion).astype(float)
    weights[~valid_obs] = 0.0
    diff_obs = np.where(valid_obs[:, None], x - observed, 0.0)
    terms["adhesion"] = float(np.sum(weights[:, None] * diff_obs ** 2)) / l2_norm

    # ── Term 2: length ────────────────────────────────────────────────────
    length_cost = 0.0
    for e_idx, edge in enumerate(edges):
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        L_ref_px = edge.L_ref * l_body_px
        if L_ref_px <= 0:
            continue
        d = float(np.linalg.norm(x[i] - x[j]))
        if d < 1e-8:
            continue
        length_cost += float(edge_lambdas[e_idx]) * ((d - L_ref_px) / L_ref_px) ** 2
    terms["length"] = length_cost

    # ── Term 3: temporal (normalized by l_body²) ─────────────────────────
    temporal_cost = 0.0
    if prev_frame is not None:
        vp = ~np.any(np.isnan(prev_frame), axis=1)
        diff_t = x - prev_frame
        diff_t[~vp] = 0.0
        temporal_cost = float(lambda_t * np.sum(diff_t[vp] ** 2)) / l2_norm
    terms["temporal"] = temporal_cost

    # ── Term 4: acceleration (normalized by l_body²) ─────────────────────
    accel_cost = 0.0
    if lambda_acc > 0 and prev_frame is not None and prev_prev_frame is not None:
        vp  = ~np.any(np.isnan(prev_frame), axis=1)
        vpp = ~np.any(np.isnan(prev_prev_frame), axis=1)
        va  = vp & vpp
        if va.any():
            acc = x - 2.0 * prev_frame + prev_prev_frame
            acc[~va] = 0.0
            accel_cost = float(lambda_acc * np.sum(acc[va] ** 2)) / l2_norm
    terms["accel"] = accel_cost

    # ── Term 5: angle ─────────────────────────────────────────────────────
    angle_cost = 0.0
    if lambda_angle > 0 and angle_triplets:
        l2 = max(l_body_px, 1.0) ** 2
        for (i_a, i_b, i_c) in angle_triplets:
            u = x[i_a] - x[i_b]
            v = x[i_c] - x[i_b]
            angle_cost += lambda_angle * ((u[0] * v[1] - u[1] * v[0]) / l2) ** 2
    terms["angle"] = angle_cost

    # ── Term 6: jerk (normalized by l_body²) ────────────────────────────
    jerk_cost = 0.0
    if (lambda_jerk > 0 and prev_frame is not None
            and prev_prev_frame is not None and prev_prev_prev_frame is not None):
        vp   = ~np.any(np.isnan(prev_frame), axis=1)
        vpp  = ~np.any(np.isnan(prev_prev_frame), axis=1)
        vppp = ~np.any(np.isnan(prev_prev_prev_frame), axis=1)
        vj   = vp & vpp & vppp
        if vj.any():
            jk = x - 3.0 * prev_frame + 3.0 * prev_prev_frame - prev_prev_prev_frame
            jk[~vj] = 0.0
            jerk_cost = float(lambda_jerk * np.sum(jk[vj] ** 2)) / l2_norm
    terms["jerk"] = jerk_cost

    # ── Term 7: angle continuity ──────────────────────────────────────────
    angle_cont_cost = 0.0
    if lambda_angle_cont > 0 and prev_angle_cross is not None and angle_triplets:
        l2 = max(l_body_px, 1.0) ** 2
        for (i_a, i_b, i_c) in angle_triplets[:1]:
            u = x[i_a] - x[i_b]
            v = x[i_c] - x[i_b]
            cross_now = (u[0] * v[1] - u[1] * v[0]) / l2
            angle_cont_cost += float(
                lambda_angle_cont * (cross_now - prev_angle_cross) ** 2
            )
    terms["angle_cont"] = angle_cont_cost

    terms["total"] = sum(v for k, v in terms.items() if k != "total")

    if verbose:
        total = max(terms["total"], 1e-12)
        print(f"  {'Term':<14} {'Value':>12} {'% of total':>12}")
        print(f"  {'-'*40}")
        for k in ["adhesion", "length", "temporal", "accel", "angle", "jerk", "angle_cont"]:
            v = terms[k]
            print(f"  {k:<14} {v:>12.4f} {v / total * 100:>11.1f}%")
        print(f"  {'total':<14} {terms['total']:>12.4f}")

    return terms
def kinematic_cost(
    x_flat: np.ndarray,
    observed: np.ndarray,
    prev_frame: Optional[np.ndarray],
    edges: List[SkeletonEdge],
    kp_index: Dict[str, int],
    l_body_px: float,
    visible_mask: np.ndarray,
    lambda_l: float = DEFAULT_LAMBDA_L,
    lambda_t: float = DEFAULT_LAMBDA_T,
) -> float:
    """Scalar wrapper without the gradient, kept for compatibility."""
    edge_lambdas = np.full(len(edges), lambda_l)
    c, _ = kinematic_cost_grad(x_flat, observed, prev_frame, edges,
                                kp_index, l_body_px, visible_mask,
                                edge_lambdas, lambda_t)
    return c


# ---------------------------------------------------------------------------
# Predefined angular triplets for the orientation constraints
# ---------------------------------------------------------------------------

def build_angle_triplets(
    skeleton: "MouseSkeleton",
    kp_index: Dict[str, int],
) -> List[Tuple[int, int, int]]:
    """Build the (i_A, i_B, i_C) triplets for the collinearity constraints.

    In each triplet B is the central joint:
      - nose, neck, tail_base:        body-axis alignment
      - ear_left, neck, ear_right:    bilateral symmetry of the head
      - hip_left, tail_base, hip_right: bilateral symmetry of the pelvis

    The penalty ``cross(A-B, C-B)^2 / L_body^4`` is zero when A, B and C are
    collinear, that is, when the body is extended, and grows with the curvature of
    the pose. Note this biases the correction towards straight postures, which is
    why the ablation in notebook 02a found that dropping this term lowers both the
    violation rate and the jerk: a curled mouse is a real pose, not an error.
    """
    candidates = [
        # Main body axis: the most important one, since it catches swaps and mirroring
        ("nose",      "neck",      "tail_base"),
        # Bilateral symmetry of the head
        ("ear_left",  "neck",      "ear_right"),
        # Bilateral symmetry of the pelvis
        ("hip_left",  "tail_base", "hip_right"),
        # Nose-to-body alignment, which reinforces the forward orientation
        ("nose",      "neck",      "hip_left"),
        ("nose",      "neck",      "hip_right"),
        # Body curvature: neck-hip_left-tail_base and neck-hip_right-tail_base
        ("neck",      "hip_left",  "tail_base"),
        ("neck",      "hip_right", "tail_base"),
        # Nose-neck-ear alignment, which reinforces the head orientation
        ("nose",      "neck",      "ear_left"),
        ("nose",      "neck",      "ear_right"),
        # Hip-to-tail symmetry, which stabilises the pelvis
        ("hip_left",  "neck",      "hip_right"),
    ]
    triplets = []
    for (a, b, c) in candidates:
        i_a = kp_index.get(a)
        i_b = kp_index.get(b)
        i_c = kp_index.get(c)
        if i_a is not None and i_b is not None and i_c is not None:
            triplets.append((i_a, i_b, i_c))
    return triplets


# ---------------------------------------------------------------------------
# Optimising a single frame
# ---------------------------------------------------------------------------

def correct_frame(
    observed: np.ndarray,
    prev_frame: Optional[np.ndarray],
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    l_body_px: float,
    outlier_mask: np.ndarray,
    edge_lambdas: Optional[np.ndarray] = None,
    lambda_l: float = DEFAULT_LAMBDA_L,
    lambda_t: float = DEFAULT_LAMBDA_T,
    outlier_adhesion: float = 0.6,
    adhesion_weights: Optional[np.ndarray] = None,
    max_iter: int = 200,
    prev_prev_frame: Optional[np.ndarray] = None,
    lambda_acc: float = DEFAULT_LAMBDA_ACC,
    lambda_angle: float = DEFAULT_LAMBDA_ANGLE,
    angle_triplets: Optional[List[Tuple[int, int, int]]] = None,
    prev_prev_prev_frame: Optional[np.ndarray] = None,
    lambda_jerk: float = DEFAULT_LAMBDA_JERK,
    prev_angle_cross: Optional[float] = None,
    lambda_angle_cont: float = 0.0,
    n_passes: int = 1,
    return_convergence: bool = False,
    staged: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, List[float]]]:
    """Correct the outlier keypoints of a single frame.

    Parameters
    ----------
    observed : np.ndarray, shape (n_kp, 2)
        Observed coordinates for this frame, outliers included.
    prev_frame : np.ndarray or None, shape (n_kp, 2)
        Positions from the previous frame.
    skeleton : MouseSkeleton
        Skeleton with its edges and L_ref already fitted.
    kp_index : dict[str, int]
        Mapping from keypoint name to index.
    l_body_px : float
        Reference body length, in pixels.
    outlier_mask : np.ndarray, shape (n_kp,), bool
        True where the keypoint is an outlier, and therefore free to move.
    edge_lambdas : np.ndarray, shape (n_edges,), opcional
        Per-edge weights. With None, ``lambda_l`` is applied uniformly.
    lambda_l : float
        Uniform fallback weight used when ``edge_lambdas`` is None.
    lambda_t : float
        Weight of the temporal smoothing term.
    outlier_adhesion : float
        Soft adhesion weight for outlier keypoints, in 0 to 1. Stops the optimiser
        dragging a keypoint with a small error into a solution that breaks other
        constraints, which is how one bad detection propagates across the skeleton.
    max_iter : int
        Maximum optimiser iterations.
    return_convergence : bool
        When True, return ``(corrected, loss_history)``, where ``loss_history`` is
        the final objective value for each pass. When False, the default, return
        only the corrected array.
    staged : bool
        When True, recommended, and n_passes >= 2, optimise in two stages:
        - Pass 0: recover the geometry, with no acceleration or jerk terms and
          lambda_l and lambda_angle raised by half.
        - Pass 1 onwards: full temporal smoothing with every lambda active.
        When False, use the original progressive adhesion schedule. Splitting the
        passes this way fixed the loss increasing between passes that the two-pass
        version suffered.

    Returns
    -------
    corrected : np.ndarray, shape (n_kp, 2)
        Keypoints corregidos.
    loss_history : list[float], opcional
        Only when ``return_convergence=True``. The final objective value per pass,
        for diagnosing whether the optimiser actually converged.
    """
    if edge_lambdas is None:
        edge_lambdas = np.full(len(skeleton.edges), lambda_l)

    visible_mask = ~outlier_mask

    x0 = observed.copy()
    if prev_frame is not None:
        nan_obs = np.any(np.isnan(observed), axis=1)
        use_prev = outlier_mask | nan_obs
        valid_prev = ~np.any(np.isnan(prev_frame), axis=1)
        x0[use_prev & valid_prev] = prev_frame[use_prev & valid_prev]

    # Fill any remaining NaN keypoints with the centroid of valid ones.
    # Suppress "Mean of empty slice" warning: on fully-corrupted frames every
    # column may be NaN; np.nanmean still returns NaN safely in that case.
    with np.errstate(all="ignore"):
        _centroid = np.nanmean(x0, axis=0)   # (2,) — NaN if all kps are NaN
    _centroid = np.nan_to_num(_centroid, nan=0.0)
    for i in range(x0.shape[0]):
        if np.any(np.isnan(x0[i])):
            x0[i] = _centroid

    # Build vectorised edge cache once (avoids kp_index lookups on every call)
    _cache = build_edge_cache(skeleton.edges, kp_index, angle_triplets)

    # Multi-pass optimization: each pass uses the previous result as warm-start.
    # Staged mode (recommended, n_passes >= 2):
    #   Pass 0: geometry recovery — λ_acc=0, λ_jerk=0, boosted λ_l/λ_angle.
    #   Pass 1+: full temporal smoothing with all lambdas.
    # Non-staged mode: adhesion decreases progressively across passes.
    adhesion_schedule = np.linspace(1.0, outlier_adhesion, n_passes + 1)[1:]
    current_x0 = x0.copy()
    loss_history: List[float] = []

    for pass_idx in range(n_passes):
        pass_adhesion = float(adhesion_schedule[pass_idx])
        # For each pass use the scheduled adhesion for the cost function
        pass_adh_weights = (
            adhesion_weights * pass_adhesion / max(outlier_adhesion, 1e-6)
            if adhesion_weights is not None
            else None
        )
        # Clip so adhesion weights stay in valid range [0, 1]
        if pass_adh_weights is not None:
            pass_adh_weights = np.clip(pass_adh_weights, 0.0, 1.0)

        # Continuation scheduling: all passes share the same objective structure
        # (all terms active throughout).  Geometry weight starts at 1.5× and
        # decays smoothly to 1×; temporal/motion weights ramp from a small floor
        # (10%) up to full.  This keeps loss values comparable across passes,
        # enabling meaningful early stopping from pass 1 onward and preventing
        # the “loss increases across passes” artifact caused by adding new terms
        # mid-optimization.
        if staged and n_passes >= 2:
            _alpha = pass_idx / max(n_passes - 1, 1)   # 0.0 at pass 0 → 1.0 at last
            _geom  = 1.5 - 0.5 * _alpha                # geometry: 1.5× → 1.0×
            _temp  = max(0.1 + 0.9 * _alpha, 0.1)      # temporal: 0.1 → 1.0
            pass_edge_lambdas  = edge_lambdas * _geom
            pass_lambda_angle  = lambda_angle * _geom
            pass_lambda_t      = lambda_t * _temp
            pass_lambda_acc    = lambda_acc * _temp
            pass_lambda_jerk   = lambda_jerk * _temp
            pass_lambda_a_cont = lambda_angle_cont * _temp
        else:
            pass_lambda_acc    = lambda_acc
            pass_lambda_jerk   = lambda_jerk
            pass_lambda_t      = lambda_t
            pass_edge_lambdas  = edge_lambdas
            pass_lambda_angle  = lambda_angle
            pass_lambda_a_cont = lambda_angle_cont

        result: OptimizeResult = minimize(
            fun=kinematic_cost_grad,
            x0=current_x0.ravel(),
            args=(observed, prev_frame, skeleton.edges, kp_index,
                  l_body_px, visible_mask, pass_edge_lambdas, pass_lambda_t,
                  pass_adhesion, pass_adh_weights,
                  prev_prev_frame, pass_lambda_acc, angle_triplets, pass_lambda_angle,
                  prev_prev_prev_frame, pass_lambda_jerk,
                  prev_angle_cross, pass_lambda_a_cont, _cache),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 20},
        )
        current_x0 = result.x.reshape(-1, 2)
        loss_history.append(float(result.fun))

        # ── Convergence early stopping ────────────────────────────────────────
        # With continuation scheduling all passes share the same objective
        # structure (just with ramped weights), so loss values are comparable
        # from pass 1 onward.  Stop early if relative improvement < 0.01%.
        if pass_idx >= 1 and len(loss_history) >= 2:
            prev_loss = loss_history[-2]
            curr_loss = loss_history[-1]
            rel_improvement = abs(prev_loss - curr_loss) / max(abs(prev_loss), 1e-12)
            if rel_improvement < 1e-4:
                break

    # ── Convergence metadata: expose iterations and status for diagnostics ───
    # result.nit = iterations in last pass; result.success = convergence status.
    converged = bool(result.success) or ("CONVERGENCE" in result.message.upper())
    conv_meta = {
        "n_opt_iters_last":  int(result.nit),
        "converged":         converged,
        "opt_message":       result.message,
    }

    corrected = current_x0.copy()
    # Only restore original observations for visible (non-outlier) keypoints
    # that actually have valid (non-NaN) observed positions.  Missing keypoints
    # (NaN in observed) keep the geometry-inferred position from the optimizer.
    _valid_obs = ~np.any(np.isnan(observed), axis=1)
    corrected[visible_mask & _valid_obs] = observed[visible_mask & _valid_obs]

    if return_convergence:
        return corrected, loss_history, conv_meta
    return corrected


# ---------------------------------------------------------------------------
# Joint optimisation over a temporal window
# ---------------------------------------------------------------------------

def correct_window(
    window_observed: np.ndarray,
    window_adhesion: np.ndarray,
    window_valid: np.ndarray,
    skeleton: "MouseSkeleton",
    kp_index: Dict[str, int],
    l_body_px: float,
    edge_lambdas: np.ndarray,
    lambda_t: float = DEFAULT_LAMBDA_T,
    lambda_acc: float = DEFAULT_LAMBDA_ACC,
    lambda_angle: float = DEFAULT_LAMBDA_ANGLE,
    lambda_jerk: float = DEFAULT_LAMBDA_JERK,
    max_iter: int = 400,
    angle_triplets: Optional[List[Tuple[int, int, int]]] = None,
    edge_cache: Optional["EdgeCache"] = None,
    return_convergence: bool = False,
    x0_warm: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float, dict]]:
    """Jointly optimize all frames in a temporal window.

    Instead of optimizing each outlier frame independently (which only has
    access to previous frames as temporal anchors), this function optimizes
    an entire short window of frames simultaneously.  Clean frames at the
    window boundary have ``window_adhesion=1.0`` and act as strong anchors;
    outlier frames have lower adhesion and are free to move.  The cross-frame
    velocity, acceleration, and jerk terms span the full window, so the
    optimizer can reason about trajectory continuity over multiple frames at
    once — making swaps, drift and speed spikes detectable.

    Args:
        window_observed : shape (T_win, n_kp, 2) — observed (or spline-init)
                          keypoints for every frame in the window.
        window_adhesion : shape (T_win, n_kp) — adhesion weight per keypoint.
                          Clean frames: 1.0 (anchor).  Outlier frames: 0–1.
        window_valid    : shape (T_win,) bool — False for zero/placeholder frames.
        skeleton        : fitted MouseSkeleton.
        kp_index        : keypoint name → index.
        l_body_px       : per-video body length in pixels.
        edge_lambdas    : shape (n_edges,) per-edge geometry weights.
        lambda_t/acc/angle/jerk : loss weights (same semantics as correct_frame).
        max_iter        : L-BFGS-B iteration budget.
        angle_triplets  : angle collinearity triplets (int indices).
        edge_cache      : pre-built EdgeCache (built once per video for speed).
        return_convergence : if True, return (corrected, final_cost, conv_meta).
        x0_warm         : shape (T_win, n_kp, 2) — warm-start initial point from
                          the previous corrected window.  When provided, L-BFGS-B
                          starts from this point instead of the observed/spline
                          values, which typically reduces the number of iterations
                          substantially for consecutive overlapping windows.

    Returns:
        corrected : shape (T_win, n_kp, 2) — jointly corrected window.
        final_cost : float (only when return_convergence=True).
        conv_meta  : dict with n_opt_iters_last, converged (only when True).
    """
    T_win, n_kp, _ = window_observed.shape
    l2_norm = max(l_body_px, 1.0) ** 2

    # Build edge cache once if not provided
    if edge_cache is None:
        edge_cache = build_edge_cache(skeleton.edges, kp_index, angle_triplets)

    # x0: prefer warm-start when provided, fall back to observed values.
    # The warm-start is the corrected output of the previous adjacent window,
    # giving L-BFGS-B a much better initial point than the raw/spline obs.
    if (x0_warm is not None
            and x0_warm.shape == (T_win, n_kp, 2)
            and np.isfinite(x0_warm).all()):
        x0 = x0_warm.copy()
        # For clean anchor frames (adhesion=1.0), the warm-start is overridden
        # with the actual observations so anchors don't drift from their values.
        anchor_mask = (window_adhesion >= 1.0) & window_valid[:, None]  # (T, K)
        valid_obs = ~np.any(np.isnan(window_observed), axis=2)           # (T, K)
        replace = anchor_mask & valid_obs
        x0[replace] = window_observed[replace]
    else:
        x0 = window_observed.copy()
    for kp_i in range(n_kp):
        for dim in range(2):
            series = x0[:, kp_i, dim]
            nan_m  = np.isnan(series) | ~window_valid
            valid_t = np.where(~nan_m)[0]
            if valid_t.size >= 2:
                bad_t = np.where(nan_m)[0]
                if bad_t.size:
                    series[bad_t] = np.interp(bad_t, valid_t, series[valid_t])
    np.nan_to_num(x0, nan=0.0, copy=False)

    # Pre-compute cross-frame validity masks (skip pairs/triplets/quads that
    # span a zero-frame so the temporal terms don't pull across dead regions).
    vp = window_valid[:-1] & window_valid[1:]      if T_win >= 2 else None  # velocity
    vt = (window_valid[:-2] & window_valid[1:-1]
          & window_valid[2:])                       if T_win >= 3 else None  # accel
    vq = (window_valid[:-3] & window_valid[1:-2]
          & window_valid[2:-1] & window_valid[3:])  if T_win >= 4 else None  # jerk

    # ── Pre-compute constants shared across all _cost_grad calls ─────────────
    # Hoisted out of the hot loop to avoid recomputation on every L-BFGS-B iter.
    _dp_adh   = 0.5 * max(l_body_px, 1.0)
    _valid_obs_mask = ~np.any(np.isnan(window_observed), axis=2)  # (T_win, n_kp)
    # Combined validity: frame valid AND observation not NaN
    _obs_active = _valid_obs_mask & window_valid[:, None]          # (T_win, n_kp)
    # Adhesion weights with NaN and invalid frames zeroed out
    _w = window_adhesion * _obs_active                             # (T_win, n_kp)
    # Safe observed values: replace NaN/invalid positions with 0 so subtraction
    # x - _obs_safe = 0 where the frame/keypoint is inactive.
    _obs_safe   = np.where(_obs_active[:, :, None], window_observed, 0.0)
    # Edge precomputation
    _L_ref_px   = edge_cache.L_refs * l_body_px                    # (n_edges,)
    _lam_arr    = edge_lambdas[edge_cache.e_idx]                   # (n_edges,)
    # Per-keypoint edge membership lists for scatter-add (built once per window)
    _src_per_kp = [np.where(edge_cache.src_idx == ki)[0] for ki in range(n_kp)]
    _dst_per_kp = [np.where(edge_cache.dst_idx == ki)[0] for ki in range(n_kp)]
    # Angle triplet indices
    _l2_angle   = max(l_body_px, 1.0) ** 2
    _has_trips  = (lambda_angle > 0.0 and edge_cache.trip_arr is not None
                   and len(edge_cache.trip_arr) > 0)
    if _has_trips:
        _ia = edge_cache.trip_arr[:, 0]
        _ib = edge_cache.trip_arr[:, 1]
        _ic = edge_cache.trip_arr[:, 2]
    # Frame validity as float column for broadcasting
    _wv_col = window_valid[:, None].astype(float)                  # (T_win, 1)

    def _cost_grad(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        x    = x_flat.reshape(T_win, n_kp, 2)
        grad = np.zeros_like(x)
        cost = 0.0

        # ── Per-frame adhesion + geometry + angle (vectorized over T_win) ─────
        # Huber adhesion: diff = x - obs where active, else 0.
        # _obs_safe already has inactive positions set to x's base value (0),
        # but we need diff = x - obs_safe when active, 0 otherwise.
        # Since _obs_safe[inactive] = 0 and we subtract from x (not 0), we use:
        diff  = x - _obs_safe                                        # (T_win, n_kp, 2)
        diff  = diff * _obs_active[:, :, None]                       # zero inactive
        _dn   = np.sqrt(np.sum(diff ** 2, axis=2))                   # (T_win, n_kp)
        _l2r  = _dn <= _dp_adh
        _hub  = np.where(_l2r, _dn ** 2,
                          2.0 * _dp_adh * _dn - _dp_adh ** 2) / l2_norm
        cost += float(np.sum(_w * _hub))
        _gs   = np.where(_l2r, 2.0 / l2_norm,
                         2.0 * _dp_adh / (np.maximum(_dn, 1e-8) * l2_norm))
        grad += _w[:, :, None] * _gs[:, :, None] * diff

        # Geometry (Huber length) — vectorized over (T_win, n_edges)
        if edge_cache.src_idx.size > 0:
            deltas = x[:, edge_cache.src_idx] - x[:, edge_cache.dst_idx]  # (T, E, 2)
            dists  = np.sqrt(np.sum(deltas ** 2, axis=2))                  # (T, E)
            nz     = dists > 1e-8
            viol   = np.where(nz, (dists - _L_ref_px) / _L_ref_px, 0.0)
            av     = np.abs(viol)
            lm     = av <= _HUBER_DELTA_L
            hl     = np.where(lm, viol ** 2,
                               2.0 * _HUBER_DELTA_L * av - _HUBER_DELTA_L ** 2)
            hl    *= _wv_col                                                # (T, E)
            cost  += float(np.dot(_lam_arr, hl.sum(axis=0)))
            gd     = np.where(nz, _L_ref_px * dists, 1.0)
            dLdv   = np.where(lm, 2.0 * viol,
                               2.0 * _HUBER_DELTA_L * np.sign(viol))
            gc     = np.where(nz & window_valid[:, None],
                               _lam_arr * dLdv / gd, 0.0)                  # (T, E)
            gc_d   = gc[:, :, None] * deltas                               # (T, E, 2)
            # Scatter-add to grad[:, ki]: sum contributions from edges
            for ki in range(n_kp):
                if _src_per_kp[ki].size:
                    grad[:, ki] += gc_d[:, _src_per_kp[ki]].sum(axis=1)
                if _dst_per_kp[ki].size:
                    grad[:, ki] -= gc_d[:, _dst_per_kp[ki]].sum(axis=1)

        # Angle collinearity — vectorized over (T_win, n_trips)
        if _has_trips:
            u  = x[:, _ia] - x[:, _ib]                                    # (T, P, 2)
            v  = x[:, _ic] - x[:, _ib]
            cr = (u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]) / _l2_angle
            cr = cr * _wv_col                                               # (T, P)
            cost += lambda_angle * float(np.sum(cr ** 2))
            cf   = (2.0 * lambda_angle / _l2_angle) * cr                   # (T, P)
            for ti in range(len(_ia)):
                cf_ti = cf[:, ti]
                grad[:, _ia[ti], 0] += cf_ti *  v[:, ti, 1]
                grad[:, _ia[ti], 1] += cf_ti * -v[:, ti, 0]
                grad[:, _ib[ti], 0] += cf_ti * (-v[:, ti, 1] + u[:, ti, 1])
                grad[:, _ib[ti], 1] += cf_ti * (-u[:, ti, 0] + v[:, ti, 0])
                grad[:, _ic[ti], 0] += cf_ti * -u[:, ti, 1]
                grad[:, _ic[ti], 1] += cf_ti *  u[:, ti, 0]

        # ── Cross-frame velocity (pseudo-Huber) ──────────────────────────────────────────
        # E(dv) = lambda_t * sum_{t,kp} sqrt(||dv||^2 + eps) / l2
        # Gradient: dv / sqrt(||dv||^2 + eps)
        if lambda_t > 0.0 and vp is not None:
            dv      = (x[1:] - x[:-1]) * vp[:, None, None]       # (T-1, K, 2)
            _eps_t  = 0.05
            _dv_ph  = np.sqrt(np.sum(dv ** 2, axis=2) + _eps_t)   # (T-1, K)
            cost   += lambda_t * np.sum(_dv_ph) / l2_norm
            _g_dv   = lambda_t / l2_norm * dv / _dv_ph[:, :, None]
            grad[1:]  += _g_dv
            grad[:-1] -= _g_dv

        # ── Cross-frame acceleration (pseudo-Huber) ──────────────────────────────────────────
        # acc[s] = x[s+2] - 2*x[s+1] + x[s]
        # grad[s] += g_acc;  grad[s+1] -= 2*g_acc;  grad[s+2] += g_acc
        if lambda_acc > 0.0 and vt is not None:
            acc      = (x[2:] - 2.0 * x[1:-1] + x[:-2]) * vt[:, None, None]  # (T-2, K, 2)
            _eps_a   = 0.02
            _acc_ph  = np.sqrt(np.sum(acc ** 2, axis=2) + _eps_a)              # (T-2, K)
            cost    += lambda_acc * np.sum(_acc_ph) / l2_norm
            _g_acc   = lambda_acc / l2_norm * acc / _acc_ph[:, :, None]
            grad[:-2]  += _g_acc
            grad[1:-1] -= 2.0 * _g_acc
            grad[2:]   += _g_acc

        # ── Cross-frame jerk (pseudo-Huber) ──────────────────────────────────────────────
        # jerk[s] = x[s+3] - 3x[s+2] + 3x[s+1] - x[s]
        # grad[s]-=g_jk;  grad[s+1]+=3*g_jk;  grad[s+2]-=3*g_jk;  grad[s+3]+=g_jk
        if lambda_jerk > 0.0 and vq is not None:
            jerk     = (x[3:] - 3.0 * x[2:-1]
                        + 3.0 * x[1:-2] - x[:-3]) * vq[:, None, None]  # (T-3, K, 2)
            _eps_j   = 0.01
            _jk_ph   = np.sqrt(np.sum(jerk ** 2, axis=2) + _eps_j)     # (T-3, K)
            cost    += lambda_jerk * np.sum(_jk_ph) / l2_norm
            _g_jk    = lambda_jerk / l2_norm * jerk / _jk_ph[:, :, None]
            grad[:-3]  -= _g_jk
            grad[1:-2] += 3.0 * _g_jk
            grad[2:-1] -= 3.0 * _g_jk
            grad[3:]   += _g_jk

        # Gradient clipping
        _gn = np.linalg.norm(grad)
        if _gn > 1e3:
            grad *= 1e3 / _gn

        return cost, grad.ravel()

    result = minimize(
        fun=_cost_grad,
        x0=x0.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-9, "gtol": 1e-6, "maxls": 20},
    )
    corrected_win = result.x.reshape(T_win, n_kp, 2)

    # Restore strongly-anchored (clean) keypoints to their observed positions
    # so that window optimization cannot drift context frames.
    for t in range(T_win):
        if not window_valid[t]:
            continue
        anchor_mask = window_adhesion[t] >= 1.0 - 1e-6
        valid_obs_t = ~np.any(np.isnan(window_observed[t]), axis=1)
        restore = anchor_mask & valid_obs_t
        if restore.any():
            corrected_win[t, restore] = window_observed[t, restore]

    converged = bool(result.success) or ("CONVERGENCE" in result.message.upper())
    conv_meta = {"n_opt_iters_last": int(result.nit), "converged": converged}

    if return_convergence:
        return corrected_win, float(result.fun), conv_meta
    return corrected_win


# ---------------------------------------------------------------------------
# Detecting biomechanical constraint violations
# ---------------------------------------------------------------------------

def compute_constraint_violations(
    wide_df: pd.DataFrame,
    skeleton: MouseSkeleton,
    kp_index: Dict[str, int],
    l_body_series: "pd.Series",
    threshold: float = 0.20,
) -> "pd.DataFrame":
    """Length violation rate, per edge and per frame.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Tracking en formato ancho (columnas {bodypart}_x, {bodypart}_y).
    skeleton : MouseSkeleton
        Skeleton with L_ref fitted.
    kp_index : dict[str, int]
        Keypoint name to index. Unused here, kept for interface compatibility.
    l_body_series : pd.Series
        Nose-to-tail_base length, one per row of wide_df.
    threshold : float
        Relative variation that counts as a violation. Default 0.20.

    Returns
    -------
    pd.DataFrame
        Columnas: ``edge``, ``violation_rate``, ``mean_error_rel``.
    """
    import pandas as _pd

    results = []
    for edge in skeleton.edges:
        sx, sy = f"{edge.src}_x", f"{edge.src}_y"
        dx, dy = f"{edge.dst}_x", f"{edge.dst}_y"
        missing = {sx, sy, dx, dy} - set(wide_df.columns)
        if missing:
            continue

        seg_len = np.sqrt(
            (wide_df[sx] - wide_df[dx]) ** 2 +
            (wide_df[sy] - wide_df[dy]) ** 2
        )
        L_ref_px = edge.L_ref * l_body_series
        rel_error = (seg_len - L_ref_px).abs() / L_ref_px.replace(0, np.nan)
        abs_error_px = (seg_len - L_ref_px).abs()
        viol_mask = rel_error > threshold
        results.append({
            "edge": f"{edge.src}→{edge.dst}",
            "violation_rate": float(viol_mask.mean()),
            "mean_error_rel": float(rel_error.mean()),
            "mean_abs_dev_px": float(abs_error_px.mean()),
            "max_abs_dev_px": float(abs_error_px.max()),
            "p95_abs_dev_px": float(abs_error_px.quantile(0.95)),
            # Continuous residual metrics (replace binary threshold)
            "residual_p50": float(rel_error.quantile(0.50)),
            "residual_p75": float(rel_error.quantile(0.75)),
            "residual_p95": float(rel_error.quantile(0.95)),
            "residual_p99": float(rel_error.quantile(0.99)),
            "residual_std": float(rel_error.std()),
        })

    return _pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Residual metrics: continuous per-edge quality scoring
# ---------------------------------------------------------------------------

def compute_residual_metrics(
    poses: np.ndarray,
    skeleton: "MouseSkeleton",
    kp_index: Dict[str, int],
    l_body_px: float,
    zero_frame_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute continuous residual metrics r_e = |L_e - L̂_e| / L̂_e for each edge.

    Replaces binary violation thresholds with a continuous residual distribution.
    This is more informative: a 84% violation rate at 5px noise may indicate
    threshold calibration problems rather than optimizer failure.

    Args:
        poses          : shape (T, K, 2) — corrected poses tensor.
        skeleton       : fitted MouseSkeleton.
        kp_index       : keypoint name → index.
        l_body_px      : per-video body length in pixels.
        zero_frame_mask: shape (T,) bool — frames to exclude.

    Returns:
        dict with keys ``edge_names`` (list[str]) and per-edge arrays:
        ``mean``, ``std``, ``p50``, ``p75``, ``p95``, ``p99``, ``max``.
        All arrays have shape (n_edges,).

    Example::

        metrics = compute_residual_metrics(corrected_poses, skeleton, kp_index, l_body)
        for i, name in enumerate(metrics["edge_names"]):
            print(f"{name}: mean={metrics['mean'][i]:.3f}  p95={metrics['p95'][i]:.3f}")
    """
    if zero_frame_mask is None:
        zero_frame_mask = np.zeros(len(poses), dtype=bool)
    active = ~zero_frame_mask

    edge_names = []
    means, stds = [], []
    p50s, p75s, p95s, p99s, maxs = [], [], [], [], []

    for edge in skeleton.edges:
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        L_ref_px = edge.L_ref * l_body_px
        if L_ref_px <= 0:
            continue

        # r_e = |L_e - L̂_e| / L̂_e  (relative residual, frame-level)
        diffs = poses[active, i] - poses[active, j]
        seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
        r_e = np.abs(seg_len - L_ref_px) / max(L_ref_px, 1e-6)

        # Remove NaN/inf (e.g. from all-zero frames leaking through)
        r_e = r_e[np.isfinite(r_e)]
        if len(r_e) == 0:
            r_e = np.array([0.0])

        edge_names.append(f"{edge.src}→{edge.dst}")
        means.append(float(np.mean(r_e)))
        stds.append(float(np.std(r_e)))
        p50s.append(float(np.percentile(r_e, 50)))
        p75s.append(float(np.percentile(r_e, 75)))
        p95s.append(float(np.percentile(r_e, 95)))
        p99s.append(float(np.percentile(r_e, 99)))
        maxs.append(float(np.max(r_e)))

    return {
        "edge_names": edge_names,
        "mean":  np.array(means),
        "std":   np.array(stds),
        "p50":   np.array(p50s),
        "p75":   np.array(p75s),
        "p95":   np.array(p95s),
        "p99":   np.array(p99s),
        "max":   np.array(maxs),
    }


# ---------------------------------------------------------------------------
# Per-video adaptive threshold calibration
# ---------------------------------------------------------------------------

def compute_adaptive_edge_thresholds(
    poses: np.ndarray,
    skeleton: "MouseSkeleton",
    kp_index: Dict[str, int],
    l_body_px: float,
    k: float = 2.5,
    min_threshold: float = 0.05,
    zero_frame_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute per-video adaptive detection thresholds τ_e = μ_e + k·σ_e.

    Rather than a fixed LENGTH_VAR_THRESHOLD (e.g. 20%), this estimates the
    natural variation of each edge in the current video and sets the threshold
    relative to that.  This is critical for cross-video robustness:
    videos with different motion regimes or body sizes get calibrated thresholds.

    Formula:
        τ_e = μ_e + k · σ_e

    where μ_e and σ_e are the mean and std of r_e = |L_e - L̂_e| / L̂_e
    over clean (non-outlier) frames of this video.

    Args:
        poses          : shape (T, K, 2) — raw (uncorrected) poses.
        skeleton       : fitted MouseSkeleton (L_ref per edge).
        kp_index       : keypoint → index.
        l_body_px      : per-video body length estimate (pixels).
        k              : how many std above mean sets the threshold (default 2.5).
        min_threshold  : minimum threshold to avoid calibrating too tight.
        zero_frame_mask: shape (T,) bool — frames to exclude.

    Returns:
        dict mapping ``"{src}→{dst}"`` → threshold float.
    """
    if zero_frame_mask is None:
        zero_frame_mask = np.zeros(len(poses), dtype=bool)
    active = ~zero_frame_mask

    thresholds: Dict[str, float] = {}
    for edge in skeleton.edges:
        i = kp_index.get(edge.src)
        j = kp_index.get(edge.dst)
        if i is None or j is None:
            continue
        L_ref_px = edge.L_ref * l_body_px
        if L_ref_px <= 0:
            continue

        diffs = poses[active, i] - poses[active, j]
        seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
        r_e = np.abs(seg_len - L_ref_px) / max(L_ref_px, 1e-6)
        r_e = r_e[np.isfinite(r_e)]
        if len(r_e) < 5:
            thresholds[f"{edge.src}→{edge.dst}"] = max(edge.L_std * k, min_threshold)
            continue

        # Use only the lower 75th percentile as "clean" frames to avoid
        # outliers polluting μ and σ estimates.
        clean_mask = r_e <= np.percentile(r_e, 75)
        r_clean = r_e[clean_mask]
        mu_e    = float(np.mean(r_clean))
        sigma_e = float(np.std(r_clean))
        tau_e   = mu_e + k * sigma_e
        thresholds[f"{edge.src}→{edge.dst}"] = max(tau_e, min_threshold)

    return thresholds
