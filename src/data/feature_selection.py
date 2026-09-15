"""
src/data/feature_selection.py
=============================
B5 — Feed the redundancy/stability diagnostics back into feature selection.

The notebooks (`03_features` §8/§15) compute collinearity, VIF, drift and MI but
never act on them. These helpers turn the diagnostics into a kept-feature subset,
on the per-frame feature space the models consume. `03_features` imports and runs
``select_features`` and saves the result for the LOVO benchmark to load.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def vif_scores(Z):
    """VIF per column of standardized Z via R² of each col on the rest."""
    n_feat = Z.shape[1]
    vif = np.zeros(n_feat)
    lr = LinearRegression()
    for j in range(n_feat):
        others = np.delete(Z, j, axis=1)
        lr.fit(others, Z[:, j])
        vif[j] = 1.0 / max(1.0 - lr.score(others, Z[:, j]), 1e-9)
    return vif


def drift_scores(Xw_mean, vids):
    """Between-video / within-video variance ratio of per-window feature means."""
    groups = np.unique(vids)
    grand = Xw_mean.mean(0)
    between = np.zeros(Xw_mean.shape[1])
    within = np.zeros(Xw_mean.shape[1])
    for g in groups:
        m = vids == g
        gm = Xw_mean[m].mean(0)
        between += m.sum() * (gm - grand) ** 2
        within += ((Xw_mean[m] - gm) ** 2).sum(0)
    between /= (len(groups) - 1)
    within /= max(len(Xw_mean) - len(groups), 1)
    return between / np.maximum(within, 1e-12)


def select_features(X, y, vids, names=None, corr_thr=0.95, vif_thr=10.0,
                    max_frames=40000, seed=42):
    """Prune the (N,W,F) feature space by collinearity / VIF / stability.

    Policy: (1) drop redundant members of |r|>corr_thr clusters (keep highest-MI);
    (2) iteratively drop highest VIF>vif_thr; (3) drop high-drift & low-MI features.

    Returns
    -------
    (kept_idx, diagnostics) where diagnostics is a list of per-feature dicts
    (idx, feature, mi, drift, vif, max_abs_corr, kept, drop_reason).
    """
    X = np.nan_to_num(X.astype(np.float64))
    N, W, F = X.shape
    names = list(names) if names is not None else [f"f{i}" for i in range(F)]

    Xf = X.reshape(N * W, F)
    rng = np.random.default_rng(seed)
    if len(Xf) > max_frames:
        Xf = Xf[rng.choice(len(Xf), max_frames, replace=False)]
    Zf = np.nan_to_num(StandardScaler().fit_transform(Xf))

    Xw = X.mean(1)
    mi = mutual_info_classif(StandardScaler().fit_transform(Xw), y, random_state=seed)
    drift = drift_scores(Xw, vids)

    corr = np.corrcoef(Zf, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    max_abs_corr = np.abs(corr).max(1)

    kept = set(range(F))
    reason = {i: "" for i in range(F)}

    # 1. collinearity clusters
    adj = np.abs(corr) > corr_thr
    seen = set()
    for i in range(F):
        if i in seen:
            continue
        cluster, stack = {i}, [i]
        while stack:
            a = stack.pop()
            for b in np.where(adj[a])[0]:
                if b not in cluster:
                    cluster.add(b); stack.append(b)
        seen |= cluster
        if len(cluster) > 1:
            keep = max(cluster, key=lambda c: mi[c])
            for c in cluster:
                if c != keep:
                    kept.discard(c)
                    reason[c] = f"collinear(|r|>{corr_thr}) with {names[keep]}"

    # 2. VIF
    while True:
        kept_idx = sorted(kept)
        v = vif_scores(Zf[:, kept_idx])
        worst = int(np.argmax(v))
        if v[worst] <= vif_thr:
            break
        gi = kept_idx[worst]
        kept.discard(gi)
        reason[gi] = f"vif={v[worst]:.1f}>{vif_thr}"

    # 3. stability: high drift AND low MI
    drift_hi, mi_lo = np.quantile(drift, 0.75), np.quantile(mi, 0.25)
    for i in sorted(kept):
        if drift[i] >= drift_hi and mi[i] <= mi_lo:
            kept.discard(i)
            reason[i] = f"unstable(drift={drift[i]:.1f}) & low-mi({mi[i]:.3f})"

    kept_idx = sorted(kept)
    vif_full = vif_scores(Zf)
    diagnostics = [{
        "idx": i, "feature": names[i], "mi": float(mi[i]), "drift": float(drift[i]),
        "vif": float(vif_full[i]), "max_abs_corr": float(max_abs_corr[i]),
        "kept": i in kept, "drop_reason": reason[i],
    } for i in range(F)]
    return kept_idx, diagnostics
