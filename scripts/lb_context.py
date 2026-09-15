"""
scripts/lb_context.py
=====================
**Long, asymmetric** temporal context, added to an already-built cache.

The motivation was measured, not assumed: blending the tree with the CNN-LSTM came out
flat (w=0.0 -> 0.4654, w=1.0 -> 0.4634, best blend 0.4667). Two completely different
model families plateau in the same place, so what is missing is not capacity but
**information**. The window is 64 frames, about 2.1 s at 30 fps, and several actions —
chase, escape, approach, disengage — are only distinguishable by looking further out
and, above all, by looking at past and future separately: a symmetric aggregate gives
exactly the same value whether the mice are closing in or moving apart.

For each of the F base features, two columns are added:

    delta_past   = mean(features over the K preceding windows) - central mean
    delta_future = mean(features over the K following windows) - central mean

as a **difference** against the window's own block of means, which is what a tree can
split on with a single cut; with absolute values it would need two.

The cache does not have to be rebuilt: windows of the same (video, agent) are
`stride` frames apart, so the context is obtained by shifting along the array itself
once it is sorted by `core_start`.
"""
from __future__ import annotations
import numpy as np


def _group_slices(vids, agents):
    """Indices grouped by (video, agent); temporal order is imposed afterwards."""
    order = np.lexsort((agents, vids))
    v_s, a_s = vids[order], agents[order]
    brk = np.flatnonzero((v_s[1:] != v_s[:-1]) | (a_s[1:] != a_s[:-1])) + 1
    return order, np.split(order, brk)


def add_context_multi(X, vids, agents, core_start, n_base, ks=(6, 24), verbose=True):
    """Several temporal scales at once.

    At k=12, that is plus or minus 96 frames, the context was worth +0.012 on average
    across 9 labs, and positive in 8 of them. A single scale forces a choice between
    resolving short transitions and capturing long sequences; stacking scales lets the
    tree decide which to look at on each split. Reuses `add_context` and discards the
    repeated base columns.
    """
    blocks = [X]
    for k in ks:
        Y = add_context(X, vids, agents, core_start, n_base, k=k, verbose=False)
        blocks.append(Y[:, X.shape[1]:])          # only the two new blocks
    out = np.concatenate(blocks, axis=1)
    if verbose:
        print("    multi-scale context k={}: {} -> {} features".format(
            list(ks), X.shape[1], out.shape[1]), flush=True)
    return out


def add_context(X, vids, agents, core_start, n_base, k=12, verbose=True, out=None):
    """Return X extended with the past-delta and future-delta of the mean block.

    `n_base` is F, the number of features per frame. The aggregated cache stores
    [mean | std | min | max], so the first F columns are the means and they are the
    only ones it makes sense to average again.
    """
    N, D = X.shape
    mean_blk = X[:, :n_base]
    if out is not None:
        # Views onto the destination, which avoids materialising `past` and `fut`
        # in full; over the whole corpus those are 1.7 GB each.
        if out.shape != (N, D + 2 * n_base):
            raise ValueError("`out` has shape {} instead of {}".format(
                out.shape, (N, D + 2 * n_base)))
        past = out[:, D:D + n_base]
        fut = out[:, D + n_base:]
        past[:] = 0.0
        fut[:] = 0.0
    else:
        past = np.zeros((N, n_base), np.float32)
        fut = np.zeros((N, n_base), np.float32)

    _, groups = _group_slices(np.asarray(vids), np.asarray(agents))
    cs = np.asarray(core_start)
    for g in groups:
        if len(g) < 2:
            continue
        g = g[np.argsort(cs[g])]          # temporal order within the (video, agent)
        M = mean_blk[g].astype(np.float32)
        n = len(g)
        # Cumulative sums give sliding-window means in O(n)
        C = np.concatenate([np.zeros((1, n_base), np.float64), np.cumsum(M, 0, dtype=np.float64)])
        idx = np.arange(n)
        lo = np.maximum(idx - k, 0)
        hi = np.minimum(idx + k + 1, n)
        np_ = (idx - lo)[:, None]                 # windows available on each side
        nf_ = (hi - idx - 1)[:, None]
        # At the edges of a group there is no context. Dividing by max(n,1) would
        # give delta = 0 - mean = -mean, which is a REAL value and different on every
        # row: the tree would read it as signal when all it says is "no data here".
        # With no context the correct delta is exactly 0.
        past[g] = np.where(np_ > 0,
                           (C[idx] - C[lo]) / np.maximum(np_, 1) - M, 0.0).astype(np.float32)
        fut[g] = np.where(nf_ > 0,
                          (C[hi] - C[idx + 1]) / np.maximum(nf_, 1) - M, 0.0).astype(np.float32)

    if verbose:
        print("    context k={} windows: {} -> {} features ({} groups)".format(
            k, D, D + 2 * n_base, len(groups)), flush=True)
    if out is not None:
        return out
    return np.concatenate([X, past, fut], axis=1)
