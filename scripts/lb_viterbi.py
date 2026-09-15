"""
scripts/lb_viterbi.py
=====================
Sequential decoding **per lab**, over the already-saved probabilities.

Measured motivation. The dominant remaining failure is not classification but
segmentation: in TranquilPanther, `mount` is predicted over 5.73x as many frames as
are real, `sniff` 2.34x and `sniffgenital` 1.86x, while `intromit` sits at 0.46x. These
are **fragmented** false positives. Today the temporal structure is handled by two
local patches — Savitzky-Golay smoothing of the probabilities and an *a posteriori*
minimum-duration filter — neither of which optimises anything globally. Viterbi
searches for the maximum-likelihood label sequence over the whole video, and the
duration prior falls out naturally from the self-transitions.

The emission formulation. It uses

    emis[t, c] = log(p[t, c]) - log(t_c)   for c != background
    emis[t, background] = 0

With uniform transitions this **exactly reproduces** the `ratio` rule that 7 of the 9
labs already use: background wins if and only if no class exceeds its threshold, since
`p_c > t_c` is equivalent to `log(p_c/t_c) > 0`. So Viterbi generalises the current
decision rather than replacing it, and the emission weight `w` interpolates between
"emission only", which is today's rule, and "dominated by the transition prior".

The transition matrix is counted at **window** level (stride 8) from the labels already
in the cache, not by re-reading annotations: `y` is exactly the window's label. It
therefore depends on the stride, and must be recounted if the stride changes.
"""
from __future__ import annotations
import numpy as np

NEG = -1e30          # A workable -inf: avoids NaN when summed with transitions


def group_sequences(vids, agents, core_start):
    """Indices of each (video, agent), in temporal order."""
    keys = {}
    for i, (v, a) in enumerate(zip(np.asarray(vids).tolist(), np.asarray(agents).tolist())):
        keys.setdefault((v, a), []).append(i)
    cs = np.asarray(core_start)
    return [np.asarray(ix)[np.argsort(cs[np.asarray(ix)])] for ix in keys.values()]


def transition_logprobs(y, vids, agents, core_start, mask, n_classes, alpha=1.0):
    """log P(c_t | c_{t-1}), counted over the windows in `mask`, Laplace-smoothed.

    `alpha` keeps the counts off zero: a transition unseen in training should be
    improbable, not impossible, or the decoder could never leave a rare state.
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.log(np.full((n_classes, n_classes), 1.0 / n_classes))
    cnt = np.full((n_classes, n_classes), float(alpha))
    for seq in group_sequences(vids[idx], agents[idx], core_start[idx]):
        lab = y[idx[seq]]
        np.add.at(cnt, (lab[:-1], lab[1:]), 1.0)
    return np.log(cnt / cnt.sum(axis=1, keepdims=True))


def emission_logscores(P, thr, bg_idx, allowed=None):
    """log(p/t) for the action classes and 0 for background; see the module docstring."""
    t = np.maximum(np.asarray(thr, np.float64), 1e-6)
    with np.errstate(divide="ignore"):
        E = np.log(np.maximum(P.astype(np.float64), 1e-12)) - np.log(t)[None, :]
    E[:, bg_idx] = 0.0
    if allowed is not None:
        # Without the mask the winning class can be one the video does not annotate,
        # and the prediction is discarded downstream, scoring neither hit nor miss.
        blocked = ~allowed.copy()
        blocked[:, bg_idx] = False
        E[blocked] = NEG
    return E


def viterbi_decode(E, logA, w=1.0):
    """Maximum-likelihood sequence. `E` is (T, C) log-emission, `logA` is (C, C)."""
    T, C = E.shape
    if T == 0:
        return np.zeros(0, np.int64)
    if T == 1:
        return np.asarray([int(np.argmax(w * E[0]))], np.int64)
    delta = w * E[0]
    back = np.empty((T, C), np.int32)
    for t in range(1, T):
        M = delta[:, None] + logA          # (C_prev, C_next)
        back[t] = M.argmax(axis=0)
        delta = w * E[t] + M.max(axis=0)
    out = np.empty(T, np.int64)
    out[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out


def decode_lab(P, thr, bg_idx, vids, agents, core_start, logA, w=1.0, allowed=None):
    """Apply Viterbi to each (video, agent) in the given block of rows."""
    E = emission_logscores(P, thr, bg_idx, allowed=allowed)
    out = np.full(len(P), bg_idx, np.int64)
    for seq in group_sequences(vids, agents, core_start):
        out[seq] = viterbi_decode(E[seq], logA, w=w)
    return out


# ─────────────────────────────── autotest ────────────────────────────────────
def _selftest():
    from scripts.lb_threshold import decide
    rng = np.random.default_rng(0)
    C, T = 6, 200
    bg = 0
    P = rng.dirichlet(np.ones(C) * 0.5, size=T).astype(np.float32)
    thr = np.clip(rng.uniform(0.05, 0.6, C), 0.05, 0.9).astype(np.float32)
    allowed = rng.random((T, C)) > 0.25
    allowed[:, bg] = True
    v = np.array(["v"] * T); ag = np.zeros(T, int); cs = np.arange(T) * 8

    # (a) Degenerate case: uniform transitions with w=1 must reproduce `ratio`
    uni = np.log(np.full((C, C), 1.0 / C))
    got = decode_lab(P, thr, bg, v, ag, cs, uni, w=1.0, allowed=allowed)
    exp = decide(P, thr, bg, mode="ratio", allowed=allowed)
    assert np.array_equal(got, exp), "degenerate Viterbi != decide(ratio): {}/{}".format(
        int((got != exp).sum()), T)
    print("  (a) uniform transitions exactly reproduce decide(mode='ratio')")

    # (b) A strong self-transition prior must fragment less
    def n_runs(a):
        return 1 + int((a[1:] != a[:-1]).sum())
    A = np.full((C, C), 0.02); np.fill_diagonal(A, 1.0 - 0.02 * (C - 1))
    strong = np.log(A / A.sum(1, keepdims=True))
    got2 = decode_lab(P, thr, bg, v, ag, cs, strong, w=1.0, allowed=allowed)
    assert n_runs(got2) < n_runs(exp), "the strong prior did not reduce fragmentation"
    print("  (b) persistence prior: {} -> {} intervals".format(n_runs(exp), n_runs(got2)))

    # (c) The active-class mask is always respected
    assert allowed[np.arange(T), got2].all(), "an inactive class was emitted"
    print("  (c) no prediction falls on an inactive class")

    # (d) The transition matrix sums to 1 per row and has no hard zeros
    y = rng.integers(0, C, T)
    lg = transition_logprobs(y, v, ag, cs, np.ones(T, bool), C, alpha=1.0)
    assert np.allclose(np.exp(lg).sum(1), 1.0), "rows do not sum to 1"
    assert np.isfinite(lg).all(), "the transition matrix contains -inf"
    print("  (d) transition matrix normalised and free of hard zeros")
    print("AUTOTEST VITERBI OK")


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    _selftest()
