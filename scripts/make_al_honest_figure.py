"""Honest active-learning figure: paired AULC, per seed.

Plots the five seeds of `results/al_honest_runs.csv` as deltas paired against random
sampling, which is exactly the contrast the thesis' Wilcoxon test makes. The left
panel shows the raw AULC per strategy and seed; the right panel shows the delta
against the same seed's random run, with its mean.

The rendered figure text stays in Spanish on purpose: this script writes into the
thesis figure directory and the result is included in Chapter 5, so regenerating it
must not change the language of the document.

    python -m scripts.make_al_honest_figure
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "al_honest_runs.csv"
SUMMARY = ROOT / "results" / "al_honest_summary.csv"
OUT = ROOT / "results" / "al_honest_aulc.png"
FIG = ROOT / "memoria" / "definitiva" / "TFM___Template" / "fig" / "skeleton"

# Spanish on purpose: these are rendered into a figure used by the thesis.
LABELS = {
    "entropy": "entropía",
    "coreset": "coreset",
    "margin": "margen",
    "least_confidence": "confianza mínima",
    "badge": "badge",
    "random": "aleatoria",
}


def main() -> None:
    runs = pd.read_csv(RUNS)
    summary = pd.read_csv(SUMMARY).set_index("strategy")
    order = [s for s in summary.sort_values("aulc_mean", ascending=False).index]
    others = [s for s in order if s != "random"]

    base = runs[runs.strategy == "random"].set_index("seed").aulc
    seeds = sorted(runs.seed.unique())

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # --- left: raw AULC, one mark per seed ---------------------------------
    for x, strat in enumerate(order):
        vals = runs[runs.strategy == strat].aulc.to_numpy()
        ax0.scatter(np.full_like(vals, x, dtype=float), vals, s=26,
                    color="#4C72B0" if strat != "random" else "#888888",
                    alpha=0.75, zorder=3)
        ax0.hlines(vals.mean(), x - 0.28, x + 0.28,
                   color="#C44E52" if strat != "random" else "#333333",
                   lw=2.2, zorder=4)
    ax0.set_xticks(range(len(order)))
    ax0.set_xticklabels([LABELS.get(s, s) for s in order], rotation=25, ha="right")
    ax0.set_ylabel("AULC")
    ax0.set_title("AULC por estrategia (5 semillas)\nla raya roja es la media")
    ax0.grid(axis="y", alpha=0.3)

    # --- right: delta paired against the same seed's random run ------------
    for x, strat in enumerate(others):
        sub = runs[runs.strategy == strat].set_index("seed").aulc
        deltas = np.array([sub[s] - base[s] for s in seeds])
        ax1.scatter(np.full_like(deltas, x, dtype=float), deltas, s=26,
                    color="#4C72B0", alpha=0.75, zorder=3)
        ax1.hlines(deltas.mean(), x - 0.28, x + 0.28, color="#C44E52", lw=2.2, zorder=4)
        p = summary.loc[strat, "wilcoxon_p"]
        ax1.annotate(f"p={p:.2f}", (x, deltas.max()), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color="#444444")
    ax1.axhline(0, color="#333333", lw=1.2, ls="--")
    ax1.set_xticks(range(len(others)))
    ax1.set_xticklabels([LABELS.get(s, s) for s in others], rotation=25, ha="right")
    ax1.set_ylabel("Δ AULC frente al azar (pareado por semilla)")
    ax1.set_title("Delta pareado y p de Wilcoxon\nninguna estrategia separa del azar")
    ax1.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    # The second copy goes into the dissertation's figure tree, which is not
    # distributed with this repository. Write it only where that tree already
    # exists; creating it from nothing would leave an orphan directory in a
    # fresh clone.
    if FIG.parent.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG / "al_honest_aulc.png", dpi=150)
        print(f"wrote {FIG / 'al_honest_aulc.png'}")
    else:
        print(f"thesis copy skipped: {FIG.parent} not present")


if __name__ == "__main__":
    main()
