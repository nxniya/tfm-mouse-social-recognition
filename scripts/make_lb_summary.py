"""
scripts/make_lb_summary.py
==========================
Consolidates the leaderboard-simulation CSVs into `results/lb_summary.md`.
Run at the end of `run_lb_experiment.py --stage all`, or by hand.
"""
from __future__ import annotations
import sys, pathlib
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))
import pandas as pd

RES = _r / "results"


def _read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def _md(df, index=True):
    """Markdown when `tabulate` is available, otherwise a plain text block."""
    try:
        return df.to_markdown(index=index)
    except ImportError:
        nl = chr(10)
        return nl.join(["```", df.to_string(index=index), "```"])


def main():
    out = ["# MABe leaderboard simulation — summary", "",
           "Protocol: multi-lab corpus (9 labs, the 7-keypoint CalMS21 schema), split by",
           "video and stratified by lab, **official** metric (F1 per action -> mean per",
           "lab -> mean across labs). Every block is run over both **raw** tracking and",
           "tracking **corrected with MouseSkeleton**.", ""]

    mods = _read("lb_models.csv")
    if mods is not None:
        out += ["## Models (official metric)", ""]
        piv = mods.pivot(index="model", columns="condition", values="f1_event_official")
        piv["delta_corr_raw"] = (piv.get("corr") - piv.get("raw")).round(4)
        out += [_md(piv.round(4)), ""]
        best = mods.loc[mods.f1_event_official.idxmax()]
        out += ["Best: **{} / {}** = {:.4f}".format(
            best.condition, best.model, best.f1_event_official), ""]

    bs = _read("lb_bootstrap.csv")
    if bs is not None:
        out += ["## Public (32%) vs private (68%)", ""]
        cols = ["condition", "model", "score_full", "score_public_32",
                "score_private_68", "boot_ci_lo", "boot_ci_hi", "boot_range"]
        out += [_md(bs[cols].round(4), index=False), ""]
        gap = (bs.score_public_32 - bs.score_private_68).abs()
        spread = bs.groupby("condition").score_full.agg(lambda s: s.max() - s.min())
        out += ["- |public - private|: mean {:.4f}, max {:.4f}".format(gap.mean(), gap.max()),
                "- Mean width of the public 95% CI: {:.4f}".format(bs.boot_range.mean()),
                "- Spread between models: {}".format(spread.round(4).to_dict()),
                "", "> When the CI width exceeds the spread between models, the public",
                "> leaderboard ordering does not distinguish architectures.", ""]

    geo = _read("lb_geometry.csv")
    if geo is not None and "viol_raw" in geo:
        ok = geo[geo.viol_raw.notna()]
        out += ["## O1 - Geometry", "",
                "Violations: raw **{:.4f}** -> corr **{:.4f}** ({:+.2f} p.p., {} videos)".format(
                    ok.viol_raw.mean(), ok.viol_corr.mean(), ok.delta_pp.mean(), len(ok)), ""]

    al = _read("lb_al_summary.csv")
    if al is not None:
        out += ["## O3 - Active learning (AULC)", "",
                _md(al.round(4), index=False), ""]
        w = al[(al.strategy != "random") & al.wilcoxon_p.notna() &
               (al.wilcoxon_p < 0.05) & (al.delta_vs_random > 0)]
        out += ["Strategies significantly beating random: **{}**".format(
            "none" if w.empty else ", ".join(w.strategy + " (" + w.condition + ")")), ""]

    cl = _read("lb_crosslab.csv")
    if cl is not None:
        out += ["## O4 - Cross-lab (mean off the diagonal)", ""]
        off = cl[cl.train_lab != cl.test_lab]
        piv = off.pivot_table(index="train_lab", columns="condition",
                              values="official", aggfunc="mean")
        out += [_md(piv.round(4)), "",
                "Mean action overlap between different labs: {:.2f}".format(
                    off.n_common_actions.mean()), ""]

    rt = _read("lb_runtime_cache.csv")
    if rt is not None:
        agg = rt.groupby("variant").agg(videos=("video_id", "nunique"),
                                        frames=("n_frames", "sum"),
                                        total_s=("elapsed_s", "sum"),
                                        ms_per_frame=("ms_per_frame", "mean")).round(2)
        out += ["## Computational cost", "", _md(agg), ""]
        if mods is not None:
            out += ["Training plus inference per model (s):", "",
                    mods.pivot(index="model", columns="condition",
                               values="fit_predict_s").round(1).pipe(_md), ""]

    (RES / "lb_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("Wrote", RES / "lb_summary.md")


if __name__ == "__main__":
    main()
