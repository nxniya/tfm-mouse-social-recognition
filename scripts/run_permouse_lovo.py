"""
scripts/run_permouse_lovo.py
============================
Runs the paired BiLSTM LOVO harness over three feature caches and measures, per
video, the effect of the correction against the raw signal:

  * raw            — no correction
  * corrected      — correction with a POOLED skeleton, one shared reference  [base]
  * permouse       — correction with a PER-MOUSE skeleton                     [ablation]

For each metric (f1_eval, f1_full, f1_event) it reports the paired delta,
variant minus raw, with a bootstrap CI and a paired Wilcoxon test. If the per-mouse
reference closes or reverses the gap left by the pooled one, the hypothesis holds.

Each (cache, fold) is an independent training run, so they are parallelised across a
process pool, with torch pinned to one thread per worker. Same protocol, seed and
EPOCHS as notebook 05_deep_model.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np, pandas as pd

LAB = "CalMS21_task1"
FEAT = _r / "dataset" / "features"
SEQ_MODEL = "bilstm"
EPOCHS = int(os.environ.get("LOVO_EPOCHS", "20"))
N_WORKERS = int(os.environ.get("LOVO_WORKERS", "12"))
CACHES = {
    "raw":       FEAT / "CalMS21_task1_multi_20_event_raw.npz",
    "corrected": FEAT / "CalMS21_task1_multi_20_event.npz",
    "permouse":  FEAT / "CalMS21_task1_multi_20_event_permouse.npz",
}


def _fold_worker(args):
    """Train one LOVO fold of one cache and return its row of metrics."""
    import torch
    torch.set_num_threads(1)
    from sklearn.model_selection import LeaveOneGroupOut
    from src.evaluation.lovo import (
        load_event_cache, evaluable_class_ids, fold_metrics, run_seq, event_f1_for_fold)
    cache_path, tag, fi = args
    X, y, vids, classes, ev = load_event_cache(cache_path)
    nC = len(classes); eids = evaluable_class_ids(y, vids, classes)
    splits = list(LeaveOneGroupOut().split(X, y, groups=vids))
    tr, te = splits[fi]
    pred, _ = run_seq(SEQ_MODEL, X[tr], y[tr], X[te], nC, epochs=EPOCHS)
    _, _, fe, ff = fold_metrics(y[te], pred, nC, eids)
    row = {"test_video": vids[te][0], "f1_eval": round(fe, 4), "f1_full": round(ff, 4)}
    if ev is not None:
        fev = event_f1_for_fold(LAB, vids[te][0], te, pred, classes, vids, ev)
        row["f1_event"] = round(fev, 4) if not np.isnan(fev) else np.nan
    return tag, fi, row


def _n_folds(cache_path):
    from src.evaluation.lovo import load_event_cache
    X, y, vids, classes, ev = load_event_cache(cache_path)
    return len(np.unique(vids))


def paired_verdict(df_raw, df_var, tag):
    metrics = [m for m in ("f1_eval", "f1_full", "f1_event") if m in df_var.columns]
    from src.skeleton.metrics import bootstrap_ci, paired_wilcoxon
    out = []
    for metric in metrics:
        m = (df_raw[["test_video", metric]].merge(
                df_var[["test_video", metric]], on="test_video", suffixes=("_raw", "_var"))
             .dropna(subset=[f"{metric}_raw", f"{metric}_var"]))
        if len(m) < 3:
            continue
        raw_v = m[f"{metric}_raw"].to_numpy(); var_v = m[f"{metric}_var"].to_numpy()
        delta = var_v - raw_v
        lo, hi = bootstrap_ci(delta, statistic="mean")
        wil = paired_wilcoxon(raw_v, var_v)
        out.append({
            "variant": tag, "metric": metric, "n_pairs": len(m),
            "raw_mean": round(float(raw_v.mean()), 4),
            "var_mean": round(float(var_v.mean()), 4),
            "delta_mean": round(float(delta.mean()), 4),
            "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
            "wilcoxon_p": round(float(wil["p_value"]), 4),
            "helps": bool(delta.mean() > 0 and (lo > 0 or hi < 0) and wil["significant"]),
        })
    return out


def main():
    from src.evaluation.lovo import load_event_cache  # noqa: F401 (import check)
    missing = [t for t, p in CACHES.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing caches: {missing}. Run scripts/build_permouse_cache.py first.")

    nf = _n_folds(str(CACHES["raw"]))
    tasks = [(str(p), tag, fi) for tag, p in CACHES.items() for fi in range(nf)]
    print(f"=== BiLSTM LOVO (EPOCHS={EPOCHS}) — raw / corrected(pooled) / permouse ===")
    print(f"{len(CACHES)} caches x {nf} folds = {len(tasks)} training runs, {N_WORKERS} workers", flush=True)

    results = {tag: {} for tag in CACHES}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_fold_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            tag, fi, row = fut.result()
            results[tag][fi] = row
            done += 1
            print(f"  [{done:2d}/{len(tasks)}] {tag:9s} fold{fi:02d} {row['test_video']:>10s} "
                  f"f1_eval={row['f1_eval']} f1_event={row.get('f1_event')}  ({time.time()-t0:.0f}s)", flush=True)

    dfs = {tag: pd.DataFrame([results[tag][fi] for fi in sorted(results[tag])]) for tag in CACHES}
    for tag, df in dfs.items():
        mrow = {m: round(float(df[m].mean()), 4) for m in ("f1_eval", "f1_full", "f1_event") if m in df.columns}
        print(f"\n[{tag}] means over folds: {mrow}")

    verdicts = paired_verdict(dfs["raw"], dfs["corrected"], "corrected(pooled)")
    verdicts += paired_verdict(dfs["raw"], dfs["permouse"], "permouse")
    res = pd.DataFrame(verdicts)
    print("\n=== Effect of the correction vs raw (per video, paired) ===")
    print(res.to_string(index=False))
    out = _r / "results" / "skeleton_permouse_ablation.csv"
    res.to_csv(out, index=False)
    for tag, df in dfs.items():
        df.to_csv(_r / "results" / f"skeleton_permouse_lovo_{tag}.csv", index=False)
    print("\nSaved:", out, flush=True)


if __name__ == "__main__":
    main()
