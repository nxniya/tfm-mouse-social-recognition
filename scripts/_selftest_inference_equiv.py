#!/usr/bin/env python3
"""
scripts/_selftest_inference_equiv.py
====================================
Checks that the **real inference** path (`src/submit/predict.py`, with `mouseN`/`self`
ids) scores the same as the **simulation** (cache windows, integer ids) over the same
videos.

Why it matters. These are two separate implementations of the same decision: the
simulation works from the already-built cache, while inference recomputes features
from the parquet and formats the ids the way the competition expects. If they diverge,
the score measured locally stops predicting anything about the submission — and the
divergence raises no error, it just yields a different number.

It runs over a few test videos per lab, which is enough to catch a broken port (0
rows, wrong ids, a misaligned decision layer) without paying the hours the full split
would cost.

Ejecutar:
  .venv/Scripts/python.exe scripts/_selftest_inference_equiv.py --per-lab 2
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.loader import DATASET_DIR, load_annotations, load_tracking  # noqa: E402
from src.eval.mabe_metric import mouse_fbeta_records  # noqa: E402
from src.submit.predict import load_bundles, predict_video  # noqa: E402
from scripts.run_lb_perlab import load_full  # noqa: E402
from scripts.run_lb_experiment import split_videos, video_coverage  # noqa: E402

_r = pathlib.Path(__file__).resolve().parents[1]
MOUSE = "mouse{}".format


def solution_rows(vid, lab, bl):
    """One video's official solution, with ids in the behaviors_labeled format."""
    ann = load_annotations(vid, lab)
    if ann is None or not len(ann):
        return []
    out = []
    for r in ann.to_dict("records"):
        a, t = int(r["agent_id"]), int(r["target_id"])
        out.append({"video_id": str(vid), "agent_id": MOUSE(a),
                    "target_id": "self" if t == a else MOUSE(t),
                    "action": str(r["action"]),
                    "start_frame": int(r["start_frame"]),
                    "stop_frame": int(r["stop_frame"]),
                    "lab_id": lab, "behaviors_labeled": bl})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v4")
    ap.add_argument("--models", default="models/lb19")
    ap.add_argument("--per-lab", type=int, default=2)
    ap.add_argument("--labs", nargs="+", default=None)
    ap.add_argument("--frame-limit", type=int, default=0,
                    help="clip both video AND solution to this many frames, to "
                         "compare against a cache built with the same cap")
    a = ap.parse_args()

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix)
    del X
    train_v, test_v, lab_of = split_videos(vids, labs)
    cov = video_coverage(vids, ev["fstop"])
    meta = pd.read_csv(DATASET_DIR / "train.csv")
    meta["video_id"] = meta["video_id"].astype(str)
    bl_of = dict(zip(meta["video_id"], meta["behaviors_labeled"]))

    bundles = load_bundles(a.models)
    by_lab = {}
    for v in test_v:
        by_lab.setdefault(lab_of[v], []).append(v)

    sol_all, sub_all = [], []
    t0 = time.perf_counter()
    for lab in sorted(by_lab):
        if a.labs and lab not in a.labs:
            continue
        b = bundles.get(lab)
        if b is None:
            continue
        for vid in sorted(by_lab[lab])[:a.per_lab]:
            bl = bl_of.get(str(vid))
            if bl is None:
                continue
            sol = solution_rows(vid, lab, bl)
            if not sol:
                continue
            tr = load_tracking(vid, lab)
            rows = predict_video(lab, str(vid), tr, bl, b,
                                 frame_limit=a.frame_limit or None)
            if a.frame_limit:
                # The cache was built with a frame cap, so its solution covers only
                # that part of the video. To compare like with like, clip here too.
                sol = [r for r in sol if r["start_frame"] < a.frame_limit]
                for r in sol:
                    r["stop_frame"] = min(r["stop_frame"], a.frame_limit)
                rows = [r for r in rows if r["start_frame"] < a.frame_limit]
            sol_all += sol
            sub_all += rows
        print("   {:22s} {:4d} solution rows  {:4d} predicted  ({:.0f}s)".format(
            lab, sum(1 for r in sol_all if r["lab_id"] == lab),
            sum(1 for r in sub_all if r["video_id"] in
                {str(v) for v in by_lab[lab][:a.per_lab]}),
            time.perf_counter() - t0), flush=True)

    if not sol_all:
        print("no evaluable videos")
        return 1

    # ── The same video set, down the SIMULATION path ────────────────────────
    # This is the comparison that counts: same videos, same decision layer, but
    # starting from the cache and using integer ids. If the two numbers disagree,
    # the local score has stopped predicting anything about the submission.
    used = sorted({r["video_id"] for r in sol_all})
    sim = None
    try:
        from scripts.run_lb_experiment import (build_solution, build_submission,
                                               official_score)
        from scripts.lb_threshold import decide, build_active_mask
        from scripts.run_lb_decision import min_duration_filter, duration_stats
        d = np.load(_r / "results" / "lb19_pred_histgb.npz")
        P = d["proba_test"].astype(np.float32)
        te = d["test_mask"]
        thr_json = json.load(open(_r / "results" / "lb19_thr_histgb.json"))
        nC = len(classes); bg = classes.index("background")
        sol_s, active_s = build_solution(used, lab_of, cov)
        te_idx = np.flatnonzero(te)
        keep = np.isin(vids[te], np.array(used, dtype=object))
        allowed = build_active_mask(vids[te], ev["agent"][te], ev["target"][te],
                                    classes, active_s)
        pred = np.full(len(P), bg, np.int64)
        tl = labs[te]
        for lab in sorted(set(tl[keep].tolist())):
            sel = np.flatnonzero(keep & (tl == lab))
            t = np.asarray(thr_json.get(lab, [0.5] * nC), np.float32)
            pred[sel] = decide(P[sel], t, bg, mode="residual", allowed=allowed[sel])
        m = np.zeros(len(vids), bool); m[te_idx[np.flatnonzero(keep)]] = True
        sub_s = build_submission(m, vids, ev, pred[np.flatnonzero(keep)], classes,
                                 active=active_s)
        mind, maxg = duration_stats(used, lab_of, cov)
        sub_s = min_duration_filter(sub_s, lab_of, mind, maxg)
        sim = official_score(sol_s, sub_s)
    except Exception as e:  # noqa: BLE001
        print("could not compute the simulation path: {}: {}".format(
            type(e).__name__, e))
    score = mouse_fbeta_records(sol_all, sub_all)
    print("\nscore down the INFERENCE path (mouseN/self ids) = {:.4f}".format(score))
    print("  {} videos, {} rows emitted".format(
        len({r["video_id"] for r in sol_all}), len(sub_all)))
    # Sanity: every emitted triplet must be present in behaviors_labeled.
    active = {}
    for r in sol_all:
        active.setdefault(r["video_id"], set()).update(
            json.loads(str(r["behaviors_labeled"])))
    bad = [r for r in sub_all
           if "{},{},{}".format(r["agent_id"], r["target_id"], r["action"])
           not in active.get(r["video_id"], set())]
    print("  triplets outside behaviors_labeled: {}".format(len(bad)))
    if sim is not None:
        print()
        print("score down the SIMULATION path (integer ids) = {:.4f}".format(sim))
        print("difference, inference minus simulation = {:+.4f}".format(score - sim))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
