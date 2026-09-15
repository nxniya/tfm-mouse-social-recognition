#!/usr/bin/env python3
"""
scripts/update_bundles.py
=========================
Fills in the model bundles with everything the **decision layer** needs at inference
time, without retraining anything.

The bundles the first submission produced held only the model and its thresholds, so
`predict_video` got no further than thresholding plus the active-class mask: no
minimum-duration filter, no revival of dead classes, and no per-lab decision variant.
Measured on the 19-lab split, that left about +0.011 on the table
(0.4583 -> 0.4689).

Fills in, per lab:
  * `min_dur` / `max_gap` per action, from `duration_stats` over the TRAIN videos;
  * `rates`, the per-action prevalence, from `train_rates`, also train only;
  * `choice` and the recalibrated thresholds, when a `run_lb_decide_perlab` run is
    passed via `--dec`.

None of this looks at the test labels: every statistic comes from the training set.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TQDM_DISABLE", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.run_lb_perlab import load_full  # noqa: E402
from scripts.run_lb_experiment import split_videos, video_coverage  # noqa: E402
from scripts.run_lb_decision import duration_stats  # noqa: E402
from scripts.lb_prevalence import train_rates  # noqa: E402

_r = pathlib.Path(__file__).resolve().parents[1]
RESULTS = _r / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v4")
    ap.add_argument("--models", default="models/lb19")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--dec", default=None,
                    help="run_lb_decide_perlab run to take `choice` and the "
                         "per-lab thresholds from")
    ap.add_argument("--full-train", action="store_true",
                    help="compute the statistics over ALL videos, for when the "
                         "model was trained without holding out a test set")
    a = ap.parse_args()

    import joblib

    X, y, vids, labs, classes, ev, agg = load_full(a.suffix)
    del X
    train_v, test_v, lab_of = split_videos(vids, labs)
    if a.full_train:
        train_v = sorted(set(vids.tolist()))
    cov = video_coverage(vids, ev["fstop"])
    print("statistics over {} train videos".format(len(train_v)), flush=True)

    mind, maxg = duration_stats(train_v, lab_of, cov)
    rates, _ = train_rates(train_v, lab_of, cov)

    choice, thr_new = {}, {}
    if a.dec:
        choice = json.load(open(RESULTS / "{}.json".format(a.dec)))["choice"]
        tf = RESULTS / "{}_thr_{}.json".format(a.dec, a.model)
        if tf.exists():
            thr_new = json.load(open(tf))
        print("decisions taken from {}: {} labs".format(a.dec, len(choice)))

    md = pathlib.Path(a.models)
    n = 0
    for p in sorted(md.glob("*.joblib")):
        b = joblib.load(p)
        lab = b["lab"]
        b["min_dur"] = {act: v for (l_, act), v in mind.items() if l_ == lab}
        b["max_gap"] = {act: v for (l_, act), v in maxg.items() if l_ == lab}
        b["rates"] = {act: v for (l_, act), v in rates.items() if l_ == lab}
        if lab in choice:
            b["choice"] = choice[lab]
        if lab in thr_new:
            b["thresholds"] = np.asarray(thr_new[lab], np.float32)
        joblib.dump(b, p, compress=3)
        n += 1
        print("   {:22s} min_dur={:2d} rates={:2d} choice={}".format(
            lab, len(b["min_dur"]), len(b["rates"]),
            b.get("choice", {}).get("mode", "-")), flush=True)
    print("\nupdated {} bundles in {}".format(n, md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
