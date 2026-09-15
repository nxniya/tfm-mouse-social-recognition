#!/usr/bin/env python3
"""
scripts/lb_compare_calib.py
===========================
Compares two runs lab by lab, using their **calibration** score.

Why calibration and not test. Since v8 training holds out `--test-frac 0.02`, that is
one or two videos per lab, so the test score a run prints is noise: ElegantMink came
out at 0.0151 on a single video. Calibration, by contrast, is computed over tens of
thousands of windows per lab, with out-of-sample k-fold on the smaller ones. It is the
only internal indicator left that means anything.

When the comparison is valid. `split_videos` draws per lab in loop order, so **two
runs are only comparable when they cover the same set of labs**. The script checks
this and warns when they do not.

Uso:
  .venv/Scripts/python.exe scripts/lb_compare_calib.py lb19_v8 lb19_v9w32
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_r = pathlib.Path(__file__).resolve().parents[1]
RESULTS = _r / "results"


def load(run: str, model: str) -> dict:
    p = RESULTS / "{}_calib_{}.json".format(run, model)
    if not p.exists():
        raise SystemExit("no existe {}".format(p))
    d = json.load(open(p))
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="reference run, e.g. lb19_v8")
    ap.add_argument("new", help="new run, e.g. lb19_v9w32")
    ap.add_argument("--model", default="histgb")
    a = ap.parse_args()

    A, B = load(a.base, a.model), load(a.new, a.model)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    if only_a or only_b:
        print("WARNING: the lab sets differ, so `split_videos` produced a "
              "different split and the comparison is NOT valid.")
        if only_a:
            print("  only in {}: {}".format(a.base, ", ".join(only_a)))
        if only_b:
            print("  only in {}: {}".format(a.new, ", ".join(only_b)))
        print()

    common = sorted(set(A) & set(B), key=lambda k: B[k] - A[k])
    if not common:
        raise SystemExit("no labs in common")

    print("%-24s %9s %9s %9s" % ("lab", a.base, a.new, "delta"))
    print("-" * 54)
    up = 0
    for k in common:
        d = B[k] - A[k]
        up += d > 0
        print("%-24s %9.4f %9.4f %+9.4f%s" % (
            k, A[k], B[k], d, "  <<<" if abs(d) >= 0.02 else ""))
    print("-" * 54)
    ma = sum(A[k] for k in common) / len(common)
    mb = sum(B[k] for k in common) / len(common)
    print("%-24s %9.4f %9.4f %+9.4f" % ("MEAN", ma, mb, mb - ma))
    print("\n{} of {} labs improve".format(up, len(common)))

    # Optimal composition: take, lab by lab, whichever of the two runs is better.
    # This is legitimate because each lab is scored only against its own videos, so
    # the coordinates are separable. Same argument that justifies tuning the
    # thresholds per lab.
    best = sum(max(A[k], B[k]) for k in common) / len(common)
    print("composing the best of each: {:.4f}  ({:+.4f} over {})".format(
        best, best - ma, a.base))
    win = [k for k in common if B[k] > A[k]]
    if win:
        print("\nlabs where {} wins:".format(a.new))
        for k in sorted(win, key=lambda k: A[k] - B[k]):
            print("   {:22s} {:+.4f}".format(k, B[k] - A[k]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
