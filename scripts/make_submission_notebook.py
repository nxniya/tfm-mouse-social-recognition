#!/usr/bin/env python3
"""
scripts/make_submission_notebook.py
===================================
Generates `notebooks/08_kaggle_submission.ipynb`.

It is generated rather than hand-written because a notebook is JSON, and editing that
by hand invites breaking it; this way the cell contents live here as readable text.

The resulting notebook is **inference only**: it trains nothing. It loads the per-lab
models from an attached dataset, walks `test.csv`, and writes `submission.csv`. That
is what a code competition requires, where the real test set is substituted at run
time and the time limit leaves no room to retrain 19 labs.
"""
from __future__ import annotations

import json
import pathlib

_r = pathlib.Path(__file__).resolve().parents[1]

MD_INTRO = """\
# 08 - Submission notebook (inference only)

Builds `submission.csv` from the **per-laboratory** models trained locally. Nothing
is trained here: in a code competition the real test set is substituted at run time
and there is no budget to retrain 19 laboratories.

Four things decide whether a submission scores anything at all, and three of them
fail silently:

1. **Identifiers must be strings.** The scorer builds
   `"{agent_id},{target_id},{action}"` and looks it up in `behaviors_labeled`, which
   uses `mouse1`, `mouse2` and `self`. With integers every prediction is discarded
   **without warning** and the score is exactly 0. Verified in
   `scripts/_selftest_submission.py`.
2. **Only triplets present in that video's `behaviors_labeled` are emitted.**
3. **Ordered pairs.** With 3 to 5 mice, `mouse1->mouse3` has to be predicted as well
   as `mouse1->mouse2`.
4. **Background is a class**, so the model can abstain instead of emitting an
   interval for every window.
"""

CELL_SETUP = '''import os, sys, json, time, pathlib
import numpy as np
import pandas as pd

# ── Paths: hosted environment or local ──────────────────────────────────────
ON_KAGGLE = pathlib.Path("/kaggle/input").exists()
if ON_KAGGLE:
    IN = pathlib.Path("/kaggle/input")

    # The listing is printed ALWAYS, not only on failure: when something is
    # missing, this is the only thing that says whether an input was not attached
    # or whether it is mounted with a different structure. It costs nothing and
    # saves a whole run.
    print("Contents of /kaggle/input:")
    for d in sorted(IN.iterdir()):
        print("  -", d.name)
        try:
            for sub in sorted(d.iterdir())[:10]:
                print("       ", sub.name + ("/" if sub.is_dir() else ""))
        except Exception as e:
            print("        (not listable:", e, ")")

    # Everything is located by CONTENT, not by name: the dataset slug is chosen by
    # whoever uploads it, and the competition's can differ in capitalisation.
    _tc = next(iter(IN.rglob("test_tracking")), None)
    COMP = _tc.parent if _tc is not None else None
    if COMP is None:
        _t = next((f for f in IN.rglob("test.csv")), None)
        COMP = _t.parent if _t is not None else None

    _pred = next(iter(IN.rglob("submit/predict.py")), None)
    CODE_DIR = _pred.parents[2] if _pred else None      # .../src/submit/predict.py
    _jl = next(iter(IN.rglob("*.joblib")), None)
    MODELS_DIR = _jl.parent if _jl else None
    OUT = pathlib.Path("/kaggle/working/submission.csv")
    print("\\ndetected -> competition:", COMP, "| code:", CODE_DIR,
          "| models:", MODELS_DIR)
else:
    ROOT = pathlib.Path.cwd()
    if not (ROOT / "src").exists():
        ROOT = ROOT.parent
    COMP = ROOT / "dataset" / "MABe-mouse-behavior-detection"
    CODE_DIR = ROOT
    MODELS_DIR = ROOT / "models" / "lb19_mix_sub"
    OUT = ROOT / "outputs" / "submission.csv"
    OUT.parent.mkdir(parents=True, exist_ok=True)

assert CODE_DIR is not None, (
    "cannot find src/submit/predict.py: attach the dataset holding the code "
    "(Add Input) and check the listing above")
assert MODELS_DIR is not None, (
    "cannot find any *.joblib: the attached dataset does not carry the models")
assert COMP is not None and COMP.exists(), (
    "cannot find the competition data (test.csv + test_tracking/). In the "
    "notebook's Input panel press Add Input -> Competitions tab -> MABe "
    "Challenge, then run again. See the listing above for what is currently "
    "mounted.")
sys.path.insert(0, str(CODE_DIR))
os.environ["MABE_DATASET_DIR"] = str(COMP)

print("\\ncompetition :", COMP)
print("code        :", CODE_DIR)
print("models      :", MODELS_DIR)
print("output      :", OUT)
'''

CELL_LOAD = '''\
from src.submit.predict import load_bundles, predict_video

bundles = load_bundles(MODELS_DIR)
test = pd.read_csv(COMP / "test.csv")
test["video_id"] = test["video_id"].astype(str)

print("models loaded: {} laboratories".format(len(bundles)))
print("test.csv: {} videos, {} laboratories".format(len(test), test.lab_id.nunique()))
missing_model = sorted(set(test.lab_id) - set(bundles))
if missing_model:
    # No predictions are invented for a lab with no model: nothing is emitted and
    # the fact is recorded. That lab scores 0, which is the honest cost.
    print("NO MODEL (nothing will be emitted):", missing_model)
'''

CELL_PREDICT = '''\
rows, t0, no_model = [], time.perf_counter(), 0
for i, r in enumerate(test.itertuples(), 1):
    b = bundles.get(r.lab_id)
    if b is None:
        no_model += 1
        continue
    p = COMP / "test_tracking" / r.lab_id / "{}.parquet".format(r.video_id)
    if not p.exists():
        continue
    try:
        tr = pd.read_parquet(p)
        rows += predict_video(r.lab_id, r.video_id, tr, r.behaviors_labeled, b)
    except Exception as e:
        # One failing video must not bring down the whole submission.
        print("  failed on {}/{}: {}: {}".format(r.lab_id, r.video_id,
                                                 type(e).__name__, e))
    if i % 25 == 0 or i == len(test):
        print("  [{}/{}] {} rows accumulated ({:.0f}s)".format(
            i, len(test), len(rows), time.perf_counter() - t0), flush=True)

print("\\ntotal: {} rows | {} videos without a model | {:.0f}s".format(
    len(rows), no_model, time.perf_counter() - t0))
'''

CELL_WRITE = '''\
cols = ["row_id", "video_id", "agent_id", "target_id", "action",
        "start_frame", "stop_frame"]

sub = pd.DataFrame(rows, columns=[c for c in cols if c != "row_id"]) if rows \\
      else pd.DataFrame(columns=[c for c in cols if c != "row_id"])
sub.insert(0, "row_id", np.arange(len(sub), dtype=np.int64))
sub["start_frame"] = sub["start_frame"].astype(np.int64)
sub["stop_frame"] = sub["stop_frame"].astype(np.int64)

# Checks before writing: the last chance to catch a mistake.
ss = pd.read_csv(COMP / "sample_submission.csv")
assert list(sub.columns) == list(ss.columns), (list(sub.columns), list(ss.columns))
assert sub.row_id.is_unique, "duplicate row_id"
assert (sub.stop_frame > sub.start_frame).all(), "empty or inverted intervals"

sub.to_csv(OUT, index=False)
print("wrote {} with {} rows".format(OUT, len(sub)))
sub.head(10)
'''


def cell(src, kind="code"):
    d = {"cell_type": kind, "metadata": {},
         "source": src.splitlines(keepends=True)}
    if kind == "code":
        d["execution_count"] = None
        d["outputs"] = []
    return d


def main():
    nb = {
        "cells": [
            cell(MD_INTRO, "markdown"),
            cell(CELL_SETUP),
            cell(CELL_LOAD),
            cell(CELL_PREDICT),
            cell(CELL_WRITE),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = _r / "notebooks" / "08_kaggle_submission.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
