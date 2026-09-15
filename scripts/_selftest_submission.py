#!/usr/bin/env python3
"""
scripts/_selftest_submission.py
===============================
Validates the submission's **identifier convention** and the CSV schema.

Why this is needed. The scorer builds the key `"{agent_id},{target_id},{action}"` and
looks it up in `behaviors_labeled` (`src/eval/mabe_metric.py:76`). If the format does
not match, the prediction **raises no error: it is discarded silently** and scores 0.
That is the most expensive failure a submission can have and the hardest to see, so it
is worth a test that catches it without needing any model.

Test A is the important one: both the solution AND the submission are built from the
SAME annotations, formatting the ids as the real file does (`mouse1`, `mouse2`,
`self`). If the convention is right the score must be exactly 1.0; if it were wrong it
would come out 0.0. The integer arm, which is the representation the internal
simulation uses, must give 0.0 precisely: it is the counter-proof that the convention
matters, and that a submission using integers would score zero without a single
warning.

Run with:  .venv/Scripts/python.exe scripts/_selftest_submission.py
"""
from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.loader import (DATASET_DIR, TRAIN_ANNOTATION_DIR,  # noqa: E402
                             load_annotations)
from src.eval.mabe_metric import (mouse_fbeta_records,  # noqa: E402
                                  solution_from_annotations)

MOUSE = "mouse{}".format


def _fmt_target(agent):
    """The target exactly as `behaviors_labeled` spells it: 'self' or 'mouseN'."""
    def f(t):
        return "self" if int(t) == int(agent) else MOUSE(int(t))
    return f


def _rows(ann, video_id, lab_id, bl, as_str):
    """Solution rows in the requested convention, strings or integers."""
    out = []
    for r in ann.to_dict("records"):
        a, t = int(r["agent_id"]), int(r["target_id"])
        if as_str:
            a_, t_ = MOUSE(a), ("self" if t == a else MOUSE(t))
        else:
            a_, t_ = a, ("self" if t == a else t)
        out.append({"video_id": str(video_id), "agent_id": a_, "target_id": t_,
                    "action": str(r["action"]), "start_frame": int(r["start_frame"]),
                    "stop_frame": int(r["stop_frame"]), "lab_id": lab_id,
                    "behaviors_labeled": bl})
    return out


def pick_videos(n_labs=6):
    """One annotated video per lab, preferring those with more than 2 mice."""
    meta = pd.read_csv(DATASET_DIR / "train.csv")
    meta["video_id"] = meta["video_id"].astype(str)
    idx = meta.set_index(["lab_id", "video_id"])
    picks = []
    labs = sorted(d.name for d in TRAIN_ANNOTATION_DIR.iterdir()
                  if d.is_dir() and any(d.glob("*.parquet")))
    # Multi-mouse first: those are the ones that exercise the self/pair convention
    labs.sort(key=lambda l: 0 if l in ("AdaptableSnail", "DeliriousFly",
                                       "ReflectiveManatee") else 1)
    for lab in labs[:n_labs]:
        for p in sorted((TRAIN_ANNOTATION_DIR / lab).glob("*.parquet")):
            try:
                bl = idx.loc[(lab, p.stem), "behaviors_labeled"]
            except KeyError:
                continue
            if isinstance(bl, pd.Series):
                bl = bl.iloc[0]
            picks.append((lab, p.stem, bl))
            break
    return picks


def main() -> int:
    picks = pick_videos()
    if not picks:
        print("no annotated videos: cannot validate")
        return 1
    ok = True

    # ── A. ground truth formatted as the submission must score exactly 1.0 ──
    print("A. identifier-convention round trip")
    for lab, vid, bl in picks:
        ann = load_annotations(vid, lab)
        if ann is None or not len(ann):
            continue
        # Strings must give 1.0. Integers must give 0.0: the real file's
        # `behaviors_labeled` uses "mouse1,mouse2,sniff", so a key "1,2,sniff" does
        # not match and the scorer discards it silently. The integer arm scoring 0 is
        # NOT a failure: it demonstrates that the convention matters, and that an
        # integer submission would score zero without a single warning.
        for as_str, label_txt, expected_sc in ((True, "mouseN/self strings", 1.0),
                                               (False, "integers (must give 0)", 0.0)):
            sol = _rows(ann, vid, lab, bl, as_str)
            sub = [{k: r[k] for k in ("video_id", "agent_id", "target_id", "action",
                                      "start_frame", "stop_frame")} for r in sol]
            sc = mouse_fbeta_records(sol, sub)
            good = abs(sc - expected_sc) < 1e-9
            ok &= good
            print("   {:22s} {:22s} score={:.4f} (expected {:.1f}) {}".format(
                lab, label_txt, sc, expected_sc, "OK" if good else "<-- FAILED"))
        # tripletas emitidas presentes en behaviors_labeled
        active = set(json.loads(str(bl)))
        sol = _rows(ann, vid, lab, bl, True)
        missing = {"{},{},{}".format(r["agent_id"], r["target_id"], r["action"])
                   for r in sol} - active
        if missing:
            print("      warning: {} annotated triplets are NOT in "
                  "behaviors_labeled (e.g. {})".format(len(missing), sorted(missing)[:2]))

    # ── B. the solution formatting helper, under the same convention ───────
    print("\nB. solution_from_annotations with agent_fmt/target_fmt")
    lab, vid, bl = picks[0]
    ann = load_annotations(vid, lab)
    sol = solution_from_annotations(
        ann, str(vid), lab, bl,
        agent_fmt=lambda a: MOUSE(int(a)),
        target_fmt=None)
    bad = [r for r in sol if not str(r["agent_id"]).startswith("mouse")]
    print("   agents formatted as mouseN: {}".format("OK" if not bad else "FAILED"))
    ok &= not bad

    # ── C. the exact sample_submission schema ──────────────────────────────
    print("\nC. submission.csv schema")
    ss = pd.read_csv(DATASET_DIR / "sample_submission.csv")
    expected = ["row_id", "video_id", "agent_id", "target_id", "action",
                "start_frame", "stop_frame"]
    good = list(ss.columns) == expected
    ok &= good
    print("   columns {} {}".format(list(ss.columns), "OK" if good else "<-- FAILED"))
    print("   example: {}".format(ss.iloc[0].to_dict()))

    print("\n{}".format("ALL OK" if ok else "FAILURES PRESENT"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
