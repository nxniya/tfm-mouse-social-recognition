"""A1/A2/B1 — LOVO harness metric + grouping helpers."""
import math

import numpy as np
import pandas as pd

import src.data.loader as loader
from src.evaluation.lovo import (
    fold_metrics, evaluable_class_ids, aggregate_windows, event_f1_for_fold,
)


def test_evaluable_classes_need_two_videos():
    # class 0 in vids {a,b}; class 1 only in {a}; class 2 in {b,c}
    y = np.array([0, 0, 1, 2, 2])
    vids = np.array(["a", "b", "a", "b", "c"])
    ids = evaluable_class_ids(y, vids, classes=["c0", "c1", "c2"])
    assert ids == [0, 2]                       # class 1 excluded (single video)


def test_aggregate_windows_shape():
    X = np.random.RandomState(0).randn(5, 8, 4).astype(np.float32)
    A = aggregate_windows(X)
    assert A.shape == (5, 16)                   # mean/std/min/max × 4 features


def test_fold_metrics_perfect_prediction():
    y = np.array([0, 0, 1, 1, 2, 2])
    per_class, support, f1_eval, f1_full = fold_metrics(
        y, y, n_classes=3, eval_ids=[0, 1, 2])
    assert f1_full == 1.0 and f1_eval == 1.0
    assert list(support) == [2, 2, 2]


def test_fold_metrics_eval_only_counts_present_eval_classes():
    # eval_ids includes class 2, but it has no test support → excluded from f1_eval
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    _, _, f1_eval, _ = fold_metrics(y_true, y_pred, n_classes=3, eval_ids=[0, 1, 2])
    assert f1_eval == 1.0                       # averaged over the 2 present eval classes


# ── D5: event/interval F-beta per fold (ported from run_lovo_benchmark) ──────
def _fold_setup():
    """4 contiguous 10-frame windows for (agent 1, target 2) in video 'v1'."""
    vids = np.array(["v1"] * 4)
    te = np.arange(4)
    ev = {"agent": np.array([1, 1, 1, 1]), "target": np.array([2, 2, 2, 2]),
          "fstart": np.array([0, 10, 20, 30]), "fstop": np.array([10, 20, 30, 40])}
    classes = ["background", "sniff"]
    return classes, vids, te, ev


def test_event_f1_for_fold_perfect(monkeypatch):
    classes, vids, te, ev = _fold_setup()
    ann = pd.DataFrame([{"agent_id": 1, "target_id": 2, "action": "sniff",
                         "start_frame": 0, "stop_frame": 40}])
    monkeypatch.setattr(loader, "load_annotations", lambda vid, lab: ann)
    pred = np.array([1, 1, 1, 1])               # all windows → "sniff"
    score = event_f1_for_fold("labA", "v1", te, pred, classes, vids, ev)
    assert abs(score - 1.0) < 1e-9


def test_event_f1_for_fold_wrong_action_is_zero(monkeypatch):
    classes, vids, te, ev = _fold_setup()
    ann = pd.DataFrame([{"agent_id": 1, "target_id": 2, "action": "sniff",
                         "start_frame": 0, "stop_frame": 40}])
    monkeypatch.setattr(loader, "load_annotations", lambda vid, lab: ann)
    pred = np.array([0, 0, 0, 0])               # all "background" → empty submission
    score = event_f1_for_fold("labA", "v1", te, pred, classes, vids, ev)
    assert score == 0.0


def test_event_f1_for_fold_missing_annotations_is_nan(monkeypatch):
    classes, vids, te, ev = _fold_setup()
    monkeypatch.setattr(loader, "load_annotations", lambda vid, lab: None)
    pred = np.array([1, 1, 1, 1])
    score = event_f1_for_fold("labA", "v1", te, pred, classes, vids, ev)
    assert math.isnan(score)
