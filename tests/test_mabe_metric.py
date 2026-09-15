"""D5 — official MABe event metric + window->interval bridge.

The scoring core is validated against the competition's own doctests (the exact
cases shipped in the Kaggle metric / the XGBoost baseline notebook); the bridge
is validated on synthetic windows.
"""
import json

from src.eval.mabe_metric import mouse_fbeta_records, windows_to_submission


def _S(v, a, t, act, s, e, lab, bl):
    return dict(video_id=v, agent_id=a, target_id=t, action=act,
                start_frame=s, stop_frame=e, lab_id=lab, behaviors_labeled=bl)


def _P(v, a, t, act, s, e):
    return dict(video_id=v, agent_id=a, target_id=t, action=act,
                start_frame=s, stop_frame=e)


def test_official_doctests():
    cases = [
        ([_S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]')],
         [_P(1, 1, 2, "attack", 0, 10)], 1.0),
        ([_S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]')],
         [_P(1, 1, 2, "mount", 0, 10)], 0.0),
        ([_S(123, 1, 2, "attack", 0, 9, 1, '["1,2,attack"]'),
          _S(123, 1, 2, "mount", 15, 24, 1, '["1,2,attack"]')],
         [_P(123, 1, 2, "attack", 0, 9)], 0.5),
        ([_S(123, 1, 2, "attack", 0, 9, 1, '["1,2,attack"]'),
          _S(123, 1, 2, "mount", 15, 24, 1, '["1,2,attack"]'),
          _S(345, 1, 2, "attack", 0, 9, 2, '["1,2,attack"]'),
          _S(345, 1, 2, "mount", 15, 24, 2, '["1,2,attack"]')],
         [_P(123, 1, 2, "attack", 0, 9)], 0.25),
        ([_S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]'),
          _S(1, 1, 2, "attack", 10, 20, 1, '["1,2,attack"]')],
         [_P(1, 1, 2, "attack", 0, 20)], 1.0),
        ([_S(1, 1, 2, "attack", 0, 10, 1, '["1,2,attack"]'),
          _S(1, 1, 2, "attack", 30, 40, 1, '["1,2,attack"]')],
         [_P(1, 1, 2, "attack", 0, 40)], 2 / 3),
    ]
    for sol, sub, expected in cases:
        assert abs(mouse_fbeta_records(sol, sub) - expected) < 1e-9


def test_windows_merge_consecutive_same_action():
    sub = windows_to_submission(
        win_video=["v1"] * 5, win_agent=[1] * 5, win_target=[2] * 5,
        win_fstart=[0, 10, 20, 30, 40], win_fstop=[10, 20, 30, 40, 50],
        y_pred_names=["sniff", "sniff", "sniff", "sniff", "background"])
    assert len(sub) == 1
    assert (sub[0]["start_frame"], sub[0]["stop_frame"]) == (0, 40)


def test_bridge_perfect_and_wrong():
    meta = dict(win_video=["v1"] * 5, win_agent=[1] * 5, win_target=[2] * 5,
                win_fstart=[0, 10, 20, 30, 40], win_fstop=[10, 20, 30, 40, 50])
    sol = [_S("v1", 1, 2, "sniff", 0, 40, "labA", json.dumps(["1,2,sniff"]))]

    perfect = windows_to_submission(y_pred_names=["sniff"] * 4 + ["background"], **meta)
    assert abs(mouse_fbeta_records(sol, perfect) - 1.0) < 1e-9

    wrong = windows_to_submission(y_pred_names=["mount"] * 4 + ["background"], **meta)
    assert abs(mouse_fbeta_records(sol, wrong) - 0.0) < 1e-9


def test_background_only_is_empty():
    sub = windows_to_submission(
        win_video=["v1"] * 3, win_agent=[1] * 3, win_target=[2] * 3,
        win_fstart=[0, 10, 20], win_fstop=[10, 20, 30],
        y_pred_names=["background"] * 3)
    assert sub == []
