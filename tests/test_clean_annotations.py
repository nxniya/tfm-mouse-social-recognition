"""C4 — annotation cleaning policy."""
import pandas as pd
from src.data.loader import clean_annotations


def _df():
    # row0 overlaps row1 (same agent); row2 inverted; row3 stop>maxframe; row4 ok
    return pd.DataFrame({
        "agent_id":    [0, 0, 0, 1, 0],
        "action":      ["a", "a", "a", "b", "a"],
        "start_frame": [-5, 50, 200, 10, 400],
        "stop_frame":  [60, 100, 150, 500, 450],
    })


def test_invariants_after_clean():
    clean, rep = clean_annotations(_df(), max_frame=420)
    assert (clean["start_frame"] < clean["stop_frame"]).all()      # no degenerate/inverted
    assert (clean["stop_frame"] <= 420).all()                       # clipped to length
    assert (clean["start_frame"] >= 0).all()                        # no negatives
    assert (clean["duration_frames"] == clean["stop_frame"] - clean["start_frame"]).all()


def test_report_counts():
    _, rep = clean_annotations(_df(), max_frame=420)
    assert rep["negative_start"] == 1
    assert rep["stop_exceeds_duration"] == 2     # rows 3 and 4
    assert rep["inverted_dropped"] == 1          # row 2
    assert rep["overlaps_truncated"] == 1        # row0 vs row1
    assert rep["n_before"] == 5 and rep["n_after"] == 4


def test_overlap_truncates_earlier_segment():
    clean, _ = clean_annotations(_df(), max_frame=420)
    agent0 = clean[clean["agent_id"] == 0].sort_values("start_frame")
    # first segment's stop pulled back to the next one's start (50)
    assert agent0.iloc[0]["stop_frame"] == 50


def test_empty_df_is_safe():
    out, rep = clean_annotations(pd.DataFrame(), max_frame=100)
    assert out.empty and rep["n_after"] == 0


def test_no_maxframe_skips_range_clip():
    _, rep = clean_annotations(_df(), max_frame=None)
    assert rep["stop_exceeds_duration"] == 0     # clip disabled without max_frame
