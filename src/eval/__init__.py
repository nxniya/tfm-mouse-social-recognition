"""Evaluation utilities for the MABe behaviour-detection task.

Exposes the official competition metric (event/interval-based F-beta) and the
helpers needed to turn per-window class predictions into the scored submission
format.  See ``mabe_metric`` for details.
"""
from src.eval.mabe_metric import (  # noqa: F401
    score,
    mouse_fbeta,
    mouse_fbeta_records,
    decode_intervals,
    robustify,
    window_frame_spans,
    windows_to_submission,
    solution_from_annotations,
)
