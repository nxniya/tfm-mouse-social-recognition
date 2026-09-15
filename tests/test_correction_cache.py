"""C2 — correction-cache key semantics."""
import pandas as pd
from src.data.correction_cache import params_hash, input_fingerprint


def _df(n_frames, f0=0):
    rows = [{"video_frame": f0 + i, "mouse_id": 0, "bodypart": "nose", "x": 1.0, "y": 2.0}
            for i in range(n_frames)]
    return pd.DataFrame(rows)


def test_hash_is_deterministic():
    kw = {"lambda_l": 12.0, "lambda_t": 0.8}
    fp = input_fingerprint(_df(100))
    assert params_hash("lab", kw, fp) == params_hash("lab", kw, fp)


def test_hash_changes_with_params():
    fp = input_fingerprint(_df(100))
    assert params_hash("lab", {"lambda_l": 12.0}, fp) != params_hash("lab", {"lambda_l": 99.0}, fp)


def test_hash_changes_with_lab():
    fp = input_fingerprint(_df(100))
    assert params_hash("labA", {}, fp) != params_hash("labB", {}, fp)


def test_slice_and_full_get_distinct_keys():
    # Same video id + params but different frame extent must not collide.
    kw = {"lambda_l": 12.0}
    h_slice = params_hash("lab", kw, input_fingerprint(_df(400)))
    h_full = params_hash("lab", kw, input_fingerprint(_df(5000)))
    assert h_slice != h_full


def test_fingerprint_fields():
    fp = input_fingerprint(_df(50, f0=10))
    assert fp["n_frames"] == 50 and fp["f_min"] == 10 and fp["f_max"] == 59
