"""
src/data/loader.py
==================
Efficient loading of the MABe Mouse Behavior Detection dataset: the tracking
and annotation Parquet files, and the metadata CSV.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
# In a hosted notebook the dataset lives outside the repository, so the location
# is overridable through an environment variable rather than by editing code.
# The default is the usual local path.
DATASET_DIR = Path(os.environ.get(
    "MABE_DATASET_DIR",
    _REPO_ROOT / "dataset" / "MABe-mouse-behavior-detection"))

TRAIN_CSV = DATASET_DIR / "train.csv"
TEST_CSV = DATASET_DIR / "test.csv"
TRAIN_TRACKING_DIR = DATASET_DIR / "train_tracking"
TRAIN_ANNOTATION_DIR = DATASET_DIR / "train_annotation"
TEST_TRACKING_DIR = DATASET_DIR / "test_tracking"


# ---------------------------------------------------------------------------
# Metadatos
# ---------------------------------------------------------------------------

def load_metadata(split: str = "train") -> pd.DataFrame:
    """Load the metadata CSV, either train.csv or test.csv.

    The ``body_parts_tracked`` and ``behaviors_labeled`` columns hold JSON
    lists and are parsed automatically.

    Parameters
    ----------
    split : {"train", "test"}

    Returns
    -------
    pd.DataFrame
        One row per video, plus three derived columns: ``n_body_parts``,
        ``n_behaviors_labeled`` and ``n_mice``, the last being the number of
        mouse1 to mouse4 slots that carry data.
    """
    path = TRAIN_CSV if split == "train" else TEST_CSV
    df = pd.read_csv(path, low_memory=False)

    for col in ("body_parts_tracked", "behaviors_labeled"):
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    # Derived columns, useful during the exploratory analysis
    df["n_body_parts"] = df["body_parts_tracked"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    df["n_behaviors_labeled"] = df["behaviors_labeled"].apply(
        lambda v: len(v) if isinstance(v, list) else 0
    )
    # Mice with at least a strain or a sex recorded
    mouse_cols = [f"mouse{i}_strain" for i in range(1, 5)]
    df["n_mice"] = df[mouse_cols].notna().sum(axis=1)

    return df


def _safe_parse_list(value) -> list:
    """Parse a JSON or Python literal string into a list; return [] on failure."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def load_tracking(
    video_id: str | int,
    lab_id: str,
    split: str = "train",
    normalize_coords: bool = False,
    arena_width_pix: Optional[float] = None,
    arena_height_pix: Optional[float] = None,
) -> pd.DataFrame:
    """Load one video's tracking Parquet file.

    Schema returned
    ---------------
    video_frame : int32
    mouse_id    : int8
    bodypart    : str
    x           : float32
    y           : float32

    Parameters
    ----------
    video_id : str or int
        Video identifier, which is the file name without its extension.
    lab_id : str
        Lab name, which is the subdirectory inside train_tracking/.
    split : {"train", "test"}
    normalize_coords : bool
        When True, scale x and y to [0, 1] using ``arena_width_pix`` and
        ``arena_height_pix``. Both must then be supplied.
    arena_width_pix : float, optional
        Arena width in pixels, the CSV's ``video_width_pix`` column.
    arena_height_pix : float, optional
        Arena height in pixels, the CSV's ``video_height_pix`` column.

    Returns
    -------
    pd.DataFrame
    """
    base_dir = TRAIN_TRACKING_DIR if split == "train" else TEST_TRACKING_DIR
    path = base_dir / lab_id / f"{video_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Tracking file not found: {path}")

    df = pq.read_table(str(path)).to_pandas()

    if normalize_coords:
        if arena_width_pix is None or arena_height_pix is None:
            raise ValueError(
                "Provide arena_width_pix and arena_height_pix to normalize."
            )
        df["x"] = df["x"] / arena_width_pix
        df["y"] = df["y"] / arena_height_pix

    return df


def load_tracking_wide(
    video_id: str | int,
    lab_id: str,
    split: str = "train",
    normalize_coords: bool = False,
    arena_width_pix: Optional[float] = None,
    arena_height_pix: Optional[float] = None,
) -> pd.DataFrame:
    """Return the tracking in wide format: one row per (frame, mouse_id).

    The columns are ``{bodypart}_x`` and ``{bodypart}_y`` for each body part the
    lab tracks, plus ``video_frame`` and ``mouse_id``.

    This is the layout the feature engineering downstream expects.
    """
    df_long = load_tracking(
        video_id,
        lab_id,
        split=split,
        normalize_coords=normalize_coords,
        arena_width_pix=arena_width_pix,
        arena_height_pix=arena_height_pix,
    )

    df_wide = df_long.pivot_table(
        index=["video_frame", "mouse_id"],
        columns="bodypart",
        values=["x", "y"],
        aggfunc="first",
    )
    # Flatten the MultiIndex columns into "nose_x", "nose_y" and so on
    df_wide.columns = [f"{bp}_{coord}" for coord, bp in df_wide.columns]
    df_wide = df_wide.reset_index().sort_values(["mouse_id", "video_frame"])
    return df_wide


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def clean_annotations(
    df: pd.DataFrame,
    max_frame: Optional[int] = None,
    group_cols: tuple = ("agent_id",),
) -> tuple[pd.DataFrame, dict]:
    """C4 — Resolve the data-quality issues found in EDA §5 (conservative clip).

    The EDA flagged 356 consistency problems (overlaps, ``stop_frame`` beyond the
    video length, inverted/negative frames) but they were never fixed, even though
    overlapping/out-of-range annotations corrupt the window labels downstream.
    This applies a conservative *clip/truncate* policy that preserves as much
    labelled data as possible:

    * ``start_frame < 0``                       → clip start to 0
    * ``stop_frame  > max_frame`` (if given)    → clip stop to ``max_frame``
    * ``start_frame > stop_frame`` (inverted)   → drop the row
    * overlapping segments of the same agent    → truncate the earlier segment's
      ``stop_frame`` down to the later segment's ``start_frame``
    * any row left degenerate (``start >= stop``) after clipping → drop

    Parameters
    ----------
    df : raw annotations (needs ``start_frame``/``stop_frame``; ``agent_id`` optional).
    max_frame : video length in frames; if None, the out-of-range clip is skipped.
    group_cols : columns defining "same actor" for overlap resolution. Falls back
        to global ordering if the columns are absent.

    Returns
    -------
    (cleaned_df, report) — report counts: negative_start, stop_exceeds_duration,
    inverted_dropped, overlaps_truncated, degenerate_dropped, n_before, n_after.
    """
    report = {"negative_start": 0, "stop_exceeds_duration": 0,
              "inverted_dropped": 0, "overlaps_truncated": 0,
              "degenerate_dropped": 0,
              "n_before": int(len(df)), "n_after": int(len(df))}
    if df.empty or not {"start_frame", "stop_frame"}.issubset(df.columns):
        return df.copy(), report

    df = df.copy()
    df["start_frame"] = df["start_frame"].astype("int64")
    df["stop_frame"] = df["stop_frame"].astype("int64")

    neg = df["start_frame"] < 0
    report["negative_start"] = int(neg.sum())
    df.loc[neg, "start_frame"] = 0

    if max_frame is not None:
        over = df["stop_frame"] > int(max_frame)
        report["stop_exceeds_duration"] = int(over.sum())
        df.loc[over, "stop_frame"] = int(max_frame)

    inverted = df["start_frame"] > df["stop_frame"]
    report["inverted_dropped"] = int(inverted.sum())
    df = df[~inverted].copy()

    # Overlap truncation within each actor group, ordered by start_frame.
    gcols = [c for c in group_cols if c in df.columns]
    keys = ([df[c] for c in gcols] if gcols else [pd.Series(0, index=df.index)])
    df["_g"] = list(zip(*keys)) if gcols else 0
    df = df.sort_values(["_g", "start_frame"] if gcols else ["start_frame"]).reset_index(drop=True)
    for _, idx in (df.groupby("_g").groups.items() if gcols else [(0, df.index)]):
        idx = list(idx)
        for a, b in zip(idx, idx[1:]):
            if df.at[a, "stop_frame"] > df.at[b, "start_frame"]:
                df.at[a, "stop_frame"] = df.at[b, "start_frame"]
                report["overlaps_truncated"] += 1
    df = df.drop(columns="_g")

    degenerate = df["start_frame"] >= df["stop_frame"]
    report["degenerate_dropped"] = int(degenerate.sum())
    df = df[~degenerate].copy()

    df["duration_frames"] = df["stop_frame"] - df["start_frame"]
    report["n_after"] = int(len(df))
    return df.reset_index(drop=True), report


def load_annotations(
    video_id: str | int,
    lab_id: str,
    clean: bool = True,
    max_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Load one video's behaviour-annotation Parquet file.

    Schema returned
    ---------------
    agent_id    : int8   — the mouse performing the behaviour
    target_id   : int8   — the mouse it is performed on; equal to the agent for
                           a self-directed behaviour such as selfgroom
    action      : str    — behaviour label
    start_frame : int32
    stop_frame  : int32
    duration_frames : int32  (derived column)

    Parameters
    ----------
    clean : bool
        When True, the default, run :func:`clean_annotations` first, which
        resolves overlaps, inverted intervals and out-of-range frames. Pass
        ``clean=False`` to get the raw annotations.
    max_frame : int | None
        Video length in frames. Supplying it lets the cleaning step clip a
        ``stop_frame`` that runs past the end of the video.

    Returns
    -------
    pd.DataFrame, empty if the video has no annotations.
    """
    path = TRAIN_ANNOTATION_DIR / lab_id / f"{video_id}.parquet"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "agent_id", "target_id", "action",
                "start_frame", "stop_frame", "duration_frames",
            ]
        )

    df = pq.read_table(str(path)).to_pandas()
    df["duration_frames"] = df["stop_frame"] - df["start_frame"]
    if clean:
        df, _report = clean_annotations(df, max_frame=max_frame)
    return df


# ---------------------------------------------------------------------------
# Per-lab iterators
# ---------------------------------------------------------------------------

def iter_lab_videos(
    lab_id: str,
    split: str = "train",
    metadata: Optional[pd.DataFrame] = None,
) -> list[dict]:
    """List every video in a lab as a {video_id, lab_id, meta_row} dict.

    Parameters
    ----------
    lab_id : str
    split : {"train", "test"}
    metadata : pd.DataFrame, optional
        When given, each video's metadata row is attached to its dict.
    """
    base_dir = TRAIN_TRACKING_DIR if split == "train" else TEST_TRACKING_DIR
    lab_dir = base_dir / lab_id
    if not lab_dir.exists():
        raise FileNotFoundError(f"Lab directory not found: {lab_dir}")

    video_ids = [p.stem for p in lab_dir.glob("*.parquet")]

    if metadata is not None:
        meta_idx = metadata.set_index("video_id")
    else:
        meta_idx = None

    result = []
    for vid in video_ids:
        entry = {"video_id": vid, "lab_id": lab_id}
        if meta_idx is not None and int(vid) in meta_idx.index:
            entry["meta"] = meta_idx.loc[int(vid)]
        result.append(entry)

    return result


def load_lab_annotations(lab_id: str) -> pd.DataFrame:
    """Load and concatenate every annotation file of one lab.

    Adds a ``video_id`` column identifying the source video.

    Returns
    -------
    pd.DataFrame, empty if the lab has no annotations.
    """
    ann_dir = TRAIN_ANNOTATION_DIR / lab_id
    if not ann_dir.exists():
        return pd.DataFrame()

    frames = []
    for path in ann_dir.glob("*.parquet"):
        df = pq.read_table(str(path)).to_pandas()
        df["video_id"] = path.stem
        df["duration_frames"] = df["stop_frame"] - df["start_frame"]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_annotations() -> pd.DataFrame:
    """Load and concatenate the annotations of every lab.

    Adds ``video_id`` and ``lab_id`` columns.

    Returns
    -------
    pd.DataFrame
    """
    frames = []
    for lab_dir in TRAIN_ANNOTATION_DIR.iterdir():
        if not lab_dir.is_dir():
            continue
        for path in lab_dir.glob("*.parquet"):
            df = pq.read_table(str(path)).to_pandas()
            df["video_id"] = path.stem
            df["lab_id"] = lab_dir.name
            df["duration_frames"] = df["stop_frame"] - df["start_frame"]
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Keypoint quality analysis
# ---------------------------------------------------------------------------

def keypoint_quality_report(
    video_id: str | int,
    lab_id: str,
    speed_sigma_threshold: float = 3.0,
    length_variation_threshold: float = 0.20,
) -> dict:
    """Assess the keypoint quality of one video.

    Detects the three failure modes the tracking data actually exhibits:
    1. NaN values.
    2. Sudden speed jumps, above ``speed_sigma_threshold`` standard deviations.
    3. Violations of the nose-to-tail_base body length, meaning a deviation from
       the median above ``length_variation_threshold``.

    Returns
    -------
    dict with keys:
        n_frames        : int   — total frames in the video
        n_mice          : int   — number of distinct mice
        nan_rate        : float — fraction of coordinates that are NaN
        speed_outlier_rate : float — fraction of frames with a sudden jump
        length_violation_rate : float — fraction with an anomalous body length
        has_nose_tail   : bool  — whether the lab tracks nose and tail_base
    """
    df = load_tracking(video_id, lab_id)
    n_frames = df["video_frame"].nunique()
    n_mice = df["mouse_id"].nunique()

    # 1. NaN rate
    nan_rate = float(df[["x", "y"]].isna().any(axis=1).mean())

    # 2. Speed, per keypoint and per mouse
    df_sorted = df.sort_values(["mouse_id", "bodypart", "video_frame"])
    df_sorted["dx"] = df_sorted.groupby(["mouse_id", "bodypart"])["x"].diff()
    df_sorted["dy"] = df_sorted.groupby(["mouse_id", "bodypart"])["y"].diff()
    df_sorted["speed"] = np.sqrt(df_sorted["dx"] ** 2 + df_sorted["dy"] ** 2)

    speed_mean = df_sorted["speed"].mean()
    speed_std = df_sorted["speed"].std()
    if speed_std > 0:
        speed_outlier_rate = float(
            (df_sorted["speed"] > speed_mean + speed_sigma_threshold * speed_std).mean()
        )
    else:
        speed_outlier_rate = 0.0

    # 3. Nose-to-tail_base body length
    has_nose_tail = (
        "nose" in df["bodypart"].values and "tail_base" in df["bodypart"].values
    )
    if has_nose_tail:
        nose = df[df["bodypart"] == "nose"][["video_frame", "mouse_id", "x", "y"]].rename(
            columns={"x": "nx", "y": "ny"}
        )
        tail = df[df["bodypart"] == "tail_base"][
            ["video_frame", "mouse_id", "x", "y"]
        ].rename(columns={"x": "tx", "y": "ty"})
        merged = nose.merge(tail, on=["video_frame", "mouse_id"])
        merged["body_len"] = np.sqrt(
            (merged["nx"] - merged["tx"]) ** 2 + (merged["ny"] - merged["ty"]) ** 2
        )
        median_len = merged.groupby("mouse_id")["body_len"].transform("median")
        violations = (
            (merged["body_len"] - median_len).abs() / median_len
            > length_variation_threshold
        )
        length_violation_rate = float(violations.mean())
    else:
        length_violation_rate = float("nan")

    return {
        "n_frames": n_frames,
        "n_mice": n_mice,
        "nan_rate": nan_rate,
        "speed_outlier_rate": speed_outlier_rate,
        "length_violation_rate": length_violation_rate,
        "has_nose_tail": has_nose_tail,
    }
