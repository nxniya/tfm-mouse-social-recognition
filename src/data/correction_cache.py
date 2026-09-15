"""
src/data/correction_cache.py
============================
C2 — Disk cache for skeleton-corrected tracking.

``smooth_video`` runs a per-frame L-BFGS-B optimiser (~76 s for a 6k-frame video),
which is why the notebooks cap everything at one video / 400 frames. The correction
is *deterministic* in its inputs, so it only needs to run once per (video, params):
this module caches the corrected long-format tracking to Parquet and reloads it on
subsequent calls.

Cache layout::

    dataset/corrected/<lab>/<video>__<params_hash>.parquet

The 10-char ``params_hash`` is derived from the lab id and the ``smooth_video``
keyword arguments, so changing any correction parameter transparently invalidates
the cache (a new file is written; the old one is ignored, never silently reused).

Used by:
  * ``feature_pipeline.load_and_correct`` (transparent speed-up for notebook 03 / batch)
  * ``scripts/build_correction_cache.py`` (parallel pre-warming across many videos)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "dataset" / "corrected"

# Bump when the correction algorithm changes in a way that invalidates old caches
# but the kwargs stay the same (keeps stale Parquet from being reused).
CACHE_VERSION = "v1"


def input_fingerprint(df_raw) -> Dict:
    """Lightweight fingerprint of the raw input extent.

    Distinguishes e.g. a 400-frame slice from the full video of the same id, so a
    frame-limited correction is never reused as if it were the full-video one."""
    frames = df_raw["video_frame"]
    return {
        "n_rows": int(len(df_raw)),
        "n_frames": int(frames.nunique()),
        "f_min": int(frames.min()) if len(df_raw) else 0,
        "f_max": int(frames.max()) if len(df_raw) else 0,
    }


def params_hash(lab_id: str, smooth_kwargs: Dict,
                input_fp: Optional[Dict] = None) -> str:
    """Stable 10-char hash of lab + correction parameters (+ input extent)."""
    payload = {
        "_v": CACHE_VERSION,
        "lab": lab_id,
        "kwargs": {k: smooth_kwargs[k] for k in sorted(smooth_kwargs)},
        "input": input_fp or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def corrected_path(lab_id: str, video_id: str, phash: str,
                   cache_dir: Optional[Path] = None) -> Path:
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    return cache_dir / lab_id / f"{video_id}__{phash}.parquet"


def load_or_correct(
    df_raw: pd.DataFrame,
    lab_id: str,
    video_id: str,
    skeleton,
    smooth_kwargs: Optional[Dict] = None,
    *,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Return corrected tracking, from disk cache if available.

    Parameters
    ----------
    df_raw : long-format tracking for a single video.
    lab_id, video_id : identifiers (form the cache key + filename).
    skeleton : a fitted ``MouseSkeleton`` (correction needs it; cheap to fit).
    smooth_kwargs : kwargs forwarded to ``smooth_video`` (also part of the cache key).
    cache_dir : override the default ``dataset/corrected``.
    force : recompute and overwrite even if a cached file exists.

    Returns
    -------
    (df_corrected, meta) where meta has keys: from_cache, path, hash, n_frames.
    """
    smooth_kwargs = dict(smooth_kwargs or {})
    phash = params_hash(lab_id, smooth_kwargs, input_fingerprint(df_raw))
    path = corrected_path(lab_id, video_id, phash, cache_dir)

    if path.exists() and not force:
        df = pd.read_parquet(path)
        return df, {"from_cache": True, "path": str(path), "hash": phash,
                    "n_frames": int(df["video_frame"].nunique())}

    # Cache miss → run the optimiser and persist.
    from src.skeleton.keypoint_smoother import smooth_video
    df_corr, _report = smooth_video(df_raw, lab_id, skeleton=skeleton, **smooth_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish write: temp then replace, so an interrupted run never leaves a
    # half-written Parquet that a later run would treat as a valid cache hit.
    tmp = path.with_suffix(".parquet.tmp")
    df_corr.to_parquet(tmp, index=False)
    tmp.replace(path)
    return df_corr, {"from_cache": False, "path": str(path), "hash": phash,
                     "n_frames": int(df_corr["video_frame"].nunique())}


def correct_one_video(args: tuple) -> dict:
    """Spawn-safe worker: load + fit + correct + cache a single video.

    Defined at module level (not in the CLI script) so ``ProcessPoolExecutor``
    can pickle it on Windows ('spawn').

    Args tuple:
        (video_id, lab_id, smooth_kwargs, frame_limit, force, cache_dir, repo_root)
    Returns dict: video_id, n_frames, elapsed_s, from_cache, error.
    """
    import sys
    import time

    (video_id, lab_id, smooth_kwargs, frame_limit, force, cache_dir, repo_root) = args
    try:
        if repo_root and repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.data.loader import load_tracking
        from src.skeleton import build_skeleton, fit_skeleton

        df = load_tracking(video_id, lab_id)
        if df is None or df.empty:
            return {"video_id": video_id, "error": "empty tracking"}
        if frame_limit and frame_limit > 0:
            df = df[df["video_frame"] < frame_limit].copy()

        skl = build_skeleton(lab_id)
        fit_skeleton(skl, df)

        t0 = time.perf_counter()
        _df_corr, meta = load_or_correct(
            df, lab_id, str(video_id), skl, smooth_kwargs,
            cache_dir=cache_dir, force=force,
        )
        return {
            "video_id": video_id,
            "n_frames": meta["n_frames"],
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "from_cache": meta["from_cache"],
            "hash": meta["hash"],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"video_id": video_id, "error": str(exc)}


def build_correction_cache(video_ids, lab_id, smooth_kwargs=None, *,
                           frame_limit=0, force=False, workers=4,
                           cache_dir=None, repo_root=None):
    """C2 — pre-warm the correction cache across many videos, in parallel.

    Used by ``02a_pipeline`` to make the A1 "scale to many videos" step practical:
    each (video, params) is corrected once, written to Parquet, and reloaded
    instantly afterwards. Returns a list of per-video result dicts.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    repo_root = repo_root or str(REPO_ROOT)
    cache_dir = str(cache_dir) if cache_dir else str(DEFAULT_CACHE_DIR)
    jobs = [(v, lab_id, dict(smooth_kwargs or {}), frame_limit, force, cache_dir, repo_root)
            for v in video_ids]
    rows = []
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(correct_one_video, j) for j in jobs]
            for fut in as_completed(futs):
                rows.append(fut.result())
    else:
        rows = [correct_one_video(j) for j in jobs]
    return rows
