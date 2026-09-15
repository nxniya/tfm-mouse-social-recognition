"""
scripts/build_permouse_cache.py
===============================
Experimental ablation: does a POOLED skeleton reference, meaning both mice sharing
one L_ref, rather than a PER-MOUSE reference, explain the single-video versus LOVO
gap in the MouseSkeleton correction?

Builds a third feature cache, `CalMS21_task1_multi_20_event_permouse.npz`, identical
to the 'corrected' one from notebook 03_features except for a single variable: the
skeleton is fitted PER MOUSE (`fit_skeleton_per_mouse`) and each individual is
corrected against its own bone-length reference. Everything inter-mouse (swap,
overlap) and the canonical correction config are held constant, so any difference is
attributable to that one variable.

Parallelised per video, since videos are independent. Each worker pins its BLAS and
OMP threads to 1 to avoid oversubscribing the process pool.
"""
from __future__ import annotations
import os
# Pin the thread counts BEFORE importing numpy or scipy; afterwards has no effect.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, pathlib, time
from concurrent.futures import ProcessPoolExecutor, as_completed
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np

LAB = "CalMS21_task1"
FEAT = _r / "dataset" / "features"
SRC = FEAT / "CalMS21_task1_multi_20_event.npz"          # defines the 20 videos + classes
OUT = FEAT / "CalMS21_task1_multi_20_event_permouse.npz"
N_WORKERS = int(os.environ.get("PERMOUSE_WORKERS", "10"))


def _process_video(vid: str, classes: list):
    """Correct one video with per-mouse skeletons and return its windows."""
    from src.data import features as F
    from src.data.features import extract_features
    from src.data.loader import load_tracking, load_annotations
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton, fit_skeleton_per_mouse
    from src.skeleton.keypoint_smoother import smooth_video
    from src.skeleton.config import skeleton_smooth_kwargs
    from src.eval.mabe_metric import window_frame_spans

    cti = {c: i for i, c in enumerate(classes)}
    kw = skeleton_smooth_kwargs(); kw.pop("skeleton", None)

    raw = load_tracking(vid, LAB)
    try:
        ann = load_annotations(vid, LAB); ann = ann if ann is not None and len(ann) else None
    except Exception:
        ann = None
    sk = build_skeleton(LAB); fit_skeleton(sk, raw)          # pooled: topology/swap/overlap
    pm = fit_skeleton_per_mouse(LAB, raw)                    # <-- the experimental variable
    track, _rep = smooth_video(raw, LAB, skeleton=sk, per_mouse_skeletons=pm, **kw)
    res = extract_features(track, sk, ann); mice = sorted(res.keys())

    Xs, ys, vv, ag, tg, fs, fe = [], [], [], [], [], [], []
    for mid in mice:
        r = res[mid]; Xm, ym, fdf = r["X"], r["y"], r["features_df"]
        if Xm is None or len(Xm) == 0 or ym is None:
            continue
        spans = window_frame_spans(fdf, F.WINDOW_SIZE, F.STRIDE, F.MIN_WINDOW_FILL)
        if len(spans) != len(Xm):
            continue
        other = next((m for m in mice if m != mid), -1) if len(mice) == 2 else -1
        for i in range(len(Xm)):
            lab = str(ym[i])
            if lab not in cti:
                continue
            Xs.append(Xm[i]); ys.append(cti[lab]); vv.append(str(vid)); ag.append(int(mid))
            tg.append(int(other)); fs.append(spans[i][0]); fe.append(spans[i][1])
    return vid, (Xs, ys, vv, ag, tg, fs, fe)


def main():
    if not SRC.exists():
        raise SystemExit(f"Base cache {SRC.name} is missing.")
    d = np.load(SRC, allow_pickle=True)
    videos = sorted(set(str(v) for v in d["video_ids"]))
    classes = [str(c) for c in d["classes"]]
    print(f"Building per-mouse cache: {len(videos)} videos, {N_WORKERS} workers…", flush=True)

    acc = {k: [] for k in ("X", "y", "vv", "ag", "tg", "fs", "fe")}
    t0 = time.time(); done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_process_video, v, classes): v for v in videos}
        for fut in as_completed(futs):
            vid, (Xs, ys, vv, ag, tg, fs, fe) = fut.result()
            acc["X"] += Xs; acc["y"] += ys; acc["vv"] += vv; acc["ag"] += ag
            acc["tg"] += tg; acc["fs"] += fs; acc["fe"] += fe
            done += 1
            print(f"  [{done:2d}/{len(videos)}] vid {vid:>10s}  +{len(Xs)} win  "
                  f"(total {len(acc['X'])})  {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(
        OUT,
        X=np.asarray(acc["X"], np.float32), y=np.asarray(acc["y"], np.int64),
        video_ids=np.asarray(acc["vv"], object), classes=np.asarray(classes),
        win_agent=np.asarray(acc["ag"], np.int64), win_target=np.asarray(acc["tg"], np.int64),
        win_fstart=np.asarray(acc["fs"], np.int64), win_fstop=np.asarray(acc["fe"], np.int64),
    )
    print(f"\nSaved {OUT.name}  X={np.asarray(acc['X']).shape}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
