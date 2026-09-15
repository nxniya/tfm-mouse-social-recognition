"""
scripts/build_lb_cache.py
=========================
Builds the feature caches for the *leaderboard simulation* (notebook 07).

A **multi-lab** corpus over the labs sharing the CalMS21 schema of 7 keypoints and
2 mice, with `background` included as a class: the official metric discards background
predictions, so the model needs to be able to abstain.

**Core labelling (`--label-core`, the default).** Scoring emits not the whole window
but its S central frames, so the label is decided over that same core, by simple
majority, background included. This decouples the *context* (W=64 frames of features)
from the *temporal granularity* (S frames), and the latter is what sets the ceiling of
the interval metric: emitting the whole window makes the minimum interval W frames
whatever the stride is, and since 75% of annotated behaviours last under 64 frames,
the ceiling collapses.

It produces two paired caches differing only in the correction:
  * `lb_raw<suffix>.npz`  — raw tracking, no `smooth_video`
  * `lb_corr<suffix>.npz` — canonical MouseSkeleton correction (`configs/skeleton.yaml`)

The correction is cached on disk per (lab, video, config hash), so changing the stride
does **not** pay the optimiser's roughly 27 ms/frame again.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# smooth_video runs tqdm per window: with 10 workers that floods stdout.
os.environ.setdefault("TQDM_DISABLE", "1")

import sys, pathlib, time, json, argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import json
import numpy as np
import pandas as pd

def _annotated_labs():
    """Labs that have annotations. This used to be 9 hard-coded ones, but the dataset
    ships 19 annotated labs (847 videos): since the metric averages PER lab, every lab
    without a model contributes 0 to a submission, so restricting to 9 sank the real
    score. MABe22_keypoints and MABe22_movies drop out on their own, having no
    annotations."""
    from src.data.loader import TRAIN_ANNOTATION_DIR
    return sorted(d.name for d in TRAIN_ANNOTATION_DIR.iterdir()
                  if d.is_dir() and any(d.glob("*.parquet")))


LABS = _annotated_labs()
FEAT = _r / "dataset" / "features"
RESULTS = _r / "results"
BACKGROUND = "background"


def select_videos(n_per_lab, labs=None):
    """Deterministic (lab, video_id): annotated videos only, in a stable order."""
    from src.data.loader import TRAIN_ANNOTATION_DIR, iter_lab_videos
    pairs = []
    for lab in (labs or LABS):
        ann_dir = TRAIN_ANNOTATION_DIR / lab
        annotated = {p.stem for p in ann_dir.glob("*.parquet")} if ann_dir.exists() else set()
        vids = sorted(v["video_id"] for v in iter_lab_videos(lab)
                      if v["video_id"] in annotated)
        pairs += [(lab, v) for v in vids[:n_per_lab]]
    return pairs


def collect_classes(pairs, frame_limit):
    """Vocabulary = the union of the chosen videos' actions, plus background."""
    from src.data.loader import load_annotations
    acts = set()
    for lab, vid in pairs:
        try:
            # frame_limit=0 means "no cap"; passing it straight through as max_frame
            # clips the annotations to nothing and the vocabulary comes out empty.
            ann = load_annotations(vid, lab,
                                   max_frame=frame_limit if frame_limit else None)
        except Exception:
            continue
        if ann is not None and len(ann):
            acts |= {str(a) for a in ann["action"].unique()}
    return [BACKGROUND] + sorted(acts)


def _label_array(ann, mid, n, target=None):
    """Per-frame labels taken from the ANNOTATIONS, over the full range.

    This has to be independent of the variant: the correction imputes keypoints, so
    `labels_per_frame`, which is defined only over the frames that survive filtering,
    differs between raw and corr and breaks the pairing. Deriving it from the
    annotations makes it identical by construction. Precedence: the longest annotation
    wins.
    """
    arr = np.full(max(n, 0), BACKGROUND, dtype=object)
    if ann is None or not len(ann):
        return arr
    sub = ann[ann["agent_id"] == mid]
    if target is not None and "target_id" in sub.columns:
        # With 3 or more mice, a mouse1->mouse3 annotation must not also label the
        # mouse1->mouse2 row. Self-directed ones (agent == target) belong to the agent
        # and are kept across all of its pairs.
        sub = sub[(sub["target_id"] == target) | (sub["target_id"] == mid)]
    sub = sub.copy()
    if not len(sub):
        return arr
    sub["_dur"] = sub["stop_frame"] - sub["start_frame"]
    for r in sub.sort_values("_dur").itertuples():
        a, b = max(0, int(r.start_frame)), min(n, int(r.stop_frame))
        if b > a:
            arr[a:b] = str(r.action)
    return arr


def canonical_keypoints(lab, videos):
    """The keypoints common to ALL of a lab's annotated videos.

    AdaptableSnail mixes two schemas: 18 keypoints in the union, 10 in the
    intersection. If each video used its own, the same column would mean different
    things within one lab and the per-lab model would stop being valid. The
    intersection is fixed as the canonical schema; for the other 18 labs that is
    exactly what they already used.
    """
    import json as _json
    import pandas as _pd
    from src.data.loader import DATASET_DIR
    meta = _pd.read_csv(DATASET_DIR / "train.csv")
    meta = meta[(meta.lab_id == lab) & (meta.video_id.astype(str).isin(set(videos)))]
    sets = []
    for v in meta.body_parts_tracked.dropna():
        try:
            sets.append(set(_json.loads(v)))
        except Exception:
            pass
    if not sets:
        return None
    common = set.intersection(*sets)
    return sorted(common) if common else None


def _process_video(args):
    """Extract one video's windows for one variant. Returns rows plus timing."""
    (lab, vid, classes, variant, frame_limit, W, S, label_core, min_fill, agg,
     center, kp_hint, vid_cap) = args
    t0 = time.perf_counter()
    from src.data.features import extract_features
    from src.data.loader import load_tracking, load_annotations
    from src.skeleton.mouse_skeleton import build_skeleton, fit_skeleton
    from src.skeleton.config import skeleton_smooth_kwargs
    from src.data.correction_cache import load_or_correct
    from src.eval.mabe_metric import window_frame_spans

    cti = {c: i for i, c in enumerate(classes)}
    n_frames = 0
    try:
        raw = load_tracking(vid, lab)
        if frame_limit:
            raw = raw[raw["video_frame"] < frame_limit].reset_index(drop=True)
        n_frames = int(raw["video_frame"].max()) + 1 if len(raw) else 0
        ann = load_annotations(vid, lab, max_frame=n_frames)
        ann = ann if ann is not None and len(ann) else None

        # These labs share the 7 CalMS21 keypoints, but `build_skeleton` recognises
        # only the three CalMS21 names: without the hint, the rest fall through to the
        # MABe22 schema (12 kp) and lose 4 hip dz features, 60 against 64.
        _kps = sorted(raw["bodypart"].astype(str).unique())
        # The lab's canonical schema, restricted to what this video actually has.
        if kp_hint:
            _sel = [k for k in kp_hint if k in set(_kps)]
            if _sel:
                _kps = _sel
        sk = build_skeleton(lab, keypoints_hint=_kps)
        fit_skeleton(sk, raw)
        if variant == "corr":
            kw = skeleton_smooth_kwargs()
            kw.pop("skeleton", None)
            track, _meta = load_or_correct(raw, lab, str(vid), sk, kw)
        else:
            track = raw

        # center="none": centring per mouse would cancel the inter-mouse geometry
        # (dist_centroid, approach_rate, speed_relative caian a ~1e-7).
        res = extract_features(track, sk, ann, window_size=W, stride=S,
                               min_fill=min_fill, center=center, pairs=True)
        mice = sorted(res.keys(), key=str)
    except Exception as e:  # noqa: BLE001
        return lab, vid, variant, None, {"error": "{}: {}".format(type(e).__name__, e)}

    off = (W - S) // 2
    rows = {k: [] for k in ("X", "y", "vid", "lab", "ag", "tg", "fs", "fe", "cs", "ce")}
    for _key in mice:
        r = res[_key]
        mid, tgt = r["agent"], r["target"]
        Xm, ym, fdf = r["X"], r["y"], r["features_df"]
        if Xm is None or len(Xm) == 0 or ym is None:
            continue
        spans = window_frame_spans(fdf, W, S, min_fill)
        if len(spans) != len(Xm):
            continue
        # With `agg` only the window statistics are stored (mean, std, min, max ->
        # 4F): the trees use nothing else, and the full sequential tensor is about 16x
        # the bytes, 46 GB over the whole corpus.
        # nan_to_num BEFORE aggregating: the tree pipeline does
        # aggregate_windows(nan_to_num(X)), and aggregating with NaN propagates it.
        _Xc = np.nan_to_num(Xm, nan=0.0, posinf=0.0, neginf=0.0) if agg else None
        Xagg = (np.concatenate([_Xc.mean(1), _Xc.std(1), _Xc.min(1), _Xc.max(1)],
                               axis=1).astype(np.float32) if agg else None)
        lab_arr = _label_array(ann, mid, n_frames, target=tgt)
        other = -1 if tgt is None else int(tgt)
        for i in range(len(Xm)):
            fstart, fstop = spans[i]
            core_a, core_b = int(fstart) + off, int(fstart) + off + S
            if label_core:
                # Label = a SIMPLE majority, background included, over the core that
                # is actually emitted, derived from the annotations so that raw and
                # corr are identical.
                core = lab_arr[max(0, core_a):max(0, core_b)]
                lb = str(Counter(core).most_common(1)[0][0]) if len(core) else BACKGROUND
            else:
                lb = str(ym[i])
            if lb not in cti:      # action outside the vocabulary -> discard
                continue
            rows["X"].append(Xagg[i] if agg else Xm[i])
            rows["y"].append(cti[lb])
            rows["vid"].append(str(vid))
            rows["lab"].append(lab)
            rows["ag"].append(int(mid))
            rows["tg"].append(int(other))
            rows["fs"].append(int(fstart))
            rows["fe"].append(int(fstop))
            rows["cs"].append(core_a)
            rows["ce"].append(core_b)

    el = time.perf_counter() - t0
    stats = {"n_frames": n_frames, "n_windows": len(rows["X"]),
             "elapsed_s": round(el, 3),
             "ms_per_frame": round(el / max(n_frames, 1) * 1000, 3)}
    if vid_cap and len(rows["y"]) > vid_cap:
        # UNIFORM decimation, not random sampling. The temporal context
        # (`lb_context.add_context`) assumes that consecutive windows of a
        # (video, agent) are `stride` frames apart: removing them at random breaks
        # that spacing and their past/future deltas stop measuring what they claim.
        # Measured: with random sampling BoisterousParrot kept only 37% of its steps
        # correct and lost 0.335 of score. Decimating uniformly keeps the spacing
        # constant, merely coarser, and the context keeps its meaning.
        step = int(np.ceil(len(rows["y"]) / vid_cap))
        sel = list(range(0, len(rows["y"]), step))[:vid_cap]
        rows = {k: [v[i] for i in sel] for k, v in rows.items()}
    return lab, vid, variant, rows, stats


def build(variant, pairs, classes, frame_limit, workers, W, S, label_core, suffix,
          min_fill, agg, center, max_per_lab=0):
    out = FEAT / "lb_{}{}.npz".format(variant, suffix)
    acc = {k: [] for k in ("y", "vid", "lab", "ag", "tg", "fs", "fe", "cs", "ce")}
    blocks = []
    timing, failures = [], []
    _by_lab = {}
    for _l, _v in pairs:
        _by_lab.setdefault(_l, []).append(str(_v))
    _hints = {l: canonical_keypoints(l, vs) for l, vs in _by_lab.items()}
    # The per-lab cap is divided among its videos: capping per video bounds the total
    # without biasing towards whichever finish first, since the pool sets that order.
    _cap_v = {}
    if max_per_lab:
        for _l, _vs in _by_lab.items():
            _cap_v[_l] = max(1, max_per_lab // max(len(_vs), 1))
    tasks = [(lab, vid, classes, variant, frame_limit, W, S, label_core, min_fill,
              agg, center, _hints.get(lab), _cap_v.get(lab, 0))
             for lab, vid in pairs]
    t0 = time.perf_counter()
    done = 0
    print("\n=== Building {} — {} videos, W={} S={} core={} ({} workers) ===".format(
        out.name, len(tasks), W, S, label_core, workers), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_video, t): t for t in tasks}
        for fut in as_completed(futs):
            lab, vid, _v, rows, stats = fut.result()
            done += 1
            if rows is None:
                failures.append(dict(lab=lab, video_id=vid, **stats))
                print("  [{:3d}/{}] {:22s} {:>12s}  FALLO: {}".format(
                    done, len(tasks), lab, vid, stats["error"]), flush=True)
                continue
            blocks.append(np.asarray(rows["X"], np.float32))
            for k in acc:
                if k != "X":
                    acc[k] += rows[k]
            timing.append(dict(lab=lab, video_id=vid, variant=variant, **stats))
            if done % 10 == 0 or done == len(tasks):
                print("  [{:3d}/{}] {:22s}  accumulated {} windows  ({:.0f}s)".format(
                    done, len(tasks), lab, len(acc["y"]), time.perf_counter() - t0), flush=True)

    blocks = [b for b in blocks if len(b)]
    widths = {b.shape[1:] for b in blocks}
    if len(widths) != 1:
        # The 19 labs do NOT share a keypoint schema, from 4 in GroovyShrew to 18 in
        # AdaptableSnail, so F ranges from 31 to 74 and this used to be a fatal error.
        # Everything is zero-padded out to the maximum width.
        #
        # This is valid ONLY because the models are per lab: within one lab the
        # columns are homogeneous and mean the same thing, and the padding is constant,
        # so a tree ignores it. This cache is NOT usable for pooling labs together,
        # which was measured to be harmful anyway.
        tail = max(int(np.prod(w)) for w in widths)
        shapes = sorted({tuple(w) for w in widths})
        # With `--aggregate` each row is [mean | std | min | max], each block F
        # columns wide. Padding happens PER BLOCK, not at the end: appended at the end,
        # the first F_max columns of a lab with small F would mix mean, std, min and
        # max, and the temporal context, which assumes the first block is the means,
        # would operate on a meaningless mixture.
        nb = 4 if agg else 1
        fmax = tail // nb
        print("Heterogeneous widths {} -> per-block padding to {} columns "
              "({} bloques x {})".format(shapes, tail, nb, fmax), flush=True)
        pad = []
        for b in blocks:
            flat = b.reshape(len(b), -1)
            if flat.shape[1] < tail:
                f = flat.shape[1] // nb
                _buf = np.zeros((len(flat), tail), flat.dtype)
                for j in range(nb):
                    _buf[:, j * fmax:j * fmax + f] = flat[:, j * f:(j + 1) * f]
                flat = _buf
            pad.append(flat)
        blocks = pad
    # np.concatenate would keep `blocks` and `Xall` alive at once, doubling the peak
    # memory. With no frame cap the corpus is around 6-7 GB, so the array is allocated
    # once and copied block by block, freeing each one as it goes.
    _n = sum(len(b) for b in blocks)
    _w = blocks[0].reshape(len(blocks[0]), -1).shape[1]
    Xall = np.empty((_n, _w), np.float32)
    _o = 0
    for _i in range(len(blocks)):
        _b = blocks[_i].reshape(len(blocks[_i]), -1)
        Xall[_o:_o + len(_b)] = _b
        _o += len(_b)
        blocks[_i] = None          # libera segun avanza
    del blocks
    np.savez_compressed(
        out,
        X=Xall, y=np.asarray(acc["y"], np.int64), aggregated=np.bool_(agg),
        video_ids=np.asarray(acc["vid"], object), lab_ids=np.asarray(acc["lab"], object),
        classes=np.asarray(classes),
        win_agent=np.asarray(acc["ag"], np.int64), win_target=np.asarray(acc["tg"], np.int64),
        win_fstart=np.asarray(acc["fs"], np.int64), win_fstop=np.asarray(acc["fe"], np.int64),
        win_core_start=np.asarray(acc["cs"], np.int64),
        win_core_stop=np.asarray(acc["ce"], np.int64),
        window_size=np.int64(W), stride=np.int64(S), label_core=np.bool_(label_core),
        # The lab's canonical keypoint schema: inference must use exactly the same
        # one, or the columns stop meaning what the model was trained on.
        lab_keypoints=np.asarray(json.dumps(_hints), dtype=object),
    )
    total = time.perf_counter() - t0
    print("Guardada {}  X={}  ({:.0f}s)".format(
        out.name, Xall.shape, total), flush=True)
    return {"timing": timing, "failures": failures, "total_s": round(total, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-lab", type=int, default=int(os.environ.get("LB_N_PER_LAB", "10")))
    ap.add_argument("--frame-limit", type=int, default=int(os.environ.get("LB_FRAME_LIMIT", "9000")))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("LB_WORKERS", "10")))
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--suffix", default="_s8")
    # min_fill=0 keeps ALL windows: dropping them leaves unpredicted gaps, which are
    # pure false negatives, and raw and corr would drop different windows, breaking the
    # pairing. The NaNs are forward-filled instead.
    ap.add_argument("--min-fill", type=float, default=0.0)
    ap.add_argument("--center", default="none", choices=["none", "centroid"])
    ap.add_argument("--aggregate", action="store_true",
                    help="store window statistics (4F) instead of the (W,F) tensor")
    ap.add_argument("--no-label-core", action="store_true",
                    help="label using the whole window (the old behaviour)")
    ap.add_argument("--variants", nargs="+", default=["raw", "corr"])
    ap.add_argument("--labs", nargs="+", default=None,
                    help="restrict to these labs, for quick tests")
    ap.add_argument("--max-windows-per-lab", type=int, default=0,
                    help="cap on windows per lab (0 = no cap). Bounds the memory: "
                         "with no frame clipping the corpus reaches about 8M windows "
                         "and the assembly peak does not fit in RAM. Training already "
                         "limits to 250k per lab, so a cap "
                         "this generous takes nothing away from it.")
    a = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pairs = select_videos(a.n_per_lab, a.labs)
    print("Videos: {} across {} labs | W={} S={} core={} frame_limit={}".format(
        len(pairs), len(set(l for l, _ in pairs)), a.window, a.stride, not a.no_label_core, a.frame_limit))

    classes = collect_classes(pairs, a.frame_limit)
    print("Vocabulario ({} clases): {}".format(len(classes), classes))
    (FEAT / "lb_classes.json").write_text(json.dumps(classes, indent=1), encoding="utf-8")

    meta = {}
    for v in a.variants:
        meta[v] = build(v, pairs, classes, a.frame_limit, a.workers,
                        a.window, a.stride, not a.no_label_core, a.suffix, a.min_fill,
                        a.aggregate, a.center, a.max_windows_per_lab)

    rt = pd.DataFrame([r for v in meta for r in meta[v]["timing"]])
    if len(rt):
        rt.to_csv(RESULTS / "lb_runtime_cache{}.csv".format(a.suffix), index=False)
        agg = rt.groupby("variant").agg(
            n_frames=("n_frames", "sum"), n_windows=("n_windows", "sum"),
            elapsed_s=("elapsed_s", "sum"), ms_per_frame=("ms_per_frame", "mean")).round(2)
        print("\n=== Build runtime ===")
        print(agg.to_string())
    fails = [f for v in meta for f in meta[v]["failures"]]
    if fails:
        pd.DataFrame(fails).to_csv(RESULTS / "lb_cache_failures.csv", index=False)
        print("\n{} videos failed -> results/lb_cache_failures.csv".format(len(fails)))


if __name__ == "__main__":
    main()
