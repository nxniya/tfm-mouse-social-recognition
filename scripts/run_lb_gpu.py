"""
scripts/run_lb_gpu.py
=====================
A sequential model **per lab**, on GPU, for ensembling with HistGB.

Why CNN-LSTM and not the TCN: `BehaviorTCN` uses `_CausalDilatedBlock`, a causal
convolution padded only on the left, so it cannot see future frames. In *offline*
scoring the whole video is available, which makes causality a free restriction to
give up. `BehaviorCNNLSTM` is bidirectional and sees the entire window.

Memory: the sequential tensor for the whole corpus is about 46 GB in float32, so it is
never materialised. One lab is processed at a time, converted to fp16 as soon as it
leaves the extractor, and training subsamples. The probabilities are saved indexed by
(video, agent, core_start) so they can be aligned afterwards with the trees'
aggregated cache, whose row order depends on the completion order of the process pool
and is therefore not reproducible.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys, pathlib, time, json, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))

import numpy as np

from scripts.build_lb_cache import _process_video, select_videos, collect_classes
from scripts.run_lb_experiment import split_videos, SEED

RESULTS = _r / "results"


def build_lab_tensor(lab, videos, classes, W, S, frame_limit, workers, cap=None, seed=SEED):
    """One lab's (N, W, F) fp16 tensor, plus its alignment keys."""
    tasks = [(lab, v, classes, "raw", frame_limit, W, S, True, 0.0, False, "none")
             for v in videos]
    Xs, ys, keys = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_process_video, t) for t in tasks]):
            _l, _v, _var, rows, _st = fut.result()
            if rows is None or not rows["X"]:
                continue
            # min_fill=0 lets windows containing NaN through. The tree path cleans
            # them when loading the cache; here it has to happen before the fp16 cast,
            # or the NaN propagates into mu/sd and the whole loss comes out NaN.
            Xs.append(np.nan_to_num(np.asarray(rows["X"], np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16))
            ys.append(np.asarray(rows["y"], np.int64))
            keys += list(zip(rows["vid"], rows["ag"], rows["cs"]))
    if not Xs:
        return None, None, []
    X = np.concatenate(Xs); y = np.concatenate(ys)
    del Xs
    if cap and len(y) > cap:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(y), cap, replace=False))
        X, y = X[idx], y[idx]
        keys = [keys[i] for i in idx]
    return X, y, keys


def train_lab(X, y, nC, epochs, batch, lr, device, seed=SEED):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.rnn import BehaviorCNNLSTM
    torch.manual_seed(seed)
    _flat = np.nan_to_num(X.astype(np.float32).reshape(-1, X.shape[-1]),
                          nan=0.0, posinf=0.0, neginf=0.0)
    mu = _flat.mean(0)
    sd = _flat.std(0) + 1e-6
    del _flat
    model = BehaviorCNNLSTM(input_size=X.shape[-1], n_classes=nC).to(device)
    cnt = np.bincount(y, minlength=nC).astype(np.float64)
    w = np.zeros(nC, np.float32); pres = cnt > 0
    w[pres] = cnt[pres].sum() / (pres.sum() * cnt[pres])
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    tmu = torch.tensor(mu, device=device); tsd = torch.tensor(sd, device=device)
    model.train()
    for ep in range(epochs):
        tot = n = 0
        for xb, yb in dl:
            xb = ((xb.to(device, non_blocking=True).float() - tmu) / tsd)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss) * len(yb); n += len(yb)
        print("      epoch {:2d}/{}  loss={:.4f}".format(ep + 1, epochs, tot / max(n, 1)),
              flush=True)
    return model, mu, sd


def predict_lab(model, mu, sd, X, nC, device, batch=512):
    import torch
    model.eval()
    tmu = torch.tensor(mu, device=device); tsd = torch.tensor(sd, device=device)
    out = np.zeros((len(X), nC), np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device).float()
            xb = (xb - tmu) / tsd
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out[i:i + batch] = torch.softmax(model(xb).float(), 1).cpu().numpy()
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--labs", nargs="+", default=None)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--frame-limit", type=int, default=15000)
    ap.add_argument("--cap-train", type=int, default=150000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="lb_gpu")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

    from scripts.run_lb_perlab import load_full
    X0, y0, vids, labs, classes, ev, _agg = load_full(a.suffix)
    del X0
    nC = len(classes)
    train_v, test_v, lab_of = split_videos(vids, labs)
    all_pairs = select_videos(1000)
    by_lab = {}
    for l, v in all_pairs:
        by_lab.setdefault(l, []).append(v)

    todo = sorted(set(labs.tolist())) if not a.labs else a.labs
    store = {}
    for lab in todo:
        tr = [v for v in by_lab.get(lab, []) if v in set(train_v)]
        teq = [v for v in by_lab.get(lab, []) if v in set(test_v)]
        if not tr or not teq:
            continue
        t0 = time.perf_counter()
        print("\n=== {} — {} train vid / {} test vid ===".format(lab, len(tr), len(teq)),
              flush=True)
        Xtr, ytr, _ = build_lab_tensor(lab, tr, classes, a.window, a.stride,
                                       a.frame_limit, a.workers, cap=a.cap_train)
        if Xtr is None:
            continue
        print("    train tensor {} ({:.1f} GB fp16)".format(
            Xtr.shape, Xtr.nbytes / 1e9), flush=True)
        model, mu, sd = train_lab(Xtr, ytr, nC, a.epochs, a.batch, a.lr, dev)
        del Xtr, ytr
        Xte, yte, keys = build_lab_tensor(lab, teq, classes, a.window, a.stride,
                                          a.frame_limit, a.workers)
        P = predict_lab(model, mu, sd, Xte, nC, dev)
        del Xte
        for k, p in zip(keys, P):
            store[(str(k[0]), int(k[1]), int(k[2]))] = p
        print("    {} done: {} test windows ({:.0f}s)".format(
            lab, len(keys), time.perf_counter() - t0), flush=True)

    # Save indexed by key, to align later with the trees' cache
    ks = list(store.keys())
    np.savez_compressed(
        RESULTS / "{}_proba.npz".format(a.out),
        vid=np.asarray([k[0] for k in ks], dtype=object),
        agent=np.asarray([k[1] for k in ks], np.int64),
        core_start=np.asarray([k[2] for k in ks], np.int64),
        proba=np.asarray([store[k] for k in ks], np.float32),
        classes=np.asarray(classes))
    print("\nSaved results/{}_proba.npz ({} windows)".format(a.out, len(ks)), flush=True)


if __name__ == "__main__":
    main()
