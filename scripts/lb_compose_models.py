#!/usr/bin/env python3
"""
scripts/lb_compose_models.py
============================
Composes a model directory by taking, **lab by lab**, the bundle from whichever run
calibrates best.

Why this is legitimate. The official metric scores each lab using only that lab's own
videos and then averages, so the coordinates are separable: picking the best model for
one lab cannot make any other lab worse. It is the same argument that justifies tuning
the thresholds per lab, and the one that allows mixing window sizes (W=64 for long
actions, W=32 for short ones) within a single submission: `predict.py` reads
`bundle["window"]` from each bundle, so a mixed directory needs no code change.

Runs are only comparable when they cover **the same set of labs**: `split_videos`
draws per lab in loop order, so a different set produces a different split for the
same lab.

Uso:
  .venv/Scripts/python.exe scripts/lb_compose_models.py \
      --runs lb19_v8 lb19_v9w32 --models models/lb19_v8 models/lb19_v9w32 \
      --out models/lb19_mix_sub
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

_r = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_r))          # los bundles despickan clases de `src`
RESULTS = _r / "results"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run names, in the same order as --models")
    ap.add_argument("--models", nargs="+", required=True,
                    help="bundle directories, one per run")
    ap.add_argument("--model", default="histgb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="ratio", choices=["residual", "ratio"],
                    help="decision variant; `ratio` is the one that was accepted")
    ap.add_argument("--no-revival", action="store_true",
                    help="desactivar la reactivacion de clases muertas (por "
                         "defecto activa: +0.0058 medido)")
    ap.add_argument("--mindur", action="store_true",
                    help="enable the duration filter; measured at -0.0109, so it "
                         "is off by default")
    a = ap.parse_args()

    if len(a.runs) != len(a.models):
        raise SystemExit("--runs and --models must have the same length")

    import joblib

    calib, dirs = {}, {}
    for run, md in zip(a.runs, a.models):
        p = RESULTS / "{}_calib_{}.json".format(run, a.model)
        if not p.exists():
            raise SystemExit("falta {}".format(p))
        calib[run] = {k: float(v) for k, v in json.load(open(p)).items()
                      if isinstance(v, (int, float))}
        dirs[run] = pathlib.Path(md)
        if not dirs[run].is_dir():
            raise SystemExit("no es un directorio: {}".format(md))

    sets = [set(calib[r]) for r in a.runs]
    common = set.intersection(*sets)
    extra = set.union(*sets) - common
    if extra:
        print("WARNING: labs absent from some runs, omitted from the "
              "comparison: {}".format(", ".join(sorted(extra))))

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.joblib"):
        old.unlink()

    picked, totals = {}, {r: 0 for r in a.runs}
    print("%-24s %-14s %9s   %s" % ("lab", "elegido", "calib", "alternativas"))
    for lab in sorted(common):
        best = max(a.runs, key=lambda r: calib[r][lab])
        src = next((p for p in dirs[best].glob("*.joblib")
                    if joblib.load(p).get("lab") == lab), None)
        if src is None:
            print("   {:22s} SIN BUNDLE en {}".format(lab, best))
            continue
        b = joblib.load(src)
        b["choice"] = {"mode": a.mode, "mindur": bool(a.mindur)}
        # `rates` alimenta la REACTIVACION DE CLASES MUERTAS (predict.py:241), que es
        # independiente del filtro de duracion: solo convierte ventanas ya predichas
        # como background, asi que no puede solapar con nada ya emitido. Atarla a
        # --mindur la desactivaba sin querer, y vale +0.0058 medido.
        b["rates"] = b.get("rates", {}) if not a.no_revival else {}
        joblib.dump(b, out / src.name, compress=3)
        picked[lab] = best
        totals[best] += 1
        others = " ".join("{}={:.4f}".format(r, calib[r][lab])
                          for r in a.runs if r != best)
        print("%-24s %-14s %9.4f   %s  (W=%s)" % (
            lab, best, calib[best][lab], others, b.get("window", "?")))

    print()
    for r in a.runs:
        print("   from {:16s} {} labs".format(r, totals[r]))
    mean_best = sum(max(calib[r][l] for r in a.runs) for l in common) / len(common)
    for r in a.runs:
        m = sum(calib[r][l] for l in common) / len(common)
        print("   mean {:16s} {:.4f}".format(r, m))
    print("   media compuesta      {:.4f}".format(mean_best))
    print("\nescritos {} bundles en {}".format(len(picked), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
