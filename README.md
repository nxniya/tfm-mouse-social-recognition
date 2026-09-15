# MABe mouse behaviour detection: skeleton correction and behaviour classification

Code for a Master's thesis (*Trabajo de Fin de Máster*) in the Máster en Ingeniería
del Software: Cloud, Datos y Gestión TI at the Universidad de Sevilla, on the
[MABe Mouse Behavior Detection](https://www.aicrowd.com/challenges/multi-agent-behavior-challenge)
corpus, working on the CalMS21 subset.

The pipeline goes from raw 2D keypoints to behaviour labels: a biomechanical skeleton
correction that re-optimises implausible poses, kinematic, implicit-3D and relational
feature extraction over sliding windows, window classification with classical and
sequence models, an active-learning layer on top, and a multi-laboratory simulation
that scores the whole thing the way the competition leaderboard would.

This is academic work. The written dissertation it accompanies is in Spanish and is
submitted through the university, not distributed here; this repository holds the code,
the experiment summaries and the tests.

## The results are negative, and that is the point

Under an evaluation protocol without information leakage:

- **One honest protocol changes the conclusions.** Under leave-one-video-out over 20
  videos, the four model families tie at 0.53 to 0.58 official `f1_event`. The earlier
  "BiLSTM beats the shallow models" was a protocol artefact: leaky cross-validation
  compared against a single-video hold-out.
- **The skeleton correction helps geometry and hurts classification.** Length
  violations drop from 17% to 9%, but relocating keypoints injects jitter that the
  classifier had been reading as signal. Reproduced six ways, all at or below raw.
- **Active learning does not beat random.** Once validation uses held-out videos
  instead of a random split of overlapping windows, no strategy separates from chance
  over five seeds.
- **At leaderboard scale the system scores 0.45515 public / 0.43412 private** on the
  real Kaggle test set, above the median competing team (0.4455). The private score
  clears the bronze threshold of 0.43241 by 0.0017 — but this was a late submission,
  scored normally yet excluded from the ranking, so the accurate claim is that it
  *would have* placed in bronze territory, not that it earned a medal.
- **None of the improvement came from better models.** Every modelling lever measured
  at or below noise. All three real gains were inherited constants that had been set
  once for iteration speed and never re-examined: `frame_limit`, `--cap-per-lab` and
  `window_size`. That is the most transferable finding here, and it is a negative
  result about modelling effort as much as a positive one about the score.

Each of those contradicts what an earlier, leakier protocol suggested. The fixes are in
the code: `train_val_test_split_temporal(purge=...)` in `src/data/dataset.py` removes
the window-overlap leakage, `src/evaluation/lovo.py` replaces the single-video hold-out,
and `src/eval/mabe_metric.py` scores with the official interval metric rather than a
per-window F1.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt    # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux, macOS
```

Run every command from the repository root; the modules import as `src.*` and the
scripts put the root on `sys.path` themselves.

The corpus is distributed by the challenge and is **not** in this repository. It is
roughly 25 GB. See [`dataset/README.md`](dataset/README.md) for the expected layout.

## Running it

Four entry points, in increasing order of cost.

| what | command | cost |
|---|---|---|
| Unit tests | `.venv/Scripts/python -m pytest tests/ -q` | seconds, no data needed |
| Contract self-tests | `python -m scripts._selftest_mabe_metric` (and `_selftest_event_bridge`, `_selftest_lb_scoring`, `_selftest_submission`, `_selftest_inference_equiv`) | seconds to minutes; the last three need the corpus |
| Notebooks | see the order below | hours |
| Leaderboard simulation | see the four commands below | many hours |

### Notebooks

```
01_eda → 02a_pipeline → 02b_visualization, 02c_benchmarks, 02d_statistics
       → 03_features → 04_baseline → 05_deep_model → 06_active_learning
       → 07_leaderboard_simulation → 08_kaggle_submission
```

`02a` writes the artefacts that `02b`, `02c` and `02d` read, so it has to run first.
`03_features` builds the multi-video caches that `04`, `05` and `06` load. The honest
multi-video evaluation lives in the appendix sections at the end of `04_baseline` and
`05_deep_model`, not in separate scripts.

The expensive cells default to **loading cached CSVs** from `results/`; each has a
`RECOMPUTE = True` flag that forces the real computation.

### Leaderboard simulation

The only part that runs outside the notebooks, because it trains one model per
laboratory.

```bash
python scripts/build_lb_cache.py --aggregate --center none --stride 8 --min-fill 0 --suffix _v3
python scripts/run_lb_perlab.py  --suffix _v3 --models histgb --context 12 --smooth 0 --out lb_ctx12
python scripts/run_lb_decide_perlab.py --pred lb_ctx12c --out lb_decperlab
python scripts/run_lb_finalize.py      --dec lb_decperlab --out lb_best
```

### What it costs

Measured on this project's hardware, from `results/lb_runtime_cache.csv` and
`results/lb_models.csv`:

| step | cost |
|---|---|
| Cache build, raw tracking | 0.7 ms/frame — about 2 minutes for 90 videos |
| Cache build, **corrected** tracking | **27 ms/frame — about 5.8 hours** for the same 90 videos |
| Per-lab fit + predict, `rf` / `histgb` | 20 s / 50 s |
| Per-lab fit + predict, `gru` | about 5 minutes |
| Per-lab fit + predict, `bilstm` / `tcn` | about 34 minutes |
| Per-lab fit + predict, `cnn_lstm` | about 65 minutes |
| Notebook 03, rebuilding the 20-video caches | about 1 hour |

The L-BFGS-B correction dominates everything else by a factor of ~40. Both the
correction and the feature caches are written to disk and reused, so a second run skips
them.

## What you should see

Numbers to check a run against. All of these are in `results/`.

**Leave-one-video-out over 20 videos, official `f1_event`** — the four families tie, and
the spread between folds is much larger than the spread between models:

| model | `f1_event` | `f1_eval` |
|---|---|---|
| `histgb` | 0.579 ± 0.155 | 0.703 ± 0.176 |
| `bilstm` | 0.577 ± 0.143 | 0.716 ± 0.152 |
| `tcn` | 0.562 ± 0.129 | 0.706 ± 0.144 |
| `rf` | 0.533 ± 0.183 | 0.645 ± 0.214 |

**Skeleton correction, 20 paired folds, BiLSTM** (`results/skeleton_effect_deep.csv`) —
at or below raw on every metric:

| metric | delta (corrected − raw) | 95% CI | Wilcoxon p |
|---|---|---|---|
| `f1_eval` | −0.027 | [−0.049, −0.004] | 0.070 |
| `f1_full` | −0.033 | [−0.070, +0.004] | 0.090 |
| `f1_event` | −0.005 | [−0.018, +0.008] | 0.622 |

**Active learning, 5 seeds, group split by video** (`results/al_honest_summary.csv`) —
the best strategy is `entropy` at AULC 0.174 ± 0.032 against random 0.138 ± 0.023,
Wilcoxon p = 0.063. No strategy clears significance.

**Leaderboard simulation, nine laboratories** (historical) — 0.494 mean
(`results/lb_best.json`), but 0.411 over the five that resemble the real test set. The
two were reported separately because that test set excludes CalMS21 and CRIM13, four of
the nine and also the four easiest. This is the 9-lab system; it was superseded once
every annotated laboratory was trained.

**Leaderboard simulation, nineteen laboratories** (current) — composed calibration
**0.4861** over all 19, mixing window sizes per laboratory (16 labs at `W=32`, 3 at
`W=64`). Reproduce the comparison and the composition with:

```bash
.venv/Scripts/python scripts/lb_compare_calib.py lb19_v8 lb19_v9w32
.venv/Scripts/python scripts/lb_compose_models.py --runs lb19_v8 lb19_v9w32     --models models/lb19_v8 models/lb19_v9w32 --out models/lb19_mix_sub --no-revival
```

### On the real leaderboard

Four submissions, each isolating one change:

| submission | what changed | public | private |
|---|---|---|---|
| v4 | video truncated to 15,000 frames | 0.38674 | 0.35554 |
| v5 | full-length video, 16 labs | 0.41142 | 0.38527 |
| v8 | no frame cap, 19 labs, 821 training videos | 0.43154 | 0.41346 |
| **mixed `W=32`/`W=64`** | per-lab window size | **0.45515** | **0.43412** |

Every step is a data-volume or windowing change; not one is a change of model. The
three levers were `frame_limit` (a debug cap that reached production), `--cap-per-lab`
(which was letting only 53% of available windows reach training), and `window_size`
(fixed at 64 across all fourteen cache builds, never once varied).

Two results worth stating plainly, because both contradict what was expected:

* **The emission grid was never the bottleneck.** Predicted interval boundaries are
  quantised to the stride, capping the metric at 0.9520 for `S=8`. Halving the stride
  lifts that ceiling to 0.9769 but bought only **+0.002** — the errors are whole missed
  intervals, not misplaced edges.
* **The `W=32` experiment succeeded while its hypothesis failed.** It was motivated by
  one laboratory whose actions last 9 frames inside a 64-frame window; that laboratory
  did not move (−0.0007), and none of the sixteen that improved had been predicted.

## Layout

```
src/
  data/             loader, features, feature_pipeline, feature_selection,
                    dataset, correction_cache
  skeleton/         keypoint_smoother, mouse_skeleton, kinematic_constraints,
                    pose_lifting, metrics, viz, config, _parallel_worker
  models/           baseline (RF, HistGB, SVM), rnn (BiLSTM, CNN-LSTM, GRU, TCN,
                    Transformer), graph (GATv2 + BiLSTM)
  active_learning/  query_strategies, oracle
  eval/             mabe_metric: the official interval F-beta
  evaluation/       lovo (leave-one-video-out harness), downstream
  submit/           inference-only prediction path
  train.py          training loop with early stopping
  evaluate.py       metrics, confusion matrices, calibration
notebooks/          01_eda -> 02a-02d skeleton -> 03_features -> 04_baseline
                    -> 05_deep_model -> 06_active_learning
                    -> 07_leaderboard_simulation -> 08_kaggle_submission
scripts/            runnable harnesses, mostly the multi-lab leaderboard simulation
configs/            YAML for the correction pipeline and its variants
tests/              unit tests for the high-risk pure functions
results/            CSV and JSON summaries of every experiment
```

## Scripts

| script | what it does |
|---|---|
| `build_lb_cache.py` | build the multi-lab feature cache (shared 7-keypoint schema) |
| `run_lb_experiment.py` | train and score under the leaderboard-shaped protocol |
| `run_lb_perlab.py` | one model per laboratory, with out-of-sample threshold calibration |
| `lb_threshold.py` | per-action decision thresholds by coordinate ascent |
| `lb_context.py` | asymmetric past and future context features |
| `lb_prevalence.py` | revive dead classes by prevalence |
| `run_lb_decision.py`, `run_lb_decide_perlab.py` | decision layer: active mask, minimum duration, tie-break mode |
| `run_lb_finalize.py`, `run_lb_select_arm.py` | pick the configuration per lab on calibration videos, never on test |
| `run_lb_gpu.py`, `run_lb_ensemble.py` | GPU sequence model per lab, and its blend with the trees |
| `lb_viterbi.py` | sequential decoding (a negative result, kept for the record) |
| `lb_diagnose.py` | per (lab, action) failure diagnosis: dead class, over- or under-prediction |
| `lb_compare_calib.py`, `lb_compose_models.py` | compare two runs lab by lab and compose the best of each |
| `make_lb_summary.py` | consolidate the CSVs into `results/lb_summary.md` |
| `make_al_honest_figure.py` | the paired active-learning figure |
| `build_permouse_cache.py`, `run_permouse_lovo.py` | per-mouse skeleton reference ablation |
| `update_bundles.py` | fill model bundles with what the decision layer needs at inference |
| `make_submission_notebook.py` | generate `notebooks/08_kaggle_submission.ipynb` |
| `_selftest_*.py` | contract checks: official metric, event bridge, submission schema, inference equivalence |

## Tests

```bash
.venv/Scripts/python -m pytest tests/ -q
```

34 tests covering label encoding, the temporal-split purge, LOVO metrics, the
correction-cache key, feature selection, window building, the annotation cleaner and
the official metric. They need no data and run in about ten seconds.

## What is not in this repository

- **The corpus.** Distributed by the challenge; see `dataset/README.md`.
- **Trained weights and model bundles.** `results/checkpoints*`, `models/` and the
  intermediate `.npz` feature caches run to well over a gigabyte and exceed GitHub's
  per-file limit. Every one is regenerated by re-running the step that wrote it; the
  CSV and JSON summaries beside them are tracked, and they are the evidence behind the
  numbers above.
- **The written dissertation and the defence slides**, which are Spanish documents
  submitted through the university.

Because of the second point, `notebooks/08_kaggle_submission.ipynb` and `src/submit/` will
not run end to end from a fresh clone: they load per-laboratory bundles that are not
distributed. The inference path itself is covered by `_selftest_submission.py` and
`_selftest_inference_equiv.py`.

A handful of plot labels and one LaTeX table caption are deliberately still in Spanish,
because those figures are rendered into the Spanish dissertation. They are marked with
a comment where they occur.
