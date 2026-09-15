# Dataset

The MABe mouse-behaviour corpus is distributed by the challenge, not by this
repository, so nothing under `dataset/` is tracked except this file.

To reproduce the experiments, download the competition data and unpack it here so
that the layout is:

```
dataset/
  MABe-mouse-behavior-detection/
    train.csv                 metadata: one row per video (lab, fps, arena size)
    train_tracking/<lab>/<video_id>.parquet
    train_annotation/<lab>/<video_id>.parquet
  features/                   generated: HDF5 and NPZ feature caches
  corrected/                  generated: skeleton-corrected tracking
```

`features/` and `corrected/` are written by the pipeline and are regenerable; see
the README at the repository root for the commands that build them.
