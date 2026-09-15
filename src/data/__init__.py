"""src/data — loading, features and dataset for the MABe pipeline."""

from src.data.loader import (
    load_metadata,
    load_tracking,
    load_tracking_wide,
    load_annotations,
    iter_lab_videos,
)
from src.data.features import (
    extract_features,
    normalize_coordinates,
    compute_kinematics,
    compute_segment_angles,
    compute_dz_features,
    compute_relational_features,
    build_windows,
    window_statistics,
    WINDOW_SIZE,
    STRIDE,
)
try:
    from src.data.dataset import (
        MouseBehaviorDataset,
        LabelEncoder,
        TARGET_CLASSES,
        train_val_test_split,
    )
except ImportError:
    # torch not installed — dataset classes unavailable; all other data utilities work
    pass

__all__ = [
    "load_metadata", "load_tracking", "load_tracking_wide",
    "load_annotations", "iter_lab_videos",
    "extract_features", "normalize_coordinates",
    "compute_kinematics", "compute_segment_angles",
    "compute_dz_features", "compute_relational_features",
    "build_windows", "window_statistics",
    "WINDOW_SIZE", "STRIDE",
    "MouseBehaviorDataset", "LabelEncoder", "TARGET_CLASSES",
    "train_val_test_split",
]
