"""src/skeleton — MouseSkeleton: kinematic correction and dz estimation."""

from src.skeleton.mouse_skeleton import (
    MouseSkeleton,
    SkeletonEdge,
    build_skeleton,
    fit_skeleton,
    skeleton_to_adjacency,
)
from src.skeleton.kinematic_constraints import (
    kinematic_cost,
    correct_frame,
    compute_constraint_violations,
    compute_residual_metrics,
    compute_adaptive_edge_thresholds,
    compute_edge_lambdas,
    compute_term_magnitudes,
    build_edge_cache,
    EdgeCache,
    DEFAULT_LAMBDA_L,
    DEFAULT_LAMBDA_T,
    DEFAULT_LAMBDA_ACC,
    DEFAULT_LAMBDA_ANGLE,
    DEFAULT_LAMBDA_JERK,
)
from src.skeleton.keypoint_smoother import (
    smooth_video, violation_rate, detect_and_fix_swaps,
    compute_observation_confidence, hampel_correct_video,
)
from src.skeleton.pose_lifting import estimate_delta_z, dz_segment_features, dz_by_behavior
from src.skeleton.viz import (
    draw_skeleton,
    frame_kp,
    get_violated_edges,
    find_best_frames,
)
from src.skeleton._parallel_worker import process_one_video, process_one_video_population

__all__ = [
    "MouseSkeleton", "SkeletonEdge", "build_skeleton", "fit_skeleton",
    "skeleton_to_adjacency",
    "kinematic_cost", "correct_frame", "compute_constraint_violations",
    "compute_residual_metrics", "compute_adaptive_edge_thresholds",
    "compute_edge_lambdas", "compute_term_magnitudes", "build_edge_cache", "EdgeCache",
    "DEFAULT_LAMBDA_L", "DEFAULT_LAMBDA_T",
    "DEFAULT_LAMBDA_ACC", "DEFAULT_LAMBDA_ANGLE", "DEFAULT_LAMBDA_JERK",
    "smooth_video", "violation_rate", "detect_and_fix_swaps",
    "compute_observation_confidence", "hampel_correct_video",
    "estimate_delta_z", "dz_segment_features", "dz_by_behavior",
    "draw_skeleton", "frame_kp", "get_violated_edges", "find_best_frames",
    "process_one_video", "process_one_video_population",
]
