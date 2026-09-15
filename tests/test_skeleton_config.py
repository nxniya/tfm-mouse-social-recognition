"""A1: skeleton.yaml is the single source of truth for correction params.

Guards the unification: skeleton_smooth_kwargs() must reflect configs/skeleton.yaml
(not code defaults or default.yaml), and downstream.smooth_kwargs_from_cfg must
delegate to it.
"""
from src.skeleton.config import load_skeleton_cfg, skeleton_smooth_kwargs


def test_kwargs_reflect_skeleton_yaml():
    cfg = load_skeleton_cfg()
    kw = skeleton_smooth_kwargs(cfg)
    # Values that must come from skeleton.yaml, not the divergent code/default.yaml.
    assert kw["lambda_l"] == float(cfg.weights.L)
    assert kw["outlier_adhesion"] == float(cfg.solver.outlier_adhesion)
    assert kw["n_passes"] == int(cfg.solver.n_passes)
    assert kw["non_outlier_adhesion_scale"] == float(cfg.solver.non_outlier_adhesion_scale)
    assert kw["length_threshold"] == float(cfg.outlier.length_threshold)


def test_kwargs_are_valid_smooth_video_signature():
    import inspect
    from src.skeleton.keypoint_smoother import smooth_video
    params = set(inspect.signature(smooth_video).parameters)
    assert set(skeleton_smooth_kwargs()).issubset(params)


def test_downstream_wrapper_delegates():
    from src.evaluation.downstream import smooth_kwargs_from_cfg
    cfg = load_skeleton_cfg()
    assert smooth_kwargs_from_cfg(cfg) == skeleton_smooth_kwargs(cfg)


def test_default_yaml_no_longer_defines_correction_params():
    # The default.yaml skeleton: block must not re-introduce diverging numerics.
    from omegaconf import OmegaConf
    from src.skeleton.config import REPO
    cfg = OmegaConf.load(str(REPO / "configs" / "default.yaml"))
    sk = cfg.skeleton
    for leaked in ("lambda_l", "outlier_adhesion", "n_passes", "speed_sigma"):
        assert leaked not in sk, f"default.yaml skeleton: still defines {leaked}"
    assert "use_correction_cache" in sk  # the surviving pipeline toggle
