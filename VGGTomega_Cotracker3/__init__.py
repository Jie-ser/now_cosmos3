"""
VGGTomega + CoTracker3 组合重建后端。

提供与 4RC 兼容的几何数据输出和 GeoReward 评分。
"""

from .vggt_omega_adapter import (
    load_vggt_omega,
    preprocess_frames,
    run_vggt_omega_inference,
)

from .cotracker3_adapter import (
    load_cotracker3,
    frames_to_video_tensor,
    run_dense_tracking,
)

from .combo_adapter import run_combo_inference

from .recon_reward_vggt import (
    VGGTReconstructionReward,
    VGGTReconRewardConfig,
)

from .bon_pipeline_vggt import GeoRewardBoNVGGT

from .utils import save_video_from_pil

__all__ = [
    # Adapters
    "load_vggt_omega",
    "preprocess_frames",
    "run_vggt_omega_inference",
    "load_cotracker3",
    "frames_to_video_tensor",
    "run_dense_tracking",
    "run_combo_inference",
    # Reward
    "VGGTReconstructionReward",
    "VGGTReconRewardConfig",
    # Pipeline
    "GeoRewardBoNVGGT",
    # Utils
    "save_video_from_pil",
]
