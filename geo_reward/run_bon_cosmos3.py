"""
Cosmos3 I2V + GeoReward Best-of-N Pipeline.

Generates N candidate videos with Cosmos3 (via Diffusers), scores each with
4RC-based GeoReward, and selects the geometrically most consistent one.

Usage:
    # Online BoN (generate + select)
    python -m geo_reward.run_bon_cosmos3 \
        --model nvidia/Cosmos3-Nano \
        --fourrc_model Luo-Yihang/4RC \
        --image /path/to/first_frame.png \
        --prompt "robot picks up the red block" \
        --N 8 --num_frames 189 --fps 24

    # Offline scoring
    python -m geo_reward.run_bon_cosmos3 \
        --mode score \
        --fourrc_model Luo-Yihang/4RC \
        --video_dir /path/to/videos/
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cosmos3 I2V + GeoReward Best-of-N Pipeline"
    )

    # Mode
    parser.add_argument(
        "--mode", type=str, default="bon", choices=["bon", "score"],
        help="'bon': generate + select; 'score': score existing videos.",
    )

    # Cosmos3 generation args
    parser.add_argument(
        "--model", type=str, default="nvidia/Cosmos3-Nano",
        help="HuggingFace model ID or local path for Cosmos3.",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to first frame image.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--num_frames", type=int, default=189, help="Number of output frames.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument("--height", type=int, default=720, help="Output height.")
    parser.add_argument("--width", type=int, default=1280, help="Output width.")
    parser.add_argument("--num_inference_steps", type=int, default=35, help="Denoising steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--flow_shift", type=float, default=10.0, help="Scheduler flow shift.")

    # BoN args
    parser.add_argument("--N", type=int, default=8, help="Number of candidates.")
    parser.add_argument("--seed_base", type=int, default=None, help="Base seed.")

    # Memory management
    parser.add_argument(
        "--offload", action="store_true",
        help="Enable Cosmos3/4RC alternating offload (for limited VRAM).",
    )
    parser.add_argument(
        "--save_all", action="store_true",
        help="Save all candidate videos to disk (default: only save best).",
    )

    # 4RC model args
    parser.add_argument(
        "--fourrc_model", type=str, required=True,
        help="4RC model: HuggingFace repo ID (e.g., 'Luo-Yihang/4RC') or local path.",
    )
    parser.add_argument("--image_size", type=int, default=518, help="4RC input resolution.")
    parser.add_argument("--max_frames", type=int, default=20, help="Keyframes for reward.")

    # V2 reward weights
    parser.add_argument("--static_weight", type=float, default=0.40)
    parser.add_argument("--dynamic_weight", type=float, default=0.40)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # Reward hyperparameters
    parser.add_argument("--dynamic_threshold_ratio", type=float, default=0.01)
    parser.add_argument("--tau_reproj", type=float, default=0.10)
    parser.add_argument("--occlusion_margin", type=float, default=1.05)
    parser.add_argument("--tau_accel", type=float, default=0.05)
    parser.add_argument("--tau_speed", type=float, default=3.0)
    parser.add_argument("--max_sample_pixels", type=int, default=1000)
    parser.add_argument("--tau_cam", type=float, default=0.02)
    parser.add_argument("--tau_rot", type=float, default=0.05)
    parser.add_argument("--min_motion", type=float, default=0.005)
    parser.add_argument("--tau_motion", type=float, default=0.005)
    parser.add_argument("--conf_valid_quantile", type=float, default=0.20)

    # Offline scoring args
    parser.add_argument("--video_dir", type=str, default=None, help="Video directory (for score mode).")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/geo_reward_bon_cosmos3")

    return parser.parse_args()


def build_recon_config(args):
    from .recon_reward import ReconRewardConfig
    return ReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        dynamic_threshold_ratio=args.dynamic_threshold_ratio,
        tau_reproj=args.tau_reproj,
        occlusion_margin=args.occlusion_margin,
        tau_accel=args.tau_accel,
        tau_speed=args.tau_speed,
        max_sample_pixels=args.max_sample_pixels,
        tau_cam=args.tau_cam,
        tau_rot=args.tau_rot,
        min_motion=args.min_motion,
        tau_motion=args.tau_motion,
        conf_valid_quantile=args.conf_valid_quantile,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )


def load_4rc_model(model_path, device="cpu"):
    """Load 4RC (Arc) model from checkpoint path or HuggingFace repo."""
    from .fourrc_adapter import _ensure_4rc_importable
    _ensure_4rc_importable()
    from arc.models.arc.arc import Arc

    logger.info(f"Loading 4RC model from: {model_path}")
    model = Arc.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def load_cosmos3_pipeline(args):
    """Load Cosmos3 Diffusers pipeline."""
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    logger.info(f"Loading Cosmos3 pipeline from: {args.model}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )
    return pipe


def run_bon(args):
    """Full Best-of-N pipeline: generate candidates and select best."""
    assert args.image is not None, "--image is required for BoN mode."
    assert args.prompt is not None, "--prompt is required for BoN mode."

    cfg = build_recon_config(args)

    # Load 4RC (start on CPU if offloading, otherwise GPU)
    fourrc_device = "cpu" if args.offload else "cuda"
    fourrc_model = load_4rc_model(args.fourrc_model, device=fourrc_device)

    from .recon_reward import ReconstructionReward
    recon_reward = ReconstructionReward(
        model=fourrc_model, device="cuda", cfg=cfg
    )

    # Load Cosmos3
    pipe = load_cosmos3_pipeline(args)

    # Load input image
    img = Image.open(args.image).convert("RGB")
    logger.info(f"Input image: {args.image} ({img.size[0]}x{img.size[1]})")
    logger.info(f"Prompt: {args.prompt}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_stem = Path(args.image).stem
    case_dir = os.path.join(args.output_dir, f"{image_stem}_{timestamp}")
    os.makedirs(case_dir, exist_ok=True)

    # Build BoN pipeline
    from .bon_pipeline import Cosmos3GeoRewardBoN
    bon = Cosmos3GeoRewardBoN(
        pipe=pipe,
        recon_reward=recon_reward,
        max_frames=args.max_frames,
        offload=args.offload,
    )

    # Run BoN
    t0 = time.time()
    best_frames, rewards, best_idx = bon.generate(
        prompt=args.prompt,
        image=img,
        N=args.N,
        num_frames=args.num_frames,
        fps=args.fps,
        seed_base=args.seed_base,
        save_all=args.save_all,
        output_dir=case_dir,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        negative_prompt=args.negative_prompt,
    )
    total_time = time.time() - t0

    # Save results log
    results = {
        "mode": "bon",
        "prompt": args.prompt,
        "image": os.path.abspath(args.image),
        "N": args.N,
        "best_index": best_idx,
        "best_seed": args.seed_base + best_idx if args.seed_base is not None else None,
        "best_reward": rewards[best_idx]["total"],
        "total_time_sec": total_time,
        "config": {
            "model": args.model,
            "fourrc_model": args.fourrc_model,
            "reward_version": "v2_4rc",
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "flow_shift": args.flow_shift,
            "offload": args.offload,
            "max_frames": args.max_frames,
            "image_size": args.image_size,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
        },
        "candidates": [
            {"index": i, "reward": r, "is_best": i == best_idx}
            for i, r in enumerate(rewards)
        ],
    }
    log_path = os.path.join(case_dir, "rewards.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {case_dir}")

    return best_frames, rewards


def run_score(args):
    """Score pre-generated videos offline."""
    assert args.video_dir is not None, "--video_dir is required for score mode."

    cfg = build_recon_config(args)
    fourrc_model = load_4rc_model(args.fourrc_model, device="cuda")

    from .recon_reward import ReconstructionReward
    recon_reward = ReconstructionReward(
        model=fourrc_model, device="cuda", cfg=cfg
    )

    from .bon_pipeline import Cosmos3GeoRewardOffline
    scorer = Cosmos3GeoRewardOffline(recon_reward=recon_reward, max_frames=args.max_frames)

    # Collect video files
    video_dir = Path(args.video_dir)
    video_files = sorted(
        list(video_dir.glob("*.mp4"))
        + list(video_dir.glob("*.avi"))
        + list(video_dir.glob("*.mov"))
    )

    if not video_files:
        logger.error(f"No video files found in {args.video_dir}")
        return

    logger.info(f"Found {len(video_files)} videos to score.")

    best_idx, rewards = scorer.select_best(
        [str(f) for f in video_files],
        num_frames=args.num_frames,
    )

    logger.info(
        f"\nBest video: {video_files[best_idx].name} "
        f"(reward={rewards[best_idx]['total']:.4f})"
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "mode": "score",
        "video_dir": str(args.video_dir),
        "reward_version": "v2_4rc",
        "best_file": video_files[best_idx].name,
        "best_reward": rewards[best_idx],
        "scores": [
            {"file": f.name, **r} for f, r in zip(video_files, rewards)
        ],
        "config": {
            "fourrc_model": args.fourrc_model,
            "image_size": args.image_size,
            "max_frames": args.max_frames,
            "static_weight": args.static_weight,
            "dynamic_weight": args.dynamic_weight,
            "motion_weight": args.motion_weight,
        },
    }
    log_path = os.path.join(
        args.output_dir,
        f"scores_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Scores saved to: {log_path}")


def main():
    args = parse_args()

    if args.mode == "bon":
        run_bon(args)
    elif args.mode == "score":
        run_score(args)


if __name__ == "__main__":
    main()
