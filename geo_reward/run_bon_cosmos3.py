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

    # --- Advanced BoN modes ---
    parser.add_argument("--progressive", action="store_true",
                        help="Enable progressive elimination BoN.")
    parser.add_argument("--tree_branching", action="store_true",
                        help="Enable tree branching BoN.")
    parser.add_argument("--guidance", action="store_true",
                        help="Enable gradient guidance (requires tree_branching + multi-GPU).")

    # Progressive elimination params
    parser.add_argument("--sigma_checkpoints", nargs="+", type=float,
                        default=[0.83, 0.63])
    parser.add_argument("--elimination_ratio", type=float, default=0.5)
    parser.add_argument("--min_survivors", type=int, default=2)
    parser.add_argument("--score_epsilon", type=float, default=0.02)
    parser.add_argument("--early_max_frames", type=int, default=12)

    # Tree branching params
    parser.add_argument("--num_trunks", type=int, default=2)
    parser.add_argument("--branches_per_trunk", type=int, default=4)
    parser.add_argument("--branch_sigma", type=float, default=0.90)
    parser.add_argument("--branch_eta", type=float, default=0.10)

    # Gradient guidance params (geo_ prefix to avoid conflict with CFG --guidance_scale)
    parser.add_argument("--geo_guidance_scale", type=float, default=0.001,
                        help="Geometric guidance strength (NOT CFG scale).")
    parser.add_argument("--geo_guidance_frequency", type=int, default=5,
                        help="Apply guidance every N-th denoising step.")
    parser.add_argument("--geo_guidance_sigma_min", type=float, default=0.08)
    parser.add_argument("--geo_guidance_sigma_max", type=float, default=0.83,
                        help="Max sigma for guidance window (0.83 = after first elimination).")
    parser.add_argument("--guidance_frames", type=int, default=8)

    return parser.parse_args()


def build_recon_config(args):
    from .recon_reward import ReconRewardConfig
    kwargs = dict(
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
    if hasattr(args, "geo_guidance_scale"):
        kwargs.update(
            geo_guidance_scale=args.geo_guidance_scale,
            geo_guidance_frequency=args.geo_guidance_frequency,
            sigma_min=args.geo_guidance_sigma_min,
            sigma_max=args.geo_guidance_sigma_max,
        )
    return ReconRewardConfig(**kwargs)


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
    """Load Cosmos3 Diffusers pipeline.

    For progressive / tree_branching modes, uses device_map=None so that
    manual model offloading works correctly (device_map="cuda" installs
    accelerate dispatch hooks that conflict with manual .to() calls).
    """
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    logger.info(f"Loading Cosmos3 pipeline from: {args.model}")

    needs_manual_offload = getattr(args, "progressive", False) or getattr(
        args, "tree_branching", False
    )

    if needs_manual_offload:
        pipe = Cosmos3OmniPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
        pipe = pipe.to("cuda")
    else:
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

    # Validate guidance compatibility (matches now project logic)
    if args.guidance and args.progressive and not args.tree_branching:
        logger.warning(
            "--guidance is incompatible with progressive elimination "
            "(without --tree_branching). Gradient guidance only works with "
            "tree_branching or sequential BoN. Ignoring --guidance flag."
        )
        args.guidance = False

    # Tree_branching + guidance: auto-adjust defaults
    if args.guidance and args.tree_branching:
        if "--geo_guidance_frequency" not in sys.argv:
            args.geo_guidance_frequency = 3
            cfg.geo_guidance_frequency = 3
        if "--geo_guidance_sigma_max" not in sys.argv:
            args.geo_guidance_sigma_max = 0.83
            cfg.sigma_max = 0.83

    # Load 4RC
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

    # Shared adapter kwargs for progressive modes
    adapter_kwargs = dict(
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    )

    t0 = time.time()

    if args.tree_branching and args.guidance:
        # --- Tree Branching + Gradient Guidance ---
        # Requires multi-GPU: transformer stays on GPU0, VAE+4RC on other GPUs.
        # With fewer than 3 GPUs, fall back to offload mode with a warning.
        from .cosmos3_adapter import Cosmos3ProgressiveAdapter
        from .guidance import GeometricGuidance
        from .bon_pipeline import Cosmos3GeoRewardBoNTreeBranchingGuided

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            logger.warning(
                "Gradient guidance with single GPU is likely to OOM. "
                "Consider using --tree_branching without --guidance, "
                "or use a multi-GPU setup (>= 3 GPUs recommended, 4 ideal)."
            )
            vae_device = None
            fourrc_device = None
        elif num_gpus < 4:
            logger.info(
                f"Detected {num_gpus} GPUs. Guidance in reduced multi-GPU mode: "
                f"transformer=cuda:0, VAE+4RC=cuda:{num_gpus - 1}."
            )
            vae_device = f"cuda:{num_gpus - 1}"
            fourrc_device = f"cuda:{num_gpus - 1}"
        else:
            logger.info(
                f"Detected {num_gpus} GPUs. Guidance in 4-GPU resident mode: "
                f"transformer=cuda:0, VAE=cuda:1, 4RC=cuda:2."
            )
            vae_device = "cuda:1"
            fourrc_device = "cuda:2"

        adapter = Cosmos3ProgressiveAdapter(pipe)
        guidance = GeometricGuidance(
            model_4rc=fourrc_model,
            vae=pipe.vae,
            cfg=cfg,
            guidance_frames=args.guidance_frames,
            vae_latents_mean=pipe._vae_latents_mean,
            vae_latents_inv_std=pipe._vae_latents_inv_std,
            vae_device=vae_device,
            fourrc_device=fourrc_device,
        )

        # In multi-GPU resident mode, no offload needed during guidance steps.
        # In single-GPU mode, offload_models=True handles DiT↔4RC swapping.
        use_offload = args.offload or (num_gpus < 2)

        bon = Cosmos3GeoRewardBoNTreeBranchingGuided(
            adapter=adapter,
            recon_reward=recon_reward,
            guidance=guidance,
            num_trunks=args.num_trunks,
            branches_per_trunk=args.branches_per_trunk,
            branch_sigma=args.branch_sigma,
            branch_eta=args.branch_eta,
            max_frames=args.max_frames,
            sigma_checkpoints=args.sigma_checkpoints,
            elimination_ratio=args.elimination_ratio,
            min_survivors=args.min_survivors,
            score_epsilon=args.score_epsilon,
            early_max_frames=args.early_max_frames,
            offload_models=use_offload,
        )
        N = args.num_trunks * args.branches_per_trunk
        best_frames, result_log, best_seed = bon.generate(
            prompt=args.prompt,
            image=img,
            N=N,
            seed_base=args.seed_base,
            output_dir=case_dir,
            **adapter_kwargs,
        )
        rewards = result_log

    elif args.tree_branching:
        # --- Tree Branching ---
        from .cosmos3_adapter import Cosmos3ProgressiveAdapter
        from .bon_pipeline import Cosmos3GeoRewardBoNTreeBranching

        adapter = Cosmos3ProgressiveAdapter(pipe)
        bon = Cosmos3GeoRewardBoNTreeBranching(
            adapter=adapter,
            recon_reward=recon_reward,
            num_trunks=args.num_trunks,
            branches_per_trunk=args.branches_per_trunk,
            branch_sigma=args.branch_sigma,
            branch_eta=args.branch_eta,
            max_frames=args.max_frames,
            sigma_checkpoints=args.sigma_checkpoints,
            elimination_ratio=args.elimination_ratio,
            min_survivors=args.min_survivors,
            score_epsilon=args.score_epsilon,
            early_max_frames=args.early_max_frames,
            offload_models=args.offload,
        )
        N = args.num_trunks * args.branches_per_trunk
        best_frames, result_log, best_seed = bon.generate(
            prompt=args.prompt,
            image=img,
            N=N,
            seed_base=args.seed_base,
            output_dir=case_dir,
            **adapter_kwargs,
        )
        rewards = result_log

    elif args.progressive:
        # --- Progressive Elimination ---
        from .cosmos3_adapter import Cosmos3ProgressiveAdapter
        from .bon_pipeline import Cosmos3GeoRewardBoNProgressiveV2

        adapter = Cosmos3ProgressiveAdapter(pipe)
        bon = Cosmos3GeoRewardBoNProgressiveV2(
            adapter=adapter,
            recon_reward=recon_reward,
            max_frames=args.max_frames,
            sigma_checkpoints=args.sigma_checkpoints,
            elimination_ratio=args.elimination_ratio,
            min_survivors=args.min_survivors,
            score_epsilon=args.score_epsilon,
            early_max_frames=args.early_max_frames,
            offload_models=args.offload,
        )
        best_frames, result_log, best_seed = bon.generate(
            prompt=args.prompt,
            image=img,
            N=args.N,
            seed_base=args.seed_base,
            output_dir=case_dir,
            **adapter_kwargs,
        )
        rewards = result_log

    else:
        # --- Default: basic sequential BoN (backward compatible) ---
        from .bon_pipeline import Cosmos3GeoRewardBoN
        bon = Cosmos3GeoRewardBoN(
            pipe=pipe,
            recon_reward=recon_reward,
            max_frames=args.max_frames,
            offload=args.offload,
        )
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
        best_seed = args.seed_base + best_idx if args.seed_base is not None else None

    total_time = time.time() - t0

    # Save results log
    if isinstance(rewards, dict):
        # Progressive / tree modes return a result_log dict
        results = rewards
        results["prompt"] = args.prompt
        results["image"] = os.path.abspath(args.image)
        results["total_time_sec"] = total_time
    else:
        # Basic sequential BoN returns a list of reward dicts
        results = {
            "mode": "bon",
            "prompt": args.prompt,
            "image": os.path.abspath(args.image),
            "N": args.N,
            "best_seed": best_seed,
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
                {"index": i, "reward": r}
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
