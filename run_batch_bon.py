"""
Batch BoN inference script for Cosmos3 I2V + GeoReward.

Iterates over all image+prompt pairs defined in a JSON config file,
runs BoN (Best-of-N) for each case, and saves ALL N candidate videos.

Usage (on server):
    cd /pfs/mayuema/spj/now_cosmos3
    conda activate now_cosmos3
    CUDA_VISIBLE_DEVICES=0 python run_batch_bon.py \
        --batch_json batch_prompts_inputs_real_6.json \
        --image_dir /pfs/mayuema/spj/now/inputs/inputs_real_6 \
        --output_dir outputs/bon_inputs_real_6 \
        --N 8 \
        --seed_base 42
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
        description="Batch BoN inference: Cosmos3 I2V + GeoReward"
    )

    # Batch config
    parser.add_argument(
        "--batch_json", type=str, required=True,
        help="Path to JSON file mapping image stems to prompts.",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--image_ext", type=str, default=".jpg",
        help="Image file extension (default: .jpg).",
    )

    # Cosmos3 model
    parser.add_argument(
        "--model", type=str, default="/pfs/mayuema/spj/now_cosmos3/Cosmos3-Nano",
        help="Cosmos3 model path.",
    )

    # Generation params
    parser.add_argument("--num_frames", type=int, default=189)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--negative_prompt", type=str, default="")

    # BoN params
    parser.add_argument("--N", type=int, default=8, help="Number of candidates per case.")
    parser.add_argument("--seed_base", type=int, default=42, help="Base seed.")

    # 4RC model
    parser.add_argument(
        "--fourrc_model", type=str,
        default="/pfs/mayuema/spj/now/4RC-main/4RC-main/checkpoints/4RC",
        help="4RC model path.",
    )
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--max_frames", type=int, default=20)

    # Reward weights
    parser.add_argument("--static_weight", type=float, default=0.40)
    parser.add_argument("--dynamic_weight", type=float, default=0.40)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # Memory management
    parser.add_argument("--offload", action="store_true",
                        help="Alternate offload Cosmos3/4RC for limited VRAM.")

    # Output
    parser.add_argument("--output_dir", type=str, default="outputs/bon_inputs_real_6")

    # Resume support
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases that already have output directories.")

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

    # Gradient guidance params
    parser.add_argument("--geo_guidance_scale", type=float, default=0.001)
    parser.add_argument("--geo_guidance_frequency", type=int, default=5)
    parser.add_argument("--geo_guidance_sigma_min", type=float, default=0.08)
    parser.add_argument("--geo_guidance_sigma_max", type=float, default=0.83)
    parser.add_argument("--guidance_frames", type=int, default=8)

    return parser.parse_args()


def _disable_cosmos_guardrail():
    """Mock cosmos_guardrail to avoid downloading gated Cosmos-1.0-Guardrail repo."""
    import types
    import importlib.machinery
    mock = types.ModuleType("cosmos_guardrail")
    mock.__spec__ = importlib.machinery.ModuleSpec("cosmos_guardrail", None)
    mock.__version__ = "0.0.0"

    class _NoOpSafetyChecker:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return args[0] if args else None

    mock.CosmosSafetyChecker = _NoOpSafetyChecker
    sys.modules["cosmos_guardrail"] = mock


def load_cosmos3_pipeline(args):
    _disable_cosmos_guardrail()

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

    pipe.safety_checker = None
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )
    return pipe


def load_4rc_model(model_path, device="cpu"):
    sys.path.insert(0, "/pfs/mayuema/spj/now_cosmos3/4RC-main/4RC-main")
    from arc.models.arc.arc import Arc

    logger.info(f"Loading 4RC model from: {model_path}")
    model = Arc.from_pretrained(model_path)
    model = model.to(device).eval()
    return model


def main():
    args = parse_args()

    # Load batch config
    with open(args.batch_json, "r", encoding="utf-8") as f:
        batch_config = json.load(f)

    total_cases = len(batch_config)
    logger.info(f"Loaded {total_cases} cases from {args.batch_json}")
    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"N={args.N}, seed_base={args.seed_base}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    fourrc_device = "cpu" if args.offload else "cuda"
    fourrc_model = load_4rc_model(args.fourrc_model, device=fourrc_device)

    from geo_reward.recon_reward import ReconRewardConfig, ReconstructionReward
    cfg = ReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )
    recon_reward = ReconstructionReward(model=fourrc_model, device="cuda", cfg=cfg)

    pipe = load_cosmos3_pipeline(args)

    # Validate guidance compatibility
    if args.guidance and args.progressive and not args.tree_branching:
        logger.warning(
            "--guidance incompatible with progressive (without --tree_branching). "
            "Ignoring --guidance."
        )
        args.guidance = False

    if args.guidance and args.tree_branching:
        if "--geo_guidance_frequency" not in sys.argv:
            args.geo_guidance_frequency = 3
        if "--geo_guidance_sigma_max" not in sys.argv:
            args.geo_guidance_sigma_max = 0.83

    # Build the appropriate BoN pipeline
    use_advanced = args.progressive or args.tree_branching

    if use_advanced:
        from geo_reward.cosmos3_adapter import Cosmos3ProgressiveAdapter
        adapter = Cosmos3ProgressiveAdapter(pipe)

        adapter_kwargs = dict(
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.guidance_scale,
            negative_prompt=args.negative_prompt,
        )

        if args.tree_branching and args.guidance:
            from geo_reward.guidance import GeometricGuidance
            from geo_reward.bon_pipeline import Cosmos3GeoRewardBoNTreeBranchingGuided

            guidance_cfg = ReconRewardConfig(
                static_weight=cfg.static_weight,
                dynamic_weight=cfg.dynamic_weight,
                motion_weight=cfg.motion_weight,
                dynamic_threshold_ratio=cfg.dynamic_threshold_ratio,
                tau_reproj=cfg.tau_reproj,
                occlusion_margin=cfg.occlusion_margin,
                tau_accel=cfg.tau_accel,
                tau_speed=cfg.tau_speed,
                max_sample_pixels=cfg.max_sample_pixels,
                tau_cam=cfg.tau_cam,
                tau_rot=cfg.tau_rot,
                min_motion=cfg.min_motion,
                tau_motion=cfg.tau_motion,
                conf_valid_quantile=cfg.conf_valid_quantile,
                max_frames=cfg.max_frames,
                image_size=cfg.image_size,
                geo_guidance_scale=args.geo_guidance_scale,
                geo_guidance_frequency=args.geo_guidance_frequency,
                sigma_min=args.geo_guidance_sigma_min,
                sigma_max=args.geo_guidance_sigma_max,
            )

            guidance = GeometricGuidance(
                model_4rc=fourrc_model,
                vae=pipe.vae,
                cfg=guidance_cfg,
                guidance_frames=args.guidance_frames,
                vae_latents_mean=pipe._vae_latents_mean,
                vae_latents_inv_std=pipe._vae_latents_inv_std,
            )
            bon = Cosmos3GeoRewardBoNTreeBranchingGuided(
                adapter=adapter, recon_reward=recon_reward, guidance=guidance,
                num_trunks=args.num_trunks, branches_per_trunk=args.branches_per_trunk,
                branch_sigma=args.branch_sigma, branch_eta=args.branch_eta,
                max_frames=args.max_frames, sigma_checkpoints=args.sigma_checkpoints,
                elimination_ratio=args.elimination_ratio,
                min_survivors=args.min_survivors,
                score_epsilon=args.score_epsilon,
                early_max_frames=args.early_max_frames,
                offload_models=args.offload,
            )
            effective_N = args.num_trunks * args.branches_per_trunk
        elif args.tree_branching:
            from geo_reward.bon_pipeline import Cosmos3GeoRewardBoNTreeBranching
            bon = Cosmos3GeoRewardBoNTreeBranching(
                adapter=adapter, recon_reward=recon_reward,
                num_trunks=args.num_trunks, branches_per_trunk=args.branches_per_trunk,
                branch_sigma=args.branch_sigma, branch_eta=args.branch_eta,
                max_frames=args.max_frames, sigma_checkpoints=args.sigma_checkpoints,
                elimination_ratio=args.elimination_ratio,
                min_survivors=args.min_survivors,
                score_epsilon=args.score_epsilon,
                early_max_frames=args.early_max_frames,
                offload_models=args.offload,
            )
            effective_N = args.num_trunks * args.branches_per_trunk
        else:
            from geo_reward.bon_pipeline import Cosmos3GeoRewardBoNProgressiveV2
            bon = Cosmos3GeoRewardBoNProgressiveV2(
                adapter=adapter, recon_reward=recon_reward,
                max_frames=args.max_frames, sigma_checkpoints=args.sigma_checkpoints,
                elimination_ratio=args.elimination_ratio,
                min_survivors=args.min_survivors,
                score_epsilon=args.score_epsilon,
                early_max_frames=args.early_max_frames,
                offload_models=args.offload,
            )
            effective_N = args.N
    else:
        from geo_reward.bon_pipeline import Cosmos3GeoRewardBoN
        bon = Cosmos3GeoRewardBoN(
            pipe=pipe,
            recon_reward=recon_reward,
            max_frames=args.max_frames,
            offload=args.offload,
        )

    # Run batch
    all_results = []
    start_time = time.time()

    for idx, (image_stem, prompt) in enumerate(batch_config.items()):
        case_dir = os.path.join(args.output_dir, image_stem)

        # Resume: skip if already done
        if args.resume and os.path.exists(os.path.join(case_dir, "rewards.json")):
            logger.info(f"[{idx+1}/{total_cases}] SKIP (already done): {image_stem}")
            continue

        image_path = os.path.join(args.image_dir, f"{image_stem}{args.image_ext}")
        if not os.path.exists(image_path):
            logger.warning(f"[{idx+1}/{total_cases}] Image not found: {image_path}, skipping.")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"[{idx+1}/{total_cases}] Processing: {image_stem}")
        logger.info(f"  Image: {image_path}")
        logger.info(f"  Prompt: {prompt[:80]}...")
        logger.info(f"{'='*60}")

        img = Image.open(image_path).convert("RGB")
        os.makedirs(case_dir, exist_ok=True)

        t0 = time.time()
        try:
            if use_advanced:
                best_frames, result_log, best_seed = bon.generate(
                    prompt=prompt,
                    image=img,
                    N=effective_N,
                    seed_base=args.seed_base,
                    output_dir=case_dir,
                    **adapter_kwargs,
                )
                elapsed = time.time() - t0

                result = {
                    "image_stem": image_stem,
                    "prompt": prompt,
                    "best_seed": best_seed,
                    "time_sec": elapsed,
                    **{k: v for k, v in result_log.items()
                       if k not in ("prompt", "image")},
                }
            else:
                best_frames, rewards, best_idx = bon.generate(
                    prompt=prompt,
                    image=img,
                    N=args.N,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    seed_base=args.seed_base,
                    save_all=True,
                    output_dir=case_dir,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    negative_prompt=args.negative_prompt,
                )
                elapsed = time.time() - t0

                result = {
                    "image_stem": image_stem,
                    "prompt": prompt,
                    "best_index": best_idx,
                    "best_seed": args.seed_base + best_idx,
                    "best_reward": rewards[best_idx]["total"],
                    "time_sec": elapsed,
                    "candidates": [
                        {"index": i, "reward": r, "is_best": i == best_idx}
                        for i, r in enumerate(rewards)
                    ],
                }
            all_results.append(result)

            # Save per-case results
            with open(os.path.join(case_dir, "rewards.json"), "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            if use_advanced:
                logger.info(
                    f"  => Best: seed_{result['best_seed']} "
                    f"time={elapsed:.1f}s"
                )
            else:
                logger.info(
                    f"  => Best: candidate {best_idx+1}/{args.N} "
                    f"(seed={args.seed_base + best_idx}, "
                    f"reward={rewards[best_idx]['total']:.4f}), "
                    f"time={elapsed:.1f}s"
                )

        except Exception as e:
            logger.error(f"  ERROR processing {image_stem}: {e}")
            import traceback
            traceback.print_exc()
            continue

        del img, best_frames
        torch.cuda.empty_cache()

    # Save overall summary
    total_time = time.time() - start_time
    summary = {
        "batch_json": args.batch_json,
        "image_dir": args.image_dir,
        "total_cases": total_cases,
        "completed_cases": len(all_results),
        "N": args.N,
        "seed_base": args.seed_base,
        "total_time_sec": total_time,
        "avg_time_per_case_sec": total_time / max(len(all_results), 1),
        "model": args.model,
        "fourrc_model": args.fourrc_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": all_results,
    }

    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {len(all_results)}/{total_cases} cases")
    logger.info(f"Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
