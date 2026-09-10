"""
批量 Best-of-N CLI（Cosmos3 + VGGTomega + CoTracker3）。

用法：
    python VGGTomega_Cotracker3/run_bon_batch_vggt_cosmos3.py \
        --batch_json batch_prompts_inputs_real_6.json \
        --image_dir /path/to/inputs/inputs_real_6 \
        --output_dir outputs/bon_vggt_inputs_real_6 \
        --model /path/to/Cosmos3-Nano \
        --vggt_model /path/to/vggt_omega.pth \
        --N 8
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量 GeoReward VGGT+CoTracker3 Best-of-N Pipeline (Cosmos3)"
    )

    # 批次配置
    parser.add_argument("--batch_json", type=str, required=True,
                        help="JSON 文件路径，映射图片 stem 到 prompt。")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="输入图片目录。")
    parser.add_argument("--image_ext", type=str, default=".jpg",
                        help="图片文件扩展名。")

    # Cosmos3 模型
    parser.add_argument("--model", type=str, required=True,
                        help="Cosmos3 模型路径。")

    # 生成参数
    parser.add_argument("--num_frames", type=int, default=189)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_inference_steps", type=int, default=35)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=10.0)
    parser.add_argument("--negative_prompt", type=str, default="")

    # BoN 参数
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--seed_base", type=int, default=42)

    # VGGTomega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGTomega 权重路径。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--image_resolution", type=int, default=512)

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50)
    parser.add_argument("--dynamic_weight", type=float, default=0.30)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/bon_vggt_batch")

    return parser.parse_args()


def main():
    args = parse_args()

    # ⚠️ 必须在导入 Cosmos3OmniPipeline 之前 mock cosmos_guardrail
    import sys
    import types
    import importlib.machinery

    # Mock cosmos_guardrail 模块以避免下载 gated Cosmos-1.0-Guardrail
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

    # 延迟导入（避免 --help 时加载依赖）
    import torch
    from PIL import Image
    from diffusers import Cosmos3OmniPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    from VGGTomega_Cotracker3.bon_pipeline_vggt import GeoRewardBoNVGGT
    from VGGTomega_Cotracker3.recon_reward_vggt import VGGTReconRewardConfig
    from VGGTomega_Cotracker3.utils import save_video_from_pil

    # 1. 加载 batch JSON
    with open(args.batch_json, "r") as f:
        batch_prompts = json.load(f)

    logger.info(f"加载 {len(batch_prompts)} 个测试用例")

    # 2. 加载 Cosmos3 模型（只加载一次）
    logger.info(f"加载 Cosmos3 模型: {args.model}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    pipe.safety_checker = None

    # ✅ flow_shift 在 scheduler 上设置
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )

    # 3. 配置 Reward
    reward_cfg = VGGTReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        image_size=args.image_resolution,
        chunk_size=args.chunk_size,
        max_frames=args.num_frames_for_reward,
    )

    # 4. 创建 BoN pipeline（模型延迟加载）
    bon = GeoRewardBoNVGGT(
        cosmos3_pipe=pipe,
        vggt_model_path=args.vggt_model,
        cotracker_model_path=args.cotracker_model,
        N=args.N,
        num_frames_for_reward=args.num_frames_for_reward,
        reward_config=reward_cfg,
        chunk_size=args.chunk_size,
    )

    # 5. 批量处理
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    for case_idx, (image_stem, prompt) in enumerate(batch_prompts.items(), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Case {case_idx}/{len(batch_prompts)}: {image_stem}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"{'='*60}")

        # 加载输入图片
        image_path = os.path.join(args.image_dir, image_stem + args.image_ext)
        if not os.path.exists(image_path):
            logger.warning(f"图片不存在: {image_path}，跳过")
            continue

        image = Image.open(image_path).convert("RGB")

        # 执行 BoN 生成
        case_output_dir = os.path.join(args.output_dir, image_stem)
        os.makedirs(case_output_dir, exist_ok=True)

        try:
            result = bon.generate(
                image=image,
                prompt=prompt,
                N=args.N,
                num_frames=args.num_frames,
                fps=args.fps,
                seed_base=args.seed_base,
                output_dir=case_output_dir,
                save_fn=save_video_from_pil,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.negative_prompt,
                # ⚠️ 不传 flow_shift
            )

            logger.info(f"最优: seed={result['best_seed']}, score={result['best_score']:.4f}")

            # 保存结果
            case_result = {
                "case": image_stem,
                "prompt": prompt,
                "best_seed": result["best_seed"],
                "best_score": result["best_score"],
                "all_scores": result["all_scores"],
                "all_seeds": result["seeds"],
                "time_sec": result["total_time_sec"],
            }
            all_results.append(case_result)

            # 保存 case 级别 JSON
            with open(os.path.join(case_output_dir, "result.json"), "w") as f:
                json.dump(case_result, f, indent=2)

        except Exception as e:
            logger.error(f"Case {image_stem} 失败: {e}", exc_info=True)
            continue

    # 6. 保存汇总结果
    summary = {
        "total_cases": len(batch_prompts),
        "completed_cases": len(all_results),
        "results": all_results,
        "config": vars(args),
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n批量处理完成: {len(all_results)}/{len(batch_prompts)} 成功")
    logger.info(f"汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
