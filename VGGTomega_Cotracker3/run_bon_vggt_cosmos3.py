"""
单条 Best-of-N CLI（Cosmos3 + VGGTomega + CoTracker3）。

用法：
    python VGGTomega_Cotracker3/run_bon_vggt_cosmos3.py \
        --model /path/to/Cosmos3-Nano \
        --vggt_model /path/to/vggt_omega_checkpoint.pth \
        --image /path/to/first_frame.png \
        --prompt "robot arm picks up the red block" \
        --N 8 --num_frames 189 --height 720 --width 1280
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoReward VGGT+CoTracker3 Best-of-N Pipeline (Cosmos3 单条)"
    )

    # Cosmos3 生成参数
    parser.add_argument("--model", type=str, required=True,
                        help="Cosmos3 模型路径（Cosmos3-Nano 等）。")
    parser.add_argument("--image", type=str, required=True,
                        help="首帧图片路径。")
    parser.add_argument("--prompt", type=str, required=True,
                        help="动作指令文本。")
    parser.add_argument("--num_frames", type=int, default=189,
                        help="输出帧数。")
    parser.add_argument("--fps", type=int, default=24,
                        help="帧率。")
    parser.add_argument("--height", type=int, default=720,
                        help="输出高度。")
    parser.add_argument("--width", type=int, default=1280,
                        help="输出宽度。")
    parser.add_argument("--num_inference_steps", type=int, default=35,
                        help="去噪步数。")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                        help="CFG scale。")
    parser.add_argument("--flow_shift", type=float, default=10.0,
                        help="Scheduler flow shift。")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="负面提示词。")

    # BoN 参数
    parser.add_argument("--N", type=int, default=8,
                        help="候选数。")
    parser.add_argument("--seed_base", type=int, default=None,
                        help="基础种子（候选使用 seed_base+i）。")

    # VGGTomega 模型
    parser.add_argument("--vggt_model", type=str, required=True,
                        help="VGGTomega 权重路径（.pth 文件）。")

    # CoTracker3 模型
    parser.add_argument("--cotracker_model", type=str, default=None,
                        help="CoTracker3 权重路径。None 则用 torch.hub 自动下载。")

    # 评分参数
    parser.add_argument("--num_frames_for_reward", type=int, default=20,
                        help="评分用帧数。")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="CoTracker3 每批追踪点数。")
    parser.add_argument("--image_resolution", type=int, default=512,
                        help="VGGTomega 输入分辨率（长边参考）。")

    # Reward 权重
    parser.add_argument("--static_weight", type=float, default=0.50)
    parser.add_argument("--dynamic_weight", type=float, default=0.30)
    parser.add_argument("--motion_weight", type=float, default=0.20)

    # 输出
    parser.add_argument("--output_dir", type=str, default="outputs/bon_vggt_cosmos3")

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

    # 1. 加载 Cosmos3 模型（参考 geo_reward/run_bon_cosmos3.py:191-223）
    logger.info(f"加载 Cosmos3 模型: {args.model}")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,  # ← 使用 bfloat16，与现有代码一致
    )
    pipe = pipe.to("cuda")

    # ✅ 加载后再禁用 safety_checker（避免下载 Guardrail）
    pipe.safety_checker = None

    # ✅ flow_shift 在 scheduler 上设置，不在 pipe() 调用时传入
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift
    )

    # 2. 加载输入图片
    image = Image.open(args.image).convert("RGB")

    # 3. 配置 Reward
    reward_cfg = VGGTReconRewardConfig(
        static_weight=args.static_weight,
        dynamic_weight=args.dynamic_weight,
        motion_weight=args.motion_weight,
        image_size=args.image_resolution,
        chunk_size=args.chunk_size,
        max_frames=args.num_frames_for_reward,
    )

    # 4. 创建 BoN pipeline
    bon = GeoRewardBoNVGGT(
        cosmos3_pipe=pipe,
        vggt_model_path=args.vggt_model,
        cotracker_model_path=args.cotracker_model,
        N=args.N,
        num_frames_for_reward=args.num_frames_for_reward,
        reward_config=reward_cfg,
        chunk_size=args.chunk_size,
    )

    # 5. 执行 BoN 生成（⚠️ 不传 flow_shift）
    logger.info(f"开始 Best-of-N 生成: N={args.N}")
    result = bon.generate(
        image=image,
        prompt=args.prompt,
        N=args.N,
        num_frames=args.num_frames,
        fps=args.fps,
        seed_base=args.seed_base,
        output_dir=args.output_dir,
        save_fn=save_video_from_pil,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        # ⚠️ 不传 flow_shift（已在 scheduler 上设置）
    )

    # 6. 输出结果
    logger.info(f"最优候选: seed={result['best_seed']}, score={result['best_score']:.4f}")
    logger.info(f"所有分数: {result['all_scores']}")
    logger.info(f"总用时: {result['total_time_sec']:.1f}s")

    # 7. 保存结果 JSON
    result_json = {
        "prompt": args.prompt,
        "best_seed": result["best_seed"],
        "best_score": result["best_score"],
        "all_scores": result["all_scores"],
        "all_seeds": result["seeds"],
        "config": vars(args),
    }
    json_path = os.path.join(args.output_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    logger.info(f"结果已保存: {json_path}")


if __name__ == "__main__":
    main()
