"""
VGGT-Omega + CoTracker3 后端的简单 Best-of-N pipeline。

全量生成 N 个候选视频，统一评分，选出最优。
不含渐进淘汰、Tree Branching、梯度引导。

显存管理：
- 生成阶段: DiT (Cosmos3) + VAE 在 GPU
- 评分阶段: VGGT-Omega + CoTracker3 在 GPU，DiT + VAE 卸载到 CPU
"""

import os
import random
import time

import numpy as np
import torch

from .vggt_omega_adapter import load_vggt_omega
from .cotracker3_adapter import load_cotracker3
from .recon_reward_vggt import VGGTReconstructionReward, VGGTReconRewardConfig

# 复用现有工具函数（geo_reward 不做修改，直接 import）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from geo_reward.utils import cosmos3_output_to_pil, sample_frames


class GeoRewardBoNVGGT:
    """
    简单 Best-of-N pipeline（VGGT-Omega + CoTracker3 后端）。

    流程：
    1. Cosmos3 生成 N 个候选视频（完整去噪）
    2. VAE decode 所有候选
    3. 对每个候选：均匀抽帧 → VGGT+CoTracker3 推理 → 计算 reward
    4. 选 reward 最高的候选
    """

    def __init__(
        self,
        cosmos3_pipe,
        vggt_model_path,
        cotracker_model_path=None,
        N=8,
        num_frames_for_reward=20,
        reward_config=None,
        chunk_size=4096,
        device="cuda",
    ):
        """
        Args:
            cosmos3_pipe: Cosmos3OmniPipeline 实例
            vggt_model_path: VGGT-Omega 权重路径
            cotracker_model_path: CoTracker3 权重路径（None 用 torch.hub）
            N: 候选数
            num_frames_for_reward: reward 评分用的帧数
            reward_config: VGGTReconRewardConfig 实例
            chunk_size: CoTracker3 per-chunk 点数
            device: 设备
        """
        self.pipe = cosmos3_pipe
        self.N = N
        self.num_frames_for_reward = num_frames_for_reward
        self.chunk_size = chunk_size
        self.device = device

        # 模型路径（延迟加载，与 DiT 交替占用 GPU）
        self.vggt_model_path = vggt_model_path
        self.cotracker_model_path = cotracker_model_path

        # Reward 计算器
        cfg = reward_config or VGGTReconRewardConfig()
        cfg.chunk_size = chunk_size
        self.reward = VGGTReconstructionReward(device=device, cfg=cfg)

        # 模型实例缓存（加载后保留在 CPU）
        self._vggt_model = None
        self._cotracker_model = None

    def _ensure_models_loaded(self):
        """确保 VGGT-Omega 和 CoTracker3 模型已加载（CPU 上）。"""
        if self._vggt_model is None:
            print("[BoNVGGT] 加载 VGGT-Omega 模型...")
            self._vggt_model = load_vggt_omega(self.vggt_model_path, device="cpu")
        if self._cotracker_model is None:
            print("[BoNVGGT] 加载 CoTracker3 模型...")
            self._cotracker_model = load_cotracker3(
                self.cotracker_model_path, device="cpu"
            )

    def _load_reward_models(self):
        """将 VGGT-Omega + CoTracker3 加载到 GPU。"""
        self._ensure_models_loaded()
        self._vggt_model.to(self.device)
        self._cotracker_model.to(self.device)
        self.reward.vggt_model = self._vggt_model
        self.reward.cotracker_model = self._cotracker_model

    def _offload_reward_models(self):
        """将 VGGT-Omega + CoTracker3 卸载到 CPU。"""
        if self._vggt_model is not None:
            self._vggt_model.cpu()
        if self._cotracker_model is not None:
            self._cotracker_model.cpu()
        torch.cuda.empty_cache()

    def _offload_dit(self):
        """将 DiT 卸载到 CPU。"""
        # Cosmos3 的 DiT 模型名称是 transformer
        if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
            self.pipe.transformer.cpu()
        torch.cuda.empty_cache()

    def _offload_vae(self):
        """将 VAE 卸载到 CPU。"""
        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
            if hasattr(self.pipe.vae, 'model'):
                self.pipe.vae.model.cpu()
            else:
                self.pipe.vae.cpu()
            torch.cuda.empty_cache()

    def _load_vae(self):
        """将 VAE 加载到 GPU。"""
        if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
            if hasattr(self.pipe.vae, 'model'):
                self.pipe.vae.model.cuda()
            else:
                self.pipe.vae.cuda()

    def generate(self, image, prompt, N=None, num_frames=189,
                 fps=24, seed_base=None, output_dir=None, save_fn=None,
                 **cosmos3_kwargs):
        """
        执行 BoN 生成。

        Args:
            image: 首帧图片（PIL.Image）
            prompt: 动作指令文本
            N: 候选数（覆盖 self.N）
            num_frames: 视频帧数
            fps: 帧率
            seed_base: 基础种子
            output_dir: 输出目录（保存所有候选视频）
            save_fn: callable(frames_pil, path) 保存视频
            **cosmos3_kwargs: 传给 Cosmos3OmniPipeline 的其他参数
                             (height, width, guidance_scale, num_inference_steps)
                             ⚠️ 不要传 flow_shift（在 scheduler 上已设置）

        Returns:
            dict:
              - best_video: 最优视频 (List[PIL.Image])
              - best_score: 最优分数
              - best_seed: 最优种子
              - all_scores: 所有候选分数列表
              - all_details: 所有候选的详细 reward 分解
        """
        N = N or self.N
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        indices = sample_frames(num_frames, self.num_frames_for_reward)

        t_start = time.time()

        # 生成器设备兼容性检测（参考 geo_reward/bon_pipeline.py:100-105）
        pipe_device = getattr(self.pipe, "device", None)
        if pipe_device is None or str(pipe_device) == "meta":
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            gen_device = pipe_device

        # ===== 阶段 1: 生成 N 个候选 =====
        print(f"\n[BoNVGGT] 开始生成 {N} 个候选视频 (seeds {seed_base}..{seed_base+N-1})")

        candidates = []  # (frames_pil, seed)
        for i in range(N):
            seed = seed_base + i
            t0 = time.time()

            # Cosmos3 生成
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                image=image,
                num_frames=num_frames,
                fps=fps,
                generator=generator,
                **cosmos3_kwargs,
            )
            # ✅ 使用 cosmos3_output_to_pil 统一处理输出
            frames_pil = cosmos3_output_to_pil(result)
            gen_time = time.time() - t0

            if frames_pil is None or len(frames_pil) == 0:
                print(f"  候选 {i+1}/{N} (seed={seed}): 生成失败，跳过")
                continue

            candidates.append((frames_pil, seed))
            print(f"  候选 {i+1}/{N} (seed={seed}): 生成完成 [{gen_time:.1f}s]")

            torch.cuda.empty_cache()

        if not candidates:
            raise RuntimeError("所有候选生成失败。")

        # ===== 阶段 2: 卸载 DiT+VAE，加载评分模型 =====
        print(f"\n[BoNVGGT] 卸载 DiT+VAE，加载 VGGT-Omega + CoTracker3...")
        self._offload_dit()
        self._offload_vae()
        self._load_reward_models()

        # ===== 阶段 3: 评分 =====
        print(f"\n[BoNVGGT] 对 {len(candidates)} 个候选评分...")
        all_scores = []
        all_details = []

        for idx, (frames_pil, seed) in enumerate(candidates):
            t0 = time.time()

            sampled = [frames_pil[i] for i in indices if i < len(frames_pil)]

            r = self.reward.compute_reward(sampled)
            reward_time = time.time() - t0

            all_scores.append(r["total"])
            all_details.append(r)

            print(f"  候选 {idx+1}/{len(candidates)} (seed={seed}): "
                  f"total={r['total']:.4f} "
                  f"(R_static={r['R_static']:.4f}, "
                  f"R_dynamic={r['R_dynamic']:.4f}, "
                  f"R_motion={r['R_motion']:.4f}, "
                  f"G_anchor={r['G_anchor']:.2f}) "
                  f"[{reward_time:.1f}s]")

        # ===== 阶段 4: 选出最优 =====
        best_idx = max(range(len(all_scores)),
                       key=lambda i: all_scores[i] if np.isfinite(all_scores[i]) else -float("inf"))
        best_frames, best_seed = candidates[best_idx]
        best_score = all_scores[best_idx]

        elapsed = time.time() - t_start
        print(f"\n[BoNVGGT] 最优: seed_{best_seed} (total={best_score:.4f}) "
              f"总用时 {elapsed:.1f}s")

        # ===== 阶段 5: 卸载评分模型，恢复 VAE =====
        self._offload_reward_models()
        self._load_vae()

        # ===== 保存视频 =====
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            ranked_indices = sorted(range(len(all_scores)),
                                    key=lambda i: all_scores[i], reverse=True)
            for rank, orig_idx in enumerate(ranked_indices):
                frames, s = candidates[orig_idx]
                reward_val = all_scores[orig_idx]
                suffix = "_BEST" if orig_idx == best_idx else ""
                filename = f"candidate_{rank+1:02d}_r{reward_val:.4f}_seed{s}{suffix}.mp4"
                path = os.path.join(output_dir, filename)
                if save_fn is not None:
                    save_fn(frames, path)

        return {
            "best_video": best_frames,
            "best_score": best_score,
            "best_seed": best_seed,
            "all_scores": all_scores,
            "all_details": all_details,
            "seeds": [s for _, s in candidates],
            "total_time_sec": elapsed,
        }
