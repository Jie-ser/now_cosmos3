"""
Best-of-N sampling pipeline with GeoReward for Cosmos3 I2V.

Generates N candidate videos with Cosmos3 Diffusers pipeline, scores each
with 4RC-based GeoReward, and selects the geometrically most consistent one.

Classes:
- Cosmos3GeoRewardBoN: online BoN (generate + score + select)
- Cosmos3GeoRewardOffline: offline scoring of pre-generated videos
"""

import json
import os
import random
import time

import numpy as np
import torch

from .utils import cosmos3_output_to_pil, sample_frames


class Cosmos3GeoRewardBoN:
    """
    Best-of-N pipeline for Cosmos3 I2V with GeoReward selection.

    Workflow:
      1. Generate N candidate videos with different seeds.
      2. For each candidate, sample keyframes and compute GeoReward.
      3. Select the candidate with highest total reward.

    Memory management:
      When offload=True, after each video is generated and scored, models
      are swapped between CPU/GPU. Only the best video is kept in memory;
      other candidates are released after scoring (or saved to disk if
      save_all=True).
    """

    def __init__(self, pipe, recon_reward, max_frames=20, offload=False):
        """
        Args:
            pipe: Cosmos3OmniPipeline instance (from diffusers).
            recon_reward: ReconstructionReward instance (4RC V2).
            max_frames: Number of keyframes to sample for reward scoring.
            offload: If True, swap Cosmos3/4RC between CPU/GPU at each step.
        """
        self.pipe = pipe
        self.reward = recon_reward
        self.max_frames = max_frames
        self.offload = offload

    def generate(
        self,
        prompt,
        image,
        N=8,
        num_frames=189,
        fps=24,
        seed_base=None,
        save_all=False,
        output_dir=None,
        save_fn=None,
        **pipe_kwargs,
    ):
        """
        Generate N candidates and select the best by GeoReward.

        Args:
            prompt: Text prompt for video generation.
            image: PIL Image (first frame / conditioning image).
            N: Number of candidate videos to generate.
            num_frames: Number of output video frames.
            fps: Frames per second.
            seed_base: Base seed (candidates use seed_base + i).
            save_all: If True, save all candidate videos to output_dir.
            output_dir: Directory for saving videos.
            save_fn: callable(frames_pil, path) to save a video.
            **pipe_kwargs: Additional arguments for Cosmos3OmniPipeline
                           (e.g., height, width, guidance_scale, num_inference_steps).

        Returns:
            Tuple of (best_video_frames, all_rewards, best_index):
              - best_video_frames: list[PIL.Image] for the best candidate.
              - all_rewards: list[dict] with reward breakdown for each candidate.
              - best_index: int index of the best candidate.
        """
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        rewards = []
        best_frames = None
        best_idx = -1
        best_score = -float("inf")
        timings = []

        # Detect device from pipeline for Generator
        pipe_device = getattr(self.pipe, "device", None)
        if pipe_device is None or str(pipe_device) == "meta":
            gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            gen_device = pipe_device

        print(f"[Cosmos3GeoRewardBoN] Generating {N} candidates "
              f"(seeds {seed_base}..{seed_base + N - 1})")

        for i in range(N):
            seed = seed_base + i
            generator = torch.Generator(device=gen_device).manual_seed(seed)

            # --- Generation phase ---
            if self.offload:
                self._load_cosmos3()
                self._offload_4rc()

            t0 = time.time()
            result = self.pipe(
                prompt=prompt,
                image=image,
                num_frames=num_frames,
                fps=fps,
                generator=generator,
                **pipe_kwargs,
            )
            gen_time = time.time() - t0

            frames_pil = cosmos3_output_to_pil(result)
            del result
            torch.cuda.empty_cache()

            # --- Scoring phase ---
            if self.offload:
                self._offload_cosmos3()
                self._load_4rc()

            # Sample indices based on actual output length, not requested num_frames
            indices = sample_frames(len(frames_pil), self.max_frames)
            sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]

            t1 = time.time()
            r = self.reward.compute_reward(sampled)
            reward_time = time.time() - t1
            rewards.append(r)
            timings.append({"gen": gen_time, "reward": reward_time})

            total = float(r.get("total", float("nan")))
            print(
                f"  Candidate {i + 1}/{N} (seed={seed}): "
                f"total={r['total']:.4f} "
                f"(R_static={r['R_static']:.4f}, "
                f"R_dynamic={r['R_dynamic']:.4f}, "
                f"R_motion={r['R_motion']:.4f}, "
                f"G_anchor={r['G_anchor']:.2f}) "
                f"[gen={gen_time:.1f}s, reward={reward_time:.1f}s]"
            )

            # Save to disk if requested
            if save_all and output_dir is not None:
                self._save_video(frames_pil, seed, r["total"], output_dir, save_fn, fps=fps)

            # Keep only the best in memory
            selection_score = total if np.isfinite(total) else -float("inf")
            if selection_score > best_score:
                best_frames = frames_pil
                best_idx = i
                best_score = selection_score
            else:
                del frames_pil

            torch.cuda.empty_cache()

        if best_frames is None:
            raise RuntimeError("No valid candidates generated.")

        # Save best video
        if output_dir is not None:
            best_seed = seed_base + best_idx
            self._save_video(
                best_frames, best_seed, best_score, output_dir, save_fn,
                suffix="_BEST", fps=fps,
            )

        print(
            f"\n[Cosmos3GeoRewardBoN] Selected candidate {best_idx + 1}/{N} "
            f"(seed={seed_base + best_idx}) "
            f"with reward {best_score:.4f}"
        )

        return best_frames, rewards, best_idx

    def _save_video(self, frames_pil, seed, reward_total, output_dir, save_fn,
                    suffix="", fps=24):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"seed_{seed}_r{reward_total:.4f}{suffix}.mp4"
        path = os.path.join(output_dir, filename)
        if save_fn is not None:
            save_fn(frames_pil, path)
        else:
            try:
                from diffusers.utils import export_to_video
                export_to_video(frames_pil, path, fps=fps)
            except ImportError:
                import imageio
                imageio.mimwrite(
                    path,
                    [np.array(f) for f in frames_pil],
                    fps=fps,
                    macro_block_size=1,
                )

    def _offload_cosmos3(self):
        """Move Cosmos3 pipeline to CPU."""
        try:
            self.pipe.to("cpu")
        except Exception:
            pass
        torch.cuda.empty_cache()

    def _load_cosmos3(self):
        """Move Cosmos3 pipeline to GPU."""
        try:
            self.pipe.to("cuda")
        except Exception:
            pass

    def _offload_4rc(self):
        """Move 4RC model to CPU."""
        if self.reward.model is not None:
            self.reward.model.cpu()
            torch.cuda.empty_cache()

    def _load_4rc(self):
        """Move 4RC model to GPU."""
        if self.reward.model is not None:
            self.reward.model.cuda()


class Cosmos3GeoRewardOffline:
    """
    Offline (post-hoc) scoring: score pre-generated videos without
    re-generating them. Useful for ablation studies and evaluation.
    """

    def __init__(self, recon_reward, max_frames=20):
        """
        Args:
            recon_reward: ReconstructionReward instance.
            max_frames: Number of keyframes to sample for scoring.
        """
        self.reward = recon_reward
        self.max_frames = max_frames

    def score_videos(self, video_sources, num_frames=None):
        """
        Score a list of videos.

        Args:
            video_sources: List of video sources. Each can be:
              - str/Path: path to .mp4 video file
              - list[PIL.Image]: pre-loaded frames
            num_frames: Override total frame count for index sampling.
                        If None, uses the actual frame count of each video.

        Returns:
            List of reward dicts, one per video.
        """
        from pathlib import Path
        from PIL import Image

        rewards = []
        for i, src in enumerate(video_sources):
            # Load frames
            if isinstance(src, (str, Path)):
                frames_pil = self._load_video_frames(str(src))
            elif isinstance(src, list) and len(src) > 0 and isinstance(src[0], Image.Image):
                frames_pil = src
            else:
                raise ValueError(
                    f"Unsupported video source type: {type(src)}. "
                    "Expected file path or list of PIL Images."
                )

            total = num_frames or len(frames_pil)
            indices = sample_frames(total, self.max_frames)
            sampled = [frames_pil[idx] for idx in indices if idx < len(frames_pil)]

            r = self.reward.compute_reward(sampled)
            rewards.append(r)

            name = str(src) if isinstance(src, (str, Path)) else f"video_{i}"
            print(
                f"  [{i + 1}/{len(video_sources)}] {name}: "
                f"total={r['total']:.4f} "
                f"(R_static={r['R_static']:.4f}, "
                f"R_dynamic={r['R_dynamic']:.4f}, "
                f"R_motion={r['R_motion']:.4f}, "
                f"G_anchor={r['G_anchor']:.2f})"
            )

        return rewards

    def select_best(self, video_sources, **kwargs):
        """Score all videos and return the best index and all rewards."""
        rewards = self.score_videos(video_sources, **kwargs)
        if not rewards:
            raise RuntimeError("No videos to score.")
        best_idx = max(
            range(len(rewards)),
            key=lambda i: rewards[i]["total"]
            if np.isfinite(rewards[i]["total"])
            else -float("inf"),
        )
        return best_idx, rewards

    @staticmethod
    def _load_video_frames(path):
        """Load video file into list of PIL Images."""
        from PIL import Image

        try:
            import imageio.v3 as iio
            try:
                frames_np = iio.imread(path, plugin="pyav")
            except ImportError:
                frames_np = iio.imread(path, plugin="FFMPEG")
        except ImportError:
            import imageio
            reader = imageio.get_reader(path)
            frames_np = [frame for frame in reader]
            reader.close()
            frames_np = np.stack(frames_np)

        return [Image.fromarray(f) for f in frames_np]
