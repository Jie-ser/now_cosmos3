"""
Best-of-N sampling pipeline with GeoReward for Cosmos3 I2V.

Generates N candidate videos with Cosmos3 Diffusers pipeline, scores each
with 4RC-based GeoReward, and selects the geometrically most consistent one.

Classes:
- Cosmos3GeoRewardBoN: online BoN (generate + score + select)
- Cosmos3GeoRewardBoNProgressive: base progressive elimination BoN
- Cosmos3GeoRewardBoNProgressiveV2: progressive + 4RC reward + model offload
- Cosmos3GeoRewardBoNTreeBranching: tree branching + progressive elimination
- Cosmos3GeoRewardBoNTreeBranchingGuided: tree branching + gradient guidance
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


# ---------------------------------------------------------------------------
# Progressive elimination BoN
# ---------------------------------------------------------------------------

class Cosmos3GeoRewardBoNProgressive:
    """
    Base class for progressive elimination Best-of-N with Cosmos3.

    Uses sigma-based checkpoints to evaluate candidates mid-generation and
    eliminate the bottom fraction, saving ~40% compute vs naive sequential BoN.

    Subclasses implement _generate_prepared() with specific reward logic.
    """

    DEFAULT_SIGMA_CHECKPOINTS = [0.83, 0.63]
    DEFAULT_ELIMINATION_RATIO = 0.5
    DEFAULT_MIN_SURVIVORS = 2
    DEFAULT_SCORE_EPSILON = 0.02
    DEFAULT_EARLY_MAX_FRAMES = 12

    def __init__(
        self,
        adapter,
        recon_reward,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
    ):
        """
        Args:
            adapter: Cosmos3ProgressiveAdapter instance.
            recon_reward: ReconstructionReward instance.
        """
        self.adapter = adapter
        self.recon_reward = recon_reward
        self.max_frames = max_frames
        self.sigma_checkpoints = sigma_checkpoints or self.DEFAULT_SIGMA_CHECKPOINTS
        self.elimination_ratio = (
            elimination_ratio
            if elimination_ratio is not None
            else self.DEFAULT_ELIMINATION_RATIO
        )
        self.min_survivors = (
            min_survivors
            if min_survivors is not None
            else self.DEFAULT_MIN_SURVIVORS
        )
        self.score_epsilon = (
            score_epsilon
            if score_epsilon is not None
            else self.DEFAULT_SCORE_EPSILON
        )
        self.early_max_frames = (
            early_max_frames
            if early_max_frames is not None
            else self.DEFAULT_EARLY_MAX_FRAMES
        )

    def generate(
        self,
        prompt,
        image,
        N=8,
        num_frames=189,
        seed_base=None,
        output_dir=None,
        save_fn=None,
        **adapter_kwargs,
    ):
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}.")

        validated_sigmas = []
        for v in self.sigma_checkpoints:
            s = float(v)
            if not np.isfinite(s) or not 0.0 < s < 1.0:
                raise ValueError(
                    f"Each sigma checkpoint must be in (0, 1), got {v}."
                )
            validated_sigmas.append(s)

        indices = sample_frames(num_frames, self.max_frames)
        early_indices = sample_frames(num_frames, self.early_max_frames)

        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)
        seeds = [seed_base + i for i in range(N)]

        print(
            f"[BoNProgressive] Preparing {N} candidates "
            f"(seeds {seeds[0]}..{seeds[-1]})"
        )
        state = self.adapter.prepare_progressive(
            prompt=prompt,
            image=image,
            seeds=seeds,
            num_frames=num_frames,
            **adapter_kwargs,
        )
        try:
            return self._generate_prepared(
                state=state,
                seeds=seeds,
                indices=indices,
                early_indices=early_indices,
                sigma_checkpoints=validated_sigmas,
                output_dir=output_dir,
                save_fn=save_fn,
            )
        finally:
            self.adapter.cleanup_progressive(state)

    def _generate_prepared(
        self, state, seeds, indices, early_indices,
        sigma_checkpoints, output_dir, save_fn,
    ):
        raise NotImplementedError

    def _eliminate(self, scored, seeds, eliminated_at, phase_name, is_early):
        """Fixed-ratio elimination with safety margin."""
        if len(scored) <= self.min_survivors:
            return [c for c, _ in scored], []

        totals = np.array(
            [float(r.get("total", float("nan"))) for _, r in scored],
            dtype=np.float64,
        )
        finite = np.isfinite(totals)
        if not finite.any():
            print(
                f"  WARNING: no finite rewards at {phase_name}; "
                "skipping elimination."
            )
            return [c for c, _ in scored], []

        ranked = sorted(
            range(len(scored)),
            key=lambda i: totals[i] if np.isfinite(totals[i]) else -float("inf"),
            reverse=True,
        )

        keep_count = max(
            self.min_survivors,
            len(scored) - int(len(scored) * self.elimination_ratio),
        )

        if keep_count < len(scored):
            last_keep = totals[ranked[keep_count - 1]]
            first_elim = totals[ranked[keep_count]]
            if (
                np.isfinite(last_keep)
                and np.isfinite(first_elim)
                and (last_keep - first_elim) < self.score_epsilon
            ):
                keep_count = min(keep_count + 1, len(scored))

        survivors, eliminated = [], []
        survivor_set = set(ranked[:keep_count])
        for idx, (cand_idx, _) in enumerate(scored):
            if idx in survivor_set:
                survivors.append(cand_idx)
            else:
                eliminated.append(cand_idx)
                eliminated_at[f"seed_{seeds[cand_idx]}"] = phase_name

        if eliminated:
            elim_seeds = [seeds[c] for c in eliminated]
            surv_seeds = [seeds[c] for c in survivors]
            print(
                f"  Eliminated {len(eliminated)} candidates: seeds {elim_seeds} "
                f"(kept {len(survivors)}: seeds {surv_seeds})"
            )
        return survivors, eliminated

    def _save_video(self, frames_pil, seed, phase_name, output_dir, save_fn, fps=24):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"seed_{seed}_{phase_name}.mp4"
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


class Cosmos3GeoRewardBoNProgressiveV2(Cosmos3GeoRewardBoNProgressive):
    """
    Progressive elimination BoN with 4RC V2 reward + model offloading.

    All checkpoints use the same reward formula; only sampled frame count
    differs (12 at early checkpoints, 20 at mid/final).
    """

    def __init__(
        self,
        adapter,
        recon_reward,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
        offload_models=True,
    ):
        super().__init__(
            adapter=adapter,
            recon_reward=recon_reward,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
        )
        self.offload_models = offload_models

    def _generate_prepared(
        self, state, seeds, indices, early_indices,
        sigma_checkpoints, output_dir, save_fn,
    ):
        total_steps = len(state["timesteps"])
        checkpoint_steps = []
        seen = set()

        for sigma_target in sorted(sigma_checkpoints, reverse=True):
            end_step = self.adapter.find_step_for_sigma(state, sigma_target)
            if end_step is None or not 0 < end_step < total_steps:
                continue
            if end_step in seen:
                continue
            sched = state["candidates"][0]["scheduler"]
            actual_sigma = float(sched.sigmas[end_step - 1])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
            })
            seen.add(end_step)

        checkpoint_steps.sort(key=lambda c: c["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
        })

        alive = list(range(len(seeds)))
        rewards_log = {f"seed_{s}": {} for s in seeds}
        eliminated_at = {}
        cur_step = 0
        t_start = time.time()
        best_frames = None
        best_cand_idx = None
        best_score = -float("inf")

        for ckpt_idx, ckpt in enumerate(checkpoint_steps):
            end_step = ckpt["end_step"]
            actual_sigma = ckpt["actual_sigma"]
            is_final = end_step == total_steps
            is_early = ckpt_idx == 0 and not is_final
            phase = (
                "final_sigma0.00"
                if is_final
                else f"checkpoint{ckpt_idx + 1}_sigma{actual_sigma:.4f}"
            )

            print(
                f"\n[BoNProgressiveV2] Phase: {phase} "
                f"(steps {cur_step}->{end_step}, {len(alive)} alive)"
            )

            last_preds, pre_lats = self.adapter.denoise_candidates(
                state, alive, cur_step, end_step
            )
            cur_step = end_step

            frame_idx_for_phase = early_indices if is_early else indices

            # Decode all alive candidates
            decoded = {}
            for ci in alive:
                if is_final:
                    lat = state["candidates"][ci]["latent"]
                else:
                    lat = self.adapter.extract_pred_x0(
                        state, ci, last_preds[ci], pre_lats[ci]
                    )
                frames_pil = self.adapter.decode_latent(lat.unsqueeze(0))
                decoded[ci] = frames_pil
                if output_dir is not None:
                    self._save_video(
                        frames_pil, seeds[ci], phase, output_dir, save_fn
                    )
                del lat
                torch.cuda.empty_cache()

            # Offload DiT, load 4RC
            if self.offload_models:
                self.adapter.offload_transformer()
                self.adapter.offload_vae()
                self._load_4rc()

            # Score
            scored = []
            for ci in alive:
                frames_pil = decoded[ci]
                sampled = [
                    frames_pil[i]
                    for i in frame_idx_for_phase
                    if i < len(frames_pil)
                ]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seeds[ci]}"][phase] = r
                scored.append((ci, r))

                print(
                    f"  seed_{seeds[ci]}: total={r['total']:.4f} "
                    f"(R_static={r['R_static']:.4f}, "
                    f"R_dynamic={r['R_dynamic']:.4f}, "
                    f"R_motion={r['R_motion']:.4f}, "
                    f"G_anchor={r['G_anchor']:.2f})"
                )

                if is_final:
                    total = float(r.get("total", float("nan")))
                    sel = total if np.isfinite(total) else -float("inf")
                    if sel > best_score:
                        best_frames = frames_pil
                        best_cand_idx = ci
                        best_score = sel

                torch.cuda.empty_cache()

            del decoded

            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self.adapter.load_transformer()
                self.adapter.load_vae()

            if is_final:
                break

            alive, _ = self._eliminate(
                scored, seeds, eliminated_at, phase, is_early
            )

        if best_cand_idx is None:
            raise RuntimeError("No final candidate decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = {
            "mode": "progressive_elimination_v2",
            "reward_type": "v2_4rc",
            "sigma_checkpoints": sigma_checkpoints,
            "seeds": seeds,
            "best_seed": best_seed,
            "total_time_sec": elapsed,
            "rewards": rewards_log,
            "eliminated_at": eliminated_at,
        }

        print(
            f"\n[BoNProgressiveV2] Best: seed_{best_seed} "
            f"(total={best_score:.4f}) in {elapsed:.1f}s"
        )
        return best_frames, result_log, best_seed

    def _load_4rc(self):
        if self.recon_reward.model is not None:
            self.recon_reward.model.cuda()

    def _offload_4rc(self):
        if self.recon_reward.model is not None:
            self.recon_reward.model.cpu()
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tree Branching BoN
# ---------------------------------------------------------------------------

class Cosmos3GeoRewardBoNTreeBranching(Cosmos3GeoRewardBoNProgressiveV2):
    """
    Tree Branching + Progressive Elimination BoN for Cosmos3.

    Phase 1: Denoise only num_trunks candidates to branch_step.
    Phase 2: Branch into N candidates via noise injection.
    Phase 3: Progressive elimination from branch_step onward.

    Saves ~29% compute vs progressive-only (shift=10, N=8).
    """

    DEFAULT_NUM_TRUNKS = 2
    DEFAULT_BRANCHES_PER_TRUNK = 4
    DEFAULT_BRANCH_SIGMA = 0.90
    DEFAULT_BRANCH_ETA = 0.10

    def __init__(
        self,
        adapter,
        recon_reward,
        num_trunks=None,
        branches_per_trunk=None,
        branch_sigma=None,
        branch_eta=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
        offload_models=True,
    ):
        super().__init__(
            adapter=adapter,
            recon_reward=recon_reward,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.num_trunks = num_trunks or self.DEFAULT_NUM_TRUNKS
        self.branches_per_trunk = (
            branches_per_trunk or self.DEFAULT_BRANCHES_PER_TRUNK
        )
        self.branch_sigma = (
            branch_sigma if branch_sigma is not None else self.DEFAULT_BRANCH_SIGMA
        )
        self.branch_eta = (
            branch_eta if branch_eta is not None else self.DEFAULT_BRANCH_ETA
        )

    def generate(
        self,
        prompt,
        image,
        N,
        num_frames=189,
        seed_base=None,
        output_dir=None,
        save_fn=None,
        **adapter_kwargs,
    ):
        if seed_base is None:
            seed_base = random.randint(0, 2**31 - 1)

        expected_N = self.num_trunks * self.branches_per_trunk
        if N != expected_N:
            print(
                f"[TreeBranching] Warning: N={N} != "
                f"{self.num_trunks}*{self.branches_per_trunk}={expected_N}. "
                f"Using {expected_N}."
            )
            N = expected_N

        trunk_seeds = [seed_base + i for i in range(self.num_trunks)]
        branch_seeds = [seed_base + 100 + i for i in range(N)]

        max_cp = max(self.sigma_checkpoints)
        if self.branch_sigma <= max_cp:
            raise ValueError(
                f"branch_sigma({self.branch_sigma}) must be > "
                f"max(sigma_checkpoints)({max_cp})"
            )

        state = self.adapter.prepare_progressive(
            prompt=prompt,
            image=image,
            seeds=trunk_seeds,
            num_frames=num_frames,
            **adapter_kwargs,
        )

        try:
            return self._generate_tree(
                state, N, branch_seeds, num_frames, output_dir, save_fn
            )
        finally:
            self.adapter.cleanup_progressive(state)

    def _generate_tree(
        self, state, N, branch_seeds, num_frames, output_dir, save_fn
    ):
        t_start = time.time()

        branch_step = self.adapter.find_step_for_sigma(state, self.branch_sigma)
        if branch_step is None:
            raise ValueError(
                f"branch_sigma={self.branch_sigma} cannot map to a valid step."
            )

        print(
            f"[TreeBranching] branch_sigma={self.branch_sigma} "
            f"-> branch_step={branch_step}"
        )

        # Phase 1: Trunk denoising
        trunk_indices = list(range(self.num_trunks))
        print(
            f"[TreeBranching] Phase 1: Denoising {self.num_trunks} trunks "
            f"for {branch_step} steps..."
        )
        self.adapter.denoise_candidates(state, trunk_indices, 0, branch_step)

        # Phase 2: Branching
        print(
            f"[TreeBranching] Phase 2: Branching into {N} candidates "
            f"(eta={self.branch_eta})..."
        )
        state = self.adapter.branch_candidates(
            state, trunk_indices, self.branches_per_trunk,
            self.branch_eta, branch_seeds,
        )

        # Phase 3: Progressive elimination from branch_step
        print(
            f"[TreeBranching] Phase 3: Progressive elimination "
            f"from step {branch_step}..."
        )
        return self._progressive_elimination(
            state, N, branch_seeds, num_frames, output_dir, save_fn,
            start_step=branch_step, t_start=t_start,
        )

    def _progressive_elimination(
        self, state, N, seeds, num_frames, output_dir, save_fn,
        start_step, t_start,
    ):
        total_steps = len(state["timesteps"])

        checkpoint_steps = []
        seen = set()
        for sigma_target in sorted(self.sigma_checkpoints, reverse=True):
            end_step = self.adapter.find_step_for_sigma(state, sigma_target)
            if end_step is None or end_step <= start_step or end_step >= total_steps:
                continue
            if end_step in seen:
                continue
            sched = state["candidates"][0]["scheduler"]
            actual_sigma = float(sched.sigmas[end_step - 1])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
            })
            seen.add(end_step)

        checkpoint_steps.sort(key=lambda c: c["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
        })

        early_frame_indices = sample_frames(num_frames, self.early_max_frames)
        normal_frame_indices = sample_frames(num_frames, self.max_frames)

        alive = list(range(N))
        eliminated_at = {}
        rewards_log = {f"seed_{s}": {} for s in seeds}
        best_frames = None
        best_cand_idx = None
        best_score = -float("inf")
        cur_step = start_step

        for cp_idx, cp in enumerate(checkpoint_steps):
            end_step = cp["end_step"]
            is_final = cp_idx == len(checkpoint_steps) - 1
            is_early = cp_idx == 0 and not is_final
            phase = (
                "final_sigma0.00"
                if is_final
                else f"checkpoint{cp_idx + 1}_sigma{cp['actual_sigma']:.4f}"
            )
            frame_idx = early_frame_indices if is_early else normal_frame_indices

            print(
                f"\n[TreeBranching] Phase: {phase} "
                f"(steps {cur_step}->{end_step}, {len(alive)} alive)"
            )

            last_preds, pre_lats = self.adapter.denoise_candidates(
                state, alive, cur_step, end_step
            )

            # Decode
            decoded = {}
            for ci in alive:
                if is_final:
                    lat = state["candidates"][ci]["latent"]
                else:
                    lat = self.adapter.extract_pred_x0(
                        state, ci, last_preds[ci], pre_lats[ci]
                    )
                decoded[ci] = self.adapter.decode_latent(lat.unsqueeze(0))
                del lat
                torch.cuda.empty_cache()

            if self.offload_models:
                self.adapter.offload_transformer()
                self.adapter.offload_vae()
                self._load_4rc()

            # Score
            scored = []
            for ci in alive:
                frames_pil = decoded[ci]
                sampled = [
                    frames_pil[i] for i in frame_idx if i < len(frames_pil)
                ]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seeds[ci]}"][phase] = r
                scored.append((ci, r))

                print(
                    f"  seed_{seeds[ci]}: total={r['total']:.4f} "
                    f"(R_static={r['R_static']:.4f}, "
                    f"R_dynamic={r['R_dynamic']:.4f}, "
                    f"R_motion={r['R_motion']:.4f}, "
                    f"G_anchor={r['G_anchor']:.2f})"
                )

                if output_dir is not None:
                    self._save_video(
                        decoded[ci], seeds[ci], phase, output_dir, save_fn
                    )

                if is_final:
                    total = float(r.get("total", float("nan")))
                    sel = total if np.isfinite(total) else -float("inf")
                    if sel > best_score:
                        best_frames = frames_pil
                        best_cand_idx = ci
                        best_score = sel

                torch.cuda.empty_cache()

            del decoded

            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self.adapter.load_transformer()
                self.adapter.load_vae()

            if not is_final:
                alive, _ = self._eliminate(
                    scored, seeds, eliminated_at, phase, is_early
                )

            cur_step = end_step

        if best_cand_idx is None:
            raise RuntimeError("No final candidate decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = {
            "mode": "tree_branching_progressive",
            "reward_type": "v2_4rc",
            "sigma_checkpoints": self.sigma_checkpoints,
            "seeds": seeds,
            "best_seed": best_seed,
            "total_time_sec": elapsed,
            "rewards": rewards_log,
            "eliminated_at": eliminated_at,
            "tree_branching": {
                "num_trunks": self.num_trunks,
                "branches_per_trunk": self.branches_per_trunk,
                "branch_sigma": self.branch_sigma,
                "branch_eta": self.branch_eta,
                "branch_step": start_step,
            },
        }

        print(
            f"\n[TreeBranching] Best: seed_{best_seed} "
            f"(total={best_score:.4f}) in {elapsed:.1f}s"
        )
        return best_frames, result_log, best_seed


# ---------------------------------------------------------------------------
# Tree Branching + Gradient Guidance
# ---------------------------------------------------------------------------

class Cosmos3GeoRewardBoNTreeBranchingGuided(Cosmos3GeoRewardBoNTreeBranching):
    """
    Tree Branching + Gradient Guidance after first elimination.

    Guidance activates during denoising within the sigma window (default
    sigma_max=0.83 = after first elimination checkpoint). Designed for
    multi-GPU resident mode: transformer on GPU0, VAE/4RC on other GPUs.
    """

    def __init__(
        self,
        adapter,
        recon_reward,
        guidance,
        num_trunks=None,
        branches_per_trunk=None,
        branch_sigma=None,
        branch_eta=None,
        max_frames=20,
        sigma_checkpoints=None,
        elimination_ratio=None,
        min_survivors=None,
        score_epsilon=None,
        early_max_frames=None,
        offload_models=True,
    ):
        super().__init__(
            adapter=adapter,
            recon_reward=recon_reward,
            num_trunks=num_trunks,
            branches_per_trunk=branches_per_trunk,
            branch_sigma=branch_sigma,
            branch_eta=branch_eta,
            max_frames=max_frames,
            sigma_checkpoints=sigma_checkpoints,
            elimination_ratio=elimination_ratio,
            min_survivors=min_survivors,
            score_epsilon=score_epsilon,
            early_max_frames=early_max_frames,
            offload_models=offload_models,
        )
        self.guidance = guidance

    def _progressive_elimination(
        self, state, N, seeds, num_frames, output_dir, save_fn,
        start_step, t_start,
    ):
        """Same as parent but uses denoise_candidates_with_guidance."""
        total_steps = len(state["timesteps"])

        checkpoint_steps = []
        seen = set()
        for sigma_target in sorted(self.sigma_checkpoints, reverse=True):
            end_step = self.adapter.find_step_for_sigma(state, sigma_target)
            if end_step is None or end_step <= start_step or end_step >= total_steps:
                continue
            if end_step in seen:
                continue
            sched = state["candidates"][0]["scheduler"]
            actual_sigma = float(sched.sigmas[end_step - 1])
            checkpoint_steps.append({
                "end_step": end_step,
                "target_sigma": sigma_target,
                "actual_sigma": actual_sigma,
            })
            seen.add(end_step)

        checkpoint_steps.sort(key=lambda c: c["end_step"])
        checkpoint_steps.append({
            "end_step": total_steps,
            "target_sigma": 0.0,
            "actual_sigma": 0.0,
        })

        early_frame_indices = sample_frames(num_frames, self.early_max_frames)
        normal_frame_indices = sample_frames(num_frames, self.max_frames)

        alive = list(range(N))
        eliminated_at = {}
        rewards_log = {f"seed_{s}": {} for s in seeds}
        best_frames = None
        best_cand_idx = None
        best_score = -float("inf")
        cur_step = start_step

        for cp_idx, cp in enumerate(checkpoint_steps):
            end_step = cp["end_step"]
            is_final = cp_idx == len(checkpoint_steps) - 1
            is_early = cp_idx == 0 and not is_final
            phase = (
                "final_sigma0.00"
                if is_final
                else f"checkpoint{cp_idx + 1}_sigma{cp['actual_sigma']:.4f}"
            )
            frame_idx = early_frame_indices if is_early else normal_frame_indices

            print(
                f"\n[TreeBranchingGuided] Phase: {phase} "
                f"(steps {cur_step}->{end_step}, {len(alive)} alive)"
            )

            # Guided denoising (sigma window auto-skips high-sigma phases)
            last_preds, pre_lats = self.adapter.denoise_candidates_with_guidance(
                state, alive, cur_step, end_step,
                guidance=self.guidance,
                guidance_offload_dit=None,
                guidance_reload_dit=None,
            )

            # Decode
            decoded = {}
            for ci in alive:
                if is_final:
                    lat = state["candidates"][ci]["latent"]
                else:
                    lat = self.adapter.extract_pred_x0(
                        state, ci, last_preds[ci], pre_lats[ci]
                    )
                decoded[ci] = self.adapter.decode_latent(lat.unsqueeze(0))
                del lat
                torch.cuda.empty_cache()

            if self.offload_models:
                self.adapter.offload_transformer()
                self.adapter.offload_vae()
                self._load_4rc()

            # Score
            scored = []
            for ci in alive:
                frames_pil = decoded[ci]
                sampled = [
                    frames_pil[i] for i in frame_idx if i < len(frames_pil)
                ]
                r = self.recon_reward.compute_reward(sampled)
                rewards_log[f"seed_{seeds[ci]}"][phase] = r
                scored.append((ci, r))

                print(
                    f"  seed_{seeds[ci]}: total={r['total']:.4f} "
                    f"(R_static={r['R_static']:.4f}, "
                    f"R_dynamic={r['R_dynamic']:.4f}, "
                    f"R_motion={r['R_motion']:.4f}, "
                    f"G_anchor={r['G_anchor']:.2f})"
                )

                if output_dir is not None:
                    self._save_video(
                        decoded[ci], seeds[ci], phase, output_dir, save_fn
                    )

                if is_final:
                    total = float(r.get("total", float("nan")))
                    sel = total if np.isfinite(total) else -float("inf")
                    if sel > best_score:
                        best_frames = frames_pil
                        best_cand_idx = ci
                        best_score = sel

                torch.cuda.empty_cache()

            del decoded

            if self.offload_models:
                self._offload_4rc()
                if not is_final:
                    self.adapter.load_transformer()
                self.adapter.load_vae()

            if not is_final:
                alive, _ = self._eliminate(
                    scored, seeds, eliminated_at, phase, is_early
                )

            cur_step = end_step

        if best_cand_idx is None:
            raise RuntimeError("No final candidate decoded and scored.")

        best_seed = seeds[best_cand_idx]
        elapsed = time.time() - t_start
        result_log = {
            "mode": "tree_branching_progressive_guided",
            "reward_type": "v2_4rc",
            "sigma_checkpoints": self.sigma_checkpoints,
            "seeds": seeds,
            "best_seed": best_seed,
            "total_time_sec": elapsed,
            "rewards": rewards_log,
            "eliminated_at": eliminated_at,
            "tree_branching": {
                "num_trunks": self.num_trunks,
                "branches_per_trunk": self.branches_per_trunk,
                "branch_sigma": self.branch_sigma,
                "branch_eta": self.branch_eta,
                "branch_step": start_step,
            },
            "guidance": {
                "enabled": True,
                "scale": self.guidance.cfg.geo_guidance_scale,
                "frequency": self.guidance.cfg.geo_guidance_frequency,
                "sigma_min": self.guidance.cfg.sigma_min,
                "sigma_max": self.guidance.cfg.sigma_max,
                "guidance_frames": self.guidance.guidance_frames,
            },
        }

        print(
            f"\n[TreeBranchingGuided] Best: seed_{best_seed} "
            f"(total={best_score:.4f}) in {elapsed:.1f}s"
        )
        return best_frames, result_log, best_seed
