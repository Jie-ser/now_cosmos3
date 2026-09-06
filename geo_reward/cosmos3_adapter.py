"""
Cosmos3 Progressive Denoising Adapter.

Decomposes Cosmos3OmniPipeline's monolithic __call__ into fine-grained
step-by-step denoising control, enabling progressive elimination BoN,
tree branching, and gradient guidance.

Exposes an API parallel to WanI2V's progressive methods:
  prepare_progressive / denoise_candidates / denoise_candidates_with_guidance
  extract_pred_x0 / find_step_for_sigma / branch_candidates / decode_latent
"""

import copy
import math

import numpy as np
import torch


class Cosmos3ProgressiveAdapter:
    """
    Wraps Cosmos3OmniPipeline to expose step-by-step denoising control.

    Each candidate gets its own scheduler instance (UniPC is stateful).
    The transformer's packed-static inputs (text tokens, mRoPE, sequence
    indexes) are computed once and reused across all denoising steps.
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler

    # ------------------------------------------------------------------
    # prepare_progressive
    # ------------------------------------------------------------------
    def prepare_progressive(
        self,
        prompt,
        image,
        seeds,
        num_frames=189,
        height=720,
        width=1280,
        fps=24.0,
        num_inference_steps=35,
        cfg_scale=6.0,
        negative_prompt=None,
    ):
        """
        Prepare shared conditioning and per-candidate initial states.

        Returns a state dict consumed by denoise_candidates and friends.
        """
        device = self.pipe._get_execution_device()
        dtype = self.transformer.dtype

        # 1. Tokenize prompt
        cond_ids, uncond_ids = self.pipe.tokenize_prompt(
            prompt,
            negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
        )

        # 2. Text segments (invariant across steps)
        cond_text = self.pipe._prepare_text_segment(cond_ids, device=device)
        uncond_text = self.pipe._prepare_text_segment(uncond_ids, device=device)

        # 3. Encode conditioning image → get x0_tokens_vision + condition mask
        #    We call prepare_latents once with a throwaway generator to obtain
        #    shapes and masks, then build per-seed latents manually.
        first_gen = torch.Generator(device=device).manual_seed(seeds[0])
        prep = self.pipe.prepare_latents(
            image=image,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=fps,
            generator=first_gen,
            device=device,
            dtype=dtype,
        )
        first_latent = prep[0]
        fps_vision = prep[3]
        vision_condition_mask = prep[5]

        # Recover the clean encoded image (x0_vision).
        # Where mask==1 the latent IS x0_vision; we need the full tensor for
        # per-seed noise blending.  Re-encode the image to get it cleanly.
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            _preprocess_conditioning_image,
        )
        cond_frame = _preprocess_conditioning_image(
            image, height=height, width=width
        ).to(device=device, dtype=dtype)

        vision_tensor = torch.zeros(
            1, 3, num_frames, height, width, dtype=dtype, device=device
        )
        vision_tensor[:, :, 0] = cond_frame
        if num_frames > 1:
            vision_tensor[:, :, 1:] = cond_frame.unsqueeze(2).expand(
                -1, -1, num_frames - 1, -1, -1
            )
        x0_tokens_vision = self.pipe._encode_video(vision_tensor).contiguous().float()
        vision_shape = tuple(x0_tokens_vision.shape)

        # Free the big pixel-space tensor immediately
        del vision_tensor, cond_frame
        torch.cuda.empty_cache()

        # 4. Vision segments (static across steps; only vision_tokens / vision_timesteps vary)
        has_image_condition = True
        condition_frame_indexes = [0]

        cond_vision = self.pipe._prepare_vision_segment(
            input_vision_tokens=first_latent,
            has_image_condition=has_image_condition,
            mrope_offset=cond_text["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=cond_text["und_len"],
            device=device,
            condition_frame_indexes=condition_frame_indexes,
        )
        uncond_vision = self.pipe._prepare_vision_segment(
            input_vision_tokens=first_latent,
            has_image_condition=has_image_condition,
            mrope_offset=uncond_text["vision_start_temporal_offset"],
            vision_fps=fps_vision,
            curr=uncond_text["und_len"],
            device=device,
            condition_frame_indexes=condition_frame_indexes,
        )

        # 5. Assemble packed-static dicts
        cond_packed_static = {
            **cond_text,
            **cond_vision,
            "position_ids": torch.cat(
                [cond_text["text_mrope_ids"], cond_vision["vision_mrope_ids"]], dim=1
            ),
            "sequence_length": cond_text["und_len"]
            + cond_vision["num_vision_tokens"],
        }
        uncond_packed_static = {
            **uncond_text,
            **uncond_vision,
            "position_ids": torch.cat(
                [uncond_text["text_mrope_ids"], uncond_vision["vision_mrope_ids"]],
                dim=1,
            ),
            "sequence_length": uncond_text["und_len"]
            + uncond_vision["num_vision_tokens"],
        }

        # 6. Per-seed candidates with independent schedulers
        from diffusers.utils.torch_utils import randn_tensor

        candidates = []
        for i, seed in enumerate(seeds):
            scheduler_i = copy.deepcopy(self.pipe.scheduler)
            if hasattr(self.pipe, "config") and getattr(
                self.pipe.config, "use_native_flow_schedule", False
            ):
                sigmas = np.linspace(
                    1.0 - 1.0 / scheduler_i.config.num_train_timesteps,
                    0.0,
                    num_inference_steps + 1,
                )[:-1]
                scheduler_i.set_timesteps(
                    num_inference_steps, device=device, sigmas=sigmas
                )
            else:
                scheduler_i.set_timesteps(num_inference_steps, device=device)

            gen_i = torch.Generator(device=device).manual_seed(seed)

            if i == 0:
                latent_i = first_latent
            else:
                noise = randn_tensor(
                    vision_shape,
                    generator=gen_i,
                    device=device,
                    dtype=torch.float32,
                )
                latent_i = (
                    vision_condition_mask
                    * x0_tokens_vision.to(device=device, dtype=torch.float32)
                    + (1.0 - vision_condition_mask) * noise
                )

            candidates.append(
                {
                    "latent": latent_i,
                    "scheduler": scheduler_i,
                    "generator": gen_i,
                    "seed": seed,
                    "step_index": 0,
                }
            )

        timesteps = candidates[0]["scheduler"].timesteps
        num_noisy_vision_tokens = cond_vision["num_noisy_vision_tokens"]

        return {
            "candidates": candidates,
            "timesteps": timesteps,
            "cond_packed_static": cond_packed_static,
            "uncond_packed_static": uncond_packed_static,
            "vision_condition_mask": vision_condition_mask,
            "x0_tokens_vision": x0_tokens_vision,
            "cfg_scale": cfg_scale,
            "num_noisy_vision_tokens": num_noisy_vision_tokens,
            "device": device,
            "dtype": dtype,
        }

    # ------------------------------------------------------------------
    # denoise_candidates
    # ------------------------------------------------------------------
    def denoise_candidates(self, state, alive_indices, start_step, end_step):
        """
        Run denoising steps [start_step, end_step) for alive candidates.

        Returns (last_model_outputs, pre_step_latents) for the final step
        — both dicts keyed by candidate index.
        """
        cond_s = state["cond_packed_static"]
        uncond_s = state["uncond_packed_static"]
        v_mask = state["vision_condition_mask"]
        cfg = state["cfg_scale"]
        timesteps = state["timesteps"]
        n_noisy = state["num_noisy_vision_tokens"]
        device = state["device"]
        dtype = state["dtype"]

        last_outputs = {}
        pre_step_lats = {}

        with torch.no_grad():
            for step_idx in range(start_step, end_step):
                t = timesteps[step_idx]
                ts_val = t.item()
                vis_ts = torch.full((n_noisy,), ts_val, device=device)
                is_last = step_idx == end_step - 1

                for ci in alive_indices:
                    cand = state["candidates"][ci]
                    lat = cand["latent"]
                    vt = lat.to(device=device, dtype=dtype)

                    velocity = self._single_step_forward(
                        vt, vis_ts, cond_s, uncond_s, v_mask, cfg
                    )

                    if is_last:
                        pre_step_lats[ci] = lat.clone()
                        last_outputs[ci] = velocity.clone()

                    result = cand["scheduler"].step(
                        velocity.unsqueeze(0), t, lat.unsqueeze(0),
                        return_dict=False,
                    )
                    cand["latent"] = result[0].squeeze(0)

        return last_outputs, pre_step_lats

    # ------------------------------------------------------------------
    # denoise_candidates_with_guidance
    # ------------------------------------------------------------------
    def denoise_candidates_with_guidance(
        self,
        state,
        alive_indices,
        start_step,
        end_step,
        guidance,
        guidance_offload_dit=None,
        guidance_reload_dit=None,
    ):
        """
        Same as denoise_candidates but with geometric gradient guidance.

        At steps where guidance is active, all candidates' DiT forward passes
        are completed first, then offload happens once, guidance runs for all
        candidates, and reload happens once. This avoids O(N) offload/reload
        cycles per step (matches now project's batched approach).
        """
        cond_s = state["cond_packed_static"]
        uncond_s = state["uncond_packed_static"]
        v_mask = state["vision_condition_mask"]
        cfg = state["cfg_scale"]
        timesteps = state["timesteps"]
        n_noisy = state["num_noisy_vision_tokens"]
        device = state["device"]
        dtype = state["dtype"]

        last_outputs = {}
        pre_step_lats = {}

        for step_idx in range(start_step, end_step):
            t = timesteps[step_idx]
            ts_val = t.item()
            vis_ts = torch.full((n_noisy,), ts_val, device=device)
            is_last = step_idx == end_step - 1

            sigma_t = self._get_current_sigma(
                state["candidates"][alive_indices[0]]["scheduler"], step_idx
            )
            needs_guidance = guidance.should_guide(sigma_t, step_idx)

            # Phase 1: DiT forward for all alive candidates (batched on GPU)
            with torch.no_grad():
                for ci in alive_indices:
                    cand = state["candidates"][ci]
                    vt = cand["latent"].to(device=device, dtype=dtype)

                    velocity = self._single_step_forward(
                        vt, vis_ts, cond_s, uncond_s, v_mask, cfg
                    )
                    cand["_pending_velocity"] = velocity

            # Phase 2: Guidance (offload DiT once, guide all, reload once)
            if needs_guidance:
                if guidance_offload_dit is not None:
                    guidance_offload_dit()

                for ci in alive_indices:
                    cand = state["candidates"][ci]
                    cand["_pending_velocity"] = guidance.guided_v_pred(
                        cand["latent"],
                        cand["_pending_velocity"],
                        sigma_t,
                        step_idx,
                    )

                if guidance_reload_dit is not None:
                    guidance_reload_dit()

            # Phase 3: Scheduler step for all candidates
            for ci in alive_indices:
                cand = state["candidates"][ci]
                lat = cand["latent"]
                velocity = cand.pop("_pending_velocity")

                if is_last:
                    pre_step_lats[ci] = lat.clone()
                    last_outputs[ci] = velocity.clone()

                result = cand["scheduler"].step(
                    velocity.unsqueeze(0), t, lat.unsqueeze(0),
                    return_dict=False,
                )
                cand["latent"] = result[0].squeeze(0)

        return last_outputs, pre_step_lats

    # ------------------------------------------------------------------
    # _single_step_forward  (shared by both denoise variants)
    # ------------------------------------------------------------------
    def _single_step_forward(
        self, vision_tokens, vision_timesteps, cond_s, uncond_s, v_mask, cfg_scale
    ):
        """Run cond + uncond transformer forward, CFG combine, return velocity."""
        preds_v, _, _ = self.transformer(
            input_ids=cond_s["input_ids"],
            text_indexes=cond_s["text_indexes"],
            position_ids=cond_s["position_ids"],
            und_len=cond_s["und_len"],
            sequence_length=cond_s["sequence_length"],
            vision_tokens=[vision_tokens],
            vision_token_shapes=cond_s["vision_token_shapes"],
            vision_sequence_indexes=cond_s["vision_sequence_indexes"],
            vision_mse_loss_indexes=cond_s["vision_mse_loss_indexes"],
            vision_timesteps=vision_timesteps,
            vision_noisy_frame_indexes=cond_s["vision_noisy_frame_indexes"],
            return_dict=False,
        )
        cond_v, _, _ = self.pipe._mask_velocity_predictions(
            preds_v, None, vision_condition_mask=[v_mask]
        )

        if cfg_scale != 1.0:
            preds_v_u, _, _ = self.transformer(
                input_ids=uncond_s["input_ids"],
                text_indexes=uncond_s["text_indexes"],
                position_ids=uncond_s["position_ids"],
                und_len=uncond_s["und_len"],
                sequence_length=uncond_s["sequence_length"],
                vision_tokens=[vision_tokens],
                vision_token_shapes=uncond_s["vision_token_shapes"],
                vision_sequence_indexes=uncond_s["vision_sequence_indexes"],
                vision_mse_loss_indexes=uncond_s["vision_mse_loss_indexes"],
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=uncond_s["vision_noisy_frame_indexes"],
                return_dict=False,
            )
            uncond_v, _, _ = self.pipe._mask_velocity_predictions(
                preds_v_u, None, vision_condition_mask=[v_mask]
            )
            velocity = uncond_v + cfg_scale * (cond_v - uncond_v)
        else:
            velocity = cond_v

        return velocity

    # ------------------------------------------------------------------
    # _get_current_sigma  (safe accessor with None fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_current_sigma(scheduler, step_idx):
        """Get sigma for the current step, handling None step_index gracefully."""
        si = scheduler.step_index
        if si is not None:
            return float(scheduler.sigmas[si])
        if step_idx < len(scheduler.sigmas):
            return float(scheduler.sigmas[step_idx])
        return float(scheduler.sigmas[-1])

    # ------------------------------------------------------------------
    # extract_pred_x0
    # ------------------------------------------------------------------
    def extract_pred_x0(self, state, cand_idx, model_output, pre_step_latent):
        """
        Compute the predicted clean latent: x0 = x_t - sigma_t * v_pred.

        Uses the latent BEFORE scheduler.step() and the sigma at that step.
        Prediction type is "flow_prediction" for Cosmos3.
        """
        cand = state["candidates"][cand_idx]
        sched = cand["scheduler"]
        step_idx = sched.step_index - 1
        if step_idx < 0:
            step_idx = 0
        sigma_t = float(sched.sigmas[step_idx])
        x0 = pre_step_latent - sigma_t * model_output
        return x0

    # ------------------------------------------------------------------
    # find_step_for_sigma
    # ------------------------------------------------------------------
    def find_step_for_sigma(self, state, target_sigma):
        """
        Find the first completed step where sigma <= target_sigma.

        Returns an exclusive end_step suitable for denoise_candidates().
        """
        sched = state["candidates"][0]["scheduler"]
        sigmas = sched.sigmas.cpu().numpy()
        for i in range(len(sigmas) - 1):
            if sigmas[i] <= target_sigma:
                return i + 1
        return None

    # ------------------------------------------------------------------
    # branch_candidates
    # ------------------------------------------------------------------
    def branch_candidates(
        self, state, trunk_indices, branches_per_trunk, eta, branch_seeds
    ):
        """
        From K trunk latents, create K * branches_per_trunk branched candidates.

        z_branch = sqrt(1 - eta^2) * z_trunk + eta * sigma_t * epsilon

        Each branch gets a deep-copied scheduler to preserve UniPC's
        multi-step history.
        """
        new_candidates = []
        branch_idx = 0

        for trunk_idx in trunk_indices:
            trunk = state["candidates"][trunk_idx]
            z_trunk = trunk["latent"].clone()
            step_idx = trunk["scheduler"].step_index
            sigma_t = float(trunk["scheduler"].sigmas[step_idx])

            for _ in range(branches_per_trunk):
                seed = branch_seeds[branch_idx]
                gen = torch.Generator(device=z_trunk.device).manual_seed(seed)
                epsilon = torch.randn(
                    z_trunk.shape,
                    generator=gen,
                    device=z_trunk.device,
                    dtype=z_trunk.dtype,
                )

                z_branch = (
                    math.sqrt(1.0 - eta**2) * z_trunk + eta * sigma_t * epsilon
                )

                new_sched = copy.deepcopy(trunk["scheduler"])

                new_candidates.append(
                    {
                        "latent": z_branch,
                        "scheduler": new_sched,
                        "generator": gen,
                        "seed": seed,
                        "step_index": step_idx,
                        "parent_trunk": trunk_idx,
                    }
                )
                branch_idx += 1

        state["candidates"] = new_candidates
        return state

    # ------------------------------------------------------------------
    # decode_latent
    # ------------------------------------------------------------------
    def decode_latent(self, latent):
        """
        VAE decode a single latent tensor → list of PIL Images.

        Handles the Cosmos3 latent normalization reversal:
        z_raw = z_normalized / inv_std + mean
        """
        with torch.no_grad():
            vae_dtype = self.vae.dtype
            mean = self.pipe._vae_latents_mean.to(
                device=latent.device, dtype=vae_dtype
            )
            inv_std = self.pipe._vae_latents_inv_std.to(
                device=latent.device, dtype=vae_dtype
            )

            z_raw = (
                latent.to(vae_dtype) / inv_std.view(1, -1, 1, 1, 1)
                + mean.view(1, -1, 1, 1, 1)
            )

            decoded = self.vae.decode(z_raw).sample

        video = self.pipe.video_processor.postprocess_video(
            decoded, output_type="pil"
        )[0]
        return video

    # ------------------------------------------------------------------
    # cleanup_progressive
    # ------------------------------------------------------------------
    def cleanup_progressive(self, state):
        """Release CUDA tensors held by the progressive state."""
        if state is None:
            return
        for cand in state.get("candidates", []):
            cand.pop("latent", None)
            cand.pop("scheduler", None)
            cand.pop("generator", None)
        state.pop("cond_packed_static", None)
        state.pop("uncond_packed_static", None)
        state.pop("vision_condition_mask", None)
        state.pop("x0_tokens_vision", None)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Model offload helpers
    # ------------------------------------------------------------------
    def offload_transformer(self):
        """Move transformer to CPU to free GPU for 4RC scoring."""
        self.transformer.cpu()
        torch.cuda.empty_cache()

    def load_transformer(self):
        """Move transformer back to GPU for denoising."""
        device = self.pipe._get_execution_device()
        self.transformer.to(device)

    def offload_vae(self):
        """Move VAE to CPU."""
        self.vae.cpu()
        torch.cuda.empty_cache()

    def load_vae(self):
        """Move VAE back to GPU."""
        device = self.pipe._get_execution_device()
        self.vae.to(device)
