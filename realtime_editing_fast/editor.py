from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageOps
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _parse_steps_mask(mask_text: str, expected_steps: int) -> list[int]:
    cleaned = mask_text.replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError("steps mask cannot be empty")
    if any(ch not in ("0", "1") for ch in cleaned):
        raise ValueError(f"steps mask must only contain 0/1, got: {mask_text}")
    mask = [int(ch) for ch in cleaned]
    if len(mask) != expected_steps:
        raise ValueError(
            f"steps mask length mismatch: got {len(mask)}, expected {expected_steps} (num_inference_steps)",
        )
    return mask


@dataclass(slots=True)
class FastFlux2Config:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    device: str = "cuda"
    gpu_id: int = 0
    dtype: str = "bfloat16"
    attention_backend: str = "sage"
    width: int = 512
    height: int = 512
    num_inference_steps: int = 2
    guidance_scale: float = 1.0
    seed: int = 0

    # Fastest benchmarked stack from this workspace:
    # DBCache + steps_mask=10 + TaylorSeer + compile + triton.cudagraphs=False.
    enable_cache_dit: bool = True
    cache_fn: int = 1
    cache_bn: int = 0
    residual_diff_threshold: float = 0.8
    single_block_rdt_scale: float = 3.0
    cache_max_warmup_steps: int = 0
    cache_warmup_interval: int = 1
    cache_max_cached_steps: int = -1
    cache_max_continuous_cached_steps: int = -1
    cache_enable_separate_cfg: bool = False
    steps_mask: str = "10"
    steps_computation_policy: str = "dynamic"
    enable_taylorseer: bool = True
    taylorseer_order: int = 1

    compile_transformer: bool = True
    compile_disable_cudagraphs: bool = True
    input_resize_mode: str = "equivalent_area"  # "equivalent_area" keeps aspect ratio with ~512x512 pixel area.
    verbose: bool = True

    @property
    def runtime_device(self) -> str:
        if self.device == "cuda":
            return f"cuda:{self.gpu_id}"
        return self.device


class FastFlux2RealtimeEditor:
    def __init__(self, config: Optional[FastFlux2Config] = None):
        self.config = config or FastFlux2Config()
        self._pipe = None
        self._cache_dit_mod = None

        self._init_lock = threading.Lock()
        self._edit_lock = threading.Lock()

        self._cached_prompt: Optional[str] = None
        self._cached_prompt_embeds: Optional[torch.Tensor] = None
        self._cached_text_ids: Optional[torch.Tensor] = None
        self._request_idx = 0

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def _log(self, msg: str) -> None:
        if not self.config.verbose:
            return
        ts = time.strftime("%H:%M:%S")
        print(f"*** [FastFlux2][{ts}] {msg}", flush=True)

    def _sync_if_cuda(self) -> None:
        if self.config.runtime_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _new_request_tag(self) -> str:
        self._request_idx += 1
        return f"req-{self._request_idx:06d}"

    @staticmethod
    def _round_to_multiple(v: float, multiple: int) -> int:
        return max(multiple, int(round(v / multiple) * multiple))

    def _compute_equivalent_resolution(
        self,
        src_w: int,
        src_h: int,
        target_area: int,
        multiple: int,
    ) -> tuple[int, int]:
        src_area = max(1.0, float(src_w * src_h))
        scale = (float(target_area) / src_area) ** 0.5
        proc_w = self._round_to_multiple(src_w * scale, multiple)
        proc_h = self._round_to_multiple(src_h * scale, multiple)
        return proc_w, proc_h

    def ensure_loaded(self) -> None:
        if self._pipe is not None:
            return

        with self._init_lock:
            if self._pipe is not None:
                return

            cfg = self.config
            runtime_device = cfg.runtime_device
            dtype = _parse_dtype(cfg.dtype)
            self._log(f"model init start: model={cfg.model_id}, device={runtime_device}, dtype={dtype}")
            init_t0 = time.perf_counter()

            if runtime_device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
                torch.cuda.set_device(runtime_device)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            pipe = AutoPipelineForImage2Image.from_pretrained(cfg.model_id, torch_dtype=dtype)
            pipe = pipe.to(runtime_device)
            pipe.set_progress_bar_config(disable=True)
            requested_attention_backend = cfg.attention_backend.strip().lower()
            if requested_attention_backend not in ("", "none", "default", "auto"):
                if not hasattr(pipe.transformer, "set_attention_backend"):
                    raise RuntimeError("Current transformer does not support set_attention_backend().")
                try:
                    pipe.transformer.set_attention_backend(requested_attention_backend)
                    self._log(f"attention backend set: {requested_attention_backend}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to set attention backend to '{requested_attention_backend}'."
                    ) from exc

            if cfg.enable_cache_dit:
                self._enable_cache_dit(pipe)

            if cfg.compile_transformer:
                if self._cache_dit_mod is not None:
                    self._cache_dit_mod.set_compile_configs()

                if cfg.compile_disable_cudagraphs:
                    compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
                else:
                    compile_kwargs = {"mode": "reduce-overhead", "fullgraph": False}

                pipe.transformer = torch.compile(pipe.transformer, **compile_kwargs)
                self._log(f"transformer compiled: kwargs={compile_kwargs}")

            self._pipe = pipe
            self._sync_if_cuda()
            self._log(f"model init done: {(time.perf_counter() - init_t0) * 1000.0:.1f} ms")

    def _enable_cache_dit(self, pipe) -> None:
        cfg = self.config
        try:
            import cache_dit as cache_dit_mod
            from cache_dit import (
                BlockAdapter,
                DBCacheConfig,
                ForwardPattern,
                ParamsModifier,
                TaylorSeerCalibratorConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "cache-dit is not available. Install it first, e.g. `uv pip install -U cache-dit`."
            ) from exc

        steps_mask = _parse_steps_mask(cfg.steps_mask, cfg.num_inference_steps)

        cache_config = DBCacheConfig(
            Fn_compute_blocks=cfg.cache_fn,
            Bn_compute_blocks=cfg.cache_bn,
            residual_diff_threshold=cfg.residual_diff_threshold,
            max_warmup_steps=cfg.cache_max_warmup_steps,
            warmup_interval=cfg.cache_warmup_interval,
            max_cached_steps=cfg.cache_max_cached_steps,
            max_continuous_cached_steps=cfg.cache_max_continuous_cached_steps,
            enable_separate_cfg=cfg.cache_enable_separate_cfg,
            num_inference_steps=cfg.num_inference_steps,
            steps_computation_mask=steps_mask,
            steps_computation_policy=cfg.steps_computation_policy,
        )

        cache_config_cls = cache_config.__class__
        params_modifiers = [
            ParamsModifier(
                cache_config=cache_config_cls().reset(
                    residual_diff_threshold=cfg.residual_diff_threshold,
                ),
            ),
            ParamsModifier(
                cache_config=cache_config_cls().reset(
                    residual_diff_threshold=cfg.residual_diff_threshold * cfg.single_block_rdt_scale,
                ),
            ),
        ]

        calibrator_config = (
            TaylorSeerCalibratorConfig(taylorseer_order=cfg.taylorseer_order)
            if cfg.enable_taylorseer
            else None
        )

        # Use transformer-only adapter because this realtime path bypasses pipe.__call__.
        cache_adapter = BlockAdapter(
            pipe=None,
            transformer=pipe.transformer,
            blocks=[
                pipe.transformer.transformer_blocks,
                pipe.transformer.single_transformer_blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_1,
                ForwardPattern.Pattern_3,
            ],
        )

        cache_dit_mod.enable_cache(
            cache_adapter,
            cache_config=cache_config,
            calibrator_config=calibrator_config,
            params_modifiers=params_modifiers,
        )
        self._cache_dit_mod = cache_dit_mod

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        pipe = self._pipe
        assert pipe is not None

        if (
            prompt == self._cached_prompt
            and self._cached_prompt_embeds is not None
            and self._cached_text_ids is not None
        ):
            return self._cached_prompt_embeds, self._cached_text_ids

        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt,
            prompt_embeds=None,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

        self._cached_prompt = prompt
        self._cached_prompt_embeds = prompt_embeds
        self._cached_text_ids = text_ids
        return prompt_embeds, text_ids

    def _decode_latents_to_pil(
        self,
        latents: torch.Tensor,
        latent_ids: torch.Tensor,
    ) -> Image.Image:
        pipe = self._pipe
        assert pipe is not None

        latents = pipe._unpack_latents_with_ids(latents, latent_ids)
        latents_bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            latents.device,
            latents.dtype,
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = pipe._unpatchify_latents(latents)

        image = pipe.vae.decode(latents, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def _prepare_inputs(
        self,
        image: Image.Image,
        prompt: str,
        seed: int,
    ) -> dict:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        prompt_embeds, text_ids = self._encode_prompt(prompt)

        image = image.convert("RGB")
        source_size = image.size
        multiple = int(pipe.vae_scale_factor * 2)
        target_area = int(cfg.width * cfg.height)
        if cfg.input_resize_mode == "equivalent_area":
            proc_w, proc_h = self._compute_equivalent_resolution(
                src_w=source_size[0],
                src_h=source_size[1],
                target_area=target_area,
                multiple=multiple,
            )
            image_for_preprocess = image.resize((proc_w, proc_h), Image.Resampling.BICUBIC)
            resize_mode = "crop"  # no crop happens because image already matches target size.
            target_w, target_h = proc_w, proc_h
        elif cfg.input_resize_mode == "crop":
            image_for_preprocess = image
            resize_mode = "crop"
            target_w, target_h = cfg.width, cfg.height
        elif cfg.input_resize_mode == "pad":
            contained = ImageOps.contain(image, (cfg.width, cfg.height), method=Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (cfg.width, cfg.height), (0, 0, 0))
            offset = ((cfg.width - contained.width) // 2, (cfg.height - contained.height) // 2)
            canvas.paste(contained, offset)
            image_for_preprocess = canvas
            resize_mode = "crop"
            target_w, target_h = cfg.width, cfg.height
        else:
            raise ValueError(f"Unsupported input_resize_mode: {cfg.input_resize_mode}")

        image_tensor = pipe.image_processor.preprocess(
            image_for_preprocess,
            height=target_h,
            width=target_w,
            resize_mode=resize_mode,
        )

        generator = torch.Generator(device=cfg.runtime_device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=target_h,
            width=target_w,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=[image_tensor],
            batch_size=1,
            generator=generator,
            device=pipe._execution_device,
            dtype=pipe.vae.dtype,
        )

        sigmas = np.linspace(1.0, 1.0 / cfg.num_inference_steps, cfg.num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=cfg.num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            cfg.num_inference_steps,
            pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )

        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "image_latents": image_latents,
            "image_latent_ids": image_latent_ids,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "source_size": source_size,
            "target_size": (target_w, target_h),
        }

    def _prepare_t2i_inputs(
        self,
        prompt: str,
        seed: int,
    ) -> dict:
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        prompt_embeds, text_ids = self._encode_prompt(prompt)
        target_h = int(cfg.height)
        target_w = int(cfg.width)

        generator = torch.Generator(device=cfg.runtime_device).manual_seed(seed)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=target_h,
            width=target_w,
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        sigmas = np.linspace(1.0, 1.0 / cfg.num_inference_steps, cfg.num_inference_steps)
        if hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas:
            sigmas = None

        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=cfg.num_inference_steps)
        timesteps, _ = retrieve_timesteps(
            pipe.scheduler,
            cfg.num_inference_steps,
            pipe._execution_device,
            sigmas=sigmas,
            mu=mu,
        )

        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
            "target_size": (target_w, target_h),
        }

    @torch.no_grad()
    def edit_image_with_meta(
        self,
        image: Image.Image,
        prompt: str,
        seed: Optional[int] = None,
    ) -> tuple[Image.Image, dict]:
        if image is None:
            raise ValueError("Input image is required.")
        if not prompt:
            raise ValueError("Prompt is required.")

        self.ensure_loaded()
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        seed = cfg.seed if seed is None else int(seed)
        req_tag = self._new_request_tag()
        total_t0 = time.perf_counter()
        self._log(f"{req_tag} begin: seed={seed}, prompt_len={len(prompt)}, resize_mode={cfg.input_resize_mode}")

        with self._edit_lock:
            stage_t0 = time.perf_counter()
            if self._cache_dit_mod is not None:
                self._cache_dit_mod.refresh_context(
                    pipe.transformer,
                    num_inference_steps=cfg.num_inference_steps,
                    verbose=False,
                )
            self._sync_if_cuda()
            refresh_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(f"{req_tag} stage refresh_context: {refresh_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            inputs = self._prepare_inputs(image=image, prompt=prompt, seed=seed)
            self._sync_if_cuda()
            prepare_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(
                f"{req_tag} stage prepare_inputs: {prepare_ms:.1f} ms, source={inputs['source_size']}, "
                f"target={inputs['target_size']}, latents={tuple(inputs['latents'].shape)}"
            )
            latents = inputs["latents"]
            latent_ids = inputs["latent_ids"]
            image_latents = inputs["image_latents"]
            image_latent_ids = inputs["image_latent_ids"]

            pipe.scheduler.set_begin_index(0)
            pipe.scheduler._step_index = None

            step_ms: list[float] = []
            for timestep_value in inputs["timesteps"]:
                step_t0 = time.perf_counter()
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)

                latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=inputs["prompt_embeds"],
                        txt_ids=inputs["text_ids"],
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1), :]
                latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                self._sync_if_cuda()
                one_step_ms = (time.perf_counter() - step_t0) * 1000.0
                step_ms.append(one_step_ms)
                self._log(f"{req_tag} stage denoise_step[{len(step_ms)}]: {one_step_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            decoded = self._decode_latents_to_pil(latents, latent_ids)
            if decoded.size != inputs["source_size"]:
                decoded = decoded.resize(inputs["source_size"], Image.Resampling.BICUBIC)
            self._sync_if_cuda()
            decode_ms = (time.perf_counter() - stage_t0) * 1000.0
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            self._log(f"{req_tag} stage decode: {decode_ms:.1f} ms")
            self._log(f"{req_tag} done total: {total_ms:.1f} ms")

            meta = {
                "request_tag": req_tag,
                "refresh_ms": refresh_ms,
                "prepare_ms": prepare_ms,
                "step_ms": step_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "source_size": inputs["source_size"],
                "target_size": inputs["target_size"],
                "resize_mode": cfg.input_resize_mode,
            }
            return decoded, meta

    @torch.no_grad()
    def edit_image(self, image: Image.Image, prompt: str, seed: Optional[int] = None) -> Image.Image:
        edited, _ = self.edit_image_with_meta(image=image, prompt=prompt, seed=seed)
        return edited

    @torch.no_grad()
    def generate_text_to_image_with_meta(
        self,
        prompt: str,
        seed: Optional[int] = None,
    ) -> tuple[Image.Image, dict]:
        if not prompt:
            raise ValueError("Prompt is required.")

        self.ensure_loaded()
        pipe = self._pipe
        cfg = self.config
        assert pipe is not None

        seed = cfg.seed if seed is None else int(seed)
        req_tag = self._new_request_tag()
        total_t0 = time.perf_counter()
        self._log(f"{req_tag} begin txt2img: seed={seed}, prompt_len={len(prompt)}")

        with self._edit_lock:
            stage_t0 = time.perf_counter()
            if self._cache_dit_mod is not None:
                self._cache_dit_mod.refresh_context(
                    pipe.transformer,
                    num_inference_steps=cfg.num_inference_steps,
                    verbose=False,
                )
            self._sync_if_cuda()
            refresh_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(f"{req_tag} stage refresh_context: {refresh_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            inputs = self._prepare_t2i_inputs(prompt=prompt, seed=seed)
            self._sync_if_cuda()
            prepare_ms = (time.perf_counter() - stage_t0) * 1000.0
            self._log(
                f"{req_tag} stage prepare_t2i_inputs: {prepare_ms:.1f} ms, "
                f"target={inputs['target_size']}, latents={tuple(inputs['latents'].shape)}"
            )

            latents = inputs["latents"]
            latent_ids = inputs["latent_ids"]

            pipe.scheduler.set_begin_index(0)
            pipe.scheduler._step_index = None

            step_ms: list[float] = []
            for timestep_value in inputs["timesteps"]:
                step_t0 = time.perf_counter()
                timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(pipe.transformer.dtype)

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()

                with pipe.transformer.cache_context("cond"):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=inputs["prompt_embeds"],
                        txt_ids=inputs["text_ids"],
                        img_ids=latent_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred[:, : latents.size(1), :]
                latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
                self._sync_if_cuda()
                one_step_ms = (time.perf_counter() - step_t0) * 1000.0
                step_ms.append(one_step_ms)
                self._log(f"{req_tag} stage denoise_step[{len(step_ms)}]: {one_step_ms:.1f} ms")

            stage_t0 = time.perf_counter()
            decoded = self._decode_latents_to_pil(latents, latent_ids)
            self._sync_if_cuda()
            decode_ms = (time.perf_counter() - stage_t0) * 1000.0
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            self._log(f"{req_tag} stage decode: {decode_ms:.1f} ms")
            self._log(f"{req_tag} done txt2img total: {total_ms:.1f} ms")

            meta = {
                "request_tag": req_tag,
                "refresh_ms": refresh_ms,
                "prepare_ms": prepare_ms,
                "step_ms": step_ms,
                "decode_ms": decode_ms,
                "total_ms": total_ms,
                "source_size": None,
                "target_size": inputs["target_size"],
                "resize_mode": "txt2img",
            }
            return decoded, meta

    @torch.no_grad()
    def generate_text_to_image(self, prompt: str, seed: Optional[int] = None) -> Image.Image:
        image, _ = self.generate_text_to_image_with_meta(prompt=prompt, seed=seed)
        return image
