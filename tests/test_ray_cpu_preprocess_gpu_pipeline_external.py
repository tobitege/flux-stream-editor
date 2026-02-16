from __future__ import annotations

import base64
import gc
import io
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import ray
from ray.job_config import JobConfig
import torch
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.editor import FastFlux2RealtimeEditor, compute_empirical_mu, retrieve_timesteps
from realtime_editing_fast.realtime_img2img_server import build_default_config


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _make_data_url_jpeg(width: int = 640, height: int = 360) -> str:
    image = Image.new("RGB", (width, height), (118, 136, 158))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _base64_to_pil(base64_image: str) -> Image.Image:
    encoded = base64_image
    if "," in encoded and "base64" in encoded[:40].lower():
        encoded = encoded.split(",", 1)[1]
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _build_config():
    attention_backend = os.getenv("FLUX_RAY_ATTENTION_BACKEND", "native")
    num_steps = int(os.getenv("FLUX_RAY_NUM_STEPS", "2"))
    width = int(os.getenv("FLUX_RAY_WIDTH", "512"))
    height = int(os.getenv("FLUX_RAY_HEIGHT", "512"))
    compile_transformer = os.getenv("FLUX_RAY_COMPILE", "0") == "1"
    enable_cache = os.getenv("FLUX_RAY_ENABLE_CACHE", "1") == "1"

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=num_steps)
    return replace(
        cfg,
        width=width,
        height=height,
        compile_transformer=compile_transformer,
        enable_cache_dit=enable_cache,
        verbose=False,
        input_resize_mode="equivalent_area",
    )


@ray.remote
class CpuPreprocessActor:
    def __init__(self, width: int, height: int, input_resize_mode: str, multiple: int):
        self.width = int(width)
        self.height = int(height)
        self.input_resize_mode = str(input_resize_mode)
        self.multiple = int(multiple)

    @staticmethod
    def _round_to_multiple(v: float, multiple: int) -> int:
        return max(multiple, int(round(v / multiple) * multiple))

    @classmethod
    def _compute_equivalent_resolution(
        cls,
        src_w: int,
        src_h: int,
        target_area: int,
        multiple: int,
    ) -> tuple[int, int]:
        src_area = max(1.0, float(src_w * src_h))
        scale = (float(target_area) / src_area) ** 0.5
        proc_w = cls._round_to_multiple(src_w * scale, multiple)
        proc_h = cls._round_to_multiple(src_h * scale, multiple)
        return proc_w, proc_h

    def preprocess(self, frame_b64: str, frame_id: int) -> dict:
        t0_decode = time.perf_counter()
        image = _base64_to_pil(frame_b64)
        decode_ms = (time.perf_counter() - t0_decode) * 1000.0

        t0_pre = time.perf_counter()
        image = image.convert("RGB")
        source_size = image.size
        target_area = int(self.width * self.height)

        if self.input_resize_mode == "equivalent_area":
            proc_w, proc_h = self._compute_equivalent_resolution(
                src_w=source_size[0],
                src_h=source_size[1],
                target_area=target_area,
                multiple=self.multiple,
            )
            # Bilinear is slightly faster than bicubic in our benchmark.
            image_for_preprocess = image.resize((proc_w, proc_h), Image.Resampling.BILINEAR)
            target_w, target_h = proc_w, proc_h
        elif self.input_resize_mode == "crop":
            image_for_preprocess = image
            target_w, target_h = self.width, self.height
        elif self.input_resize_mode == "pad":
            contained = ImageOps.contain(image, (self.width, self.height), method=Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
            offset = ((self.width - contained.width) // 2, (self.height - contained.height) // 2)
            canvas.paste(contained, offset)
            image_for_preprocess = canvas
            target_w, target_h = self.width, self.height
        else:
            raise ValueError(f"Unsupported input_resize_mode: {self.input_resize_mode}")

        arr = np.array(image_for_preprocess, dtype=np.uint8, copy=True)
        image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor * 2.0 - 1.0
        preprocess_ms = (time.perf_counter() - t0_pre) * 1000.0

        return {
            "frame_id": int(frame_id),
            "source_size": tuple(source_size),
            "target_size": (int(target_w), int(target_h)),
            "decode_input_ms": float(decode_ms),
            "preprocess_cpu_ms": float(preprocess_ms),
            "image_tensor": image_tensor,
        }


@ray.remote(num_gpus=1)
class GpuInferenceActor:
    def __init__(self, cfg):
        self.editor = FastFlux2RealtimeEditor(cfg)
        self.editor.ensure_loaded()

    def get_multiple(self) -> int:
        pipe = self.editor._pipe
        assert pipe is not None
        return int(pipe.vae_scale_factor * 2)

    @torch.no_grad()
    def infer(self, packet: dict, prompt: str, seed: int) -> dict:
        pipe = self.editor._pipe
        cfg = self.editor.config
        assert pipe is not None

        prompt_embeds, text_ids = self.editor._encode_prompt(prompt)
        target_w, target_h = packet["target_size"]
        generator = torch.Generator(device=cfg.runtime_device).manual_seed(int(seed))

        t0_prepare = time.perf_counter()
        num_channels_latents = pipe.transformer.config.in_channels // 4
        latents, latent_ids = pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels_latents,
            height=int(target_h),
            width=int(target_w),
            dtype=prompt_embeds.dtype,
            device=pipe._execution_device,
            generator=generator,
            latents=None,
        )

        image_latents, image_latent_ids = pipe.prepare_image_latents(
            images=[packet["image_tensor"]],
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prepare_gpu_ms = (time.perf_counter() - t0_prepare) * 1000.0

        if self.editor._cache_dit_mod is not None:
            self.editor._cache_dit_mod.refresh_context(
                pipe.transformer,
                num_inference_steps=cfg.num_inference_steps,
                verbose=False,
            )

        pipe.scheduler.set_begin_index(0)
        pipe.scheduler._step_index = None

        t0_denoise = time.perf_counter()
        step_ms: list[float] = []
        for timestep_value in timesteps:
            ts = time.perf_counter()
            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)
            latent_model_input = torch.cat([latents, image_latents], dim=1).to(pipe.transformer.dtype)
            latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]

            noise_pred = noise_pred[:, : latents.size(1), :]
            latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_ms.append((time.perf_counter() - ts) * 1000.0)
        denoise_ms = (time.perf_counter() - t0_denoise) * 1000.0

        t0_decode = time.perf_counter()
        decoded = self.editor._decode_latents_to_pil(latents, latent_ids)
        if decoded.size != tuple(packet["source_size"]):
            decoded = decoded.resize(tuple(packet["source_size"]), Image.Resampling.BICUBIC)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_model_ms = (time.perf_counter() - t0_decode) * 1000.0

        return {
            "frame_id": int(packet["frame_id"]),
            "decode_input_ms": float(packet["decode_input_ms"]),
            "preprocess_cpu_ms": float(packet["preprocess_cpu_ms"]),
            "prepare_gpu_ms": float(prepare_gpu_ms),
            "denoise_ms": float(denoise_ms),
            "decode_model_ms": float(decode_model_ms),
            "step_ms": step_ms,
            "stage_total_ms": float(
                packet["decode_input_ms"] + packet["preprocess_cpu_ms"] + prepare_gpu_ms + denoise_ms + decode_model_ms
            ),
        }


@pytest.mark.skipif(
    os.getenv("RUN_RAY_CPU_GPU_PIPELINE_TEST") != "1",
    reason="Set RUN_RAY_CPU_GPU_PIPELINE_TEST=1 to run heavy Ray CPU+GPU pipeline benchmark.",
)
def test_ray_cpu_preprocess_gpu_pipeline_external() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    frame_count = int(os.getenv("FLUX_RAY_FRAME_COUNT", "8"))
    warmup_count = int(os.getenv("FLUX_RAY_WARMUP", "1"))
    prompt = "Convert this live frame into a cinematic anime illustration with clean lines and rich color."
    frame_b64 = _make_data_url_jpeg()
    cfg = _build_config()

    # Serial baseline.
    serial_editor = FastFlux2RealtimeEditor(cfg)
    serial_editor.ensure_loaded()
    for i in range(max(0, warmup_count)):
        serial_editor.edit_image_with_meta(image=_base64_to_pil(frame_b64), prompt=prompt, seed=i)

    serial_total_ms: list[float] = []
    serial_t0 = time.perf_counter()
    for i in range(max(1, frame_count)):
        _, meta = serial_editor.edit_image_with_meta(
            image=_base64_to_pil(frame_b64),
            prompt=prompt,
            seed=10000 + i,
        )
        serial_total_ms.append(float(meta["total_ms"]))
    serial_elapsed = time.perf_counter() - serial_t0
    serial_fps = max(1, frame_count) / max(1e-9, serial_elapsed)

    # Free GPU memory from baseline before Ray actors.
    del serial_editor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        logging_level="ERROR",
        job_config=JobConfig(code_search_path=[str(REPO_ROOT), str(REPO_ROOT / "tests")]),
        runtime_env={
            "env_vars": {
                "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'tests'}",
                "TOKENIZERS_PARALLELISM": "false",
            }
        },
    )
    try:
        gpu_actor = GpuInferenceActor.remote(cfg)
        multiple = ray.get(gpu_actor.get_multiple.remote())
        cpu_actor = CpuPreprocessActor.remote(
            int(cfg.width),
            int(cfg.height),
            str(cfg.input_resize_mode),
            int(multiple),
        )

        for i in range(max(0, warmup_count)):
            pre = cpu_actor.preprocess.remote(frame_b64, i)
            out = gpu_actor.infer.remote(pre, prompt, i)
            ray.get(out)

        submit_ts: dict[int, float] = {}
        futures = []
        pipe_t0 = time.perf_counter()
        for i in range(max(1, frame_count)):
            frame_id = 20000 + i
            submit_ts[frame_id] = time.perf_counter()
            pre = cpu_actor.preprocess.remote(frame_b64, frame_id)
            out = gpu_actor.infer.remote(pre, prompt, frame_id)
            futures.append(out)

        results: list[dict] = []
        e2e_ms: list[float] = []
        remaining = list(futures)
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1)
            item = ray.get(ready[0])
            results.append(item)
            e2e_ms.append((time.perf_counter() - submit_ts[item["frame_id"]]) * 1000.0)

        pipe_elapsed = time.perf_counter() - pipe_t0
        pipe_fps = max(1, frame_count) / max(1e-9, pipe_elapsed)
    finally:
        ray.shutdown()

    result = {
        "frame_count": max(1, frame_count),
        "warmup_count": max(0, warmup_count),
        "config": {
            "width": int(cfg.width),
            "height": int(cfg.height),
            "num_inference_steps": int(cfg.num_inference_steps),
            "attention_backend": str(cfg.attention_backend),
            "compile_transformer": bool(cfg.compile_transformer),
            "enable_cache_dit": bool(cfg.enable_cache_dit),
        },
        "serial_avg_total_ms": _avg(serial_total_ms),
        "serial_fps": serial_fps,
        "ray_pipeline_avg_stage_total_ms": _avg([float(x["stage_total_ms"]) for x in results]),
        "ray_pipeline_avg_e2e_ms": _avg(e2e_ms),
        "ray_pipeline_fps": pipe_fps,
        "fps_speedup_vs_serial": pipe_fps / max(1e-9, serial_fps),
        "e2e_latency_ratio_vs_serial": _avg(e2e_ms) / max(1e-9, _avg(serial_total_ms)),
        "ray_stage_breakdown_ms": {
            "decode_input_ms": _avg([float(x["decode_input_ms"]) for x in results]),
            "preprocess_cpu_ms": _avg([float(x["preprocess_cpu_ms"]) for x in results]),
            "prepare_gpu_ms": _avg([float(x["prepare_gpu_ms"]) for x in results]),
            "denoise_ms": _avg([float(x["denoise_ms"]) for x in results]),
            "decode_model_ms": _avg([float(x["decode_model_ms"]) for x in results]),
        },
    }

    print("\n[RAY_CPU_GPU_PIPELINE] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert len(serial_total_ms) == max(1, frame_count)
    assert len(results) == max(1, frame_count)
