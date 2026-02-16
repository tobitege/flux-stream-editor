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

import pytest
import ray
from ray.job_config import JobConfig
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.editor import FastFlux2RealtimeEditor
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


def _build_config() -> object:
    attention_backend = os.getenv("FLUX_RAY_ATTENTION_BACKEND", "native")
    num_steps = int(os.getenv("FLUX_RAY_NUM_STEPS", "2"))
    width = int(os.getenv("FLUX_RAY_WIDTH", "256"))
    height = int(os.getenv("FLUX_RAY_HEIGHT", "256"))
    compile_transformer = os.getenv("FLUX_RAY_COMPILE", "0") == "1"
    enable_cache = os.getenv("FLUX_RAY_ENABLE_CACHE", "0") == "1"
    verbose = os.getenv("FLUX_RAY_VERBOSE", "0") == "1"

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=num_steps)
    cfg = replace(
        cfg,
        width=width,
        height=height,
        compile_transformer=compile_transformer,
        enable_cache_dit=enable_cache,
        verbose=verbose,
    )
    return cfg


@ray.remote(num_gpus=0.33)
class PrepareActor:
    def __init__(self, cfg) -> None:
        self.editor = FastFlux2RealtimeEditor(cfg)
        self.editor.ensure_loaded()

    @torch.no_grad()
    def prepare(self, frame_b64: str, prompt: str, seed: int, frame_id: int) -> dict:
        t0_decode = time.perf_counter()
        image = _base64_to_pil(frame_b64)
        decode_ms = (time.perf_counter() - t0_decode) * 1000.0

        t0_prepare = time.perf_counter()
        inputs = self.editor._prepare_inputs(image=image, prompt=prompt, seed=seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prepare_ms = (time.perf_counter() - t0_prepare) * 1000.0

        return {
            "frame_id": frame_id,
            "decode_input_ms": decode_ms,
            "prepare_ms": prepare_ms,
            "source_size": inputs["source_size"],
            "target_size": inputs["target_size"],
            "latents": inputs["latents"].detach().cpu(),
            "latent_ids": inputs["latent_ids"].detach().cpu(),
            "image_latents": inputs["image_latents"].detach().cpu(),
            "image_latent_ids": inputs["image_latent_ids"].detach().cpu(),
            "timesteps": inputs["timesteps"].detach().cpu(),
            "prompt_embeds": inputs["prompt_embeds"].detach().cpu(),
            "text_ids": inputs["text_ids"].detach().cpu(),
        }


@ray.remote(num_gpus=0.33)
class DenoiseActor:
    def __init__(self, cfg) -> None:
        self.editor = FastFlux2RealtimeEditor(cfg)
        self.editor.ensure_loaded()

    @torch.no_grad()
    def denoise(self, packet: dict) -> dict:
        pipe = self.editor._pipe
        cfg = self.editor.config
        assert pipe is not None

        latents = packet["latents"].to(pipe._execution_device, dtype=pipe.transformer.dtype)
        latent_ids = packet["latent_ids"].to(pipe._execution_device)
        image_latents = packet["image_latents"].to(pipe._execution_device, dtype=pipe.transformer.dtype)
        image_latent_ids = packet["image_latent_ids"].to(pipe._execution_device)
        timesteps = packet["timesteps"].to(pipe._execution_device)
        prompt_embeds = packet["prompt_embeds"].to(pipe._execution_device, dtype=pipe.transformer.dtype)
        text_ids = packet["text_ids"].to(pipe._execution_device)

        if self.editor._cache_dit_mod is not None:
            self.editor._cache_dit_mod.refresh_context(
                pipe.transformer,
                num_inference_steps=cfg.num_inference_steps,
                verbose=False,
            )

        pipe.scheduler.set_begin_index(0)
        pipe.scheduler._step_index = None

        t0 = time.perf_counter()
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

        denoise_ms = (time.perf_counter() - t0) * 1000.0

        return {
            **packet,
            "latents": latents.detach().cpu(),
            "latent_ids": latent_ids.detach().cpu(),
            "denoise_ms": denoise_ms,
            "step_ms": step_ms,
        }


@ray.remote(num_gpus=0.33)
class DecodeActor:
    def __init__(self, cfg) -> None:
        self.editor = FastFlux2RealtimeEditor(cfg)
        self.editor.ensure_loaded()

    @torch.no_grad()
    def decode(self, packet: dict) -> dict:
        pipe = self.editor._pipe
        assert pipe is not None

        latents = packet["latents"].to(pipe._execution_device, dtype=pipe.transformer.dtype)
        latent_ids = packet["latent_ids"].to(pipe._execution_device)

        t0 = time.perf_counter()
        decoded = self.editor._decode_latents_to_pil(latents, latent_ids)
        if decoded.size != packet["source_size"]:
            decoded = decoded.resize(packet["source_size"], Image.Resampling.BICUBIC)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "frame_id": packet["frame_id"],
            "decode_input_ms": float(packet["decode_input_ms"]),
            "prepare_ms": float(packet["prepare_ms"]),
            "denoise_ms": float(packet["denoise_ms"]),
            "decode_model_ms": decode_ms,
            "stage_total_ms": float(packet["decode_input_ms"] + packet["prepare_ms"] + packet["denoise_ms"] + decode_ms),
        }


@pytest.mark.skipif(
    os.getenv("RUN_RAY_SINGLE_GPU_TEST") != "1",
    reason="Set RUN_RAY_SINGLE_GPU_TEST=1 to run heavy single-GPU Ray pipeline benchmark.",
)
def test_single_gpu_ray_three_stage_pipeline() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    frame_count = int(os.getenv("FLUX_RAY_FRAME_COUNT", "8"))
    warmup_count = int(os.getenv("FLUX_RAY_WARMUP", "1"))
    prompt = "Convert this live frame into a cinematic anime illustration with clean lines and rich color."
    frame_b64 = _make_data_url_jpeg()
    cfg = _build_config()

    # Serial baseline (single process, single editor)
    serial_editor = FastFlux2RealtimeEditor(cfg)
    serial_editor.ensure_loaded()

    for i in range(max(0, warmup_count)):
        serial_editor.edit_image_with_meta(image=_base64_to_pil(frame_b64), prompt=prompt, seed=i)

    serial_totals: list[float] = []
    serial_t0 = time.perf_counter()
    for i in range(max(1, frame_count)):
        _, meta = serial_editor.edit_image_with_meta(image=_base64_to_pil(frame_b64), prompt=prompt, seed=1000 + i)
        serial_totals.append(float(meta["total_ms"]))
    serial_elapsed = time.perf_counter() - serial_t0
    serial_fps = max(1, frame_count) / max(1e-9, serial_elapsed)
    del serial_editor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ray 3-stage pipeline on the same single GPU
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
        prep_actor = PrepareActor.remote(cfg)
        denoise_actor = DenoiseActor.remote(cfg)
        decode_actor = DecodeActor.remote(cfg)

        for i in range(max(0, warmup_count)):
            out = decode_actor.decode.remote(
                denoise_actor.denoise.remote(
                    prep_actor.prepare.remote(frame_b64, prompt, i, i),
                )
            )
            ray.get(out)

        submit_ts: dict[int, float] = {}
        futures = []
        pipe_t0 = time.perf_counter()
        for i in range(max(1, frame_count)):
            frame_id = 2000 + i
            submit_ts[frame_id] = time.perf_counter()
            fut = decode_actor.decode.remote(
                denoise_actor.denoise.remote(
                    prep_actor.prepare.remote(frame_b64, prompt, frame_id, frame_id),
                )
            )
            futures.append(fut)

        completed: list[dict] = []
        e2e_ms_samples: list[float] = []
        remaining = list(futures)
        while remaining:
            ready, remaining = ray.wait(remaining, num_returns=1)
            result = ray.get(ready[0])
            completed.append(result)
            e2e_ms_samples.append((time.perf_counter() - submit_ts[result["frame_id"]]) * 1000.0)

        pipe_elapsed = time.perf_counter() - pipe_t0
        pipe_fps = max(1, frame_count) / max(1e-9, pipe_elapsed)
    finally:
        ray.shutdown()

    stage_total_ms = [float(x["stage_total_ms"]) for x in completed]

    result = {
        "frame_count": max(1, frame_count),
        "warmup_count": max(0, warmup_count),
        "serial_avg_total_ms": _avg(serial_totals),
        "serial_fps": serial_fps,
        "ray_pipeline_avg_stage_total_ms": _avg(stage_total_ms),
        "ray_pipeline_avg_e2e_ms": _avg(e2e_ms_samples),
        "ray_pipeline_fps": pipe_fps,
        "fps_speedup_vs_serial": pipe_fps / max(1e-9, serial_fps),
        "e2e_latency_ratio_vs_serial": _avg(e2e_ms_samples) / max(1e-9, _avg(serial_totals)),
    }

    print("\n[RAY_SINGLE_GPU_PIPELINE] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert len(serial_totals) == max(1, frame_count)
    assert len(completed) == max(1, frame_count)
