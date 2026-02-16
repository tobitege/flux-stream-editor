from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.editor import FastFlux2RealtimeEditor
from realtime_editing_fast.realtime_img2img_server import build_default_config


def _apply_env_bool(name: str, current: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return current
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _avg_step_vectors(step_vectors: list[list[float]]) -> list[float]:
    if not step_vectors:
        return []
    num_steps = len(step_vectors[0])
    sums = [0.0] * num_steps
    for vec in step_vectors:
        for i, value in enumerate(vec):
            sums[i] += float(value)
    return [value / len(step_vectors) for value in sums]


def _sync_if_cuda(editor: FastFlux2RealtimeEditor) -> None:
    if editor.config.runtime_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_test_frame(index: int, width: int = 640, height: int = 360) -> Image.Image:
    base = 110 + (index * 17) % 70
    return Image.new("RGB", (width, height), (base, base + 12, base + 28))


def _merge_batch_tensor(items: list[torch.Tensor], expected_batch: int) -> torch.Tensor:
    first = items[0]
    if first.dim() == 0:
        return first

    if first.shape[0] == 1:
        try:
            merged = torch.cat(items, dim=0)
            if merged.shape[0] == expected_batch:
                return merged
        except Exception:
            pass
        return first.repeat(expected_batch, *([1] * (first.dim() - 1)))

    return first


def _clone_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def _build_batched_inputs(single_inputs: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(single_inputs)
    if batch_size < 2:
        raise ValueError("batch_size must be >= 2 for batched denoise benchmark.")

    first = single_inputs[0]
    return {
        "latents": torch.cat([one["latents"] for one in single_inputs], dim=0),
        "latent_ids": _merge_batch_tensor([one["latent_ids"] for one in single_inputs], expected_batch=batch_size),
        "image_latents": torch.cat([one["image_latents"] for one in single_inputs], dim=0),
        "image_latent_ids": _merge_batch_tensor(
            [one["image_latent_ids"] for one in single_inputs],
            expected_batch=batch_size,
        ),
        "timesteps": first["timesteps"],
        "prompt_embeds": _merge_batch_tensor([one["prompt_embeds"] for one in single_inputs], expected_batch=batch_size),
        "text_ids": _merge_batch_tensor([one["text_ids"] for one in single_inputs], expected_batch=batch_size),
        "source_size": [one["source_size"] for one in single_inputs],
        "target_size": first["target_size"],
    }


@torch.no_grad()
def _run_denoise_only(editor: FastFlux2RealtimeEditor, inputs: dict[str, Any]) -> dict[str, Any]:
    pipe = editor._pipe
    cfg = editor.config
    assert pipe is not None

    if editor._cache_dit_mod is not None:
        editor._cache_dit_mod.refresh_context(
            pipe.transformer,
            num_inference_steps=cfg.num_inference_steps,
            verbose=False,
        )

    pipe.scheduler.set_begin_index(0)
    pipe.scheduler._step_index = None

    latents = inputs["latents"]
    latent_ids = inputs["latent_ids"]
    image_latents = inputs["image_latents"]
    image_latent_ids = inputs["image_latent_ids"]
    prompt_embeds = inputs["prompt_embeds"]
    text_ids = inputs["text_ids"]
    timesteps = inputs["timesteps"]

    step_ms: list[float] = []
    _sync_if_cuda(editor)
    t0 = time.perf_counter()

    for timestep_value in timesteps:
        _sync_if_cuda(editor)
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
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

        noise_pred = noise_pred[:, : latents.size(1), :]
        latents = pipe.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]

        _sync_if_cuda(editor)
        step_ms.append((time.perf_counter() - step_t0) * 1000.0)

    _sync_if_cuda(editor)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "batch_size": int(latents.shape[0]),
        "total_ms": float(total_ms),
        "step_ms": [float(x) for x in step_ms],
    }


@pytest.mark.skipif(
    os.getenv("RUN_FLUX_BATCH_DENOISE_TEST") != "1",
    reason="Set RUN_FLUX_BATCH_DENOISE_TEST=1 to run heavyweight batch denoise benchmark.",
)
def test_batch_denoise_external() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    attention_backend = os.getenv("FLUX_BATCH_DENOISE_ATTN", "sage")
    warmup_runs = max(0, int(os.getenv("FLUX_BATCH_DENOISE_WARMUP", "3")))
    measure_runs = max(1, int(os.getenv("FLUX_BATCH_DENOISE_RUNS", "10")))
    batch_size = max(2, int(os.getenv("FLUX_BATCH_DENOISE_SIZE", "2")))
    num_inference_steps = max(1, int(os.getenv("FLUX_BATCH_DENOISE_NUM_STEPS", "2")))
    prompt = os.getenv(
        "FLUX_BATCH_DENOISE_PROMPT",
        "Convert this live frame into a cinematic anime illustration with clean lines and rich color.",
    ).strip()

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=num_inference_steps)
    cfg.verbose = False
    cfg.compile_transformer = _apply_env_bool("FLUX_BATCH_DENOISE_COMPILE_TRANSFORMER", cfg.compile_transformer)
    cfg.enable_vae_encoder_compile = _apply_env_bool("FLUX_VAE_ENCODE_COMPILE", cfg.enable_vae_encoder_compile)
    cfg.vae_encoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS",
        cfg.vae_encoder_compile_disable_cudagraphs,
    )
    cfg.vae_encoder_compile_mode = os.getenv("FLUX_VAE_ENCODE_COMPILE_MODE", cfg.vae_encoder_compile_mode)
    cfg.enable_vae_decoder_compile = _apply_env_bool("FLUX_VAE_DECODE_COMPILE", cfg.enable_vae_decoder_compile)
    cfg.vae_decoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_DECODE_DISABLE_CUDAGRAPHS",
        cfg.vae_decoder_compile_disable_cudagraphs,
    )
    cfg.vae_decoder_compile_mode = os.getenv("FLUX_VAE_DECODE_COMPILE_MODE", cfg.vae_decoder_compile_mode)

    editor = FastFlux2RealtimeEditor(config=cfg)
    editor.ensure_loaded()

    frames = [_make_test_frame(i) for i in range(batch_size)]
    single_total_ms: list[float] = []
    batch_total_ms: list[float] = []
    single_step_ms_per_frame: list[list[float]] = []
    batch_step_ms_total: list[list[float]] = []

    def _prepare_inputs_for_run(run_idx: int) -> list[dict[str, Any]]:
        seeds = [run_idx * 10_000 + i for i in range(batch_size)]
        return [
            editor._prepare_inputs(image=frames[i], prompt=prompt, seed=seeds[i])
            for i in range(batch_size)
        ]

    for run_idx in range(warmup_runs + measure_runs):
        single_inputs = _prepare_inputs_for_run(run_idx=run_idx)
        batch_inputs = _build_batched_inputs(single_inputs)

        run_single_total = 0.0
        run_single_step_totals: list[float] = []
        for one in single_inputs:
            one_metrics = _run_denoise_only(editor, _clone_inputs(one))
            run_single_total += float(one_metrics["total_ms"])
            step_values = list(one_metrics["step_ms"])
            if not run_single_step_totals:
                run_single_step_totals = [0.0] * len(step_values)
            for i, value in enumerate(step_values):
                run_single_step_totals[i] += float(value)

        run_single_step_per_frame = [value / batch_size for value in run_single_step_totals]
        run_batch_metrics = _run_denoise_only(editor, _clone_inputs(batch_inputs))

        if run_idx < warmup_runs:
            continue

        single_total_ms.append(run_single_total)
        batch_total_ms.append(float(run_batch_metrics["total_ms"]))
        single_step_ms_per_frame.append(run_single_step_per_frame)
        batch_step_ms_total.append(list(run_batch_metrics["step_ms"]))

    avg_single_total = _avg(single_total_ms)
    avg_batch_total = _avg(batch_total_ms)
    single_per_frame = avg_single_total / batch_size
    batch_per_frame = avg_batch_total / batch_size

    result = {
        "runs": measure_runs,
        "warmup_runs": warmup_runs,
        "batch_size": batch_size,
        "num_inference_steps": num_inference_steps,
        "attention_backend_requested": attention_backend,
        "attention_backend_loaded": str(editor.config.attention_backend),
        "single_mode": {
            "denoise_total_ms_for_batch_frames": avg_single_total,
            "denoise_ms_per_frame": single_per_frame,
            "step_ms_per_frame": _avg_step_vectors(single_step_ms_per_frame),
        },
        "batch_mode": {
            "denoise_total_ms_for_batch_frames": avg_batch_total,
            "denoise_ms_per_frame": batch_per_frame,
            "step_ms_total_batch": _avg_step_vectors(batch_step_ms_total),
            "step_ms_per_frame": [x / batch_size for x in _avg_step_vectors(batch_step_ms_total)],
        },
        "comparison": {
            "total_speedup_single_vs_batch": (avg_single_total / avg_batch_total) if avg_batch_total > 0 else 0.0,
            "per_frame_speedup_single_vs_batch": (single_per_frame / batch_per_frame) if batch_per_frame > 0 else 0.0,
            "total_ms_saved_per_batch": avg_single_total - avg_batch_total,
            "ms_saved_per_frame": single_per_frame - batch_per_frame,
        },
    }

    print("\n[BATCH_DENOISE_EXTERNAL] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert len(single_total_ms) == measure_runs
    assert len(batch_total_ms) == measure_runs
    assert avg_single_total > 0.0
    assert avg_batch_total > 0.0
