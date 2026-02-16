from __future__ import annotations

import base64
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageOps
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.realtime_img2img_server import RealtimeImg2ImgApi, build_default_config
from realtime_editing_fast.editor import compute_empirical_mu, retrieve_timesteps


def _make_data_url_jpeg(width: int = 640, height: int = 360) -> str:
    image = Image.new("RGB", (width, height), (118, 136, 158))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


@pytest.mark.skipif(
    os.getenv("RUN_FLUX_TIMING_TEST") != "1",
    reason="Set RUN_FLUX_TIMING_TEST=1 to run heavyweight realtime timing benchmark.",
)
def test_img2img_pipeline_timing_external_hack(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_backend = os.getenv("FLUX_TIMING_ATTENTION_BACKEND", "native")
    num_inference_steps = int(os.getenv("FLUX_TIMING_NUM_STEPS", "2"))
    warmup_runs = int(os.getenv("FLUX_TIMING_WARMUP", "1"))
    measure_runs = int(os.getenv("FLUX_TIMING_RUNS", "3"))
    preprocess_impl = os.getenv("FLUX_PREPROCESS_IMPL", "baseline").strip().lower()
    preprocess_impl_choices = {
        "baseline",
        "baseline_if_needed_rgb",
        "baseline_bilinear",
        "pil_bilinear_tensor",
        "opencv_linear_tensor",
    }
    if preprocess_impl not in preprocess_impl_choices:
        raise ValueError(f"Unsupported FLUX_PREPROCESS_IMPL={preprocess_impl}. choices={sorted(preprocess_impl_choices)}")

    config = build_default_config(
        attention_backend=attention_backend,
        num_inference_steps=num_inference_steps,
    )
    api = RealtimeImg2ImgApi(config=config)

    decode_input_ms_samples: list[float] = []
    server_total_ms_samples: list[float] = []
    server_decode_stage_ms_samples: list[float] = []
    server_denoise_ms_samples: list[float] = []
    client_transport_rtt_ms_samples: list[float] = []
    preprocess_cpu_ms_samples: list[float] = []
    h2d_ms_samples: list[float] = []
    vae_encode_ms_samples: list[float] = []
    pack_cond_ms_samples: list[float] = []

    original_base64_to_pil = api._base64_to_pil

    def timed_base64_to_pil(base64_image: str) -> Image.Image:
        t0 = time.perf_counter()
        image = original_base64_to_pil(base64_image)
        decode_input_ms_samples.append((time.perf_counter() - t0) * 1000.0)
        return image

    monkeypatch.setattr(api, "_base64_to_pil", timed_base64_to_pil)

    original_edit = api.editor.edit_image_with_meta

    def timed_edit_image_with_meta(*args, **kwargs):
        edited, meta = original_edit(*args, **kwargs)
        step_ms = meta.get("step_ms") or []
        server_total_ms_samples.append(float(meta.get("total_ms", 0.0)))
        server_decode_stage_ms_samples.append(float(meta.get("decode_ms", 0.0)))
        server_denoise_ms_samples.append(float(sum(step_ms)))
        return edited, meta

    monkeypatch.setattr(api.editor, "edit_image_with_meta", timed_edit_image_with_meta)

    def _sync_cuda() -> None:
        if api.editor.config.runtime_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def timed_prepare_inputs(image: Image.Image, prompt: str, seed: int) -> dict:
        pipe = api.editor._pipe
        cfg = api.editor.config
        assert pipe is not None

        prompt_embeds, text_ids = api.editor._encode_prompt(prompt)

        t0_pre = time.perf_counter()
        if preprocess_impl == "baseline_if_needed_rgb":
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            image = image.convert("RGB")
        source_size = image.size
        multiple = int(pipe.vae_scale_factor * 2)
        target_area = int(cfg.width * cfg.height)
        if cfg.input_resize_mode == "equivalent_area":
            proc_w, proc_h = api.editor._compute_equivalent_resolution(
                src_w=source_size[0],
                src_h=source_size[1],
                target_area=target_area,
                multiple=multiple,
            )
            if preprocess_impl == "baseline_bilinear":
                resample = Image.Resampling.BILINEAR
            else:
                resample = Image.Resampling.BICUBIC
            image_for_preprocess = image.resize((proc_w, proc_h), resample)
            resize_mode = "crop"
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

        if preprocess_impl in {"baseline", "baseline_if_needed_rgb", "baseline_bilinear"}:
            image_tensor = pipe.image_processor.preprocess(
                image_for_preprocess,
                height=target_h,
                width=target_w,
                resize_mode=resize_mode,
            )
        elif preprocess_impl == "pil_bilinear_tensor":
            resized = image.resize((target_w, target_h), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.uint8, copy=True)
            image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            image_tensor = image_tensor / 255.0
            image_tensor = image_tensor * 2.0 - 1.0
        elif preprocess_impl == "opencv_linear_tensor":
            arr = np.asarray(image, dtype=np.uint8)
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            arr = np.ascontiguousarray(arr)
            image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
            image_tensor = image_tensor / 255.0
            image_tensor = image_tensor * 2.0 - 1.0
        else:
            raise ValueError(f"Unsupported preprocess_impl={preprocess_impl}")
        preprocess_cpu_ms = (time.perf_counter() - t0_pre) * 1000.0

        pack_cond_ms = 0.0
        t0_pack = time.perf_counter()
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
        pack_cond_ms += (time.perf_counter() - t0_pack) * 1000.0

        h2d_ms = 0.0
        vae_encode_ms = 0.0
        image_latents = []
        images = [image_tensor]
        for one_image in images:
            _sync_cuda()
            t0_h2d = time.perf_counter()
            one_image = one_image.to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            _sync_cuda()
            h2d_ms += (time.perf_counter() - t0_h2d) * 1000.0

            _sync_cuda()
            t0_vae = time.perf_counter()
            image_latent = pipe._encode_vae_image(image=one_image, generator=generator)
            _sync_cuda()
            vae_encode_ms += (time.perf_counter() - t0_vae) * 1000.0
            image_latents.append(image_latent)

        t0_pack = time.perf_counter()
        image_latent_ids = pipe._prepare_image_ids(image_latents)
        packed_latents = []
        for latent in image_latents:
            packed = pipe._pack_latents(latent)
            packed = packed.squeeze(0)
            packed_latents.append(packed)

        image_latents = torch.cat(packed_latents, dim=0)
        image_latents = image_latents.unsqueeze(0)
        image_latents = image_latents.repeat(1, 1, 1)
        image_latent_ids = image_latent_ids.repeat(1, 1, 1)
        image_latent_ids = image_latent_ids.to(pipe._execution_device)

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
        pack_cond_ms += (time.perf_counter() - t0_pack) * 1000.0

        preprocess_cpu_ms_samples.append(preprocess_cpu_ms)
        h2d_ms_samples.append(h2d_ms)
        vae_encode_ms_samples.append(vae_encode_ms)
        pack_cond_ms_samples.append(pack_cond_ms)

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

    monkeypatch.setattr(api.editor, "_prepare_inputs", timed_prepare_inputs)

    payload = {
        "base64_image": _make_data_url_jpeg(),
        "prompt": "Convert this live frame into a cinematic anime illustration with clean lines and rich color.",
        "seed": 0,
    }

    with TestClient(api.app) as client:
        load = client.post("/api/load", json={"attention_backend": attention_backend})
        assert load.status_code == 200, load.text

        for _ in range(max(0, warmup_runs)):
            warmup = client.post("/api/predict", json=payload)
            assert warmup.status_code == 200, warmup.text

        for _ in range(max(1, measure_runs)):
            t0 = time.perf_counter()
            response = client.post("/api/predict", json=payload)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            assert response.status_code == 200, response.text
            client_transport_rtt_ms_samples.append(rtt_ms)

    # External-hack metrics:
    # transport_ms_client_rtt: total client RTT for /api/predict
    # transport_overhead_ms: client RTT - server total_ms
    # decode_ms_input_b64: request-side base64->PIL decode in server route
    # decode_ms_model_stage: VAE decode stage reported by editor meta
    # denoise_ms: sum(meta["step_ms"]) reported by editor meta
    measured_decode_input = decode_input_ms_samples[-max(1, measure_runs):]
    measured_server_total = server_total_ms_samples[-max(1, measure_runs):]
    measured_server_decode = server_decode_stage_ms_samples[-max(1, measure_runs):]
    measured_server_denoise = server_denoise_ms_samples[-max(1, measure_runs):]
    measured_preprocess_cpu = preprocess_cpu_ms_samples[-max(1, measure_runs):]
    measured_h2d = h2d_ms_samples[-max(1, measure_runs):]
    measured_vae_encode = vae_encode_ms_samples[-max(1, measure_runs):]
    measured_pack_cond = pack_cond_ms_samples[-max(1, measure_runs):]

    avg_transport_rtt = _avg(client_transport_rtt_ms_samples)
    avg_server_total = _avg(measured_server_total)
    result = {
        "runs": max(1, measure_runs),
        "warmup_runs": max(0, warmup_runs),
        "attention_backend": attention_backend,
        "num_inference_steps": num_inference_steps,
        "preprocess_impl": preprocess_impl,
        "transport_ms_client_rtt": avg_transport_rtt,
        "transport_overhead_ms": avg_transport_rtt - avg_server_total,
        "decode_ms_input_b64": _avg(measured_decode_input),
        "decode_ms_model_stage": _avg(measured_server_decode),
        "denoise_ms": _avg(measured_server_denoise),
        "prepare_preprocess_cpu_ms": _avg(measured_preprocess_cpu),
        "prepare_h2d_ms": _avg(measured_h2d),
        "prepare_vae_encode_ms": _avg(measured_vae_encode),
        "prepare_pack_cond_ms": _avg(measured_pack_cond),
        "server_total_ms": avg_server_total,
    }

    print("\n[PIPELINE_TIMING_HACK] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert len(client_transport_rtt_ms_samples) == max(1, measure_runs)
    assert len(server_total_ms_samples) >= max(1, measure_runs)
    assert result["denoise_ms"] >= 0.0
