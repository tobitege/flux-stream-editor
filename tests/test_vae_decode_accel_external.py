from __future__ import annotations

import base64
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.realtime_img2img_server import RealtimeImg2ImgApi, build_default_config


def _apply_env_bool(name: str, current: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return current
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


@pytest.mark.skipif(
    os.getenv("RUN_FLUX_VAE_DECODE_ACCEL_TEST") != "1",
    reason="Set RUN_FLUX_VAE_DECODE_ACCEL_TEST=1 to run heavyweight external benchmark.",
)
def test_vae_decode_accel_external() -> None:
    attention_backend = os.getenv("FLUX_VAE_DECODE_ATTN", "sage")
    warmup_runs = int(os.getenv("FLUX_VAE_DECODE_WARMUP", "3"))
    measure_runs = int(os.getenv("FLUX_VAE_DECODE_RUNS", "5"))

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=2)
    cfg.verbose = False
    cfg.profile_stage_timing = True
    cfg.compile_transformer = _apply_env_bool("FLUX_VAE_DECODE_COMPILE_TRANSFORMER", cfg.compile_transformer)
    cfg.enable_vae_encoder_compile = _apply_env_bool("FLUX_VAE_ENCODE_COMPILE", cfg.enable_vae_encoder_compile)
    cfg.vae_encoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS",
        cfg.vae_encoder_compile_disable_cudagraphs,
    )
    cfg.vae_encoder_compile_mode = os.getenv("FLUX_VAE_ENCODE_COMPILE_MODE", cfg.vae_encoder_compile_mode)
    cfg.enable_vae_decoder_compile = _apply_env_bool("FLUX_VAE_DECODE_ENABLE_COMPILE", cfg.enable_vae_decoder_compile)
    cfg.vae_decoder_channels_last = _apply_env_bool("FLUX_VAE_DECODE_CHANNELS_LAST", cfg.vae_decoder_channels_last)
    cfg.vae_decoder_input_channels_last = _apply_env_bool(
        "FLUX_VAE_DECODE_INPUT_CHANNELS_LAST",
        cfg.vae_decoder_input_channels_last,
    )
    cfg.vae_decoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_DECODE_DISABLE_CUDAGRAPHS",
        cfg.vae_decoder_compile_disable_cudagraphs,
    )
    cfg.vae_decoder_compile_mode = os.getenv("FLUX_VAE_DECODE_COMPILE_MODE", "reduce-overhead")

    api = RealtimeImg2ImgApi(config=cfg)
    payload = {
        "base64_image": _make_data_url_jpeg(),
        "prompt": "Convert this live frame into a cinematic anime illustration with clean lines and rich color.",
        "seed": 0,
    }

    transport_ms: list[float] = []
    total_ms: list[float] = []
    decode_ms: list[float] = []
    prepare_ms: list[float] = []
    denoise_ms: list[float] = []
    output_b64: str | None = None

    with TestClient(api.app) as client:
        load = client.post("/api/load", json={"attention_backend": attention_backend})
        assert load.status_code == 200, load.text

        for _ in range(max(0, warmup_runs)):
            warmup = client.post("/api/predict", json=payload)
            assert warmup.status_code == 200, warmup.text

        for _ in range(max(1, measure_runs)):
            t0 = time.perf_counter()
            resp = client.post("/api/predict", json=payload)
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status_code == 200, resp.text
            data = resp.json()
            output_b64 = data["base64_image"]
            transport_ms.append(rtt_ms)
            total_ms.append(float(data["total_ms"]))
            decode_ms.append(float(data["decode_ms"]))
            prepare_ms.append(float(data["prepare_ms"]))
            denoise_ms.append(float(data["total_ms"]) - float(data["prepare_ms"]) - float(data["decode_ms"]))

    result = {
        "runs": max(1, measure_runs),
        "warmup_runs": max(0, warmup_runs),
        "attention_backend": attention_backend,
        "compile_transformer": cfg.compile_transformer,
        "enable_vae_encoder_compile": cfg.enable_vae_encoder_compile,
        "vae_encoder_compile_mode": cfg.vae_encoder_compile_mode,
        "vae_encoder_compile_disable_cudagraphs": cfg.vae_encoder_compile_disable_cudagraphs,
        "enable_vae_decoder_compile": cfg.enable_vae_decoder_compile,
        "vae_decoder_channels_last": cfg.vae_decoder_channels_last,
        "vae_decoder_input_channels_last": cfg.vae_decoder_input_channels_last,
        "vae_decoder_compile_mode": cfg.vae_decoder_compile_mode,
        "transport_ms_client_rtt": _avg(transport_ms),
        "transport_overhead_ms": _avg(transport_ms) - _avg(total_ms),
        "server_total_ms": _avg(total_ms),
        "prepare_ms": _avg(prepare_ms),
        "denoise_ms": _avg(denoise_ms),
        "decode_ms_model_stage": _avg(decode_ms),
    }

    output_path = os.getenv("FLUX_VAE_DECODE_OUTPUT_PATH", "").strip()
    if output_path:
        assert output_b64 is not None
        image = Image.open(io.BytesIO(base64.b64decode(output_b64))).convert("RGB")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        result["output_path"] = output_path

    print("\n[VAE_DECODE_ACCEL] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert len(total_ms) == max(1, measure_runs)
    assert result["decode_ms_model_stage"] >= 0.0
