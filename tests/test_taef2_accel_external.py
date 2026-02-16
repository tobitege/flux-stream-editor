from __future__ import annotations

import gc
import importlib.util
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import safetensors.torch as stt
import torch
from huggingface_hub import hf_hub_download
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


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def _make_test_frame(width: int = 640, height: int = 360) -> Image.Image:
    return Image.new("RGB", (width, height), (118, 136, 158))


def _download_if_missing(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    req = urllib.request.Request(url, headers={"User-Agent": "flux-stream-editor/taef2-benchmark"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    output_path.write_bytes(data)
    return output_path


def _ensure_taef2_artifacts(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    taesd_py_path = cache_dir / "taesd.py"
    _download_if_missing(
        "https://raw.githubusercontent.com/madebyollin/taesd/refs/heads/main/taesd.py",
        taesd_py_path,
    )

    # Use HF helper to maximize cache hit rate across runs.
    hf_hub_download(
        repo_id="madebyollin/taef2",
        filename="taef2.safetensors",
        local_dir=str(cache_dir),
    )
    taef2_weight_path = cache_dir / "taef2.safetensors"
    if not taef2_weight_path.exists():
        raise FileNotFoundError(f"TAEF2 weight not found at: {taef2_weight_path}")

    return taesd_py_path, taef2_weight_path


def _load_taesd_class(taesd_py_path: Path):
    spec = importlib.util.spec_from_file_location("taesd_dynamic", str(taesd_py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {taesd_py_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "TAESD"):
        raise AttributeError(f"Module {taesd_py_path} does not export TAESD")
    return module.TAESD


def _convert_diffusers_sd_to_taesd(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        parts = key.split(".")
        if len(parts) < 3:
            out[key] = value
            continue

        encdec, _layers, index, *suffix = parts
        if not index.isdigit():
            out[key] = value
            continue

        offset = 1 if encdec == "decoder" else 0
        mapped_key = ".".join([encdec, str(int(index) + offset), *suffix])
        out[mapped_key] = value
    return out


class _DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _build_taef2_vae_wrapper(
    taesd_py_path: Path,
    taef2_weight_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from diffusers.utils.accelerate_utils import apply_forward_hook

    taesd_cls = _load_taesd_class(taesd_py_path)

    class DiffusersTAEF2Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dtype = dtype
            self.taesd = taesd_cls(
                encoder_path=None,
                decoder_path=None,
                latent_channels=32,
                arch_variant="flux_2",
            ).to(dtype=self.dtype)
            self.taesd.load_state_dict(
                _convert_diffusers_sd_to_taesd(stt.load_file(str(taef2_weight_path), device="cpu"))
            )
            # FLUX.2 VAE API expects these BN stats and config key to exist.
            self.bn = torch.nn.BatchNorm2d(128, affine=False, eps=0.0)
            self.config = _DotDict(batch_norm_eps=float(self.bn.eps))

        @apply_forward_hook
        def encode(self, x: torch.Tensor):
            encoded = self.taesd.encoder(x.to(dtype=self.dtype).mul(0.5).add_(0.5)).to(dtype=x.dtype)
            latent_dist = _DotDict(
                sample=lambda generator=None: encoded,
                mode=lambda: encoded,
            )
            return _DotDict(latent_dist=latent_dist)

        @apply_forward_hook
        def decode(self, x: torch.Tensor, return_dict: bool = True):
            decoded = self.taesd.decoder(x.to(dtype=self.dtype)).mul(2).sub_(1).clamp_(-1, 1).to(dtype=x.dtype)
            if return_dict:
                return {"sample": decoded}
            return (decoded,)

    return DiffusersTAEF2Wrapper().to(device=device).eval().requires_grad_(False)


def _rebuild_editor_vae_paths(editor: FastFlux2RealtimeEditor) -> None:
    pipe = editor._pipe
    cfg = editor.config
    if pipe is None:
        raise RuntimeError("Editor pipeline is not loaded.")

    def _encode_fn(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return pipe._encode_vae_image(image=image, generator=generator)

    if cfg.enable_vae_encoder_compile:
        if cfg.vae_encoder_compile_disable_cudagraphs:
            encode_compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
        else:
            encode_compile_kwargs = {"mode": cfg.vae_encoder_compile_mode, "fullgraph": False}
        try:
            editor._vae_encode_fn = torch.compile(_encode_fn, **encode_compile_kwargs)
        except Exception:
            editor._vae_encode_fn = _encode_fn
    else:
        editor._vae_encode_fn = _encode_fn

    def _decode_fn(latents: torch.Tensor) -> torch.Tensor:
        return pipe.vae.decode(latents, return_dict=False)[0]

    if cfg.enable_vae_decoder_compile:
        if cfg.vae_decoder_compile_disable_cudagraphs:
            decode_compile_kwargs = {"fullgraph": False, "options": {"triton.cudagraphs": False}}
        else:
            decode_compile_kwargs = {"mode": cfg.vae_decoder_compile_mode, "fullgraph": False}
        try:
            editor._vae_decode_fn = torch.compile(_decode_fn, **decode_compile_kwargs)
        except Exception:
            editor._vae_decode_fn = _decode_fn
    else:
        editor._vae_decode_fn = _decode_fn


def _benchmark_editor(
    editor: FastFlux2RealtimeEditor,
    image: Image.Image,
    prompt: str,
    warmup_runs: int,
    measure_runs: int,
    seed_base: int,
) -> tuple[dict[str, float], Image.Image]:
    total_ms: list[float] = []
    prepare_ms: list[float] = []
    denoise_ms: list[float] = []
    decode_ms: list[float] = []

    output_image: Image.Image | None = None

    for run_idx in range(max(0, warmup_runs) + max(1, measure_runs)):
        out_image, meta = editor.edit_image_with_meta(
            image=image,
            prompt=prompt,
            seed=seed_base + run_idx,
        )
        if run_idx < max(0, warmup_runs):
            continue

        output_image = out_image

        total = float(meta["total_ms"])
        prepare = float(meta["prepare_ms"])
        decode = float(meta["decode_ms"])
        denoise = float(sum(meta.get("step_ms") or []))

        total_ms.append(total)
        prepare_ms.append(prepare)
        denoise_ms.append(denoise)
        decode_ms.append(decode)

    if output_image is None:
        raise RuntimeError("Failed to generate output image during benchmark.")

    metrics = {
        "server_total_ms": _avg(total_ms),
        "prepare_ms": _avg(prepare_ms),
        "denoise_ms": _avg(denoise_ms),
        "decode_ms": _avg(decode_ms),
    }
    return metrics, output_image


def _save_image_if_needed(image: Image.Image, path_text: str | None) -> str | None:
    if not path_text:
        return None

    output_path = Path(path_text).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


@pytest.mark.skipif(
    os.getenv("RUN_FLUX_TAEF2_ACCEL_TEST") != "1",
    reason="Set RUN_FLUX_TAEF2_ACCEL_TEST=1 to run heavyweight external TAEF2 benchmark.",
)
def test_taef2_accel_external() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    attention_backend = os.getenv("FLUX_TAEF2_ATTN", "sage")
    warmup_runs = max(0, int(os.getenv("FLUX_TAEF2_WARMUP", "2")))
    measure_runs = max(1, int(os.getenv("FLUX_TAEF2_RUNS", "5")))
    baseline_force_eager_vae = _apply_env_bool("FLUX_TAEF2_BASELINE_FORCE_EAGER_VAE", False)
    taef2_force_eager_vae = _apply_env_bool("FLUX_TAEF2_FORCE_EAGER_VAE", True)
    prompt = os.getenv(
        "FLUX_TAEF2_PROMPT",
        "Convert this live frame into a cinematic anime illustration with clean lines and rich color.",
    ).strip()
    cache_dir = Path(os.getenv("FLUX_TAEF2_CACHE_DIR", str(REPO_ROOT / ".cache" / "taef2"))).resolve()

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=2)
    cfg.verbose = False
    cfg.profile_stage_timing = True
    cfg.compile_transformer = _apply_env_bool("FLUX_TAEF2_COMPILE_TRANSFORMER", cfg.compile_transformer)
    cfg.enable_vae_encoder_compile = _apply_env_bool("FLUX_VAE_ENCODE_COMPILE", cfg.enable_vae_encoder_compile)
    cfg.enable_vae_decoder_compile = _apply_env_bool("FLUX_VAE_DECODE_COMPILE", cfg.enable_vae_decoder_compile)
    if baseline_force_eager_vae:
        cfg.enable_vae_encoder_compile = False
        cfg.enable_vae_decoder_compile = False
    cfg.vae_encoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS",
        cfg.vae_encoder_compile_disable_cudagraphs,
    )
    cfg.vae_decoder_compile_disable_cudagraphs = _apply_env_bool(
        "FLUX_VAE_DECODE_DISABLE_CUDAGRAPHS",
        cfg.vae_decoder_compile_disable_cudagraphs,
    )
    cfg.vae_encoder_compile_mode = os.getenv("FLUX_VAE_ENCODE_COMPILE_MODE", cfg.vae_encoder_compile_mode)
    cfg.vae_decoder_compile_mode = os.getenv("FLUX_VAE_DECODE_COMPILE_MODE", cfg.vae_decoder_compile_mode)

    editor = FastFlux2RealtimeEditor(config=cfg)
    editor.ensure_loaded()

    image = _make_test_frame()

    baseline_metrics, baseline_image = _benchmark_editor(
        editor=editor,
        image=image,
        prompt=prompt,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        seed_base=1000,
    )

    taesd_py_path, taef2_weight_path = _ensure_taef2_artifacts(cache_dir)

    pipe = editor._pipe
    if pipe is None:
        raise RuntimeError("Editor pipeline is not loaded after ensure_loaded().")

    taef2_vae = _build_taef2_vae_wrapper(
        taesd_py_path=taesd_py_path,
        taef2_weight_path=taef2_weight_path,
        device=torch.device(pipe._execution_device),
        dtype=pipe.vae.dtype,
    )

    pipe.vae = taef2_vae
    editor._image_latent_ids_cache.clear()
    if taef2_force_eager_vae:
        cfg.enable_vae_encoder_compile = False
        cfg.enable_vae_decoder_compile = False
    _rebuild_editor_vae_paths(editor)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    taef2_metrics, taef2_image = _benchmark_editor(
        editor=editor,
        image=image,
        prompt=prompt,
        warmup_runs=warmup_runs,
        measure_runs=measure_runs,
        seed_base=2000,
    )

    baseline_fps = _safe_ratio(1000.0, baseline_metrics["server_total_ms"])
    taef2_fps = _safe_ratio(1000.0, taef2_metrics["server_total_ms"])

    result: dict[str, Any] = {
        "runs": measure_runs,
        "warmup_runs": warmup_runs,
        "attention_backend_requested": attention_backend,
        "attention_backend_loaded": str(editor.config.attention_backend),
        "compile_transformer": bool(cfg.compile_transformer),
        "enable_vae_encoder_compile": bool(cfg.enable_vae_encoder_compile),
        "enable_vae_decoder_compile": bool(cfg.enable_vae_decoder_compile),
        "baseline_force_eager_vae": bool(baseline_force_eager_vae),
        "taef2_force_eager_vae": bool(taef2_force_eager_vae),
        "taef2_cache_dir": str(cache_dir),
        "baseline": baseline_metrics,
        "taef2": taef2_metrics,
        "speedup": {
            "server_total_x": _safe_ratio(baseline_metrics["server_total_ms"], taef2_metrics["server_total_ms"]),
            "prepare_x": _safe_ratio(baseline_metrics["prepare_ms"], taef2_metrics["prepare_ms"]),
            "denoise_x": _safe_ratio(baseline_metrics["denoise_ms"], taef2_metrics["denoise_ms"]),
            "decode_x": _safe_ratio(baseline_metrics["decode_ms"], taef2_metrics["decode_ms"]),
            "fps_baseline": baseline_fps,
            "fps_taef2": taef2_fps,
            "fps_gain_pct": _safe_ratio((taef2_fps - baseline_fps) * 100.0, baseline_fps),
        },
    }

    output_baseline = _save_image_if_needed(baseline_image, os.getenv("FLUX_TAEF2_BASELINE_OUTPUT_PATH", "").strip())
    output_taef2 = _save_image_if_needed(taef2_image, os.getenv("FLUX_TAEF2_OUTPUT_PATH", "").strip())
    if output_baseline:
        result["baseline_output_path"] = output_baseline
    if output_taef2:
        result["taef2_output_path"] = output_taef2

    print("\n[TAEF2_ACCEL] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert result["baseline"]["server_total_ms"] > 0.0
    assert result["taef2"]["server_total_ms"] > 0.0
    assert result["taef2"]["decode_ms"] >= 0.0
