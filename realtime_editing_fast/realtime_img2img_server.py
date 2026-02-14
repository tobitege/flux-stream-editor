from __future__ import annotations

import argparse
import asyncio
import base64
import os
import secrets
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .editor import FastFlux2Config, FastFlux2RealtimeEditor, normalize_attention_backend_name


DEFAULT_PROMPT = "Convert this live frame into a cinematic anime illustration with clean lines and rich color."
ATTENTION_BACKEND_CHOICES = ["auto", "sage", "native", "none", "sage_hub", "_flash_3", "fa3"]
STATIC_DIR = Path(__file__).resolve().parent / "static" / "realtime_img2img"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class LoadInputModel(BaseModel):
    attention_backend: str | None = Field(default=None, description="Use server default if not provided")


class LoadResponseModel(BaseModel):
    status: str
    attention_backend: str


class PredictInputModel(BaseModel):
    base64_image: str
    prompt: str = Field(default=DEFAULT_PROMPT)
    seed: int = Field(default=0, description="Use -1 for random seed")


class PredictResponseModel(BaseModel):
    base64_image: str
    seed: int
    request_tag: str
    total_ms: float
    refresh_ms: float
    prepare_ms: float
    decode_ms: float
    source_size: tuple[int, int]
    target_size: tuple[int, int]


class HealthResponseModel(BaseModel):
    status: str
    model_loaded: bool


class SettingsResponseModel(BaseModel):
    default_prompt: str
    width: int
    height: int
    num_inference_steps: int


class GPUInfoResponseModel(BaseModel):
    device_name: str
    device_count: int
    cuda_available: bool
    cuda_version: str | None


class RealtimeImg2ImgApi:
    def __init__(self, config: FastFlux2Config) -> None:
        self.editor = FastFlux2RealtimeEditor(config)
        self.app = FastAPI(title="Flux2 Realtime Img2Img")
        self._model_lock = asyncio.Lock()
        self._setup_routes()

    @staticmethod
    def _normalize_attention_backend(attention_backend: str) -> str:
        return normalize_attention_backend_name(attention_backend)

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        if int(seed) >= 0:
            return int(seed)
        return secrets.randbelow(2**31 - 1)

    @staticmethod
    def _pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
        buffered = BytesIO()
        image.convert("RGB").save(buffered, format=format, quality=92)
        return base64.b64encode(buffered.getvalue()).decode("ascii")

    @staticmethod
    def _base64_to_pil(base64_image: str) -> Image.Image:
        encoded = base64_image
        if "," in encoded and "base64" in encoded[:40].lower():
            encoded = encoded.split(",", 1)[1]
        data = base64.b64decode(encoded)
        return Image.open(BytesIO(data)).convert("RGB")

    def _setup_routes(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/api/health", response_model=HealthResponseModel)
        async def health() -> HealthResponseModel:
            return HealthResponseModel(status="ok", model_loaded=self.editor.is_loaded)

        @self.app.get("/api/settings", response_model=SettingsResponseModel)
        async def settings() -> SettingsResponseModel:
            cfg = self.editor.config
            return SettingsResponseModel(
                default_prompt=DEFAULT_PROMPT,
                width=int(cfg.width),
                height=int(cfg.height),
                num_inference_steps=int(cfg.num_inference_steps),
            )

        @self.app.get("/api/gpu_info", response_model=GPUInfoResponseModel)
        async def gpu_info() -> GPUInfoResponseModel:
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count() if cuda_available else 0
            device_name = "N/A"
            cuda_version = None
            
            if cuda_available and device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
            
            return GPUInfoResponseModel(
                device_name=device_name,
                device_count=device_count,
                cuda_available=cuda_available,
                cuda_version=cuda_version,
            )

        @self.app.post("/api/load", response_model=LoadResponseModel)
        async def load_model(inp: LoadInputModel) -> LoadResponseModel:
            async with self._model_lock:
                backend_from_input = inp.attention_backend or self.editor.config.attention_backend
                selected_backend = self._normalize_attention_backend(backend_from_input)
                if selected_backend not in ATTENTION_BACKEND_CHOICES:
                    selected_backend = self.editor.config.attention_backend

                if selected_backend != self.editor.config.attention_backend:
                    self.editor = FastFlux2RealtimeEditor(
                        replace(
                            self.editor.config,
                            attention_backend=selected_backend,
                        )
                    )

                self.editor.ensure_loaded()
                return LoadResponseModel(status="loaded", attention_backend=self.editor.config.attention_backend)

        @self.app.post("/api/predict", response_model=PredictResponseModel)
        async def predict(inp: PredictInputModel) -> PredictResponseModel:
            prompt = (inp.prompt or "").strip() or DEFAULT_PROMPT
            seed = self._resolve_seed(inp.seed)
            frame = self._base64_to_pil(inp.base64_image)

            async with self._model_lock:
                edited, meta = self.editor.edit_image_with_meta(image=frame, prompt=prompt, seed=seed)

            return PredictResponseModel(
                base64_image=self._pil_to_base64(edited),
                seed=seed,
                request_tag=meta["request_tag"],
                total_ms=float(meta["total_ms"]),
                refresh_ms=float(meta["refresh_ms"]),
                prepare_ms=float(meta["prepare_ms"]),
                decode_ms=float(meta["decode_ms"]),
                source_size=tuple(meta["source_size"]),
                target_size=tuple(meta["target_size"]),
            )

        if not STATIC_DIR.exists():
            raise RuntimeError(f"Static frontend directory not found: {STATIC_DIR}")

        self.app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="frontend")


def build_default_config(attention_backend: str = "auto", num_inference_steps: int = 2) -> FastFlux2Config:
    attention_backend = normalize_attention_backend_name(attention_backend)
    steps_mask = "10" if int(num_inference_steps) == 2 else "1" * int(num_inference_steps)
    profile_stage_timing = os.getenv("FLUX_PROFILE_STAGE", "0") == "1"
    enable_vae_decoder_compile = _env_bool("FLUX_VAE_DECODE_COMPILE", True)
    vae_decoder_compile_disable_cudagraphs = _env_bool("FLUX_VAE_DECODE_DISABLE_CUDAGRAPHS", False)
    vae_decoder_channels_last = _env_bool("FLUX_VAE_DECODE_CHANNELS_LAST", False)
    vae_decoder_input_channels_last = _env_bool("FLUX_VAE_DECODE_INPUT_CHANNELS_LAST", False)
    vae_decoder_compile_mode = os.getenv("FLUX_VAE_DECODE_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    enable_vae_encoder_compile = _env_bool("FLUX_VAE_ENCODE_COMPILE", True)
    vae_encoder_compile_disable_cudagraphs = _env_bool("FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS", True)
    vae_encoder_compile_mode = os.getenv("FLUX_VAE_ENCODE_COMPILE_MODE", "reduce-overhead").strip() or "reduce-overhead"
    cache_timesteps = _env_bool("FLUX_CACHE_TIMESTEPS", True)
    cache_image_latent_ids = _env_bool("FLUX_CACHE_IMAGE_LATENT_IDS", True)
    return FastFlux2Config(
        attention_backend=attention_backend,
        width=512,
        height=512,
        input_resize_mode="equivalent_area",
        num_inference_steps=int(num_inference_steps),
        guidance_scale=1.0,
        seed=0,
        enable_cache_dit=True,
        cache_fn=1,
        cache_bn=0,
        residual_diff_threshold=0.8,
        steps_mask=steps_mask,
        steps_computation_policy="dynamic",
        enable_taylorseer=True,
        taylorseer_order=1,
        compile_transformer=True,
        compile_disable_cudagraphs=True,
        cache_timesteps=cache_timesteps,
        cache_image_latent_ids=cache_image_latent_ids,
        enable_vae_encoder_compile=enable_vae_encoder_compile,
        vae_encoder_compile_mode=vae_encoder_compile_mode,
        vae_encoder_compile_disable_cudagraphs=vae_encoder_compile_disable_cudagraphs,
        enable_vae_decoder_compile=enable_vae_decoder_compile,
        vae_decoder_compile_mode=vae_decoder_compile_mode,
        vae_decoder_compile_disable_cudagraphs=vae_decoder_compile_disable_cudagraphs,
        vae_decoder_channels_last=vae_decoder_channels_last,
        vae_decoder_input_channels_last=vae_decoder_input_channels_last,
        profile_stage_timing=profile_stage_timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime img2img demo for FLUX.2 with fast config.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7870)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="auto",
    )
    parser.add_argument("--num-inference-steps", type=int, default=2)
    args = parser.parse_args()

    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be >= 1")

    api = RealtimeImg2ImgApi(
        build_default_config(
            attention_backend=args.attention_backend,
            num_inference_steps=args.num_inference_steps,
        )
    )

    uvicorn.run(
        api.app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
