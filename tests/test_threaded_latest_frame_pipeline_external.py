from __future__ import annotations

import base64
import io
import json
import os
import queue
import statistics
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
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


def _build_config():
    attention_backend = os.getenv("FLUX_THREAD_ATTENTION_BACKEND", "native")
    num_steps = int(os.getenv("FLUX_THREAD_NUM_STEPS", "2"))
    width = int(os.getenv("FLUX_THREAD_WIDTH", "512"))
    height = int(os.getenv("FLUX_THREAD_HEIGHT", "512"))
    compile_transformer = os.getenv("FLUX_THREAD_COMPILE", "0") == "1"
    enable_cache = os.getenv("FLUX_THREAD_ENABLE_CACHE", "1") == "1"

    cfg = build_default_config(attention_backend=attention_backend, num_inference_steps=num_steps)
    return replace(
        cfg,
        width=width,
        height=height,
        compile_transformer=compile_transformer,
        enable_cache_dit=enable_cache,
        verbose=False,
    )


@pytest.mark.skipif(
    os.getenv("RUN_THREAD_LATEST_FRAME_TEST") != "1",
    reason="Set RUN_THREAD_LATEST_FRAME_TEST=1 to run heavy threaded latest-frame benchmark.",
)
def test_threaded_latest_frame_pipeline_external() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    frame_count = int(os.getenv("FLUX_THREAD_FRAME_COUNT", "20"))
    warmup_count = int(os.getenv("FLUX_THREAD_WARMUP", "1"))
    producer_interval_ms = float(os.getenv("FLUX_THREAD_PRODUCER_INTERVAL_MS", "0"))
    prompt = "Convert this live frame into a cinematic anime illustration with clean lines and rich color."
    frame_b64 = _make_data_url_jpeg()
    cfg = _build_config()

    # Serial baseline.
    serial_editor = FastFlux2RealtimeEditor(cfg)
    serial_editor.ensure_loaded()
    for i in range(max(0, warmup_count)):
        serial_editor.edit_image_with_meta(image=_base64_to_pil(frame_b64), prompt=prompt, seed=i)

    serial_e2e_ms: list[float] = []
    serial_decode_ms: list[float] = []
    serial_model_total_ms: list[float] = []
    serial_t0 = time.perf_counter()
    for i in range(max(1, frame_count)):
        t0 = time.perf_counter()
        d0 = time.perf_counter()
        frame = _base64_to_pil(frame_b64)
        serial_decode_ms.append((time.perf_counter() - d0) * 1000.0)
        _, meta = serial_editor.edit_image_with_meta(image=frame, prompt=prompt, seed=1000 + i)
        serial_model_total_ms.append(float(meta["total_ms"]))
        serial_e2e_ms.append((time.perf_counter() - t0) * 1000.0)
    serial_elapsed = time.perf_counter() - serial_t0
    serial_fps = max(1, frame_count) / max(1e-9, serial_elapsed)

    # Threaded latest-frame pipeline.
    # Reuse the already-loaded editor to avoid cache-dit double-patching in one process.
    threaded_editor = serial_editor

    latest_queue: queue.Queue = queue.Queue(maxsize=1)
    producer_done = threading.Event()

    lock = threading.Lock()
    submitted = 0
    dropped = 0
    processed = 0
    threaded_e2e_ms: list[float] = []
    threaded_decode_ms: list[float] = []
    threaded_model_total_ms: list[float] = []
    threaded_queue_wait_ms: list[float] = []

    def producer() -> None:
        nonlocal submitted, dropped
        for i in range(max(1, frame_count)):
            submit_ts = time.perf_counter()
            d0 = time.perf_counter()
            frame = _base64_to_pil(frame_b64)
            decode_ms = (time.perf_counter() - d0) * 1000.0
            item = {
                "frame_id": i,
                "submit_ts": submit_ts,
                "frame": frame,
                "decode_ms": decode_ms,
            }

            # Latest-frame overwrite policy.
            try:
                latest_queue.put_nowait(item)
            except queue.Full:
                try:
                    latest_queue.get_nowait()
                    with lock:
                        dropped += 1
                except queue.Empty:
                    pass
                latest_queue.put_nowait(item)

            with lock:
                submitted += 1

            if producer_interval_ms > 0:
                time.sleep(producer_interval_ms / 1000.0)
        producer_done.set()

    def consumer() -> None:
        nonlocal processed
        while True:
            if producer_done.is_set() and latest_queue.empty():
                break
            try:
                item = latest_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            start_ts = time.perf_counter()
            _, meta = threaded_editor.edit_image_with_meta(
                image=item["frame"],
                prompt=prompt,
                seed=3000 + int(item["frame_id"]),
            )
            end_ts = time.perf_counter()

            with lock:
                processed += 1
                threaded_decode_ms.append(float(item["decode_ms"]))
                threaded_model_total_ms.append(float(meta["total_ms"]))
                threaded_queue_wait_ms.append((start_ts - float(item["submit_ts"])) * 1000.0)
                threaded_e2e_ms.append((end_ts - float(item["submit_ts"])) * 1000.0)

    t_pipeline0 = time.perf_counter()
    producer_thread = threading.Thread(target=producer, daemon=True)
    consumer_thread = threading.Thread(target=consumer, daemon=True)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    threaded_elapsed = time.perf_counter() - t_pipeline0
    threaded_fps = processed / max(1e-9, threaded_elapsed)

    result = {
        "frame_count": max(1, frame_count),
        "warmup_count": max(0, warmup_count),
        "producer_interval_ms": producer_interval_ms,
        "config": {
            "width": int(cfg.width),
            "height": int(cfg.height),
            "num_inference_steps": int(cfg.num_inference_steps),
            "attention_backend": str(cfg.attention_backend),
            "compile_transformer": bool(cfg.compile_transformer),
            "enable_cache_dit": bool(cfg.enable_cache_dit),
        },
        "serial": {
            "fps": serial_fps,
            "avg_e2e_ms": _avg(serial_e2e_ms),
            "avg_decode_ms": _avg(serial_decode_ms),
            "avg_model_total_ms": _avg(serial_model_total_ms),
        },
        "threaded_latest_frame": {
            "submitted": submitted,
            "processed": processed,
            "dropped": dropped,
            "drop_ratio": (dropped / submitted) if submitted else 0.0,
            "fps_processed": threaded_fps,
            "avg_e2e_ms": _avg(threaded_e2e_ms),
            "avg_decode_ms": _avg(threaded_decode_ms),
            "avg_model_total_ms": _avg(threaded_model_total_ms),
            "avg_queue_wait_ms": _avg(threaded_queue_wait_ms),
        },
        "comparison": {
            "fps_ratio_threaded_vs_serial": threaded_fps / max(1e-9, serial_fps),
            "e2e_ratio_threaded_vs_serial": _avg(threaded_e2e_ms) / max(1e-9, _avg(serial_e2e_ms)),
        },
    }

    print("\n[THREADED_LATEST_FRAME_PIPELINE] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert submitted == max(1, frame_count)
    assert processed > 0
