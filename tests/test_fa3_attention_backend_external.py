from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from realtime_editing_fast.realtime_img2img_server import RealtimeImg2ImgApi, build_default_config


def _make_data_url_jpeg(width: int = 640, height: int = 360) -> str:
    image = Image.new("RGB", (width, height), (118, 136, 158))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    payload = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@pytest.mark.skipif(
    os.getenv("RUN_FLUX_FA3_BACKEND_TEST") != "1",
    reason="Set RUN_FLUX_FA3_BACKEND_TEST=1 to run FA3 compatibility external test.",
)
def test_fa3_attention_backend_external() -> None:
    requested_backend = os.getenv("FLUX_FA3_BACKEND", "fa3")
    cfg = build_default_config(attention_backend=requested_backend, num_inference_steps=2)
    cfg.verbose = False

    api = RealtimeImg2ImgApi(config=cfg)
    payload = {
        "base64_image": _make_data_url_jpeg(),
        "prompt": "Convert this live frame into a cinematic anime illustration with clean lines and rich color.",
        "seed": 0,
    }

    with TestClient(api.app) as client:
        load = client.post("/api/load", json={"attention_backend": requested_backend})
        assert load.status_code == 200, load.text
        load_data = load.json()
        loaded_backend = str(load_data["attention_backend"])

        infer = client.post("/api/predict", json=payload)
        assert infer.status_code == 200, infer.text
        infer_data = infer.json()

    result = {
        "requested_backend": requested_backend,
        "loaded_backend": loaded_backend,
        "total_ms": float(infer_data["total_ms"]),
        "prepare_ms": float(infer_data["prepare_ms"]),
        "decode_ms": float(infer_data["decode_ms"]),
    }
    print("\n[FA3_BACKEND_EXTERNAL] " + json.dumps(result, ensure_ascii=False, indent=2))

    assert loaded_backend in {"_flash_3", "sage", "native"}
