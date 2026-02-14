# Realtime Editing Fast Module

This folder keeps a reusable, cleaner FLUX.2 realtime editing wrapper without changing `flux2.py`.

## Files

- `editor.py`: `FastFlux2RealtimeEditor` core wrapper.
- `realtime_txt2img_server.py`: FastAPI realtime txt2img demo server.
- `realtime_img2img_server.py`: FastAPI realtime img2img demo server.
- `static/realtime_txt2img/index.html`: 4x4 realtime txt2img frontend.
- `static/realtime_img2img/index.html`: webcam/screen realtime img2img frontend.
- `__init__.py`: package export.

## Config

Default config inside `FastFlux2Config` is aligned to the fastest benchmarked setup in this workspace:

- `512x512`
- `2-step`
- `DBCache + steps_mask=10 + TaylorSeer`
- `torch.compile` with `triton.cudagraphs=False`

## Gradio Entry

Use root script:

```bash
python gradio_realtime_editing.py --host 0.0.0.0 --port 7860
```

## FastAPI Realtime Txt2Img Entry

Run a StreamDiffusion-style realtime txt2img demo (without reusing StreamDiffusion code):

```bash
uv pip install fastapi uvicorn
python -m realtime_editing_fast.realtime_txt2img_server --host 0.0.0.0 --port 7860
```

Then open `http://localhost:7860`.

## FastAPI Realtime Img2Img Entry

Run a StreamDiffusion-style realtime img2img demo:

```bash
uv pip install fastapi uvicorn
python -m realtime_editing_fast.realtime_img2img_server --host 0.0.0.0 --port 7870 --num-inference-steps 2
```

Then open `http://localhost:7870`.

If `gradio` is missing:

```bash
uv pip install gradio
```
