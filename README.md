# ⚡ FastFlux2 Realtime Editor

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.6.0+cu126-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FLUX.2-klein--4B-8A2BE2" alt="FLUX.2">
  <img src="https://img.shields.io/badge/SageAttention-2.2.0-00CED1" alt="SageAttention">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Turn your webcam into a real-time AI art studio.</b>
</p>

<p align="center">
  <img src="assets/demo.gif" width="900" alt="Demo GIF">
</p>



---

## 🎯 What is this?

**FastFlux2 Realtime Editor** is a blazing-fast, browser-based realtime image-to-image generation tool powered by **FLUX.2-klein-4B** and optimized with **FlashAttention-3 / SageAttention**.

Whether you're livestreaming, creating content, or just having fun with AI art, this tool transforms your webcam or screen capture into stylized artwork **in real-time** with just **2 inference steps**.

### Key Highlights
- ⚡ **Ultra-low latency**: ~66-75ms per frame on single H100 with 2-step inference
- 🚀 **Single-GPU H100 throughput**: **15.6 FPS** with FA3 (latest measured)
- 🎨 **21 built-in presets**: Anime, Pixar, Ghibli, LEGO, Neon, Accessories & more
- 🖥️ **Webcam & Screen support**: Stream your camera or entire desktop
- 🧠 **Smart caching**: Prompt embeddings cached for repeated use
- 🧠 **Auto attention backend selection**: `FA3 > Sage > Native`
- 🚀 **FA3 recommended**: best measured throughput in current setup
- 🌐 **Zero client install**: Runs entirely in your browser

---

## 👤 Author

**Tian Ye**  
PhD Student @ HKUST(Guangzhou)  
🐙 [About ME](https://owen718.github.io/)

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12.6+ (RTX 4090/H100 recommended)
- Python 3.10+
- 24GB VRAM (for FP16 inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/Owen718/flux-stream-editor.git
cd flux-stream-editor

# Install dependencies
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install diffusers transformers accelerate pillow fastapi uvicorn
pip install flash_attn_3==3.0.0b1  # Recommended (FA3)
pip install sageattention==2.2.0 --no-build-isolation  # Fallback backend
pip install cache-dit  # For transformer caching optimization
```

### Start the Server

```bash
# Recommended: auto backend selection (FA3 > Sage > Native), single GPU #1
CUDA_VISIBLE_DEVICES=1 python -m realtime_editing_fast.realtime_img2img_server \
  --host 127.0.0.1 \
  --port 6006 \
  --num-inference-steps 2

# Optional: force FA3
CUDA_VISIBLE_DEVICES=1 python -m realtime_editing_fast.realtime_img2img_server \
  --host 127.0.0.1 \
  --port 6006 \
  --num-inference-steps 2 \
  --attention-backend fa3
```

### Open in Browser

Navigate to `http://localhost:6006` and click **"Load Model"** → **"Start"**.

---

## ⚙️ Configuration Options

### Server Arguments

```bash
python -m realtime_editing_fast.realtime_img2img_server \
  --host 0.0.0.0 \              # Server host
  --port 6006 \                 # Server port
  --num-inference-steps 2 \     # Number of denoising steps (1-4 recommended)
  --attention-backend auto \    # Attention backend: auto, fa3, sage, native, none
  --compile-transformer \       # Enable torch.compile (faster but slower startup)
  --width 512 \                 # Output width
  --height 512                  # Output height
```

### Attention Backends

| Backend | Speed | Quality | Notes |
|---------|-------|---------|-------|
| `auto` | ⭐⭐⭐⭐ Recommended | Excellent | Auto-selects `FA3 > Sage > Native` |
| `fa3` | ⭐⭐⭐⭐ Fastest (current) | Excellent | Requires `flash_attn_3` |
| `sage` | ⭐⭐⭐ Fast | Excellent | Requires SageAttention 2.2.0+ |
| `native` | ⭐⭐ Compatible | Excellent | PyTorch native SDPA |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Browser UI    │────▶│  FastAPI Server  │────▶│  FLUX.2 Model   │
│  (Webcam/Screen)│     │  (GPU Optimized) │     │  (2-Step Infer) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │ SageAttention│  ◀── 30% Speedup
                        │ Cache-DiT    │  ◀── Skip redundant blocks
                        │ Torch.Compile│  ◀── Graph optimization
                        └──────────────┘
```

### Performance Tips

1. **Use `auto` backend (default)**: it tries `FA3 > Sage > Native`.
2. **Install FA3**: `flash_attn_3==3.0.0b1` currently gives the best throughput.
2. **Enable torch.compile**: Essential for reaching latest H100 throughput targets (RTX 4090 figures are old reference values)
3. **Prompt Caching**: Same prompts reuse cached embeddings (0ms overhead)
4. **2-Step Inference**: Perfect balance of speed & quality for real-time stylization

> Tip: In practice, **3-step inference** is much better than **2-step** in both visual quality and instruction following, but the FPS drops noticeably.

---

## 📊 Benchmarks

### 🎯 Measured Performance

| GPU | Configuration | Infer Latency | **Infer FPS** | Status |
|-----|--------------|---------------|---------------|--------|
| **H100 (single GPU)** | **FA3 + Compile (Transformer + VAE Encode + VAE Decode)** | ~64ms class | **15.6 FPS** 🚀 | **Latest** |
| **H100 (single GPU)** | SageAttention + Compile (Transformer + VAE Encode + VAE Decode) | ~79ms class | **~12.6 FPS** | Reference |
| **H100** | Native + Compile | ~100-120ms | **8-10 FPS** | Old |
| **RTX 4090** | SageAttention + Compile | ~150-200ms | **5+ FPS** | **Old** |
| **RTX 4090** | Native + Compile | ~180-220ms | **4-5 FPS** | **Old** |

> 🚀 **Latest**: Single H100 measured **15.6 FPS** with FA3 at 2-step inference.  
> 🕘 **Old reference**: RTX 4090 numbers are kept for historical comparison.


*Note: First inference includes model loading (~10s) and torch.compile warmup (~5-10s). Subsequent requests achieve full speed.*




## 🤝 Contributing

Contributions are welcome! Areas we'd love help with:

- [ ] Mobile UI optimization
- [ ] Better processing mode
- [ ] Quant-based acceleration
- [ ] WebRTC streaming support



---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Black Forest Labs](https://blackforestlabs.ai/) for FLUX.2 models
- [SageAttention](https://github.com/thu-ml/SageAttention) team for the optimized attention kernel
- [Diffusers](https://github.com/huggingface/diffusers) team for the inference pipeline
- [Cache-DiT](https://github.com/your-repo/cache-dit) for transformer block caching

---



---

<p align="center">
  Made with ❤️ for the AI art community
</p>
