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


<!-- <p align="center">
  <img src="https://via.placeholder.com/800x400/0f1115/7ec4ff?text=Demo+GIF+Placeholder" width="800" alt="Demo">
</p> -->

---

## 🎯 What is this?

**FastFlux2 Realtime Editor** is a blazing-fast, browser-based realtime image-to-image generation tool powered by **FLUX.2-klein-4B** and optimized with **SageAttention 2.2.0**.

Whether you're livestreaming, creating content, or just having fun with AI art, this tool transforms your webcam or screen capture into stylized artwork **in real-time** with just **2 inference steps**.

### Key Highlights
- ⚡ **Ultra-low latency**: ~50-100ms per frame with 2-step inference
- 🎨 **21 built-in presets**: Anime, Pixar, Ghibli, LEGO, Neon, Accessories & more
- 🖥️ **Webcam & Screen support**: Stream your camera or entire desktop
- 🧠 **Smart caching**: Prompt embeddings cached for repeated use
- 🚀 **SageAttention optimized**: 30-40% speedup over standard attention
- 🌐 **Zero client install**: Runs entirely in your browser

---

## 👤 Author

**Tian Ye**  
PhD Student @ HKUST(Guangzhou)  
📧 [tye610@connect.hkust-gz.edu.cn] | 🐙 [About ME](https://owen718.github.io/)

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12.6+ (RTX 4090/H100 recommended)
- Python 3.10+
- 24GB VRAM (for FP16 inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fastflux2-realtime-editor.git
cd fastflux2-realtime-editor

# Install dependencies
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install diffusers transformers accelerate pillow fastapi uvicorn
pip install sageattention==2.2.0 --no-build-isolation  # Optional but recommended for speed
pip install cache-dit  # For transformer caching optimization
```

### Start the Server

```bash
# Start with SageAttention (fastest)
python -m realtime_editing_fast.realtime_img2img_server \
  --host 0.0.0.0 \
  --port 6006 \
  --num-inference-steps 2 \
  --attention-backend sage

# Or start with native attention (no SageAttention required)
python -m realtime_editing_fast.realtime_img2img_server \
  --host 0.0.0.0 \
  --port 6006 \
  --num-inference-steps 2 \
  --attention-backend native
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
  --attention-backend sage \    # Attention backend: sage, native, none
  --compile-transformer \       # Enable torch.compile (faster but slower startup)
  --width 512 \                 # Output width
  --height 512                  # Output height
```

### Attention Backends

| Backend | Speed | Quality | Notes |
|---------|-------|---------|-------|
| `sage` | ⭐⭐⭐ Fastest | Excellent | Requires SageAttention 2.2.0+ |
| `native` | ⭐⭐ Fast | Excellent | PyTorch native SDPA, most compatible |

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

1. **Use SageAttention**: Install `sageattention==2.2.0` for **30-40% speedup** (RTX 4090: 10 FPS → 16 FPS)
2. **Enable torch.compile**: Essential for achieving target FPS on both RTX 4090 and H100
3. **Prompt Caching**: Same prompts reuse cached embeddings (0ms overhead)
4. **2-Step Inference**: Perfect balance of speed & quality for real-time stylization

---

## 📊 Benchmarks

### 🎯 Measured Performance

| GPU | Configuration | Infer Latency | **Infer FPS** |
|-----|--------------|---------------|---------------|
| **RTX 4090** | SageAttention + Compile | ~150-200ms | **5+ FPS** ✅ |
| **RTX 4090** | Native + Compile | ~180-220ms | **4-5 FPS** |
| **H100** | SageAttention + Compile | ~80-100ms | **10+ FPS** 🚀 |
| **H100** | Native + Compile | ~100-120ms | **8-10 FPS** |

> ✅ **RTX 4090**: Measured **Infer FPS 5+**  
> 🚀 **H100**: Measured **Infer FPS 10+**


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
