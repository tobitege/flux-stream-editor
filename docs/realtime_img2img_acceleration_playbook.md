# Realtime FLUX Img2Img Acceleration Playbook

## 1. Scope and Outcome

This document captures the end-to-end optimization work on:

- `realtime_editing_fast/realtime_img2img_server.py`
- `realtime_editing_fast/editor.py`

for low-latency realtime img2img inference (`num_inference_steps=2`), with production-oriented controls and benchmark methodology.

Primary objective:

- maximize steady-state FPS on a single GPU
- without changing the model architecture
- while preserving output quality and operational safety

## 2. Baseline Architecture and Critical Path

Per request (`/api/predict`):

1. transport + route parsing
2. `base64 -> PIL` decode
3. `prepare_inputs`
4. denoise (2 steps)
5. VAE decode + postprocess
6. response encode

The practical latency bottleneck is the GPU critical path:

- `prepare_gpu` (H2D + VAE encode + conditioning pack)
- denoise
- VAE decode

CPU-side request handling was measured to be comparatively small and not the primary limiter.

## 3. Measurement Framework

### 3.1 External benchmark harnesses

Used external tests/scripts under `tests/` to avoid contaminating production logic:

- `tests/test_realtime_img2img_pipeline_timing_hack.py`
- `tests/test_threaded_latest_frame_pipeline_external.py`
- `tests/test_ray_cpu_preprocess_gpu_pipeline_external.py`
- `tests/test_vae_decode_accel_external.py`

### 3.2 Stage timing toggle

Added profile-mode sync gating:

- `FastFlux2Config.profile_stage_timing`
- server env: `FLUX_PROFILE_STAGE=1`

In profile mode only, stage timers include CUDA synchronization to avoid async misattribution.

Important: sync-based profiling is for diagnosis, not production latency numbers.

## 4. What Was Tried, and Why

### 4.1 Scheduling/concurrency experiments

Tried:

- same-process producer/consumer threading
- latest-frame overwrite queue (`maxsize=1`)
- Ray-based CPU preprocess split

Result:

- no material throughput gain on single GPU
- sometimes higher latency due to queueing/scheduling overhead

Reason:

- CPU work was not dominant
- GPU critical path remained the bound

Conclusion:

- prioritize kernel path and tensor preparation optimizations over orchestration complexity.

## 5. Implemented Optimizations

## 5.1 Prepare path simplification and memory movement

### Changes

In `editor.py`:

- fast tensor preprocess path
- pin-memory and non-blocking H2D options
- reduced avoidable tensor transforms in prepare flow

### Impact

Reduced CPU preprocess and H2D overhead; improved consistency in steady-state.

## 5.2 VAE decoder compile (major win)

### Changes

Added decoder-specific compile path:

- wraps decode in dedicated function
- compiles once in `ensure_loaded()`
- independent from transformer compile
- fallback to eager on compile failure

Config knobs (all in `FastFlux2Config` / env-driven defaults):

- `enable_vae_decoder_compile`
- `vae_decoder_compile_mode`
- `vae_decoder_compile_disable_cudagraphs`
- optional channels-last controls

### Empirical result

Decoder stage moved from ~31ms class to ~18–20ms class in stable runs, with strong end-to-end impact.

## 5.3 Conditioning-pack cleanup and caching

### Changes

In `_prepare_image_latents_fast`:

- removed unnecessary batch=1 `repeat/squeeze/unsqueeze` behavior
- cached `image_latent_ids` keyed by `(latent_h, latent_w, device)`

Added timestep cache:

- `_get_timesteps_for_latent_seq_len()`
- cache keyed by `(image_seq_len, num_steps, device)`

Config toggles:

- `cache_timesteps`
- `cache_image_latent_ids`

Server env:

- `FLUX_CACHE_TIMESTEPS`
- `FLUX_CACHE_IMAGE_LATENT_IDS`

### Why this matters

This removes recurrent per-frame allocation/packing overhead and reduces long-tail behavior in prepare.

## 5.4 VAE encoder compile switch + A/B validation

### Changes

Added encoder-specific compile path in `editor.py`:

- `enable_vae_encoder_compile`
- `vae_encoder_compile_mode`
- `vae_encoder_compile_disable_cudagraphs`
- compile-once, fallback-to-eager behavior

Server env:

- `FLUX_VAE_ENCODE_COMPILE`
- `FLUX_VAE_ENCODE_DISABLE_CUDAGRAPHS`
- `FLUX_VAE_ENCODE_COMPILE_MODE`

Also integrated into external benchmark harness:

- `tests/test_vae_decode_accel_external.py`

## 5.5 TAEF2 integration (optional VAE replacement)

### Changes

Integrated TAEF2 as an optional runtime VAE path in `realtime_editing_fast`:

- new helper module: `realtime_editing_fast/taef2.py`
- editor switch: `FastFlux2Config.enable_taef2`
- automatic artifact handling (`taesd.py` + `taef2.safetensors`) with local cache
- VAE compile fallback guards retained for stability

Runtime env switches:

- `FLUX_USE_TAEF2=1`
- `FLUX_TAEF2_FORCE_EAGER_VAE` (default `1`)
- `FLUX_TAEF2_CACHE_DIR`
- optional explicit artifact paths:
  - `FLUX_TAEF2_SCRIPT_PATH`
  - `FLUX_TAEF2_WEIGHT_PATH`

### Why this matters

TAEF2 primarily reduces VAE encode/decode cost (prepare/decode stages), which directly improves end-to-end realtime FPS in 2-step workloads.

## 6. Current Defaults

Default config now enables compile across critical modules:

- attention backend: `auto` (priority: `FA3 > Sage > Native`)
- transformer compile: `on`
- VAE decoder compile: `on`
- VAE encoder compile: `on`
- timestep/image-latent-id caches: `on`

All remain runtime-overridable via env flags and API/CLI backend arguments.

## 6.1 Attention Backend Selection (Current Recommended)

Backend strategy in server/editor is now:

1. try FA3 (`_flash_3`, alias: `fa3`)
2. fallback to `sage`
3. fallback to `native`

Implementation details:

- backend aliases normalized in `editor.py`
- auto-selection done at model load time
- runtime response reports the actual loaded backend
- FA3 compatibility wrapper is applied for current Diffusers + `flash_attn_3` interface shape

## 7. Benchmark Results (Representative)

Hardware/setup:

- single GPU
- `attention_backend=sage`
- `num_inference_steps=2`
- warmup `5`, measured `30`

### 7.1 Encoder compile A/B (clean API-level measurement)

| Case | avg server total (ms) | avg prepare (ms) | avg decode (ms) | avg denoise (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| `FLUX_VAE_ENCODE_COMPILE=0` | 74.90 | 15.16 | 18.67 | 41.08 | 13.35 |
| `FLUX_VAE_ENCODE_COMPILE=1` | 66.52 | 12.60 | 17.82 | 36.10 | 15.03 |

Delta (on vs off):

- `-8.38ms` end-to-end
- `+1.68 FPS` (about `+12.6%`)

### 7.2 Best-Config Stage Breakdown (Current)

Best config:

- `attention_backend=sage`
- `num_inference_steps=2`
- transformer compile: `on`
- VAE encoder compile: `on`
- VAE decoder compile: `on`
- timestep/image-latent-id cache: `on`

Clean API-level breakdown (`warmup=5`, `runs=30`):

| Stage | avg latency (ms) | share of total |
|---|---:|---:|
| `prepare` | 12.60 | 18.9% |
| `denoise` | 36.10 | 54.3% |
| `decode` | 17.82 | 26.8% |
| `server_total` | 66.52 | 100.0% |

Derived throughput:

- **15.03 FPS** (`1000 / 66.52`)

### 7.5 Latest FA3 vs Sage External A/B (Single GPU)

External harness:

- `tests/test_vae_decode_accel_external.py`
- `warmup=3`, `runs=5`, `num_inference_steps=2`

Results:

| Backend | avg server total (ms) | avg prepare (ms) | avg denoise (ms) | avg decode (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| `sage` | 79.03 | 21.95 | 36.90 | 20.18 | 12.65 |
| `_flash_3` (FA3) | 64.49 | 13.30 | 33.70 | 17.49 | 15.51 |

Additional user-side steady-state measurement:

- **FA3 reached 15.6 FPS** on single GPU in realtime run.

Decision:

- recommend `auto` (which prefers FA3), and use `fa3` explicitly when pinning backend for controlled experiments.

### 7.6 Latest TAEF2 External A/B (FA3, 2-step, same harness)

Measurement date:

- **2026-02-16**

External harness:

- `tests/test_vae_decode_accel_external.py`
- `warmup=5`, `runs=20`, `num_inference_steps=2`, `attention_backend=fa3`

Results:

| Case | server total (ms) | prepare (ms) | denoise (ms) | decode (ms) | FPS |
|---|---:|---:|---:|---:|---:|
| Baseline (`FLUX_USE_TAEF2=0`, VAE compile on) | 70.40 | 16.82 | 34.69 | 18.89 | 14.21 |
| TAEF2 eager-VAE (`FLUX_USE_TAEF2=1`, `FLUX_TAEF2_FORCE_EAGER_VAE=1`) | 56.79 | 12.54 | 33.63 | 10.63 | 17.61 |
| TAEF2 + VAE compile (`FLUX_USE_TAEF2=1`, `FLUX_TAEF2_FORCE_EAGER_VAE=0`) | 53.55 | 11.94 | 33.69 | 7.91 | 18.67 |

Delta summary:

- TAEF2 eager vs baseline:
  - `-13.60ms` total (`70.40 -> 56.79`)
  - `+3.40 FPS` (`14.21 -> 17.61`, about `+23.9%`)
- TAEF2 + compile vs baseline:
  - `-16.85ms` total (`70.40 -> 53.55`)
  - `+4.46 FPS` (`14.21 -> 18.67`, about `+31.5%`)
- TAEF2 + compile vs TAEF2 eager:
  - extra `-3.25ms` total
  - extra `+1.06 FPS` (about `+6.0%`)

Interpretation:

- most gain still comes from VAE path shrink (prepare/decode)
- denoise remains nearly unchanged
- in this setup, **TAEF2 + compile is the best measured option**

Diagnostic profile split (`FLUX_PROFILE_STAGE=1`, sync-instrumented; use for attribution, not direct SLO):

| Prepare sub-stage | avg latency (ms) |
|---|---:|
| `preprocess_cpu` | 0.56 |
| `h2d` | 2.38 |
| `vae_encode` | 12.47 |
| `pack_cond` | 0.76 |
| `timesteps` | 1.16 |

### 7.3 Earlier decoder compile impact (same workload family)

Decoder compile reduced decode stage from ~31ms class to ~18ms class and contributed the largest single-stage gain before encoder compile was introduced.

### 7.4 Prepare cache impact

With cache toggles enabled, prepare overhead and variance dropped versus no-cache runs under the same workload shape.

## 8. Quality and Stability Validation

### Quality checks

Performed image-level comparisons for compile variants:

- intra-mode deterministic checks (same mode, same seed) remained stable
- cross-mode outputs may differ numerically (as expected with kernel/codegen changes), but no catastrophic degradation was observed in visual inspection

### Operational caveat

Observed one process-exit segmentation fault in an encoder-compile-enabled run after successful inference and output save.

Implication:

- runtime inference path is usable
- still recommended to keep env kill-switches for rapid rollback:
  - `FLUX_VAE_ENCODE_COMPILE=0`
  - `FLUX_VAE_DECODE_COMPILE=0`

## 9. Runbook

## 9.1 Start server (defaults compile-enabled)

```bash
CUDA_VISIBLE_DEVICES=1 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2
```

`--attention-backend` default is `auto` (priority: FA3 > Sage > Native).

## 9.1.1 Force FA3

```bash
CUDA_VISIBLE_DEVICES=1 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2 --attention-backend fa3
```

## 9.2 Force-disable encoder compile

```bash
FLUX_VAE_ENCODE_COMPILE=0 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2
```

## 9.3 Profile stage timings

```bash
FLUX_PROFILE_STAGE=1 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2
```

## 9.4 Enable TAEF2

Default (stability first, VAE eager):

```bash
FLUX_USE_TAEF2=1 FLUX_TAEF2_FORCE_EAGER_VAE=1 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2 --attention-backend fa3
```

TAEF2 with VAE compile enabled:

```bash
FLUX_USE_TAEF2=1 FLUX_TAEF2_FORCE_EAGER_VAE=0 python -m realtime_editing_fast.realtime_img2img_server --host 127.0.0.1 --port 6006 --num-inference-steps 2 --attention-backend fa3
```

## 10. Why We Did Not Pursue More Scheduling Complexity

Single-GPU tests showed:

- Ray/threaded producer-consumer designs did not improve true throughput
- queueing/orchestration often increased latency

Given the observed bottleneck distribution, low-level path optimization (compile, prepare simplification, cache) had substantially better ROI.

## 11. Next High-ROI Directions

1. Denoise path fusion/unroll for fixed 2-step workload (reduce Python boundary + launch overhead).
2. Optional graph-capture experiments for denoise-only path, isolated from cache-dit constraints.
3. Harden encoder-compile stability:
   - soak tests
   - process lifecycle stress tests
   - add automated fallback on repeated runtime faults.
