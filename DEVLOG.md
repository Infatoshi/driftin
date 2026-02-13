# Drifting vs Diffusion: CIFAR-10 Head-to-Head -- Dev Log

## Date: Feb 7-8, 2026

## Goal

Compare drifting (single-step generation via drift field) against DDPM (iterative denoising) on CIFAR-10, sharing the same UNet backbone. Demonstrate that drifting can produce recognizable images at 50x faster inference.

---

## 1. Architecture

Both methods use the same UNet:
- **Small config (~38M params):** base_ch=128, ch_mult=(1,2,2,2), 2 res blocks, attention at 16x16, 4 heads
- **Large config (~152M params):** base_ch=256, ch_mult=(1,2,2,2), 3 res blocks, attention at 16x16 and 8x8, 8 heads (tested, abandoned due to time constraints)

DDPM: noise -> 50 DDIM steps -> image
Drift: noise -> 1 forward pass -> image

Drift training uses a frozen feature encoder to compute the drift field V. The generator (UNet) learns to map noise to images such that `gen_feats` move toward `pos_feats` (real data features) according to the drift signal.

---

## 2. Hardware & Throughput

### Single GPU (RTX 3090, 24GB)

| Method              | Steps/sec | ms/step | img/s | VRAM  |
|---------------------|-----------|---------|-------|-------|
| DDPM                | 7.4       | 136     | 944   | 5.8GB |
| Drift (feature)     | 5.7       | 174     | 735   | 7.2GB |
| Drift (pixel)       | 7.0       | 142     | 902   | ~6GB  |

### 8x H100 (80GB each) -- DDP

| Method                       | Global BS | img/s   | GPU util | VRAM/GPU |
|------------------------------|-----------|---------|----------|----------|
| DDPM                         | 1024      | 25,800  | ~95%     | 6.7GB    |
| Drift + ResNet-18 (bs=256)   | 2048      | 12,184  | ~85%     | ~8GB     |
| Drift + ResNet-18 (bs=512)   | 4096      | 12,811  | ~85%     | ~12GB    |
| Drift + DINOv2 (bs=128)      | 1024      | 4,149   | ~85%     | 19.7GB   |
| Drift pixel-space (bs=512)   | 4096      | 12,694  | ~85%     | ~10GB    |

DINOv2 is slower per step because it resizes 32x32 -> 224x224 and runs through a ViT-B/14 (86M frozen params) for every batch. The feature quality tradeoff is worth it.

---

## 3. Experiments Run

### Exp 0: DDPM baseline (8xH100)
- Config: small UNet, bs=128/GPU (global 1024), 50k steps
- Duration: ~32 min
- Result: **Sharp, recognizable CIFAR-10 images.** Clear objects (cars, planes, horses, ships).
- Final loss: 0.022 (MSE on noise prediction -- meaningful convergence metric)

### Exp 1: Drift + ResNet-18, bs=256/GPU (8xH100)
- Config: small UNet, ResNet-18 layer2 features (128D), global BS 2048, 50k steps
- Duration: ~2.3h
- Result: Blobby shapes with varied colors. Some scene-like composition (sky+ground) but no fine detail.
- Final loss: ~8.26 (drift-normalized, not a convergence metric)

### Exp 2: Drift + ResNet-18, bs=512/GPU (8xH100)
- Config: small UNet, ResNet-18 layer2 features (128D), global BS 4096, 50k steps
- Duration: ~4.4h
- Result: **Noticeably better than Exp 1.** More scene-like, some recognizable object outlines. Still washed out.
- Final loss: ~8.16
- Takeaway: Bigger batch = better drift signal. 4096 > 2048.

### Exp 3: Drift pixel-space, bs=512/GPU (8xH100)
- Config: small UNet, raw pixel features (3072D), global BS 4096, ~10.8k steps (killed early)
- Result: **Mode collapse to beige blobs.** Complete failure.
- Reason: Curse of dimensionality. In 3072D, all pairwise distances concentrate around the same value, softmax kernel becomes uniform, drift signal vanishes. Feature-space is mandatory.

### Exp 4: Drift + DINOv2, bs=128/GPU (8xH100) -- RUNNING
- Config: small UNet, DINOv2 ViT-B/14 CLS token (768D), global BS 1024, 50k steps
- Status: step ~34k/50k, ~1h remaining
- 10k sample: Recognizable animal/scene shapes, dramatic improvement over ResNet-18
- 20k sample: Clear animals (dogs, horses, birds), vehicles, sky/landscapes
- 30k sample: Further refinement, better color saturation, clearer object boundaries
- Final loss: ~8.9 (drift-normalized)
- **Best drift result by far, even with 4x smaller global batch than ResNet-18 experiments**

### Abandoned: Large UNet (152M params, 8xH100)
- ETA was 12.5h for 200k steps -- too expensive for the time budget
- The GPUs were memory-underutilized (6.7GB/80GB) with the small UNet, but the large UNet was compute-bound, not memory-bound

---

## 4. Key Insight: Feature Encoder Quality Matters More Than Batch Size

| Encoder    | Dim  | Global BS | Best visual quality |
|------------|------|-----------|---------------------|
| Pixel      | 3072 | 4096      | Mode collapse (useless) |
| ResNet-18  | 128  | 4096      | Blobby, some scenes |
| DINOv2     | 768  | 1024      | Recognizable objects |

DINOv2 with 4x less batch beats ResNet-18 with 4x more batch. The drift kernel does nearest-neighbor weighting in feature space -- DINOv2's self-supervised features are specifically trained for visual nearest-neighbor retrieval, which is exactly what the drift computation needs.

Pixel-space fails because of the curse of dimensionality: in 3072D, exp(-||x-y||^2 / tau) collapses to a uniform distribution (all distances are similar), so the weighted drift field carries no information.

---

## 5. Issues Encountered & Fixes

### Issue 1: CIFAR-10 download race condition (DDP)
- **Bug:** All 8 ranks tried to download CIFAR-10 simultaneously. Non-rank-0 processes would find a partial download and crash.
- **Fix:** Rank 0 downloads first, then `dist.barrier()`, then all ranks load with `download=False`.

### Issue 2: torch.randn doesn't accept memory_format
- **Bug:** `torch.randn(B, 3, 32, 32, device=device, memory_format=torch.channels_last)` raises TypeError.
- **Fix:** `torch.randn(B, 3, 32, 32, device=device).to(memory_format=torch.channels_last)`

### Issue 3: DINOv2 non-contiguous tensors in all_gather
- **Bug:** DINOv2's output CLS token is not contiguous in memory. `dist.all_gather` requires contiguous tensors, raising `ValueError: Tensors must be contiguous`.
- **Fix:** Add `tensor = tensor.contiguous()` at the start of `all_gather_flat()`.

### Issue 4: torch.compile state_dict key prefix
- **Bug:** After `torch.compile(model)`, `state_dict()` keys get `_orig_mod.` prefix. Loading into an uncompiled model fails.
- **Fix:** Strip prefix when loading, or always save `model.module.state_dict()` (unwrap DDP first).

### Issue 5: Drift loss is always ~1.0 (or ~8-9 with DINOv2)
- **Not a bug.** Drift normalization (lambda_j) rescales V to unit per-dimension variance, so MSE(x, x+V) = E[||V||^2]/C_j ~ 1.0 per temperature. With 3 temperatures x higher-D features, it's ~8-9. Gradients are still correct. Loss value is NOT a convergence metric for drift.

### Issue 6: bf16 underflow in softmax with small temperatures
- **Bug:** With tau=0.02 and high-dimensional features, logits in the softmax are very negative. bf16 range can't represent them, leading to all-zero softmax.
- **Fix:** Cast to float32 for cdist and softmax computation, cast back after.

---

## 6. Inference Benchmarks (RTX 3090)

### Drift vs DDPM -- Same 38M UNet

| Method          | 1 image   | 8 images       | 64 images       |
|-----------------|-----------|----------------|-----------------|
| Drift (1 step)  | **8.3 ms** | 16 ms (2 ms/img) | 74 ms (1.2 ms/img) |
| DDPM (50 DDIM)  | 418 ms    | 873 ms (109 ms/img) | 4834 ms (76 ms/img) |

**Drift is 50x faster.** Both use the same UNet; DDPM runs it 50 times, drift runs it once.

### Resolution Scaling (Drift, single image, 3090)

| Resolution         | Latency  | FPS    | VRAM  |
|--------------------|----------|--------|-------|
| 32x32 (pixel)      | 8.5 ms   | 117    | 0.3GB |
| 64x64 (pixel)      | 13.0 ms  | 77     | 0.3GB |
| 128x128 (pixel)    | 57.8 ms  | 17     | 0.9GB |
| 256x256 (pixel)    | OOM      | --     | >24GB |
| 64x64 latent -> 512x512 | ~20 ms | ~49 | ~1GB  |

The latent-space approach (drift on 64x64, VAE decode to 512x512) is the practical path: ~49 FPS on a consumer 3090.

---

## 7. Inference Optimization Benchmark (RTX 3090, 38M UNet, 32x32)

Full benchmark: 5 methods x 8 batch sizes, CUDA event timing (GPU-side), 20 warmup + 100 timed iterations.

### Per-image latency (ms)

| Method | bs=1 | bs=2 | bs=4 | bs=8 | bs=16 | bs=32 | bs=64 | bs=128 |
|---|---|---|---|---|---|---|---|---|
| eager bf16 | 8.59 | 4.70 | 2.62 | 1.84 | 1.44 | 1.24 | 1.12 | 1.08 |
| eager fp32 | 7.22 | 4.21 | 2.55 | 2.02 | 1.76 | 1.57 | 1.48 | 1.42 |
| torch.compile | 3.87 | 3.14 | 1.81 | 1.29 | 0.97 | 0.81 | 0.69 | 0.64 |
| **compile max-autotune** | **3.26** | **2.61** | **1.55** | **1.08** | **0.85** | **0.75** | **0.66** | **0.64** |
| CUDA graphs (eager) | 6.80 | 3.80 | 2.27 | 1.61 | 1.35 | 1.21 | 1.11 | 1.03 |

### Throughput (img/s)

| Method | bs=1 | bs=8 | bs=32 | bs=128 |
|---|---|---|---|---|
| eager bf16 | 116 | 544 | 803 | 930 |
| eager fp32 | 138 | 495 | 636 | 706 |
| torch.compile | 258 | 778 | 1235 | 1573 |
| **compile max-autotune** | **306** | **923** | **1338** | **1574** |
| CUDA graphs (eager) | 147 | 621 | 829 | 968 |

### Observations

1. **torch.compile max-autotune wins everywhere.** 2.6x over eager at bs=1 (3.26ms vs 8.59ms). The Triton autotuner picks optimal tile sizes for every conv kernel.
2. **CUDA graphs on eager model barely help (~4%).** The model is compute-bound, not kernel-launch-bound at these sizes. Graphs would matter more for very small models or when combined with compile.
3. **bf16 is paradoxically slower than fp32 at bs=1.** The 3090's tensor cores need enough work to amortize the format conversion overhead. At bs>=8, bf16 pulls ahead.
4. **Throughput saturates around bs=32-64.** Going from bs=64 to bs=128 only gains ~4% throughput. The GPU is fully utilized by bs=32.
5. **For real-time video (bs=1):** compile max-autotune gives 306 FPS at 32x32. Extrapolating to 64x64 latent: ~150 FPS. Plenty for 60Hz video.

Raw data: `outputs/inference_bench/inference_bench.csv` and `inference_bench.json`

---

## 8. Multi-Resolution Encoder Sweep (8xH100)

After sharing our initial results, one of the paper's authors reached out with a key implementation detail: as noted in the paper (page 7), they have been unable to make the method work without a feature encoder. They recommended extracting multi-resolution features using off-the-shelf models like MoCo-v2 or ConvNeXt-v2, as detailed in Appendix A.5. The core idea is to extract per-location spatial features at multiple encoder stages rather than using a single global vector (like a CLS token or GAP output).

We took this recommendation and expanded it significantly, running a total of **8 different encoder configurations** across two rounds of experiments to find the best feature representation for the drift field.

### Multi-Resolution Feature Extraction

Instead of a single feature vector per image, multi-res extraction produces **72 feature vectors per image** per encoder stage: 16 per-location vectors (from 4x4 adaptive pooling) + 1 global mean + 1 global std, across 4 encoder stages. Features with the same channel dimension are batched into [L, N, C] tensors for efficient drift computation via a single cdist/bmm call.

### Round 1: Author-Recommended Encoders

All trained for 50K steps on 8xH100 GPUs in parallel.

| Encoder | Architecture | Input | GPUs | Speed | Duration |
|---------|-------------|-------|------|-------|----------|
| DINOv2 multi-res | ViT-B/14, layers [2,5,8,11] | 112x112 | 3 | 3.3 it/s | 4.2h |
| MoCo-v2 multi-res | ResNet-50, 4 stages | 112x112 | 2 | 4.6 it/s | 3.0h |
| ConvNeXt-v2 multi-res | ConvNeXt-v2-Base, 4 stages | 112x112 | 3 | 3.8 it/s | 3.7h |

**Results:** Surprisingly, none of the multi-res configurations clearly beat the DINOv2 CLS single-vector approach from Exp 4. MoCo-v2 produced sharper edges but less semantically coherent objects. ConvNeXt-v2 suffered severe mode collapse (repeating dark animals). DINOv2 multi-res was more fragmented than DINOv2 CLS.

### Round 2: Next-Generation Encoders

Based on the Round 1 results, we hypothesized that **encoder quality matters more than the multi-res extraction strategy**. We tested 4 more encoders, including the then-newly-released DINOv3.

| Encoder | Architecture | Input | GPUs | Speed | Duration |
|---------|-------------|-------|------|-------|----------|
| DINOv3 multi-res | ViT-B/16 (timm), layers [2,5,8,11] | 112x112 | 2 | 2.7 it/s | 3.4h |
| EVA-02 multi-res | ViT-B/14 (timm), layers [2,5,8,11] | 224x224 | 2 | 1.4 it/s | 5.1h |
| SigLIP-2 multi-res | ViT-B/16 (HF), layers [2,5,8,11] | 224x224 | 2 | 2.1 it/s | 4.5h |
| CLIP multi-res | ViT-B/16 (OpenCLIP), layers [3,6,9,12] | 224x224 | 2 | 2.1 it/s | 4.6h |

### Combined Results Ranking

| Rank | Method | Quality | Notes |
|------|--------|---------|-------|
| 1 | **DINOv3 multi-res** | Clearly recognizable objects, good class diversity | Best single-step result by far |
| 2 | DINOv2 CLS (single vector) | Blurry but semantically coherent | Still competitive despite simpler features |
| 3 | SigLIP-2 multi-res | Decent diversity, some recognizable objects | Darker/muddier than DINOv3 |
| 4 | MoCo-v2 multi-res | Sharp edges but hard to identify objects | Good low-level stats, weak semantics |
| 5 | CLIP multi-res | Mostly blobs | Contrastive language-image features don't transfer well |
| 6 | DINOv2 multi-res | Fragmented, worse than CLS | Multi-res hurt rather than helped |
| 7 | EVA-02 multi-res | Mode collapse (red barns + deer) | Limited diversity, few repeated modes |
| 8 | ConvNeXt-v2 multi-res | Mode collapse (all dark cattle) | Total failure |

![9-way comparison](outputs/comparison_all_9_methods.png)

### Key Takeaway: Encoder Representation Quality Dominates

The paper author was right that feature encoders are essential, but the specific choice of encoder matters enormously. DINOv3 (trained via self-distillation from a 7B-parameter teacher on 1.7B images) produces dramatically better drift results than all other encoders, including its predecessor DINOv2. The quality of the pretrained visual representations transfers directly into generation quality.

Interestingly, the multi-resolution extraction strategy was a mixed bag: it helped DINOv3 but actually hurt DINOv2 (where the CLS token alone was better). This suggests that spatial feature coherence varies significantly across encoders, and simply extracting more features doesn't guarantee better drift fields.

---

## 9. The Case for Drifting at Scale

**Training scales with batch size.** The drift signal quality is directly proportional to how many samples participate in the kernel estimation. This is fundamentally different from DDPM, where batch size mainly affects gradient noise.

- 1 GPU, bs=128: weak drift signal, blobby results
- 8 GPUs, bs=4096 global: scene-like compositions
- Hypothetical 1000 GPUs, bs=500k global: potentially state-of-the-art single-step generation

**Inference is fixed.** No matter how much compute you spend training, drift inference is always 1 forward pass. DDPM always needs 50-1000 sequential passes. This makes drift the only viable path for:
- Real-time video generation (30-60 FPS)
- Interactive world models / AI-generated game frames
- Edge deployment where latency matters

**The quality gap is a training problem, not an architecture problem.** The UNet is identical. The question is whether the drift field provides enough learning signal at scale.

---

## 10. TODO: Further Kernel Optimization

### Training-side optimization

**Current bottlenecks (from profiling on 3090):**
- matmul: 21% of CUDA time
- elementwise ops: 15%
- dtype conversion (bf16 <-> f32): 11%
- cdist (pairwise distances): 8.5%

**Potential optimizations:**
- [ ] Profile on H100 to identify different bottleneck distribution (tensor cores change the balance)
- [ ] Fuse the cdist + softmax + normalization into a single Triton kernel (avoid materializing the full NxN distance matrix)
- [ ] The drift V computation does cdist -> softmax -> weighted sum for each temperature. These three temps can share the cdist computation
- [ ] Investigate FlashAttention-style tiling for the NxN pairwise distance matrix (it's structurally similar to attention -- query=gen, key=pos, value=pos)
- [ ] The all_gather + compute_V + slice pattern could use a ring-reduce variant to avoid materializing the full global batch on each GPU
- [ ] DINOv2 forward pass (86M params, 224x224 input) dominates wall time. Could distill into a smaller encoder, or use intermediate features instead of CLS token to avoid the full forward pass
- [ ] Mixed-precision: currently cast to f32 for cdist/softmax. Could investigate fp8 for the distance computation on H100 (fp8 has enough range if we scale properly)

### Inference-side optimization

**Current state:** 8.3ms for 32x32, 13ms for 64x64 on 3090. Already fast, but for real-time video every ms counts.

**Potential optimizations:**
- [ ] Quantization: INT8/INT4 weight quantization of the UNet. Single-step generation is more tolerant of quantization noise than iterative methods (no error accumulation across steps)
- [ ] Kernel fusion: fuse GroupNorm + SiLU + Conv patterns that appear throughout the UNet's residual blocks. Currently each is a separate kernel launch
- [ ] TensorRT / torch.compile with max-autotune: compile the full forward pass into a single fused graph. `torch.compile(mode="max-autotune")` already helps but a TensorRT engine would be tighter
- [ ] Operator fusion for attention: the UNet's self-attention uses `F.scaled_dot_product_attention` which already picks FlashAttention, but the surrounding reshape/proj operations could be fused
- [ ] Speculative: since drift generates from pure noise (no conditioning yet), multiple frames could be batched and pipelined -- generate frame N while postprocessing frame N-1
- [ ] For video: temporal conditioning (feeding previous frame) adds one concat + slight overhead. Profile to see if this changes the bottleneck
- [ ] CUDA graphs: capture the entire forward pass as a CUDA graph to eliminate kernel launch overhead entirely. Single-step generation is perfect for this (no dynamic control flow)
- [ ] For latent-space deployment: co-optimize the VAE decoder and drift UNet as a single fused pipeline

### Estimated impact (updated with measured baselines)

| Optimization                     | Expected speedup | Effort | Notes |
|----------------------------------|------------------|--------|-------|
| torch.compile max-autotune       | **2.6x measured** | Zero  | Already done -- 3.26ms vs 8.59ms at bs=1 |
| CUDA graphs (on eager)           | ~4% measured     | Low    | Compute-bound, not launch-bound |
| CUDA graphs + compile combo      | ~10-20%          | Medium | Needs static shapes |
| TensorRT export                  | 20-40%           | Medium | |
| INT8 quantization                | 30-50%           | Medium | Single-step tolerant of quant noise |
| Triton fused cdist+softmax (train) | 20-30%        | High   | |
| Distilled feature encoder (train)  | 2-3x           | High   | |
| INT4 + kernel fusion (inference)   | 50-70%         | High   | |

With compile max-autotune already achieving 3.26ms at bs=1 (306 FPS), the remaining headroom via TensorRT + INT8 could push to:
- 32x32: ~1.5-2ms (500-650 FPS)
- 64x64 latent -> 512x512: ~6-8ms (125-166 FPS)

---

## 11. File Reference

```
drifting_vs_diffusion/
  config.py                       -- UNetConfig, UNetLargeConfig, DDPMConfig, DriftConfig, MultiResDriftConfig
  train_ddpm.py                   -- Single-GPU DDPM training
  train_ddpm_ddp.py               -- Multi-GPU DDP DDPM training
  train_drift.py                  -- Single-GPU drift training
  train_drift_ddp.py              -- Multi-GPU DDP drift training (ResNet-18 + DINOv2 encoders)
  train_drift_multires_ddp.py     -- Multi-GPU DDP training with multi-res encoders (pre-computed features)
  models/
    unet.py                       -- Shared UNet (AdaGN, SelfAttention, ResBlock)
    ema.py                        -- EMA wrapper
  training/
    compute_v.py                  -- Drift field computation (single-vector + batched multi-res)
    encoders.py                   -- Multi-res feature encoders (DINOv2/v3, MoCo-v2, ConvNeXt-v2, EVA-02, SigLIP-2, CLIP)
    ddpm_utils.py                 -- DDPM noise schedule, q_sample, DDIM sampling
  eval/
    sample.py                     -- Grid sampling utilities
    fid.py                        -- FID computation

outputs/                          -- On 8xH100 remote
  ddpm_h100/                      -- DDPM baseline (50k steps, complete)
  drift_bs256_feat/               -- Drift + ResNet-18, bs=256/GPU (50k steps, complete)
  drift_bs512_feat/               -- Drift + ResNet-18, bs=512/GPU (50k steps, complete)
  drift_bs512_pixel/              -- Drift pixel-space, bs=512/GPU (killed at 10.8k steps)
  drift_dinov2/                   -- Drift + DINOv2 CLS, bs=128/GPU (50k steps, complete)
  drift_dinov2_multires/          -- Drift + DINOv2 multi-res (50k steps, complete)
  drift_mocov2/                   -- Drift + MoCo-v2 multi-res (50k steps, complete)
  drift_convnextv2/               -- Drift + ConvNeXt-v2 multi-res (50k steps, complete)
  drift_dinov3/                   -- Drift + DINOv3 multi-res (50k steps, complete) -- BEST
  drift_eva02/                    -- Drift + EVA-02 multi-res (50k steps, complete)
  drift_siglip2/                  -- Drift + SigLIP-2 multi-res (50k steps, complete)
  drift_clip/                     -- Drift + CLIP multi-res (50k steps, complete)
  comparison_all_9_methods.png    -- 9-way visual comparison grid
```

---

## 12. Scaling to 256x256: Latent-Space Drifting

### Architecture

To generate at 256x256, we moved to latent space:
- **UNet** generates 32x32x4 latents (same architecture, just in_ch=4, out_ch=4)
- **SD VAE** (stabilityai/sd-vae-ft-mse, 83.7M params, frozen) decodes latents to 256x256 RGB
- **DINOv3 ViT-B/16** (85.6M params, frozen) extracts multi-res features from decoded images
- Gradient flows: drift loss -> DINOv3 -> VAE decode -> UNet

The VAE decoder is gradient-checkpointed to fit in memory. It's also the training bottleneck (~70ms on H100 at bs=16, vs 5.5ms for UNet and 7ms for DINOv3).

### Component Benchmarks (H100, single GPU)

| Component | bs=16 | bs=32 | bs=64 |
|---|---|---|---|
| UNet fwd (compiled) | 5.5ms | 6.0ms | 7.7ms |
| VAE decode (grad ckpt) | 70.7ms | 140.3ms | 271.9ms |
| DINOv3 real (no grad) | 7.3ms | 13.3ms | 24.6ms |
| DINOv3 gen (with grad) | 16.1ms | 22.6ms | 43.1ms |
| Drift (L=72) | 1.5ms | 1.5ms | 1.8ms |
| **Full step** | **218ms** | **399ms** | **737ms** |
| Peak memory | 14 GB | 26.5 GB | 51.6 GB |

Max per-GPU batch on H100 80GB: ~96. Optimal throughput at bs=32/GPU.

### Encoder Benchmarks (H100, bs=32, forward pass)

| Encoder | Params | No-grad | With grad | Est. step | 50K hours |
|---|---|---|---|---|---|
| DINOv3 ViT-B/16 | 85.6M | 21.8ms | 49.0ms | 218ms | 3.0h |
| DINOv3 ViT-L/16 | 303M | 25.1ms | 126.6ms | 299ms | 4.2h |
| DINOv3 ViT-H+/16 | 841M | 111.9ms | 253.3ms | 513ms | 7.1h |
| AIMv2-Large | ~300M | 37.3ms | 98.2ms | 283ms | 3.9h |
| AIMv2-Huge | ~600M | 64.6ms | 165.8ms | 378ms | 5.2h |
| SigLIP2 SO400M | ~400M | 60.9ms | 157.6ms | 366ms | 5.1h |
| RADIO v2.5-L | ~300M | 28.7ms | 74.2ms | 250ms | 3.5h |

### Exp 5: FFHQ 256x256 (8xH100)

- **Config:** 38M UNet, DINOv3 ViT-B/16, bs=32/GPU (global 256), 50K steps
- **Duration:** ~5.8h, 419ms/step, 611 img/s
- **Dataset:** FFHQ 256x256, 70K face images
- **Result:** Good single-step face generation. Faces progress from blobs (5K) to recognizable (10K) to sharp with detail (50K). Diverse poses, lighting, expressions.

### Exp 6: ImageNet 256x256, Unconditional (8xH100)

- **Config:** 38M UNet, DINOv3 ViT-B/16, bs=32/GPU (global 256), 50K steps, unconditional
- **Duration:** ~5.8h, 419ms/step, 611 img/s
- **Dataset:** ImageNet-1k 256x256, 1.28M images, 1000 classes
- **Result:** Severe mode collapse. Individual images are sharp and coherent, but the model only produces ~15-20 recurring subjects out of 1000 classes (bulldogs, propeller planes, woven baskets, jars, ships, sheep, owls). The 38M UNet without class guidance can't cover the full distribution.

### Exp 7: ImageNet 256x256, Capacity + Conditioning Ablation (8xH100)

To disentangle the effect of model capacity vs class conditioning, we ran 3 experiments at 10K steps each:

| Experiment | UNet | Params | Conditioning | ms/step | Result |
|---|---|---|---|---|---|
| Small + cond | base_ch=128, 2 res blocks | 38.8M | 1000 classes | 475 | Recognizable objects, limited diversity |
| Large uncond | base_ch=256, 3 res blocks, attn@16+8 | 210M | none | 479 | Brown blobs, hasn't converged |
| Large + cond | base_ch=256, 3 res blocks, attn@16+8 | 211M | 1000 classes | 475 | **Best -- diverse objects, coherent shapes** |

Class conditioning is implemented via `nn.Embedding(num_classes, time_dim)` added to the timestep embedding before it enters the ResBlocks.

**Key finding: class conditioning is the dominant factor** at matched step count. The 210M unconditional model produces worse results than the 38M conditioned model -- it needs far more training steps without the guidance signal to disambiguate 1000 classes. The large + conditioned model combines capacity with guidance and is the clear winner.

---

## 13. Updated File Reference

```
drifting_vs_diffusion/
  config.py                       -- UNetConfig, UNetLargeConfig, DDPMConfig, DriftConfig, LatentDriftConfig
  train_ddpm.py                   -- Single-GPU DDPM training
  train_ddpm_ddp.py               -- Multi-GPU DDP DDPM training
  train_drift.py                  -- Single-GPU drift training
  train_drift_ddp.py              -- Multi-GPU DDP drift training (ResNet-18 + DINOv2 encoders)
  train_drift_multires_ddp.py     -- Multi-GPU DDP training with multi-res encoders (CIFAR)
  train_drift_latent_ddp.py       -- 256x256 latent-space training (--large, --num-classes)
  models/
    unet.py                       -- Shared UNet (AdaGN, SelfAttention, ResBlock, class conditioning)
    ema.py                        -- EMA wrapper
  training/
    compute_v.py                  -- Drift field computation (single-vector + batched multi-res)
    encoders.py                   -- Multi-res feature encoders (DINOv3/v3-L/v3-H+, AIMv2, RADIO, DINOv2, MoCo-v2, ConvNeXt-v2, EVA-02, SigLIP-2, CLIP)
    ddpm_utils.py                 -- DDPM noise schedule, q_sample, DDIM sampling
  eval/
    sample.py                     -- Grid sampling utilities
    sample_latent.py              -- Latent-space sampling
    fid.py                        -- FID computation
```

---

## 14. Summary

Single-step generation via drift fields scales from CIFAR-10 (32x32) to 256x256 through latent-space generation. Across all experiments, two findings dominate:

1. **Encoder quality is the single most important factor.** DINOv3 dramatically outperforms all other encoders (DINOv2, SigLIP-2, MoCo-v2, CLIP, EVA-02, ConvNeXt-v2) for drift field computation. The quality of pretrained visual representations transfers directly into generation quality.

2. **Class conditioning is critical for diverse datasets.** On ImageNet-1k (1000 classes), a 38M-param conditioned UNet outperforms a 210M-param unconditional UNet at matched step count. The guidance signal helps the model allocate capacity to quality rather than mode selection.

The latent-space pipeline (UNet on 32x32x4 latents, VAE decode to 256x256) runs at 611 img/s on 8xH100 during training. FFHQ faces look good at 50K steps. ImageNet requires class conditioning and larger models for full mode coverage.
