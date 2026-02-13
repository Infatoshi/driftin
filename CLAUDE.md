# Drifting CIFAR-10 Comparison

## Active Requirement: Multi-Resolution Feature Encoder Experiments

Paper author confirmed: **multi-resolution features are critical** (Appendix A.5). Our previous
experiments used single global feature vectors (ResNet-18 GAP -> 128D, DINOv2 CLS -> 768D).
The paper uses per-location + statistical features extracted at multiple spatial scales.

### Experiments to Run (8xH100)

All three use the same 38M-param UNet, same training config, same comparison framework.

#### 1. DINOv2 Multi-Resolution (`dinov2-multires`)
- Extract patch tokens from 4 intermediate ViT layers (blocks 3, 6, 9, 12)
- Reshape to 16x16 spatial, pool to 4x4 = 16 locations per layer
- Per-location + global mean + global std = 18 feature groups per layer
- All layers share C_j=768, so one batched drift call: [72, N, 768]

#### 2. ConvNeXt-v2 Multi-Resolution (`convnextv2`)
- ConvNeXt-v2-Base (88M params, frozen) pretrained on ImageNet
- 4 stages: C=[128, 256, 512, 1024], spatial=[56, 28, 14, 7]
- Pool each stage to 4x4, extract per-location + mean + std
- 4 separate batched drift calls, one per C_j

#### 3. MoCo-v2 Multi-Resolution (`mocov2`)
- ResNet-50 (25M params, frozen) with MoCo-v2 self-supervised weights
- 4 stages: C=[256, 512, 1024, 2048], spatial=[56, 28, 14, 7]
- Pool each stage to 4x4, extract per-location + mean + std
- 4 separate batched drift calls, one per C_j

### Training Config
- Per-GPU batch: 128, 8 GPUs, global batch: 1024
- UNet: base_ch=128, ch_mult=(1,2,2,2), 38M params
- Steps: 50,000
- Temperatures: [0.02, 0.05, 0.2]
- Input resize: 32x32 -> 224x224 for all encoders
- Feature pool target: 4x4 spatial (16 locations per stage)
- EMA decay: 0.9999
- LR: 2e-4, AdamW, grad clip 2.0

### Launch Commands
```bash
# Install dependencies
pip install timm

# DINOv2 multi-res
torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_multires_ddp \
    --encoder dinov2-multires --batch-size 128 --steps 50000 \
    --output-dir outputs/drift_dinov2_multires

# ConvNeXt-v2
torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_multires_ddp \
    --encoder convnextv2 --batch-size 128 --steps 50000 \
    --output-dir outputs/drift_convnextv2

# MoCo-v2
torchrun --nproc_per_node=8 -m drifting_vs_diffusion.train_drift_multires_ddp \
    --encoder mocov2 --batch-size 128 --steps 50000 \
    --output-dir outputs/drift_mocov2
```

### Key Implementation Details
- All encoders frozen; gradient flows: loss -> gen_features -> encoder_forward -> gen_images -> UNet
- Feature maps pooled to 4x4 to keep O(N^2) drift computation tractable at global batch 1024
- DDP: all_gather per C_j feature group, then batched V computation on [L, N_global, C_j]
- float32 for cdist/softmax in drift computation, bf16 for UNet forward
- Feature normalization per location: avg pairwise dist ~ sqrt(C_j) (Appendix A.6)
- Drift normalization per location: lambda_j scales V to unit per-dim variance

## Previous Experiments (Completed)
- DDPM baseline: 50k steps, 8xH100
- Drift + ResNet-18 global (bs=256, bs=512): 50k steps
- Drift + DINOv2 CLS token (bs=128): 50k steps, 8xH100
- Results: DINOv2 CLS >> ResNet-18 global >> pixel-space
- Key finding: encoder quality matters more than batch size

## Build / Run
- Python via `uv run` or system Python with torch+torchvision+timm
- Single-GPU test: `python -m drifting_vs_diffusion.train_drift_multires_ddp --encoder dinov2-multires --batch-size 4 --steps 10 --output-dir outputs/test_multires`
