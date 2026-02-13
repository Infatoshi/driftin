"""Multi-resolution feature encoders for drift field computation.

Each encoder extracts spatial features at multiple scales from a frozen pretrained model.
Returns list of (features[B, L, C], C_j) tuples grouped by channel dimension,
ready for batched drift computation.

Supported encoders:
- DINOv2 multi-res: patch tokens from 4 intermediate ViT layers
- ConvNeXt-v2: 4 stage feature maps from ConvNeXt-v2-Base
- MoCo-v2: 4 stage feature maps from ResNet-50 with self-supervised weights
- DINOv3 multi-res: patch tokens from 4 intermediate ViT layers (DINOv3 ViT-B/16)
- DINOv3-L: DINOv3 ViT-L/16 (303M, D=1024, 24 layers)
- DINOv3-H+: DINOv3 ViT-H+/16 (841M, D=1280, 32 layers)
- AIMv2-L: Apple AIMv2 Large (309M, D=1024, autoregressive self-supervised)
- AIMv2-H: Apple AIMv2 Huge (681M, D=1536)
- RADIO: NVIDIA RADIO v2.5-L (320M, multi-teacher distillation)
- EVA-02 multi-res: intermediate features from EVA-02 ViT-B/14 via timm
- SigLIP-2 multi-res: patch tokens from 4 intermediate layers (SigLIP-2 Base)
- CLIP multi-res: patch tokens from OpenCLIP ViT-B/16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_spatial_features(feat_map, pool_size=4):
    """Extract per-location + statistical features from a feature map.

    Args:
        feat_map: [B, C, H, W] feature map
        pool_size: target spatial size for adaptive pooling

    Returns:
        List of (features[B, L, C], C_j) tuples.
        Contains: per-location vectors, global mean, global std.
    """
    B, C, H, W = feat_map.shape
    result = []

    if H != pool_size or W != pool_size:
        pooled = F.adaptive_avg_pool2d(feat_map, (pool_size, pool_size))
    else:
        pooled = feat_map

    # Per-location vectors: [B, pool_size^2, C]
    per_loc = pooled.reshape(B, C, pool_size * pool_size).permute(0, 2, 1)
    result.append((per_loc, C))

    # Global mean: [B, 1, C]
    g_mean = feat_map.mean(dim=[2, 3]).unsqueeze(1)
    result.append((g_mean, C))

    # Global std: [B, 1, C]
    g_std = feat_map.reshape(B, C, -1).std(dim=2).unsqueeze(1)
    result.append((g_std, C))

    return result


def _group_by_channel_dim(feature_groups):
    """Group feature tuples by channel dimension for batched computation."""
    by_dim = {}
    for feat, C_j in feature_groups:
        if C_j not in by_dim:
            by_dim[C_j] = []
        by_dim[C_j].append(feat)

    result = []
    for C_j in sorted(by_dim.keys()):
        cat = torch.cat(by_dim[C_j], dim=1)  # [B, sum(L_i), C_j]
        result.append((cat, C_j))
    return result


class DINOv2MultiResEncoder(nn.Module):
    """Frozen DINOv2 ViT-B/14 with multi-resolution patch token extraction."""

    def __init__(self, pool_size=4, input_size=112):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.layer_indices = [2, 5, 8, 11]

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        intermediate = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=True
        )

        all_groups = []
        for feat_map in intermediate:
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class ConvNeXtV2Encoder(nn.Module):
    """Frozen ConvNeXt-v2-Base with multi-resolution feature extraction."""

    def __init__(self, pool_size=4, input_size=112):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import timm
        self.model = timm.create_model(
            'convnextv2_base.fcmae_ft_in22k_in1k',
            pretrained=True,
            features_only=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        stage_features = self.model(x)

        all_groups = []
        for feat_map in stage_features:
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class MoCoV2Encoder(nn.Module):
    """Frozen ResNet-50 with MoCo-v2 self-supervised weights."""

    def __init__(self, pool_size=4, input_size=112):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        self._build_model()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _build_model(self):
        from torchvision.models import resnet50
        model = resnet50(weights=None)

        ckpt_url = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"
        ckpt = torch.hub.load_state_dict_from_url(ckpt_url, map_location="cpu")
        state_dict = ckpt["state_dict"]

        clean_sd = {}
        for k, v in state_dict.items():
            if k.startswith("module.encoder_q."):
                new_key = k.replace("module.encoder_q.", "")
                if new_key.startswith("fc."):
                    continue
                clean_sd[new_key] = v

        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        h = self.stem(x)

        all_groups = []
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            h = stage(h)
            groups = _extract_spatial_features(h, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class DINOv3MultiResEncoder(nn.Module):
    """Frozen DINOv3 ViT-B/16 with multi-resolution patch token extraction via timm.

    Uses timm's forward_intermediates() for clean multi-scale extraction.
    DINOv3 uses RoPE so it handles arbitrary input sizes.
    """

    def __init__(self, pool_size=4, input_size=112):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import timm
        self.model = timm.create_model(
            'vit_base_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 16
        self.hidden_size = 768
        self.layer_indices = [2, 5, 8, 11]  # 0-indexed for timm

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        final_feat, intermediates = self.model.forward_intermediates(
            x, indices=self.layer_indices
        )

        all_groups = []
        for feat_map in intermediates:
            if feat_map.dim() == 3:
                h_p = target_size // ps
                feat_map = feat_map.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class DINOv3LargeMultiResEncoder(nn.Module):
    """Frozen DINOv3 ViT-L/16 (303M params, D=1024, 24 layers).

    3.5x params over ViT-B. Distilled from ViT-7B. Uses RoPE.
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import timm
        self.model = timm.create_model(
            'vit_large_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 16
        self.hidden_size = 1024
        # 24 layers: pick 4 evenly spaced
        self.layer_indices = [5, 11, 17, 23]

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        final_feat, intermediates = self.model.forward_intermediates(
            x, indices=self.layer_indices
        )

        all_groups = []
        for feat_map in intermediates:
            if feat_map.dim() == 3:
                h_p = target_size // ps
                feat_map = feat_map.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class DINOv3HugePlusMultiResEncoder(nn.Module):
    """Frozen DINOv3 ViT-H+/16 (841M params, D=1280, 32 layers).

    10x params over ViT-B. Distilled from ViT-7B. Uses RoPE.
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import timm
        self.model = timm.create_model(
            'vit_huge_plus_patch16_dinov3.lvd1689m',
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 16
        self.hidden_size = 1280
        # 32 layers: pick 4 evenly spaced
        self.layer_indices = [7, 15, 23, 31]

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        final_feat, intermediates = self.model.forward_intermediates(
            x, indices=self.layer_indices
        )

        all_groups = []
        for feat_map in intermediates:
            if feat_map.dim() == 3:
                h_p = target_size // ps
                feat_map = feat_map.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class AIMv2MultiResEncoder(nn.Module):
    """Frozen AIMv2 with multi-resolution feature extraction via timm.

    Apple's autoregressive image model. Self-supervised (no language).
    Supports Large (309M, D=1024) and Huge (681M, D=1536).
    """

    def __init__(self, pool_size=4, input_size=224, variant="large"):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import timm

        if variant == "large":
            model_name = 'aimv2_large_patch14_224.apple_pt'
            self.hidden_size = 1024
            n_blocks = 24
        elif variant == "huge":
            model_name = 'aimv2_huge_patch14_224.apple_pt'
            self.hidden_size = 1536
            n_blocks = 32
        else:
            raise ValueError(f"Unknown AIMv2 variant: {variant}")

        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 14
        # Pick 4 evenly spaced layers
        self.layer_indices = [
            n_blocks // 4 - 1,
            n_blocks // 2 - 1,
            3 * n_blocks // 4 - 1,
            n_blocks - 1,
        ]

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        final_feat, intermediates = self.model.forward_intermediates(
            x, indices=self.layer_indices
        )

        h_p = target_size // ps
        all_groups = []
        for feat_map in intermediates:
            if feat_map.dim() == 3:
                feat_map = feat_map.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class RADIOMultiResEncoder(nn.Module):
    """Frozen NVIDIA RADIO v2.5-L (320M, D=1024) with multi-res extraction.

    Multi-teacher distillation from CLIP + DINOv2 + SAM.
    Loaded via torch.hub.
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        self.hidden_size = 1024
        self.patch_size = 16

        self.radio = torch.hub.load(
            'NVlabs/RADIO', 'radio_model', version='radio_v2.5-l',
            progress=False, trust_repo=True,
        )
        self.radio.eval()
        for param in self.radio.parameters():
            param.requires_grad = False

        # RADIO expects ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Hook intermediate layers from the inner ViT
        inner = self.radio.model
        n_blocks = len(inner.blocks)
        self.layer_indices = [
            n_blocks // 4 - 1,
            n_blocks // 2 - 1,
            3 * n_blocks // 4 - 1,
            n_blocks - 1,
        ]
        self._intermediate_features = []
        self._hooks = []
        for idx in self.layer_indices:
            h = inner.blocks[idx].register_forward_hook(self._make_hook())
            self._hooks.append(h)

    def _make_hook(self):
        storage = []
        self._intermediate_features.append(storage)
        def hook_fn(module, input, output):
            storage.clear()
            storage.append(output)
        return hook_fn

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Forward triggers hooks
        summary, features = self.radio(x)

        h_p = target_size // ps
        all_groups = []
        for storage in self._intermediate_features:
            if storage:
                feat = storage[0]  # [B, N, D]
                # RADIO inner ViT has no CLS prefix in spatial features
                # but the block output includes it; check shape
                if feat.shape[1] > h_p * h_p:
                    # Has prefix tokens, skip them
                    n_prefix = feat.shape[1] - h_p * h_p
                    feat = feat[:, n_prefix:, :]
                feat_map = feat.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
                groups = _extract_spatial_features(feat_map, self.pool_size)
                all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class EVA02MultiResEncoder(nn.Module):
    """Frozen EVA-02 ViT-B/14 with multi-resolution feature extraction via timm.

    Uses timm's forward_intermediates() for clean multi-scale extraction.
    EVA-02 uses absolute PE so input must match model size (224).
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = 224  # EVA-02 requires 224 (absolute PE)
        import timm
        self.model = timm.create_model(
            'eva02_base_patch14_224.mim_in22k',
            pretrained=True,
            num_classes=0,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.patch_size = 14
        self.hidden_size = 768
        self.layer_indices = [2, 5, 8, 11]  # 0-indexed layer indices for timm

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        final_feat, intermediates = self.model.forward_intermediates(
            x, indices=self.layer_indices
        )

        all_groups = []
        for feat_map in intermediates:
            # timm returns [B, C, H, W] spatial tensors (CLS already removed)
            if feat_map.dim() == 3:
                h_p = self.input_size // self.patch_size
                feat_map = feat_map.transpose(1, 2).reshape(-1, self.hidden_size, h_p, h_p)
            groups = _extract_spatial_features(feat_map, self.pool_size)
            all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class SigLIP2MultiResEncoder(nn.Module):
    """Frozen SigLIP-2 Base with multi-resolution feature extraction.

    Manual forward through transformer layers to collect intermediates.
    SigLIP has no CLS token (mean pooling), so all tokens are patch tokens.
    Uses 224 input (learned absolute PE).
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = 224  # SigLIP uses learned PE, must match 224
        from transformers import AutoModel
        full_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        vm = full_model.vision_model
        self.embeddings = vm.embeddings
        self.layers = vm.encoder.layers
        del full_model, vm
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        # SigLIP normalization: maps [0,1] -> [-1,1]
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.patch_size = 16
        self.hidden_size = 768
        # Collect after layers 3, 6, 9, 12 (1-indexed)
        self.layer_indices = {2, 5, 8, 11}  # 0-indexed

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)

        h_p = self.input_size // self.patch_size  # 14
        w_p = h_p

        hidden = self.embeddings(x)  # [B, num_patches, D]

        all_groups = []
        for i, layer in enumerate(self.layers):
            out = layer(hidden, attention_mask=None)
            hidden = out[0] if isinstance(out, tuple) else out
            if i in self.layer_indices:
                feat_map = hidden.transpose(1, 2).reshape(-1, self.hidden_size, h_p, w_p)
                groups = _extract_spatial_features(feat_map, self.pool_size)
                all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


class CLIPMultiResEncoder(nn.Module):
    """Frozen OpenCLIP ViT-B/16 with multi-resolution feature extraction.

    Uses open_clip library. Extracts intermediate transformer layer outputs
    and reshapes patch tokens to spatial feature maps.
    """

    def __init__(self, pool_size=4, input_size=224):
        super().__init__()
        self.pool_size = pool_size
        self.input_size = input_size
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai'
        )
        self.visual = model.visual
        del model
        self.visual.eval()
        for param in self.visual.parameters():
            param.requires_grad = False
        # CLIP uses its own normalization (close to ImageNet)
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.patch_size = 16
        self.hidden_size = 768
        self.layer_indices = [3, 6, 9, 12]

    def forward(self, x):
        x = (x + 1) / 2
        x = (x - self.mean) / self.std
        ps = self.patch_size
        target_size = (self.input_size // ps) * ps
        x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        h_p = target_size // ps
        w_p = target_size // ps

        # Manual forward through CLIP visual transformer to get intermediates
        v = self.visual
        # Patch embedding
        x = v.conv1(x)  # [B, D, h, w]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, h*w, D]
        # Prepend CLS token
        cls_token = v.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+h*w, D]
        x = x + v.positional_embedding[:x.shape[1]].unsqueeze(0)
        x = v.ln_pre(x)

        # Run through transformer layers, collecting intermediates
        x = x.permute(1, 0, 2)  # [seq, B, D] for nn.TransformerEncoder
        all_groups = []
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)
            if (i + 1) in self.layer_indices:
                # Extract patch tokens (skip CLS at index 0)
                patches = x[1:, :, :].permute(1, 0, 2)  # [B, h*w, D]
                feat_map = patches.transpose(1, 2).reshape(-1, self.hidden_size, h_p, w_p)
                groups = _extract_spatial_features(feat_map, self.pool_size)
                all_groups.extend(groups)

        return _group_by_channel_dim(all_groups)


def build_encoder(name, pool_size=4, input_size=112):
    """Factory function for multi-resolution encoders.

    Args:
        name: one of 'dinov2-multires', 'convnextv2', 'mocov2', 'dinov3',
              'dinov3-l', 'dinov3-h+', 'aimv2-l', 'aimv2-h', 'radio',
              'eva02', 'siglip2', 'clip'
        pool_size: target spatial pool size (default 4 -> 16 locations per stage)
        input_size: resize input to this before encoder (default 112, down from 224)
    """
    if name == "dinov2-multires":
        return DINOv2MultiResEncoder(pool_size=pool_size, input_size=input_size)
    elif name == "convnextv2":
        return ConvNeXtV2Encoder(pool_size=pool_size, input_size=input_size)
    elif name == "mocov2":
        return MoCoV2Encoder(pool_size=pool_size, input_size=input_size)
    elif name == "dinov3":
        return DINOv3MultiResEncoder(pool_size=pool_size, input_size=input_size)
    elif name == "dinov3-l":
        sz = input_size if input_size >= 224 else 224
        return DINOv3LargeMultiResEncoder(pool_size=pool_size, input_size=sz)
    elif name == "dinov3-h+":
        sz = input_size if input_size >= 224 else 224
        return DINOv3HugePlusMultiResEncoder(pool_size=pool_size, input_size=sz)
    elif name == "aimv2-l":
        sz = input_size if input_size >= 224 else 224
        return AIMv2MultiResEncoder(pool_size=pool_size, input_size=sz, variant="large")
    elif name == "aimv2-h":
        sz = input_size if input_size >= 224 else 224
        return AIMv2MultiResEncoder(pool_size=pool_size, input_size=sz, variant="huge")
    elif name == "radio":
        sz = input_size if input_size >= 224 else 224
        return RADIOMultiResEncoder(pool_size=pool_size, input_size=sz)
    elif name == "eva02":
        return EVA02MultiResEncoder(pool_size=pool_size, input_size=224)
    elif name == "siglip2":
        sz = input_size if input_size >= 224 else 224
        return SigLIP2MultiResEncoder(pool_size=pool_size, input_size=sz)
    elif name == "clip":
        sz = input_size if input_size >= 224 else 224
        return CLIPMultiResEncoder(pool_size=pool_size, input_size=sz)
    else:
        raise ValueError(
            f"Unknown encoder: {name}. Choose from: dinov2-multires, convnextv2, "
            f"mocov2, dinov3, dinov3-l, dinov3-h+, aimv2-l, aimv2-h, radio, "
            f"eva02, siglip2, clip"
        )
