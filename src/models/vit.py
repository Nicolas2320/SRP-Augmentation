"""
Vision Transformer (ViT) for CIFAR-style datasets.

Follows "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).
Designed for 32x32 inputs (CIFAR-10 / CIFAR-100).

Usage
-----
    from models.vit import build_vit_cifar
    model = build_vit_cifar(num_classes=100)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Sub-modules
# -----------------------------------------------------------------------------


class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and linearly project each one.

    A Conv2d with kernel_size == stride == patch_size is mathematically
    identical to flattening each patch and applying a Linear layer, but
    faster in practice.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        d_model: int,
    ) -> None:
        super().__init__()

        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )

        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x      : (B, C, H, W)
        # output : (B, N, d_model)
        x = self.proj(x)        # (B, d_model, H/P, W/P)
        x = x.flatten(2)        # (B, d_model, N)
        x = x.transpose(1, 2)   # (B, N, d_model)
        return x


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, attn_drop: float = 0.0) -> None:
        super().__init__()

        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv       = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj      = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V in one matrix multiply then split
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-LN Transformer block (LayerNorm before each sub-layer).

    Pre-LN trains more stably than Post-LN, especially on small datasets.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadSelfAttention(d_model, n_heads, attn_drop)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, mlp_ratio, proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------


class VisionTransformer(nn.Module):
    """
    Lightweight ViT for CIFAR-scale images (32x32).

    Default hyper-parameters use patch_size=4, giving 64 patches per image —
    a sequence length that is manageable on small datasets without a GPU cluster.

    Parameters
    ----------
    num_classes : int
        Output dimension (e.g. 100 for CIFAR-100).
    img_size : int
        Input spatial resolution (assumed square). Default: 32.
    patch_size : int
        Patch side length; must divide img_size. Default: 4.
    in_channels : int
        RGB input channels. Default: 3.
    d_model : int
        Token embedding dimension. Default: 256.
    depth : int
        Number of stacked TransformerBlocks. Default: 6.
    n_heads : int
        Number of self-attention heads. Default: 8.
    mlp_ratio : float
        FFN hidden-dim multiplier. Default: 4.0.
    dropout : float
        Dropout for attention weights and FFN activations. Default: 0.1.
    emb_dropout : float
        Dropout applied after positional embedding. Default: 0.1.
    """

    def __init__(
        self,
        num_classes: int = 100,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        d_model: int = 256,
        depth: int = 6,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, d_model)
        n_patches = self.patch_embed.n_patches

        # Learnable [CLS] token prepended to the patch sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # One positional embedding per patch + one for the CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        self.emb_drop = nn.Dropout(emb_dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, n_heads, mlp_ratio, dropout, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    # -------------------------------------------------------------------------
    # Weight initialisation (following the original ViT paper)
    # -------------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        # 1. Embed patches
        x = self.patch_embed(x)                    # (B, N, d_model)

        # 2. Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)     # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)             # (B, N+1, d_model)

        # 3. Add positional embedding
        x = x + self.pos_embed
        x = self.emb_drop(x)

        # 4. Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 5. Classify using only the CLS token
        cls_out = x[:, 0]                          # (B, d_model)
        return self.head(cls_out)                   # (B, num_classes)


# -----------------------------------------------------------------------------
# Builder — mirrors build_resnet50_cifar() in resnet.py
# -----------------------------------------------------------------------------


def build_vit_cifar(num_classes: int = 100) -> VisionTransformer:
    """
    Build ViT adapted for CIFAR-style datasets (32x32 images).

    Architecture
    ------------
    - patch_size = 4   ->  (32/4)^2 = 64 patches per image
    - d_model    = 256
    - depth      = 6 Transformer blocks
    - n_heads    = 8
    - ~5.8 M trainable parameters

    This works for:
    - CIFAR-10  with num_classes=10
    - CIFAR-100 with num_classes=100
    """
    return VisionTransformer(
        num_classes=num_classes,
        img_size=32,
        patch_size=4,
        in_channels=3,
        d_model=256,
        depth=6,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        emb_dropout=0.1,
    )
