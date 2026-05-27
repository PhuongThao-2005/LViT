# -*- coding: utf-8 -*-
"""
Vit.py  –  EfficientViT-powered ViT branch for EfficientLViT
Replaces the original VisionTransformer (global self-attention, O(n²))
with EfficientViTBranch (local window CGA, O(n·w²)).

Drop-in compatible: EfficientViTBranch.forward(x, skip_x, text, reconstruct)
has the same signature as the original VisionTransformer.forward().

Exported symbols used by LViT.py:
    EfficientViTBranch   ←  replaces VisionTransformer
    Reconstruct          ←  identical to original
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import torch.utils.checkpoint as ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Primitive building blocks  (EfficientViT)
# ─────────────────────────────────────────────────────────────────────────────

class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d.  fuse() merges them for faster inference."""
    def __init__(self, a, b, ks=1, stride=1, pad=0,
                 dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv2d(a, b, ks, stride, pad,
                                        dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = nn.Conv2d(w.size(1) * c.groups, w.size(0), w.shape[2:],
                      stride=c.stride, padding=c.padding,
                      dilation=c.dilation, groups=c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(
                x.size(0), 1, 1, 1, device=x.device
            ).ge_(self.drop).div(1 - self.drop).detach()
        return x + self.m(x)


class FFN_conv(nn.Module):
    """Point-wise FFN with 1×1 convs – operates in BCHW space."""
    def __init__(self, ed, expand=2):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, ed * expand)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = Conv2d_BN(ed * expand, ed, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


# ─────────────────────────────────────────────────────────────────────────────
# CascadedGroupAttention
# ─────────────────────────────────────────────────────────────────────────────

class CascadedGroupAttention(nn.Module):
    """
    Channels are split across heads.  Head i receives:
        feat_i = qkv_i( input_i + output_{i-1} )
    giving a cascaded refinement for near-zero extra cost.

    Works in BCHW format.  Window complexity: O(H·W·w²).
    """
    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4, window_resolution=7, kernels=None):
        super().__init__()
        if kernels is None:
            kernels = [5] * num_heads
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)

        qkvs, dws = [], []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // num_heads, self.key_dim * 2 + self.d))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim,
                                  kernels[i], 1, kernels[i] // 2,
                                  groups=self.key_dim))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws  = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(),
            Conv2d_BN(self.d * num_heads, dim, bn_weight_init=0)
        )

        # learnable relative-position bias at training window size
        points = list(itertools.product(range(window_resolution),
                                         range(window_resolution)))
        N = len(points)
        offsets, idxs = {}, []
        for p1 in points:
            for p2 in points:
                off = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if off not in offsets:
                    offsets[off] = len(offsets)
                idxs.append(offsets[off])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(offsets)))
        self.register_buffer('attention_bias_idxs',
                              torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x: (B, C, H, W)
        B, C, H, W = x.shape
        ab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in  = x.chunk(self.num_heads, dim=1)
        feats_out = []
        feat = feats_in[0]
        for i, qkv_layer in enumerate(self.qkvs):
            if i > 0:
                feat = feat + feats_in[i]
            feat = qkv_layer(feat)
            q, k, v = feat.split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)
            attn = (q.transpose(-2, -1) @ k) * self.scale + ab[i]
            attn = attn.softmax(dim=-1)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)
            feats_out.append(feat)
        return self.proj(torch.cat(feats_out, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# LocalWindowAttention
# ─────────────────────────────────────────────────────────────────────────────

class LocalWindowAttention(nn.Module):
    """
    Partitions (H,W) into non-overlapping windows of size window_resolution,
    applies CGA inside each window, then reverses the partition.

    Cost: O(HW·w²) instead of O((HW)²).  Essential for large images.
    """
    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4, window_resolution=7, kernels=None):
        super().__init__()
        self.window_resolution = window_resolution
        self.attn = CascadedGroupAttention(
            dim, key_dim, num_heads,
            attn_ratio=attn_ratio,
            window_resolution=window_resolution,
            kernels=kernels or [5] * num_heads
        )

    def forward(self, x):
        B, C, H, W = x.shape
        wr = self.window_resolution

        if H <= wr and W <= wr:
            return self.attn(x)

        # pad to multiple of wr
        pad_b = (wr - H % wr) % wr
        pad_r = (wr - W % wr) % wr
        if pad_b > 0 or pad_r > 0:
            x = F.pad(x, (0, pad_r, 0, pad_b))

        pH, pW = H + pad_b, W + pad_r
        nH, nW = pH // wr, pW // wr

        # (B, C, pH, pW) → (B*nH*nW, C, wr, wr)
        x = (x.view(B, C, nH, wr, nW, wr)
              .permute(0, 2, 4, 1, 3, 5)
              .reshape(B * nH * nW, C, wr, wr))
        x = self.attn(x)
        # reverse
        x = (x.view(B, nH, nW, C, wr, wr)
              .permute(0, 3, 1, 4, 2, 5)
              .reshape(B, C, pH, pW))
        if pad_b > 0 or pad_r > 0:
            x = x[:, :, :H, :W].contiguous()
        return x


# ─────────────────────────────────────────────────────────────────────────────
# EfficientViTBlock  (dw → FFN → LWA → dw → FFN sandwich)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientViTBlock(nn.Module):
    """
    One EfficientViT building block.
    Structure: DW-conv → FFN → LocalWindowAttention → DW-conv → FFN
    All residual, operates in BCHW.
    """
    def __init__(self, dim, key_dim, num_heads=4,
                 attn_ratio=4, window_resolution=7, kernels=None):
        super().__init__()
        self.dw0   = Residual(Conv2d_BN(dim, dim, 3, 1, 1,
                                         groups=dim, bn_weight_init=0.))
        self.ffn0  = Residual(FFN_conv(dim, expand=2))
        self.mixer = Residual(LocalWindowAttention(
            dim, key_dim, num_heads, attn_ratio,
            window_resolution, kernels))
        self.dw1   = Residual(Conv2d_BN(dim, dim, 3, 1, 1,
                                         groups=dim, bn_weight_init=0.))
        self.ffn1  = Residual(FFN_conv(dim, expand=2))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# EfficientPatchEmbed  (replaces Embeddings + learnable position table)
# ─────────────────────────────────────────────────────────────────────────────

class EfficientPatchEmbed(nn.Module):
    """
    Convolutional stem that downsamples by `patch_size` using stacked
    stride-2 Conv2d_BN layers.  No fixed-resolution position embedding
    → works at any input size (critical for BTRXD).
    """
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        assert patch_size in (2, 4, 8, 16), "patch_size must be 2/4/8/16"
        layers = []
        ch = in_channels
        while patch_size > 1:
            out = embed_dim if patch_size == 2 else max(embed_dim // (patch_size // 2), 8)
            layers += [Conv2d_BN(ch, out, 3, 2, 1), nn.ReLU(inplace=True)]
            ch = out
            patch_size //= 2
        self.stem = nn.Sequential(*layers)

    def forward(self, x):
        return self.stem(x)   # (B, embed_dim, H/patch_size, W/patch_size)


# ─────────────────────────────────────────────────────────────────────────────
# TextProject  (replaces CTBN3 in original Vit.py)
# ─────────────────────────────────────────────────────────────────────────────

class TextProject(nn.Module):
    """
    Project text tokens (B, T, text_dim) → spatial feature map (B, embed_dim, H, W).
    Uses F.interpolate so it works for any (H, W) — unlike the original
    CTBN3 which is hardcoded to 196 tokens (14×14 = 224/16).
    """
    def __init__(self, text_dim, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(text_dim, embed_dim, 1)

    def forward(self, text, H, W):
        # text: (B, T, text_dim)
        x = self.proj(text.transpose(1, 2))           # (B, embed_dim, T)
        x = x.unsqueeze(-1)                           # (B, embed_dim, T, 1)
        x = F.interpolate(x, size=(H * W, 1),
                          mode='bilinear', align_corners=False)
        x = x.squeeze(-1).view(x.size(0), -1, H, W)  # (B, embed_dim, H, W)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# EfficientViTBranch  –  drop-in replacement for VisionTransformer
# ─────────────────────────────────────────────────────────────────────────────

class EfficientViTBranch(nn.Module):
    """
    Drop-in replacement for VisionTransformer in LViT.

    forward(x, skip_x, text, reconstruct=False) has the same call signature.

    Down path (reconstruct=False):
        x      – CNN feature map   (B, C, H, W)
        skip_x – previous ViT seq  (B, N_prev, C_prev)
        text   – text tokens       (B, T, text_dim)
        →  returns token sequence  (B, N, embed_dim)

    Up / reconstruct path (reconstruct=True):
        x      – current level ViT seq  (B, N, C)
        skip_x – deeper level ViT seq   (B, N_deep, C_deep)
        →  returns updated seq          (B, N, embed_dim)

    Args:
        in_channels   : input feature-map channels (CNN branch)
        embed_dim     : ViT token dimension (== in_channels in LViT)
        patch_size    : spatial stride for patch embed (16/8/4/2)
        depth         : number of EfficientViTBlocks
        num_heads     : attention heads (must divide embed_dim)
        key_dim       : per-head Q/K dimension
        window_size   : local attention window size
        text_dim      : dimension of incoming text tokens
        finest_scale  : True for downVit (scale 0) — return tokens directly,
                        no skip merge (mirrors original LViT behaviour)
        is_bottleneck : True for upVit3 — return tokens directly
        skip_in_dim   : channel dim of skip_x on reconstruct path
    """
    def __init__(self, in_channels, embed_dim, patch_size,
                 depth=1, num_heads=4, key_dim=16,
                 window_size=7, text_dim=64,
                 finest_scale=False, is_bottleneck=False,
                 skip_in_dim=None, use_checkpoint=False):
        super().__init__()
        self.embed_dim    = embed_dim
        self.patch_size   = patch_size
        self.finest_scale = finest_scale
        self.is_bottleneck = is_bottleneck
        self.use_checkpoint = use_checkpoint

        # ── patch embed ───────────────────────────────────────────────────
        self.patch_embed = EfficientPatchEmbed(in_channels, embed_dim, patch_size)

        # ── text → spatial (finest scale only, mirrors original) ──────────
        self.text_proj = TextProject(text_dim, embed_dim) if finest_scale else None

        # ── EfficientViT blocks ───────────────────────────────────────────
        self.blocks = nn.Sequential(*[
            EfficientViTBlock(
                embed_dim, key_dim,
                num_heads=num_heads,
                attn_ratio=max(embed_dim // (key_dim * num_heads), 1),
                window_resolution=window_size
            )
            for _ in range(depth)
        ])

        # ── channel halving for skip merge (non-finest down path) ─────────
        # Mirrors original CTBN: embed_dim → embed_dim//2 → cat(skip) → embed_dim
        if not finest_scale and not is_bottleneck:
            self.chan_half = Conv2d_BN(embed_dim, embed_dim // 2, 1)

        # ── skip projection for reconstruct path (mirrors CTBN2) ──────────
        skip_in = skip_in_dim if skip_in_dim is not None else embed_dim
        self.skip_proj = nn.Linear(skip_in, embed_dim) if skip_in != embed_dim else None

        self.norm = nn.LayerNorm(embed_dim)

    def _run_blocks(self, feat):
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint_sequential(self.blocks, len(self.blocks), feat)
        return self.blocks(feat)
    
    # ── shape helpers ─────────────────────────────────────────────────────
    @staticmethod
    def to_seq(feat):
        """(B, C, H, W) → (B, N, C),  returns (seq, H, W)"""
        B, C, H, W = feat.shape
        return feat.flatten(2).transpose(1, 2), H, W

    @staticmethod
    def to_spatial(seq, H, W):
        """(B, N, C) → (B, C, H, W)"""
        return seq.transpose(1, 2).view(seq.size(0), -1, H, W)

    @staticmethod
    def _hw(seq):
        N = seq.size(1)
        h = w = int(N ** 0.5)
        return h, w

    # ── forward ───────────────────────────────────────────────────────────
    def forward(self, x, skip_x, text, reconstruct=False):
        if not reconstruct:
            # ── Down path ─────────────────────────────────────────────
            feat = self.patch_embed(x)            # (B, embed_dim, H', W')
            B, C, H, W = feat.shape
            self._last_hw = (H, W)  
            # text injection at finest scale only (matches original LViT)
            if self.text_proj is not None:
                feat = feat + self.text_proj(text, H, W)

            feat = self._run_blocks(feat)              # (B, embed_dim, H', W')

            # finest scale (downVit) or bottleneck → return tokens directly
            if self.finest_scale or self.is_bottleneck:
                seq, _, _ = self.to_seq(feat)
                return self.norm(seq)             # (B, N, embed_dim)

            # other scales: halve channels, cat with previous skip (seq)
            # mirrors: CTBN(x) → cat([x_half, skip_x]) in original Vit.py
            feat_half = self.chan_half(feat)      # (B, embed_dim//2, H', W')
            seq_half, _, _ = self.to_seq(feat_half)

            if skip_x.shape[1] != seq_half.shape[1]:
                sk_sp = self.to_spatial(skip_x, *self._get_hw(skip_x))
                sk_sp = F.interpolate(sk_sp, size=(H, W),
                                      mode='bilinear', align_corners=False)
                skip_x, _, _ = self.to_seq(sk_sp)

            merged = torch.cat([seq_half, skip_x], dim=2)  # (B, N, embed_dim)
            return self.norm(merged)

        else:
            # ── Reconstruct / Up path ─────────────────────────────────
            # mirrors: Encoder_blocks(x)  +  CTBN2(skip_x)  →  x + skip
            h, w = self._last_hw if hasattr(self, '_last_hw') else self._hw(x)
            x_sp = self.blocks(self.to_spatial(x, h, w))  # (B, C, h, w)
            x, _, _ = self.to_seq(x_sp)                   # (B, N, embed_dim)

            skip = skip_x
            if self.skip_proj is not None:
                skip = self.skip_proj(skip)
            # align token count if deeper level has different resolution
            if skip.shape[1] != x.shape[1]:
                skip_h, skip_w = self._get_hw(skip_x)
                sk_sp = self.to_spatial(skip, skip_h, skip_w)
                sk_sp = F.interpolate(sk_sp, size=(h, w),
                                      mode='bilinear', align_corners=False)
                skip, _, _ = self.to_seq(sk_sp)

            return self.norm(x + skip)            # (B, N, embed_dim)

    @staticmethod
    def _get_hw(seq):
        """Lấy H,W từ _last_hw nếu có, không thì fallback sqrt."""
        N = seq.size(1)
        s = int(N ** 0.5)
        return s, s   # chỉ dùng khi không có _last_hw

# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct  (identical to original Vit.py)
# ─────────────────────────────────────────────────────────────────────────────

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x, hw=None):
        if x is None:
            return None
        B, n_patch, hidden = x.size()
        if hw is not None:
            h, w = hw
        else:
            h = w = int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        x = F.interpolate(x, scale_factor=self.scale_factor,
                          mode='bilinear', align_corners=False)
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
