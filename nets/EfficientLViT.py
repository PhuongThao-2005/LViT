# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from .EfficientViT import EfficientViTBranch, Reconstruct
from .pixlevel import PixLevelModule


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpblockAttention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pixModule = PixLevelModule(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.pixModule(skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class LViT(nn.Module):
    """
    EfficientLViT – LViT with EfficientViT-powered transformer branch.

    Constructor signature is identical to the original LViT so train_model.py
    needs zero changes.

    Extra hyper-params (with sensible defaults) exposed via config:
        config.window_size   (int, default 7)   – local attention window
        config.vit_depth     (int, default 1)   – EfficientViTBlocks per scale
        config.vit_num_heads (int, default 4)   – attention heads
        config.vit_key_dim   (int, default 16)  – per-head Q/K dimension
    """

    def __init__(self, config, n_channels=3, n_classes=1, img_size=224, vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes  = n_classes

        C = config.base_channel   # 64

        # ── read optional EfficientViT hyper-params from config ───────────
        window_size = getattr(config, 'window_size',   7)
        depth       = getattr(config, 'vit_depth',     1)
        num_heads   = getattr(config, 'vit_num_heads', 4)
        key_dim     = getattr(config, 'vit_key_dim',   16)

        self.inc   = ConvBatchNorm(n_channels, C)
        self.downVit  = EfficientViTBranch(
            in_channels=C,     embed_dim=C,     patch_size=16,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C,
            finest_scale=True,   # scale-0: inject text, return tokens directly
        )
        self.downVit1 = EfficientViTBranch(
            in_channels=C * 2, embed_dim=C * 2, patch_size=8,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 2,
        )
        self.downVit2 = EfficientViTBranch(
            in_channels=C * 4, embed_dim=C * 4, patch_size=4,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 4,
        )
        self.downVit3 = EfficientViTBranch(
            in_channels=C * 8, embed_dim=C * 8, patch_size=2,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 8,
        )
        self.upVit  = EfficientViTBranch(
            in_channels=C,     embed_dim=C,     patch_size=16,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C,
            skip_in_dim=C * 2,              # receives tokens from upVit1 (C*2)
        )
        self.upVit1 = EfficientViTBranch(
            in_channels=C * 2, embed_dim=C * 2, patch_size=8,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 2,
            skip_in_dim=C * 4,              # receives tokens from upVit2 (C*4)
        )
        self.upVit2 = EfficientViTBranch(
            in_channels=C * 4, embed_dim=C * 4, patch_size=4,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 4,
            skip_in_dim=C * 8,              # receives tokens from upVit3 (C*8)
        )
        self.upVit3 = EfficientViTBranch(
            in_channels=C * 8, embed_dim=C * 8, patch_size=2,
            depth=depth, num_heads=num_heads, key_dim=key_dim,
            window_size=window_size, text_dim=C * 8,
            is_bottleneck=True,              # deepest: skip_x == self, no merge
            skip_in_dim=C * 8,
        )
        self.down1 = DownBlock(C,     C * 2, nb_Conv=2)
        self.down2 = DownBlock(C * 2, C * 4, nb_Conv=2)
        self.down3 = DownBlock(C * 4, C * 8, nb_Conv=2)
        self.down4 = DownBlock(C * 8, C * 8, nb_Conv=2)
        self.up4 = UpblockAttention(C * 16, C * 4, nb_Conv=2)
        self.up3 = UpblockAttention(C * 8,  C * 2, nb_Conv=2)
        self.up2 = UpblockAttention(C * 4,  C,     nb_Conv=2)
        self.up1 = UpblockAttention(C * 2,  C,     nb_Conv=2)

        self.outc = nn.Conv2d(C, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation  = nn.Sigmoid()   # for BCELoss
        self.multi_activation = nn.Softmax()

        # ── Reconstruct heads ──────────────────────────────────────────────
        self.reconstruct1 = Reconstruct(in_channels=C,     out_channels=C,     kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(in_channels=C * 2, out_channels=C * 2, kernel_size=1, scale_factor=(8,  8))
        self.reconstruct3 = Reconstruct(in_channels=C * 4, out_channels=C * 4, kernel_size=1, scale_factor=(4,  4))
        self.reconstruct4 = Reconstruct(in_channels=C * 8, out_channels=C * 8, kernel_size=1, scale_factor=(2,  2))

        # ── Pixel-level attention ───────────────────────────────────────────
        self.pix_module1 = PixLevelModule(C)
        self.pix_module2 = PixLevelModule(C * 2)
        self.pix_module3 = PixLevelModule(C * 4)
        self.pix_module4 = PixLevelModule(C * 8)

        # ── Text projections ────────────────────────────────────────────────
        self.text_module4 = nn.Conv1d(in_channels=768,   out_channels=C * 8, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=C * 8, out_channels=C * 4, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=C * 4, out_channels=C * 2, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=C * 2, out_channels=C,     kernel_size=3, padding=1)

    def forward(self, x, text):
        """
        x    : (B, n_channels, H, W)
        text : (B, T, 768)  – BERT token embeddings
        """
        x = x.float()
        x1 = self.inc(x)       # (B, C,   H,    W)

        # ── text channel projections ────────────────────────────────────────
        text4 = self.text_module4(text.transpose(1, 2)).transpose(1, 2)   # (B,T,C*8)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)  # (B,T,C*4)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)  # (B,T,C*2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)  # (B,T,C)

        y1 = self.downVit (x1, x1, text1)          # (B, N1, C) 
        x2 = self.down1(x1)                         # (B, C*2, H/2,  W/2)
        y2 = self.downVit1(x2, y1, text2)           # (B, N2, C*2)
        x3 = self.down2(x2)                         # (B, C*4, H/4,  W/4)
        y3 = self.downVit2(x3, y2, text3)           # (B, N3, C*4)
        x4 = self.down3(x3)                         # (B, C*8, H/8,  W/8)
        y4 = self.downVit3(x4, y3, text4)           # (B, N4, C*8)
        x5 = self.down4(x4)                         # (B, C*8, H/16, W/16)

        # ── ViT up path / reconstruct ───────────────────────────────────────
        y4 = self.upVit3(y4, y4,  text4, reconstruct=True)
        y3 = self.upVit2(y3, y4,  text3, reconstruct=True)
        y2 = self.upVit1(y2, y3,  text2, reconstruct=True)
        y1 = self.upVit (y1, y2,  text1, reconstruct=True)

        # ── seq → spatial, residual add ─────────────────────────────────────
        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4

        # ── CNN decoder ──────────────────────────────────────────────────────
        x = self.up4(x5, x4)
        x = self.up3(x,  x3)
        x = self.up2(x,  x2)
        x = self.up1(x,  x1)

        # ── output ────────────────────────────────────────────────────────
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        return logits
