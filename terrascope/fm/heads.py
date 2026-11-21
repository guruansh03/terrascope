"""Light task decoders shared by every backbone ("UPerNet-lite").

levels (4 x (B,D,h,w), stride = patch) -> lateral 1x1 convs -> concat + 3x3 fuse -> log2(patch) x
[x2 bilinear upsample + 3x3 conv + GroupNorm + GELU] -> concat full-resolution stem of the normalised raw input
(+ availability flags) -> 3x3 conv -> 1x1 output conv.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm2d(nn.Module):
    """GroupNorm written as centre-then-variance with plain ops (same maths as nn.GroupNorm); exports to simple
    ONNX ReduceMean/Sub/Mul nodes that every runtime supports."""

    def __init__(self, groups: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.g, self.eps = groups, eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.reshape(B, self.g, -1)
        y = y - y.mean(-1, keepdim=True)
        y = y / torch.sqrt((y * y).mean(-1, keepdim=True) + self.eps)
        return y.reshape(B, C, H, W) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _cbr(i, o, k=3):
    g = 8 if o % 8 == 0 else 1
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), GroupNorm2d(g, o), nn.GELU())


class UPerLite(nn.Module):
    def __init__(self, in_dims: list[int], patch: int, n_out: int, raw_ch: int, width: int = 128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(d, width, 1) for d in in_dims])
        self.fuse = _cbr(width * len(in_dims), width)
        n_up = int(round(math.log2(patch)))
        if 2 ** n_up != patch:
            raise ValueError(f"patch size {patch} must be a power of two")
        chans = [width]
        for i in range(n_up):
            chans.append(max(32, chans[-1] * 3 // 4 if i < n_up - 1 else 32))
        self.ups = nn.ModuleList([_cbr(chans[i], chans[i + 1]) for i in range(n_up)])
        self.stem = nn.Sequential(_cbr(raw_ch + 3, 16), _cbr(16, 16))
        self.out = nn.Sequential(_cbr(chans[-1] + 16, 32), nn.Conv2d(32, n_out, 1))

    def forward(self, levels, raw, avail):
        f = torch.cat([l(x) for l, x in zip(self.lateral, levels)], 1)
        f = self.fuse(f)
        for up in self.ups:
            f = up(F.interpolate(f, scale_factor=2.0, mode="bilinear", align_corners=False))
        H, W = raw.shape[-2:]
        if f.shape[-2:] != (H, W):
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
        a = avail.float()[:, :, None, None].expand(-1, -1, H, W)
        s = self.stem(torch.cat([raw, a], 1))
        return self.out(torch.cat([f, s], 1))
