"""Supervised baseline: a plain U-Net trained from scratch on the same labelled scenes as TerraFM.

Input = the same normalised 10 channels (missing sensors zeroed) + 3 sensor-availability channels, i.e.
exactly what TerraFM's task heads see. Outputs the same task dict (lc logits, flood logit, veg in [0,1],
cloud logit). No reconstruction decoder (it is not a masked autoencoder), so no cloud-gap RMSE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import N_CH, N_CLASSES, channel_mask, normalize


def _block(cin, cout, groups=8):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(groups, cout), nn.GELU())


class UNet(nn.Module):
    """4-level U-Net (3 down-samplings; H, W must be multiples of 8). base=32 -> 1,931,561 parameters (2.8x TerraFM)."""

    def __init__(self, base=32, norm="v2", cloud_head=True, arch="unet"):
        super().__init__()
        self.norm_version = norm
        self.has_cloud_head = bool(cloud_head)
        b = base
        self.e1, self.e2, self.e3 = _block(N_CH + 3, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.mid = _block(4 * b, 8 * b)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2), _block(8 * b, 4 * b)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2), _block(4 * b, 2 * b)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * b, b, 2, 2), _block(2 * b, b)
        self.lc = nn.Conv2d(b, N_CLASSES, 1)
        self.flood = nn.Conv2d(b, 1, 1)
        self.veg = nn.Conv2d(b, 1, 1)
        self.cloud = nn.Conv2d(b, 1, 1) if cloud_head else None

    def normalize(self, optical, sar, dem):
        return normalize(optical, sar, dem, self.norm_version)

    def forward_tasks(self, x, avail, with_feats=False):
        x = x * channel_mask(avail)
        B, _, H, W = x.shape
        if H % 8 or W % 8:
            raise ValueError("UNet needs H and W to be multiples of 8")
        h = torch.cat([x, avail.float()[:, :, None, None].expand(-1, -1, H, W)], 1)
        s1 = self.e1(h)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        m = self.mid(F.max_pool2d(s3, 2))
        f = self.d3(torch.cat([self.u3(m), s3], 1))
        f = self.d2(torch.cat([self.u2(f), s2], 1))
        f = self.d1(torch.cat([self.u1(f), s1], 1))
        out = dict(lc=self.lc(f), flood=self.flood(f)[:, 0], veg=torch.sigmoid(self.veg(f)[:, 0]))
        if self.cloud is not None:
            out["cloud"] = self.cloud(f)[:, 0]
        return out

    # evaluate(recon=False) only calls forward_tasks; forward_all kept for API symmetry (no "recon" key)
    def forward_all(self, x, avail):
        return self.forward_tasks(x, avail)
