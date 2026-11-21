"""Supervised baselines: U-Net with a ResNet-18 encoder (segmentation_models_pytorch), trained with the SAME
protocol as the foundation models (crops, epochs, early stopping, loss, tiled evaluation) but with smp's own
U-Net decoder instead of the shared UPerLite head ("head-free" w.r.t. our decoder).

* ``unet_r18_scratch``  - random initialisation
* ``unet_r18_imagenet`` - ResNet-18 ImageNet weights from smp's own Hugging Face repo
  (``smp-hub/resnet18.imagenet`` at the revision pinned by smp 0.6.0, file ``model.safetensors``), cached in
  ``data/hf_cache/hub`` by ``get_data.bat weights``.

Input = the same channels TerraFM sees for the task, with TerraFM's normalisation (v2: reflectance (x-0.2)/0.2,
SAR (dB+12)/8, DEM as relative elevation + HAND-like height):
    flood / landcover: 6 optical + VV,VH + 2 DEM = 10 channels;  burn: 6 optical;  cloud: 6 optical + VV,VH.
Absent sensors are zeroed (as TerraFM does).

First-conv adaptation for ImageNet weights (documented choice): the pretrained conv1 kernel is (64,3,7,7) in RGB
order. Our B4 (red), B3 (green), B2 (blue) slots receive the R, G, B kernels; every other channel (NIR, SWIR1/2,
VV, VH, DEM) receives the mean of the three RGB kernels; the whole kernel is then scaled by 3/C so the expected
activation magnitude matches the 3-channel network. All other ResNet-18 weights are loaded unchanged (fc dropped).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .specs import MODELS, TASK_OUT, TASK_SENSORS

IMAGENET_REPO = "smp-hub/resnet18.imagenet"
IMAGENET_FILE = "model.safetensors"
# the revision smp 0.6.0 pins in segmentation_models_pytorch/encoders/resnet.py (read from the installed package)
IMAGENET_REVISION = "3f2325ff978283d47aa6a1d6878ca20565622683"


def adapt_first_conv(w_rgb: torch.Tensor, channels: list[str]) -> torch.Tensor:
    """(64,3,k,k) RGB kernel -> (64,C,k,k) for our channel list (see module docstring)."""
    rgb = {"B4": 0, "B3": 1, "B2": 2}
    mean = w_rgb.mean(1)
    out = torch.stack([w_rgb[:, rgb[c]] if c in rgb else mean for c in channels], 1)
    return out * (3.0 / len(channels))


def normalize_v2(optical, sar, dem):
    """TerraFM's v2 input normalisation (terrascope.core.model.normalize(..., "v2")), re-implemented with an
    ONNX-exportable median (sort + lower-middle element == torch.median). A test asserts equality with core."""
    import torch.nn.functional as F
    from ..core.model import HAND_WIN
    o = (optical.float() - 0.2) / 0.2
    s = (sar.float() + 12.0) / 8.0
    e = dem[:, :1].float()
    flat = e.flatten(1)
    med = torch.sort(flat, dim=1).values[:, (flat.shape[1] - 1) // 2].view(-1, 1, 1, 1)
    rel = (e - med) / 150.0
    k = HAND_WIN
    lm = -F.max_pool2d(-e, k, 1, k // 2)
    lm = F.avg_pool2d(F.pad(lm, (3, 3, 3, 3), mode="replicate"), 7, 1)
    hand = (e - lm).clamp(min=0) / 50.0
    return torch.cat([o, s, rel.clamp(-8, 8), hand.clamp(0, 8)], 1)


class UnetEncoder(nn.Module):
    """ResNet-18 encoder of an smp U-Net; ``make_head`` hands over smp's decoder + segmentation head."""

    family = "unet"

    def __init__(self, model_id: str, task: str, weights_path: str | None = None, task_sensors=None):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.model_id = model_id
        sens = list(task_sensors or [s for s in MODELS[model_id]["sensors"] if s in TASK_SENSORS[task]])
        self.use_sar, self.use_dem = "sar" in sens, "dem" in sens
        self.channels = ["B2", "B3", "B4", "B8", "B11", "B12"] + (["VV", "VH"] if self.use_sar else []) + \
                        (["elev_rel", "hand"] if self.use_dem else [])
        C = len(self.channels)
        unet = smp.Unet("resnet18", encoder_weights=None, in_channels=C, classes=TASK_OUT[task])
        if weights_path:
            from safetensors.torch import load_file
            sd = load_file(weights_path, device="cpu")
            sd["conv1.weight"] = adapt_first_conv(sd["conv1.weight"].float(), self.channels)
            unet.encoder.load_state_dict(sd)  # smp's ResNetEncoder.load_state_dict drops fc.*
        self.enc = unet.encoder
        self._pending_head = [unet.decoder, unet.segmentation_head]  # plain list: not registered here
        self.patch = 32  # deepest feature stride; inputs must be multiples of 32
        self.dims = list(self.enc.out_channels)
        self.raw_ch = C
        self.required = ("optical",)
        self.task_sensors = sens

    def make_head(self, n_out: int):
        dec, seg = self._pending_head
        return UnetHead(dec, seg)

    def encoder_params(self) -> int:
        """Whole U-Net (encoder + smp decoder + 1x1 head): the baseline has no separate task head."""
        n = sum(p.numel() for p in self.enc.parameters())
        for m in self._pending_head:
            n += sum(p.numel() for p in m.parameters())
        return int(n)

    def forward(self, optical, sar, dem, avail):
        x = normalize_v2(optical, sar, dem)
        av = avail.float()
        parts = [x[:, 0:6] * av[:, 0:1, None, None]]
        if self.use_sar:
            parts.append(x[:, 6:8] * av[:, 1:2, None, None])
        if self.use_dem:
            parts.append(x[:, 8:10] * av[:, 2:3, None, None])
        x = torch.cat(parts, 1)
        return list(self.enc(x)), x


class UnetHead(nn.Module):
    def __init__(self, decoder, seg_head):
        super().__init__()
        self.decoder, self.segmentation_head = decoder, seg_head

    def forward(self, levels, raw, avail):
        return self.segmentation_head(self.decoder(list(levels)))
