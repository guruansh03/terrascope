"""Backbone wrappers with a common interface.

Every encoder takes PHYSICAL-UNIT tensors (so ONNX graphs include the normalisation):
    optical (B,6,H,W) reflectance 0..1 [B2,B3,B4,B8,B11,B12]
    sar     (B,2,H,W) dB [VV,VH]
    dem     (B,2,H,W) [elevation m, slope deg]
    avail   (B,3) float/bool  [optical, sar, dem] availability (batch-uniform for TerraMind)
and returns ``(levels, raw)``: ``levels`` = list of 4 feature maps (B,D,h,w) (stride = patch),
``raw`` = the model's normalised input stack (B,C_raw,H,W) used by the decoder's full-resolution stem.
"""
from __future__ import annotations

import logging, os, warnings

import torch
import torch.nn as nn

from .specs import MODELS

log = logging.getLogger("terrascope.fm.encoders")


def terratorch_registry():
    """Import terratorch's backbone registry WITHOUT keeping its global side effect: importing terratorch
    pulls in ``clay_v15`` which calls ``torch.set_float32_matmul_precision("medium")`` (bf16 matmul/conv on
    CPUs with oneDNN bf16 -> ~1e-3 relative error process-wide). We restore the previous precision."""
    prev = torch.get_float32_matmul_precision()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from terratorch.registry import BACKBONE_REGISTRY
    if torch.get_float32_matmul_precision() != prev:
        torch.set_float32_matmul_precision(prev)
    return BACKBONE_REGISTRY


def _tokens_to_map(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    B, N, D = t.shape
    return t.transpose(1, 2).reshape(B, D, h, w)


# ---------------------------------------------------------------------------------------------- Prithvi
class PrithviEncoder(nn.Module):
    def __init__(self, model_id: str, weights_path: str | None = None):
        super().__init__()
        BACKBONE_REGISTRY = terratorch_registry()
        from terratorch.models.backbones import prithvi_vit as pv
        spec = MODELS[model_id]
        self.model_id = model_id
        self.patch = spec["patch"]
        self.out_layers = list(spec["out_layers"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = BACKBONE_REGISTRY.build(
                spec["backbone"], pretrained=bool(weights_path), ckpt_path=weights_path, num_frames=1,
                bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"])
        self.dim = int(self.backbone.embed_dim)
        self.dims = [self.dim] * len(self.out_layers)
        self.register_buffer("mean", torch.tensor(pv.PRITHVI_V2_MEAN, dtype=torch.float32).view(1, 6, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(pv.PRITHVI_V2_STD, dtype=torch.float32).view(1, 6, 1, 1), persistent=False)
        self.raw_ch = 6
        self.required = ("optical",)

    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def forward(self, optical, sar, dem, avail):
        x = (optical * 10000.0 - self.mean) / self.std
        H, W = x.shape[-2:]
        feats = self.backbone.forward_features(x)
        h, w = H // self.patch, W // self.patch
        levels = [_tokens_to_map(feats[i][:, 1:, :], h, w) for i in self.out_layers]  # drop cls token
        return levels, x


# ---------------------------------------------------------------------------------------------- TerraMind
class TerraMindEncoder(nn.Module):
    """TerraMind-1.0 raw-modality encoder (S2 6-band subset + S1GRD + DEM elevation), merge = mean."""

    def __init__(self, model_id: str = "terramind_tiny", weights_path: str | None = None, optical_source: str = "l1c",
                 use_sar: bool = True, use_dem: bool = True):
        super().__init__()
        BACKBONE_REGISTRY = terratorch_registry()
        from terratorch.models.backbones.terramind.model import terramind_register as tr
        from ..real.bands import TERRATORCH_NAMES_HLS, TERRATORCH_NAMES_L1C
        spec = MODELS[model_id]
        self.model_id = model_id
        self.patch = spec["patch"]
        self.out_layers = list(spec["out_layers"])
        self.s2_mod = "S2L1C" if optical_source == "l1c" else "S2L2A"
        s2_key = "untok_sen2l1c@224" if optical_source == "l1c" else "untok_sen2l2a@224"
        names = TERRATORCH_NAMES_L1C if optical_source == "l1c" else TERRATORCH_NAMES_HLS
        self.optical_source = optical_source
        self.mods = [self.s2_mod] + (["S1GRD"] if use_sar else []) + (["DEM"] if use_dem else [])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = BACKBONE_REGISTRY.build(spec["backbone"], pretrained=False, modalities=self.mods,
                                                    merge_method="mean")
            if weights_path:
                # mirror terratorch's pretrained=True path (filter -> strict load) but from a local file
                sd = torch.load(weights_path, map_location="cpu", weights_only=True)
                if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]
                sd = tr.checkpoint_filter_fn(sd, self.backbone)
                self.backbone.load_state_dict(sd, strict=True)
            # 6-band subset of the pretrained S2 patch embedding (weights copied per band name)
            self.backbone = tr.select_modality_patch_embed_weights(self.backbone, {self.s2_mod: list(names)},
                                                                   tr.PRETRAINED_BANDS)
        idx = [tr.PRETRAINED_BANDS[s2_key].index(n) for n in names]
        mean, std = tr.v1_pretraining_mean, tr.v1_pretraining_std
        self.register_buffer("s2_mean", torch.tensor([mean[s2_key][i] for i in idx]).view(1, 6, 1, 1), persistent=False)
        self.register_buffer("s2_std", torch.tensor([std[s2_key][i] for i in idx]).view(1, 6, 1, 1), persistent=False)
        self.register_buffer("s1_mean", torch.tensor(mean["untok_sen1grd@224"]).view(1, 2, 1, 1), persistent=False)
        self.register_buffer("s1_std", torch.tensor(std["untok_sen1grd@224"]).view(1, 2, 1, 1), persistent=False)
        self.register_buffer("dem_mean", torch.tensor(mean["untok_dem@224"]).view(1, 1, 1, 1), persistent=False)
        self.register_buffer("dem_std", torch.tensor(std["untok_dem@224"]).view(1, 1, 1, 1), persistent=False)
        self.use_sar, self.use_dem = use_sar, use_dem
        self.dim = int(self.backbone.encoder_norm.weight.shape[0]) if hasattr(self.backbone, "encoder_norm") else 192
        self.dims = [self.dim] * len(self.out_layers)
        self.raw_ch = 9
        self.required = ()  # any non-empty subset of its modalities works
        # Python-level modality presence override (export / tracing): None -> decided from ``avail``
        self.force_present: tuple | None = None

    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.backbone.parameters())

    def forward(self, optical, sar, dem, avail):
        s2 = (optical * 10000.0 - self.s2_mean) / self.s2_std
        s1 = (sar - self.s1_mean) / self.s1_std
        de = (dem[:, :1] - self.dem_mean) / self.dem_std
        if self.force_present is not None:
            present = self.force_present
        else:
            a = avail.bool().all(0)  # batch-uniform presence (TerraMind takes a dict of present modalities)
            present = tuple(bool(v) for v in a.tolist())
        d = {}
        if present[0]:
            d[self.s2_mod] = s2
        if self.use_sar and present[1]:
            d["S1GRD"] = s1
        if self.use_dem and present[2]:
            d["DEM"] = de
        if not d:
            raise ValueError("TerraMind needs at least one of its modalities")
        H, W = optical.shape[-2:]
        feats = self.backbone(d)
        h, w = H // self.patch, W // self.patch
        levels = [_tokens_to_map(feats[i], h, w) for i in self.out_layers]
        used = (present[0], self.use_sar and present[1], self.use_dem and present[2])
        pm = torch.tensor([float(p) for p in used], device=optical.device)
        raw = torch.cat([s2 * pm[0], s1 * pm[1], de * pm[2]], 1)
        return levels, raw


# ---------------------------------------------------------------------------------------------- TerraFM
class TerraFMEncoder(nn.Module):
    """Encoder half of the project's TerraFM (sim-pretrained MAE weights when available)."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        from ..core import model as cm
        cfg = dict(dim=96, depth=4, heads=4, dec_dim=64, dec_depth=2, patch=4)
        sd = None
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=True)
            cfg = dict(ck.get("cfg", cfg))
            sd = ck.get("model")
        import inspect
        ok = set(inspect.signature(cm.TerraFM.__init__).parameters)
        m = cm.TerraFM(**{k: v for k, v in cfg.items() if k in ok})
        if sd is not None:
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing:
                log.warning("TerraFM encoder: %d missing keys when loading %s", len(missing), weights_path)
        # keep only the encoder parts
        self.embed, self.tag, self.blocks, self.norm = m.embed, m.tag, m.blocks, m.norm
        self.patch, self.dim = int(m.patch), int(m.dim)
        self.norm_version = getattr(m, "norm_version", "v1")
        self._normalize = cm.normalize
        self._sincos = cm.sincos_2d
        self.mod_slices = cm.MOD_SLICES
        self.mods = cm.MODS
        depth = len(self.blocks)
        self.out_layers = list(range(depth)) if depth <= 4 else [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        self.dims = [self.dim] * len(self.out_layers)
        self.raw_ch = 10
        self.required = ()
        self.cfg = cfg

    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, optical, sar, dem, avail):
        try:
            x = self._normalize(optical, sar, dem, self.norm_version)
        except TypeError:
            x = self._normalize(optical, sar, dem)
        av = avail.float()
        cols = []
        for i, m in enumerate(self.mods):
            c0, c1 = self.mod_slices[m]
            cols.append(av[:, i:i + 1].expand(-1, c1 - c0))
        x = x * torch.cat(cols, 1)[:, :, None, None]
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        e = 0
        for i, m in enumerate(self.mods):
            c0, c1 = self.mod_slices[m]
            t = self.embed[m](x[:, c0:c1]).flatten(2).transpose(1, 2) + self.tag[i]
            e = e + t * av[:, i].view(-1, 1, 1)
        t = e / av.sum(1).clamp(min=1).view(-1, 1, 1) + self._sincos(self.dim, h, w, x.device)
        outs = []
        for b in self.blocks:
            t = b(t)
            outs.append(t)
        outs[-1] = self.norm(outs[-1])
        levels = [_tokens_to_map(outs[i], h, w) for i in self.out_layers]
        return levels, x


# ---------------------------------------------------------------------------------------------- factory
def build_encoder(model_id: str, task: str | None = None, weights_path: str | None = None, **kw) -> nn.Module:
    from .specs import TASK_OPTICAL_SOURCE, TASK_SENSORS
    fam = MODELS[model_id]["family"]
    if fam == "prithvi":
        return PrithviEncoder(model_id, weights_path)
    if fam == "terramind":
        sens = kw.get("task_sensors") or (TASK_SENSORS.get(task) if task else ["optical", "sar", "dem"])
        return TerraMindEncoder(model_id, weights_path, optical_source=kw.get("optical_source") or
                                TASK_OPTICAL_SOURCE.get(task, "l1c"),
                                use_sar="sar" in sens, use_dem="dem" in sens)
    if fam == "terrafm":
        return TerraFMEncoder(weights_path)
    if fam == "unet":
        from .baselines import UnetEncoder
        return UnetEncoder(model_id, task or "flood", weights_path, task_sensors=kw.get("task_sensors"))
    raise KeyError(model_id)
