"""TaskModel = backbone encoder + UPerLite head for one task; checkpoints; tiled inference.

Checkpoint (``artifacts/ckpt/real/<task>/<model>_s<seed>.pt``)::
    {"format": "terrascope-fm-1", "model_id", "task", "seed", "encoder_frozen": bool,
     "build": {kwargs needed to rebuild the encoder without pretrained weights},
     "head": state_dict, "encoder": state_dict (fp16) | None (frozen -> rebuilt from the pretrained weights),
     "val": {...best validation metrics...}, "epoch", "dataset", "created_at"}
plus a sidecar ``<model>_s<seed>.json`` with the same metadata and the per-epoch history.
"""
from __future__ import annotations

import json, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import build_encoder
from .heads import UPerLite
from .specs import MODELS, TASK_OUT, TASK_SENSORS, TILE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT_ROOT = os.environ.get("TERRASCOPE_REAL_CKPT", os.path.join(ROOT, "artifacts", "ckpt", "real"))
SENSOR_ORDER = ("optical", "sar", "dem")


def ckpt_path(task: str, model_id: str, seed: int, root: str | None = None) -> str:
    return os.path.join(root or CKPT_ROOT, task, f"{model_id}_s{seed}.pt")


class TaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, task: str, model_id: str, build: dict | None = None):
        super().__init__()
        self.encoder = encoder
        self.task = task
        self.model_id = model_id
        self.patch = int(encoder.patch)
        if hasattr(encoder, "make_head"):  # baselines bring their own decoder (smp U-Net)
            self.head = encoder.make_head(TASK_OUT[task])
        else:
            self.head = UPerLite(encoder.dims, self.patch, TASK_OUT[task], encoder.raw_ch)
        self.build_kw = build or {}

    def encode(self, optical, sar, dem, avail):
        return self.encoder(optical, sar, dem, avail)

    def decode(self, levels, raw, optical, avail):
        y = self.head(levels, raw, avail)
        if self.task == "cloud":
            return (optical + y).clamp(0.0, 1.5)  # residual reconstruction, reflectance
        return y  # logits

    def forward(self, optical, sar, dem, avail):
        levels, raw = self.encode(optical, sar, dem, avail)
        return self.decode(levels, raw, optical, avail)

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def model_sensors(model_id: str, task: str) -> list[str]:
    """Sensors the (model, task) pair actually consumes."""
    return [s for s in MODELS[model_id]["sensors"] if s in TASK_SENSORS[task]]


def build_task_model(model_id: str, task: str, weights_path: str | None = None, **enc_kw) -> TaskModel:
    enc = build_encoder(model_id, task, weights_path, **enc_kw)
    build = dict(enc_kw)
    if hasattr(enc, "optical_source"):
        build["optical_source"] = enc.optical_source
    if MODELS[model_id]["family"] in ("terramind", "unet"):
        build["task_sensors"] = [s for s, u in (("optical", True), ("sar", enc.use_sar), ("dem", enc.use_dem)) if u]
    if MODELS[model_id]["family"] == "terrafm":
        build["terrafm_cfg"] = getattr(enc, "cfg", None)
    return TaskModel(enc, task, model_id, build)


def _half(sd):
    return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}


def save_checkpoint(path: str, model: TaskModel, seed: int, encoder_frozen: bool, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ck = {
        "format": "terrascope-fm-1", "model_id": model.model_id, "task": model.task, "seed": int(seed),
        "encoder_frozen": bool(encoder_frozen), "build": model.build_kw,
        "head": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        "encoder": None if encoder_frozen else _half({k: v.detach().cpu() for k, v in model.encoder.state_dict().items()}),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta,
    }
    tmp = path + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, weights_path: str | None = None, device="cpu") -> TaskModel:
    """Rebuild a TaskModel from a fine-tuned checkpoint. Frozen-encoder checkpoints need ``weights_path``
    (the pretrained backbone file) because they do not store the encoder."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "terrascope-fm-1":
        raise ValueError(f"{path}: not a TerraScope FM checkpoint")
    build = dict(ck.get("build") or {})
    build.pop("terrafm_cfg", None)
    if ck["encoder"] is None and not weights_path:
        raise FileNotFoundError("frozen-encoder checkpoint needs the pretrained backbone weights")
    wp = weights_path if ck["encoder"] is None else None
    if MODELS[ck["model_id"]]["family"] == "terrafm" and ck["encoder"] is None:
        wp = weights_path
    m = build_task_model(ck["model_id"], ck["task"], wp, **build)
    if ck["encoder"] is not None:
        sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["encoder"].items()}
        m.encoder.load_state_dict(sd, strict=True)
    m.head.load_state_dict(ck["head"], strict=True)
    m.meta = {k: v for k, v in ck.items() if k not in ("head", "encoder")}
    return m.to(device).eval()


# ------------------------------------------------------------------------------------------- tiling
def tile_starts(n: int, tile: int, overlap: int, align: int) -> list[int]:
    if n <= tile:
        return [0]
    stride = max(align, tile - overlap)
    k = math.ceil((n - tile) / stride) + 1
    pos = np.linspace(0, n - tile, k)
    out = sorted({int(p // align * align) for p in pos[:-1]} | {n - tile})
    return out


def _w1d(t: int, device) -> torch.Tensor:
    """Sine taper (floor 0.05) for blending overlapping tiles."""
    return 0.05 + 0.95 * torch.sin(torch.linspace(0, math.pi, t, device=device)).clamp(min=0)


def to_tensors(sc: dict, sensors, device="cpu"):
    """Scene dict (numpy) -> batch-of-one tensors + avail (1,3). Missing arrays become zeros."""
    o = torch.from_numpy(np.ascontiguousarray(sc["optical"], dtype=np.float32))[None]
    H, W = o.shape[-2:]

    def get(k, c):
        a = sc.get(k)
        if a is None:
            return torch.zeros(1, c, H, W)
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None]

    s, d = get("sar", 2), get("dem", 2)
    av = torch.tensor([[bool(x) for x in sensors]], dtype=torch.float32)
    return o.to(device), s.to(device), d.to(device), av.to(device)


@torch.no_grad()
def predict_tiled(models: dict[str, TaskModel], optical, sar, dem, avail, tile: int = TILE, overlap: int = 64,
                  batch: int = 4, on_stage=lambda s: None, want_feats: bool = True, amp_dtype=None,
                  pad_to_tile: bool = False):
    """Run several TaskModels on one (1,C,H,W) scene with tiling. Calls on_stage('encoding') before the first
    backbone forward and on_stage('decoding') before the first head. Returns {task: (C,H,W) tensor}, feats."""
    any_m = next(iter(models.values()))
    align = max(m.patch for m in models.values())
    H, W = optical.shape[-2:]
    Hp, Wp = math.ceil(H / align) * align, math.ceil(W / align) * align
    if pad_to_tile:  # fixed-shape graphs (ONNX): never run a tile smaller than ``tile``
        Hp, Wp = max(Hp, tile), max(Wp, tile)
    pad = (0, Wp - W, 0, Hp - H)
    xs = [F.pad(t, pad, mode="replicate") if (Hp != H or Wp != W) else t for t in (optical, sar, dem)]
    th, tw = min(tile, Hp), min(tile, Wp)
    ys = tile_starts(Hp, th, overlap, align)
    xs_ = tile_starts(Wp, tw, overlap, align)
    boxes = [(y, x) for y in ys for x in xs_]
    dev = optical.device
    ctx = torch.autocast(device_type=dev.type, dtype=amp_dtype) if amp_dtype is not None else _null()
    # ---- encoding (all models, all tiles)
    on_stage("encoding")
    enc: dict[str, list] = {t: [] for t in models}
    with ctx:
        for task, m in models.items():
            for i in range(0, len(boxes), batch):
                bb = boxes[i:i + batch]
                crops = [torch.cat([a[..., y:y + th, x:x + tw] for (y, x) in bb], 0) for a in xs]
                av = avail.expand(len(bb), -1)
                levels, raw = m.encode(crops[0], crops[1], crops[2], av)
                enc[task].append((bb, levels, raw, crops[0], av))
    # ---- decoding
    on_stage("decoding")
    out = {}
    win = torch.outer(_w1d(th, dev), _w1d(tw, dev))
    if len(boxes) == 1:
        win = torch.ones(th, tw, device=dev)
    with ctx:
        for task, m in models.items():
            acc = None
            wsum = torch.zeros(Hp, Wp, device=dev)
            for bb, levels, raw, opt, av in enc[task]:
                y = m.decode(levels, raw, opt, av).float()
                if acc is None:
                    acc = torch.zeros(y.shape[1], Hp, Wp, device=dev)
                for j, (yy, xx) in enumerate(bb):
                    acc[:, yy:yy + th, xx:xx + tw] += y[j] * win
                    wsum[yy:yy + th, xx:xx + tw] += win
            out[task] = (acc / wsum.clamp(min=1e-6))[:, :H, :W]
    feats = None
    if want_feats:
        task0 = next(iter(models))
        p = models[task0].patch
        hp, wp = Hp // p, Wp // p
        D = enc[task0][0][1][-1].shape[1]
        fa = torch.zeros(D, hp, wp, device=dev)
        fw = torch.zeros(1, hp, wp, device=dev)
        for bb, levels, _, _, _ in enc[task0]:
            last = levels[-1].float()
            for j, (yy, xx) in enumerate(bb):
                fa[:, yy // p:yy // p + th // p, xx // p:xx // p + tw // p] += last[j]
                fw[:, yy // p:yy // p + th // p, xx // p:xx // p + tw // p] += 1
        fa = fa / fw.clamp(min=1)
        hh, ww = math.ceil(H / p), math.ceil(W / p)
        fa = fa[:, :hh, :ww]
        feats = (fa.reshape(D, -1).T.contiguous(), (hh, ww))
    return out, feats


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
