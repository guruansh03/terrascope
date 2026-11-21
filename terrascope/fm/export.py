"""ONNX export of fine-tuned (model, task) pairs for the CPU demo laptop + the ONNX adapter used by
:class:`terrascope.fm.runners.OnnxFMRunner`.

Two graphs per task so the runner can emit honest "encoding" / "decoding" stage events:
    <task>.encoder.onnx : optical(1,6,224,224) sar(1,2,..) dem(1,2,..) avail(1,3) -> lvl0..lvl3, raw
    <task>.head.onnx    : lvl0..lvl3, raw, optical, avail -> out (logits, or reflectance for 'cloud')
    <task>.json         : metadata (patch, tile, required sensors, params, source checkpoint, int8)
Input size is FIXED to 224 x 224 (Prithvi's position embedding is interpolated with numpy at trace time, so a
dynamic size would not be exported faithfully); the runner pads/tiles scenes to 224 tiles.
``int8=True`` applies onnxruntime dynamic quantisation (MatMul/Gemm weights -> int8).

    python -m terrascope.fm.export --model prithvi_tiny --task flood --out artifacts/onnx [--int8]
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch
import torch.nn as nn

from .specs import MODELS, TILE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ONNX_ROOT = os.environ.get("TERRASCOPE_ONNX_DIR", os.path.join(ROOT, "artifacts", "onnx"))
OPSET = 17


class _Enc(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, optical, sar, dem, avail):
        levels, raw = self.m.encode(optical, sar, dem, avail)
        return (*levels, raw)


class _Head(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, *args):  # lvl0..lvl{n-1}, raw, optical, avail (n = 4 for ViTs, 6 for the U-Net)
        *levels, raw, optical, avail = args
        return self.m.decode(list(levels), raw, optical, avail)


def required_sensors(model) -> list[str]:
    fam = MODELS[model.model_id]["family"]
    if fam == "prithvi":
        return ["optical"]
    if fam == "terramind":
        e = model.encoder
        return ["optical"] + (["sar"] if e.use_sar else []) + (["dem"] if e.use_dem else [])
    if fam == "unet":
        return ["optical"]
    return []


def _export(module, args, path, in_names, out_names):
    kw = dict(input_names=in_names, output_names=out_names, opset_version=OPSET, do_constant_folding=True)
    try:
        torch.onnx.export(module, args, path, dynamo=False, **kw)
    except TypeError:  # very old torch without the dynamo kwarg
        torch.onnx.export(module, args, path, **kw)


def _quantize(path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q = path.replace(".onnx", ".int8.onnx")
    quantize_dynamic(path, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
    return q


def export_model(model, out_dir: str, int8: bool = False, source_ckpt: str | None = None, tile: int = TILE) -> dict:
    """Export an in-memory TaskModel. Returns the metadata dict (also written to <task>.json)."""
    model = model.eval().cpu()
    task = model.task
    os.makedirs(out_dir, exist_ok=True)
    req = required_sensors(model)
    enc = model.encoder
    if hasattr(enc, "force_present"):
        enc.force_present = tuple(s in req for s in ("optical", "sar", "dem"))
    g = torch.Generator().manual_seed(0)
    o = torch.rand(1, 6, tile, tile, generator=g) * 0.4
    s = torch.randn(1, 2, tile, tile, generator=g) * 3 - 14
    d = torch.stack([torch.rand(1, tile, tile, generator=g) * 300, torch.rand(1, tile, tile, generator=g) * 20], 1)
    a = torch.tensor([[1.0 if (not req or n in req) else 0.0 for n in ("optical", "sar", "dem")]])
    if not req:
        a = torch.ones(1, 3)
    enc_p = os.path.join(out_dir, f"{task}.encoder.onnx")
    head_p = os.path.join(out_dir, f"{task}.head.onnx")
    try:
        with torch.no_grad():
            levels, raw = model.encode(o, s, d, a)
            n = len(levels)
            _export(_Enc(model), (o, s, d, a), enc_p, ["optical", "sar", "dem", "avail"],
                    [f"lvl{i}" for i in range(n)] + ["raw"])
            _export(_Head(model), (*levels, raw, o, a), head_p, [f"lvl{i}" for i in range(n)] + ["raw", "optical", "avail"],
                    ["out"])
    finally:
        if hasattr(enc, "force_present"):
            enc.force_present = None
    files = dict(encoder=os.path.basename(enc_p), head=os.path.basename(head_p))
    if int8:
        files = dict(encoder=os.path.basename(_quantize(enc_p)), head=os.path.basename(_quantize(head_p)))
    meta = dict(model_id=model.model_id, task=task, **files, patch=int(model.patch), tile=tile, n_levels=n,
                required_sensors=req, int8=bool(int8), params=model.n_params(), opset=OPSET,
                source_ckpt=os.path.relpath(source_ckpt, ROOT).replace("\\", "/") if source_ckpt else None,
                exported_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(out_dir, f"{task}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def export_onnx(model_id: str, task: str, out_dir: str | None = None, int8: bool = False, ckpt_root: str | None = None,
                model=None) -> dict:
    """Contract entry point (used by OPS's make_demo_bundle). Exports the best fine-tuned checkpoint of
    ``model_id`` for ``task`` into ``out_dir/<model_id>/``. ``model`` lets callers pass an in-memory TaskModel."""
    if task == "veg":
        raise ValueError("'veg' is a derived spectral index computed at runtime; there is no network to export")
    if task not in ("flood", "burn", "landcover", "cloud"):
        raise ValueError(f"unknown task {task!r}")
    base = out_dir or ONNX_ROOT
    if not os.path.isabs(base):  # e.g. "artifacts/onnx" -> relative to the repository root, not the cwd
        base = os.path.join(ROOT, base)
    out_dir = os.path.join(base, model_id)
    src = None
    if model is None:
        from .model import CKPT_ROOT, load_checkpoint
        from .runners import best_checkpoints
        from .weights import weights_path
        cks = best_checkpoints(model_id, ckpt_root or CKPT_ROOT)
        if task not in cks:
            raise FileNotFoundError(f"no completed fine-tuned checkpoint for {model_id}/{task}; run train.bat first")
        src = cks[task]
        model = load_checkpoint(src, weights_path(model_id))
    return export_model(model, out_dir, int8=int8, source_ckpt=src)


class OnnxTask:
    """Adapter giving onnxruntime sessions the TaskModel encode/decode interface (torch tensors in/out)."""

    def __init__(self, enc_sess, head_sess, meta: dict):
        self.enc, self.head, self.meta = enc_sess, head_sess, meta
        self.patch = int(meta["patch"])
        self.task = meta["task"]
        self.enc_in = {i.name for i in enc_sess.get_inputs()}
        self.head_in = {i.name for i in head_sess.get_inputs()}

    def encode(self, optical, sar, dem, avail):
        feeds = dict(optical=optical, sar=sar, dem=dem, avail=avail)
        outs = []
        for b in range(optical.shape[0]):
            f = {k: np.ascontiguousarray(v[b:b + 1].cpu().numpy(), np.float32) for k, v in feeds.items() if k in self.enc_in}
            outs.append(self.enc.run(None, f))
        n = len(outs[0])
        cat = [torch.from_numpy(np.concatenate([o[i] for o in outs], 0)) for i in range(n)]
        return cat[:-1], cat[-1]

    def decode(self, levels, raw, optical, avail):
        res = []
        for b in range(optical.shape[0]):
            f = {f"lvl{i}": levels[i][b:b + 1].numpy() for i in range(len(levels))}
            f.update(raw=raw[b:b + 1].numpy(), optical=optical[b:b + 1].cpu().numpy().astype(np.float32),
                     avail=avail[b:b + 1].cpu().numpy().astype(np.float32))
            res.append(self.head.run(None, {k: np.ascontiguousarray(v) for k, v in f.items() if k in self.head_in})[0])
        return torch.from_numpy(np.concatenate(res, 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=[m for m in MODELS])
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", default=ONNX_ROOT)
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(export_onnx(a.model, a.task, a.out, a.int8), indent=1))


if __name__ == "__main__":
    main()
