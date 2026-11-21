"""Calibration-/domain-shift probes: the same held-out test scenes with physical perturbations applied.

Gate (phase 3): every IN-RANGE probe (shifts inside the training augmentation ranges: elevation offset
-800..+1500 m, SAR -3..+3 dB, optical gain 0.8..1.25) within 0.05 mIoU of the in-distribution score.
STRESS probes lie outside those ranges; they are reported separately and are not part of the gate.
"""
from __future__ import annotations

import os
import torch

from .data import load_raw, with_inputs
from .metrics import evaluate

PROBES = [
    ("in-distribution", {}),
    ("elevation +500 m", dict(dem_off=500.0)),
    ("elevation +1500 m", dict(dem_off=1500.0)),
    ("SAR +3 dB", dict(sar_off=3.0)),
    ("SAR -3 dB", dict(sar_off=-3.0)),
    ("optical x0.8", dict(o_scale=0.8)),
    ("optical x1.2", dict(o_scale=1.2)),
]
STRESS = [
    ("elevation +3000 m", dict(dem_off=3000.0)),
    ("SAR +6 dB", dict(sar_off=6.0)),
    ("SAR -6 dB", dict(sar_off=-6.0)),
    ("optical x0.6", dict(o_scale=0.6)),
    ("optical x1.5", dict(o_scale=1.5)),
]
GATE = 0.05


def probe(model, norm, raw=None, split="test64"):
    raw = raw if raw is not None else load_raw(split)
    base = None

    def run(name, kw, kind):
        nonlocal base
        r = evaluate(model, with_inputs(raw, norm, **kw), [True, True, True])
        if base is None:
            base = r
        drop = base["miou"] - r["miou"]
        return dict(name=name, kind=kind, shift=kw, miou=r["miou"], flood_iou=r["flood_iou"], veg_mae=r["veg_mae"],
                    miou_drop=drop, flood_iou_drop=base["flood_iou"] - r["flood_iou"], within_gate=bool(drop <= GATE))

    rows = [run(n, kw, "in_range") for n, kw in PROBES]
    stress = [run(n, kw, "stress") for n, kw in STRESS]
    return dict(split=split, gate_miou_drop=GATE, all_within_gate=all(r["within_gate"] for r in rows), rows=rows,
                stress=stress, stress_note="outside the training augmentation ranges; reported, not gated")


def print_table(res, title=""):
    print(title)
    for r in res["rows"] + [dict(x, name=x["name"] + " [stress]") for x in res.get("stress", [])]:
        print(f"  {r['name']:20s} mIoU {r['miou']:.3f} (drop {r['miou_drop']:+.3f})  flood IoU {r['flood_iou']:.3f}  "
              f"vegMAE {r['veg_mae']:.3f}  {'ok' if r['within_gate'] else 'FAIL'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--update-metrics":
        torch.set_num_threads(os.cpu_count() or 2)
        for k, v in update_metrics().items():
            print_table(v, k)
        sys.exit(0)
    from .model import load_checkpoint
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "artifacts", "ckpt", "finetuned.pt")
    torch.set_num_threads(os.cpu_count() or 2)
    m, cfg = load_checkpoint(path)
    print_table(probe(m, cfg.get("norm", "v1")), f"{path} (norm {cfg.get('norm', 'v1')})")


def update_metrics(path=None):
    """Light re-run (no training): recompute the robustness section of artifacts/metrics.json."""
    import json
    from .model import load_checkpoint
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = path or os.path.join(root, "artifacts", "metrics.json")
    ck = os.path.join(root, "artifacts", "ckpt")
    with open(path) as f:
        m = json.load(f)
    raw = load_raw("test64")
    rob = {}
    for key, p in (("finetuned_v2", "finetuned.pt"), ("scratch_v2", "scratch.pt"), ("finetuned_v1_before_fix", "v1/finetuned.pt")):
        if os.path.exists(os.path.join(ck, p)):
            mdl, cfg = load_checkpoint(os.path.join(ck, p))
            rob[key] = probe(mdl, cfg.get("norm", "v1"), raw)
    m["robustness"] = rob
    with open(path, "w") as f:
        json.dump(m, f, indent=1)
    return rob
