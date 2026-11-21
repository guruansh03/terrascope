"""Robustness probe (calibration / domain shifts) for any TerraFM checkpoint.

    python tests/ml/robustness_probe.py                                  # artifacts/ckpt/finetuned.pt
    python tests/ml/robustness_probe.py artifacts/ckpt/v1/finetuned.pt   # legacy v1 checkpoint (before the fix)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from terrascope.core.model import load_checkpoint
from terrascope.core.robustness import probe, print_table

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "artifacts", "ckpt", "finetuned.pt")
torch.set_num_threads(os.cpu_count() or 2)
m, cfg = load_checkpoint(path)
res = probe(m, cfg.get("norm", "v1"))
print_table(res, f"{path}  norm={cfg.get('norm', 'v1')}")
print("GATE (every in-range probe within %.2f mIoU):" % res["gate_miou_drop"], "PASS" if res["all_within_gate"] else "FAIL")
print("stress probes (outside training augmentation ranges) are informational only")
