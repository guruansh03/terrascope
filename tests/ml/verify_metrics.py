"""Independently regenerate the 64 px test set from its seeds and re-score the checkpoints against metrics.json."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from terrascope.core.simulator import build_dataset
from terrascope.core.model import load_checkpoint
from terrascope.core.data import with_inputs
from terrascope.core.metrics import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
t = time.time()
z = build_dataset(range(900_000, 900_300), 64)
raw = {k: torch.from_numpy(v) for k, v in z.items()}
raw["lc"] = raw["lc"].long(); raw["flood"] = raw["flood"].float(); raw["veg"] = raw["veg"].float(); raw["cloud"] = raw["cloud"].float()
print("gen", round(time.time() - t, 1), "s")
M = json.load(open(os.path.join(ROOT, "artifacts", "metrics.json")))
bad = 0
for name, key in [("finetuned", "ablation"), ("scratch", "scratch")]:
    m, cfg = load_checkpoint(os.path.join(ROOT, "artifacts", "ckpt", f"{name}.pt"))
    d = with_inputs(raw, cfg.get("norm", "v1"))
    for s, av in [("All sensors", [1, 1, 1]), ("Optical only", [1, 0, 0]), ("SAR only", [0, 1, 0])]:
        r = evaluate(m, d, [bool(a) for a in av])
        ok = abs(r["miou"] - M[key][s]["miou"]) < 1e-3
        bad += not ok
        print(f"{name:9s} {s:13s} miou {r['miou']:.4f} (claimed {M[key][s]['miou']:.4f})  flood {r['flood_iou']:.4f} ({M[key][s]['flood_iou']:.4f})  "
              f"veg {r['veg_mae']:.4f} ({M[key][s]['veg_mae']:.4f})  {'ok' if ok else 'MISMATCH'}")
y = raw["lc"]; maj = torch.bincount(y.flatten()).argmax()
print("majority-class OA", (y == maj).float().mean().item())
sys.exit(1 if bad else 0)
