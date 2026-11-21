"""Sensitivity of the heuristic risk indices to their hand-picked coefficients -> artifacts/risk_sensitivity.json

Inputs per scene come from the model's predictions (finetuned_full, seed 0 by default; all sensors) on test64,
exactly as the app computes them (engine/stats.py). Every coefficient in RISK_COEF is multiplied by a factor
in [0.5, 1.5]:
  * one-at-a-time (OAT): each coefficient alone on an 11-point grid 0.5..1.5
  * joint: 500 random draws, all coefficients independently ~ U(0.5, 1.5)
Reported: Spearman rank correlation of the composite score across scenes vs the default coefficients, and the
fraction of scenes whose risk band (0-25, 25-50, 50-75, 75-100) changes. Main analysis renormalises the three
composite weights to sum to 1 (the composite is a weighted mean by design); the un-normalised variant is also given.

    python -m terrascope.core.risk_sensitivity [--ckpt artifacts/ckpt/finetuned.pt] [--draws 500]
"""
from __future__ import annotations

import argparse, json, os, time

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import load
from .model import load_checkpoint
from .engine.stats import RISK_COEF, risk_indices, VEG_CLASSES

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BANDS = [25, 50, 75]


@torch.no_grad()
def scene_inputs(model, d, bs=50):
    rows = {k: [] for k in ("flood_frac", "urban_fl", "crop_fl", "stress", "veg_frac", "urban_frac")}
    for i in range(0, d["x"].shape[0], bs):
        x = d["x"][i:i + bs]
        out = model.forward_tasks(x, torch.ones(x.shape[0], 3, dtype=torch.bool))
        lc = out["lc"].argmax(1).numpy()
        fl = (out["flood"] > 0).numpy()
        veg = out["veg"].numpy()
        for j in range(x.shape[0]):
            l, f, v = lc[j], fl[j], veg[j]
            vegm = np.isin(l, VEG_CLASSES)
            rows["flood_frac"].append(f.mean())
            rows["urban_fl"].append((f & (l == 3)).mean())
            rows["crop_fl"].append((f & (l == 2)).mean())
            rows["stress"].append(((v < 0.30) & vegm).sum() / max(vegm.sum(), 1))
            rows["veg_frac"].append(vegm.mean())
            rows["urban_frac"].append((l == 3).mean())
    return {k: np.asarray(v, np.float64) for k, v in rows.items()}


def _rho(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else None


def compare(inp, coef, base, renorm):
    fl, dr, ur, comp = risk_indices(inp["flood_frac"], inp["urban_fl"], inp["crop_fl"], inp["stress"], inp["veg_frac"],
                                    inp["urban_frac"], coef, renormalise=renorm)
    return dict(rho=_rho(base["comp"], comp), band_change=float((np.digitize(comp, BANDS) != base["band"]).mean()),
                rho_flood=_rho(base["fl"], fl), rho_drought=_rho(base["dr"], dr), rho_urban=_rho(base["ur"], ur))


def _summ(vals):
    v = np.array([x for x in vals if x is not None], np.float64)
    if not len(v):
        return None
    return dict(median=float(np.median(v)), p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)),
                min=float(v.min()), max=float(v.max()), n=int(len(v)))


def analyse(inp, draws=500, seed=0, renorm=True):
    fl, dr, ur, comp = risk_indices(inp["flood_frac"], inp["urban_fl"], inp["crop_fl"], inp["stress"], inp["veg_frac"],
                                    inp["urban_frac"], RISK_COEF, renormalise=renorm)
    base = dict(fl=fl, dr=dr, ur=ur, comp=comp, band=np.digitize(comp, BANDS))
    grid = [f for f in np.round(np.linspace(0.5, 1.5, 11), 2) if f != 1.0]
    oat = {}
    for k in RISK_COEF:
        rows = []
        for f in grid:
            c = dict(RISK_COEF); c[k] = RISK_COEF[k] * f
            rows.append(dict(factor=float(f), **compare(inp, c, base, renorm)))
        oat[k] = dict(rows=rows, min_rho=min(r["rho"] for r in rows if r["rho"] is not None),
                      max_band_change=max(r["band_change"] for r in rows))
    rng = np.random.default_rng(seed)
    joint = []
    for _ in range(draws):
        c = {k: v * float(rng.uniform(0.5, 1.5)) for k, v in RISK_COEF.items()}
        joint.append(compare(inp, c, base, renorm))
    return dict(
        default_band_counts={b: int((base["band"] == i).sum()) for i, b in enumerate(["0-25", "25-50", "50-75", "75-100"])},
        oat=oat,
        joint=dict(draws=draws, rho=_summ([j["rho"] for j in joint]), band_change=_summ([j["band_change"] for j in joint]),
                   rho_flood=_summ([j["rho_flood"] for j in joint]), rho_drought=_summ([j["rho_drought"] for j in joint]),
                   rho_urban=_summ([j["rho_urban"] for j in joint])),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "artifacts", "ckpt", "finetuned.pt"))
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N test scenes (smoke tests)")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "risk_sensitivity.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("TERRASCOPE_THREADS", "0")) or os.cpu_count() or 2)
    t0 = time.time()
    model, cfg = load_checkpoint(a.ckpt)
    d = load("test64", cfg.get("norm", "v1"))
    if a.limit:
        d = {k: (v[:a.limit] if hasattr(v, "shape") and v.shape[:1] == d["x"].shape[:1] else v) for k, v in d.items()}
    inp = scene_inputs(model, d)
    res = dict(
        ckpt=os.path.relpath(a.ckpt, ROOT).replace("\\", "/"), split="test64", n_scenes=int(len(inp["flood_frac"])),
        sensors="all", coefficients=RISK_COEF, factor_range=[0.5, 1.5], bands=[0, 25, 50, 75, 100],
        renormalised=analyse(inp, a.draws, renorm=True),
        raw_weights=analyse(inp, a.draws, renorm=False),
        note="Simulated test scenes; model predictions as inputs. 'renormalised' = composite weights rescaled to sum 1 "
             "after perturbation (main analysis); 'raw_weights' = not rescaled.",
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for key in ("renormalised", "raw_weights"):
        j = res[key]["joint"]
        print(f"[risk] {key:12s} joint Spearman median {j['rho']['median']:.3f} p5 {j['rho']['p5']:.3f} | "
              f"band change median {j['band_change']['median']:.3f} p95 {j['band_change']['p95']:.3f}")
    worst = min(res["renormalised"]["oat"].items(), key=lambda kv: kv[1]["min_rho"])
    print(f"[risk] OAT most sensitive coefficient: {worst[0]} (min rho {worst[1]['min_rho']:.3f}, "
          f"max band change {worst[1]['max_band_change']:.3f})  ({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    main()
