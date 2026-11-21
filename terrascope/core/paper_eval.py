"""Paper evaluation on the simulator -> artifacts/metrics_paper_sim.json

    python -m terrascope.core.evaluate --paper [--seeds 0,1,2] [--device auto] [--limit N] [--ckpt-dir DIR] [--out FILE]

Models (checkpoint names as written by ``train paper_sim``):
  finetuned_full (finetuned[_sN]), finetuned_nodrop / _nocloud / _nomask (finetuned_<v>_sN), scratch[_sN], unet_sN.
Per model x seed:
  test64   six sensor subsets: oa, miou, flood_iou, veg_mae (+ cloud IoU for models with a cloud head)
  heavy    heavy-cloud subset of test64 (mean cloud opacity > 0.3): optical-only and all sensors
  test128  all sensors (every model, incl. scratch and U-Net)
  cloud_rmse  shift-ensemble reconstruction RMSE on cloudy pixels (TerraFM models only; the U-Net has no decoder)
  bootstrap   per-seed 95% CI over test scenes (1000 scene-level resamples) for oa/miou/flood_iou/veg_mae
              (test64 all sensors, optical-only, test128)
Aggregates over seeds: mean, std (ddof=1), 95% CI (Student t, n-1 dof); null where n < 2.
Paired comparisons per seed: finetuned_full vs every other model (positive = finetuned_full better).
Robustness probes (in-range gate + stress) for finetuned_full per seed.
Metric definitions are identical to metrics.evaluate (mIoU = mean IoU over all 6 classes on pooled pixels).
"""
from __future__ import annotations

import json, os, time

import numpy as np
import torch
from scipy import stats as sps

from . import train as T
from .data import load_raw, with_inputs
from .metrics import SENSOR_SETS, evaluate
from .model import N_CLASSES, load_checkpoint
from .robustness import probe

MAIN = ("oa", "miou", "flood_iou", "veg_mae")
HIGHER_BETTER = dict(oa=True, miou=True, flood_iou=True, veg_mae=False)


def model_ckpts(seed):
    out = {"finetuned_full": T.finetuned_name("full", seed)}
    for v in ("nodrop", "nocloud", "nomask"):
        out[f"finetuned_{v}"] = T.finetuned_name(v, seed)
    out["scratch"] = T.scratch_name(seed)
    out["unet"] = T.unet_name(seed)
    return out


@torch.no_grad()
def scene_stats(model, d, avail, bs=50):
    """Per-scene sufficient statistics: confusion (N,36), flood inter/union, veg abs-error sum, pixel count."""
    model.eval()
    dev = next(model.parameters()).device
    conf, fi, fu, ve, npx = [], [], [], [], []
    for i in range(0, d["x"].shape[0], bs):
        x = d["x"][i:i + bs].to(dev)
        out = model.forward_tasks(x, torch.tensor([avail] * x.shape[0], device=dev))
        pred = out["lc"].argmax(1).cpu()
        y = d["lc"][i:i + bs]
        idx = (y * N_CLASSES + pred).flatten(1)
        conf.append(torch.stack([torch.bincount(r, minlength=N_CLASSES ** 2) for r in idx]).numpy())
        pf = (out["flood"] > 0).float().cpu()
        yf = d["flood"][i:i + bs]
        fi.append((pf * yf).sum((1, 2)).numpy()); fu.append(((pf + yf) > 0).float().sum((1, 2)).numpy())
        ve.append((out["veg"].cpu() - d["veg"][i:i + bs]).abs().sum((1, 2)).numpy())
        npx.append(np.full(x.shape[0], y[0].numel()))
    return dict(conf=np.concatenate(conf).astype(np.float64), fi=np.concatenate(fi).astype(np.float64),
                fu=np.concatenate(fu).astype(np.float64), ve=np.concatenate(ve).astype(np.float64),
                npx=np.concatenate(npx).astype(np.float64))


def metrics_from_sums(conf, fi, fu, ve, npx):
    """conf (...,36) etc. summed over scenes -> metrics (vectorised over leading dims)."""
    c = conf.reshape(conf.shape[:-1] + (N_CLASSES, N_CLASSES))
    diag = np.diagonal(c, axis1=-2, axis2=-1)
    iou = diag / np.maximum(c.sum(-2) + c.sum(-1) - diag, 1)
    return dict(oa=diag.sum(-1) / np.maximum(c.sum((-2, -1)), 1), miou=iou.mean(-1),
                flood_iou=fi / np.maximum(fu, 1), veg_mae=ve / np.maximum(npx, 1))


def summarize(st, sel=None):
    sel = np.arange(len(st["fi"])) if sel is None else sel
    m = metrics_from_sums(*(st[k][sel].sum(0) for k in ("conf", "fi", "fu", "ve", "npx")))
    out = {k: float(v) for k, v in m.items()}
    out["n_scenes"] = int(len(sel))
    return out


def bootstrap(st, n_boot=1000, seed=0):
    """Scene-level bootstrap: 95% percentile CI of each main metric."""
    N = len(st["fi"])
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, N, N), minlength=N) for _ in range(n_boot)]).astype(np.float64)  # (B,N)
    m = metrics_from_sums(W @ st["conf"], W @ st["fi"], W @ st["fu"], W @ st["ve"], W @ st["npx"])
    return {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), se=float(np.std(v, ddof=1)))
            for k, v in m.items()} | {"n_boot": n_boot, "n_scenes": N}


def agg(values):
    v = np.array([x for x in values if x is not None], np.float64)
    n = len(v)
    if n == 0:
        return dict(n=0, mean=None, std=None, ci95=None, values=[])
    mean = float(v.mean())
    if n < 2:
        return dict(n=1, mean=mean, std=None, ci95=None, values=v.tolist())
    sd = float(v.std(ddof=1))
    h = float(sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return dict(n=n, mean=mean, std=sd, ci95=[mean - h, mean + h], values=v.tolist())


def _limit(d, n):
    if not n:
        return d
    N = d["x"].shape[0] if "x" in d else d["optical"].shape[0]
    return {k: (v[:n] if hasattr(v, "shape") and v.shape[:1] == (N,) else v) for k, v in d.items()}


def eval_model(model, norm, raw64, raw128, n_boot, do_probe=False):
    te64, te128 = with_inputs(raw64, norm), with_inputs(raw128, norm)
    r = dict(test64={}, bootstrap={})
    st = {}
    for name, av in SENSOR_SETS.items():
        st[name] = scene_stats(model, te64, av)
        r["test64"][name] = summarize(st[name])
    heavy = np.where(te64["cloud"].float().mean((1, 2)).numpy() > 0.3)[0]
    r["heavy_cloud"] = dict(n=int(len(heavy)), optical_only=summarize(st["Optical only"], heavy) if len(heavy) else None,
                            all_sensors=summarize(st["All sensors"], heavy) if len(heavy) else None)
    st128 = scene_stats(model, te128, [True, True, True], bs=20)
    r["test128"] = summarize(st128)
    r["bootstrap"] = {"test64|All sensors": bootstrap(st["All sensors"], n_boot),
                      "test64|Optical only": bootstrap(st["Optical only"], n_boot, seed=1),
                      "test128": bootstrap(st128, n_boot, seed=2)}
    ev = evaluate(model, te64, [True, True, True], recon=hasattr(model, "reconstruct"), shift_ens=True)
    r["cloud_iou"] = ev.get("cloud_iou")
    r["cloud_rmse"] = ev.get("cloud_rmse")          # None for the U-Net (no reconstruction decoder)
    r["params"] = int(sum(p.numel() for p in model.parameters()))
    if do_probe:
        r["robustness"] = probe(model, norm, raw64)
    return r


def flat(r):
    """Scalar metrics of one model/seed result keyed 'section|set|metric'."""
    out = {}
    for s, m in r["test64"].items():
        for k in MAIN:
            out[f"test64|{s}|{k}"] = m[k]
    for s in ("optical_only", "all_sensors"):
        if r["heavy_cloud"][s]:
            for k in MAIN:
                out[f"heavy_cloud|{s}|{k}"] = r["heavy_cloud"][s][k]
    for k in MAIN:
        out[f"test128|{k}"] = r["test128"][k]
    for k in ("cloud_iou", "cloud_rmse"):
        if r.get(k) is not None:
            out[k] = r[k]
    return out


PAIRED_KEYS = ("test64|All sensors|miou", "test64|All sensors|flood_iou", "test64|All sensors|veg_mae",
               "test64|Optical only|miou", "test64|SAR only|miou", "heavy_cloud|all_sensors|miou", "test128|miou",
               "test128|flood_iou")


def paired(per, ref="finetuned_full"):
    out = {}
    for other in per:
        if other == ref:
            continue
        common = sorted(set(per[ref]) & set(per[other]))
        if not common:
            continue
        comp = {}
        for key in PAIRED_KEYS:
            metric = key.split("|")[-1]
            sign = 1 if HIGHER_BETTER[metric] else -1
            diffs = []
            for s in common:
                a, b = flat(per[ref][s]).get(key), flat(per[other][s]).get(key)
                if a is not None and b is not None:
                    diffs.append(sign * (a - b))
            if diffs:
                g = agg(diffs)
                comp[key] = dict(per_seed=dict(zip(map(str, common), diffs)), mean=g["mean"], std=g["std"], ci95=g["ci95"],
                                 n_positive=int(sum(d > 0 for d in diffs)), n=len(diffs))
        out[f"{ref}_vs_{other}"] = dict(seeds=common, sign="positive = finetuned_full better", metrics=comp)
    return out


def main(seeds=(0, 1, 2), limit=None, n_boot=1000, out=None, device="auto"):
    torch.set_num_threads(int(os.environ.get("TERRASCOPE_THREADS", "0")) or os.cpu_count() or 2)
    device = T.pick_device(device)
    t0 = time.time()
    raw64, raw128 = _limit(load_raw("test64"), limit), _limit(load_raw("test128"), limit)
    per, missing, timing = {}, [], {}
    for s in seeds:
        for mname, ck in model_ckpts(s).items():
            p = T.path_of(ck)
            if not os.path.exists(p):
                missing.append(ck)
                continue
            t = time.time()
            model, cfg = load_checkpoint(p)
            model.to(device)
            res = eval_model(model, cfg.get("norm", "v1"), raw64, raw128, n_boot, do_probe=(mname == "finetuned_full"))
            res["ckpt"] = os.path.basename(p)
            per.setdefault(mname, {})[s] = res
            timing[f"{mname}|s{s}"] = round(time.time() - t, 1)
            print(f"[paper-eval] {mname} s{s}: mIoU {res['test64']['All sensors']['miou']:.4f} "
                  f"(128px {res['test128']['miou']:.4f})  {timing[f'{mname}|s{s}']}s", flush=True)
    aggregate = {}
    for mname, by_seed in per.items():
        keys = sorted(set().union(*(flat(r).keys() for r in by_seed.values())))
        aggregate[mname] = dict(seeds=sorted(by_seed), **{k: agg([flat(r).get(k) for r in by_seed.values()]) for k in keys})
    rob = {str(s): r["robustness"] for s, r in per.get("finetuned_full", {}).items()}
    rob_agg = None
    if rob:
        names = [x["name"] for x in next(iter(rob.values()))["rows"] + next(iter(rob.values()))["stress"]]
        rob_agg = {n: agg([next(x["miou"] for x in r["rows"] + r["stress"] if x["name"] == n) for r in rob.values()]) for n in names}
        rob_agg["all_seeds_within_gate"] = all(r["all_within_gate"] for r in rob.values())
    doc = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"), device=str(device), torch=torch.__version__,
        seeds_requested=list(seeds), missing_checkpoints=missing, limit_scenes=limit, n_boot=n_boot,
        splits=dict(test64=int(raw64["optical"].shape[0]), test128=int(raw128["optical"].shape[0])),
        definitions=dict(miou="mean IoU over the 6 classes on pixels pooled over scenes", flood_iou="pooled IoU of flood>0.5",
                         veg_mae="mean absolute error of the vegetation index", heavy_cloud="test64 scenes with mean cloud opacity > 0.3",
                         ci95_seeds="Student-t over seeds (n-1 dof)", bootstrap="per-seed scene-level percentile bootstrap"),
        per_seed={m: {str(s): r for s, r in v.items()} for m, v in per.items()},
        aggregate=aggregate, paired=paired(per), robustness=dict(per_seed=rob, miou_over_seeds=rob_agg),
        eval_seconds=timing, data_note="Simulated scenes only (core/simulator.py); not real-world accuracy.",
    )
    out = out or os.path.join(T.ROOT, "artifacts", "metrics_paper_sim.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"[paper-eval] wrote {out}; missing {len(missing)} checkpoints; {time.time() - t0:.0f}s", flush=True)
    return doc
