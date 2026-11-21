"""Full benchmark -> artifacts/metrics.json (consumed by /api/benchmarks and the web UI).

Keeps the phase-2 keys (ablation, scratch, size128, heavy_cloud, curves, config) for seed 0 and adds
  seeds       mean/std over every seed checkpoint found (finetuned[_sN].pt, scratch[_sN].pt)
  robustness  calibration/domain-shift probes (core/robustness.py) for the v2 model and, if present, the v1 one
  recon       cloud-gap RMSE single pass vs 4-shift ensemble (64 px and 128 px)
  cloud       cloud head vs the legacy reflectance-threshold rule (IoU and composite RMSE)
"""
from __future__ import annotations

import glob, json, os, re, time
import numpy as np
import torch

from .data import load_raw, with_inputs
from .model import load_checkpoint
from .metrics import evaluate, SENSOR_SETS, threshold_cloud_mask
from .robustness import probe
from .train import CKPT, LOGS, ROOT, DEFAULTS

SEED_KEYS = ("miou", "oa", "flood_iou", "veg_mae")


def read_log(name):
    p = os.path.join(LOGS, f"{name}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def seeds_available():
    out = [0] if os.path.exists(os.path.join(CKPT, "finetuned.pt")) else []
    for p in sorted(glob.glob(os.path.join(CKPT, "finetuned_s*.pt"))):
        m = re.search(r"_s(\d+)\.pt$", p)
        if m and os.path.exists(os.path.join(CKPT, f"scratch_s{m.group(1)}.pt")):
            out.append(int(m.group(1)))
    return out


def ck(name, seed):
    return os.path.join(CKPT, f"{name}{'' if seed == 0 else f'_s{seed}'}.pt")


@torch.no_grad()
def composite_scores(model, d, bs=50):
    """RMSE (reflectance) of the cloud-free *composite* shown in the app vs the true clean image,
    using the cloud-head blend weights vs the legacy threshold rule (all sensors, shift-ensemble recon)."""
    from .engine.render import blend_weights, threshold_blend_weights
    se_head = se_thr = 0.0
    n = 0
    for i in range(0, d["x"].shape[0], bs):
        x = d["x"][i:i + bs]
        av = torch.ones(x.shape[0], 3, dtype=torch.bool)
        out = model.forward_tasks(x, av)
        rec = (model.reconstruct(x, av)[:, :6] * 0.2 + 0.2).clamp(0, 1).numpy()
        cp = torch.sigmoid(out["cloud"]).numpy()
        o = d["optical"][i:i + bs].float().numpy()
        clean = d["optical_clean"][i:i + bs].float().numpy()
        for j in range(x.shape[0]):
            for w, acc in ((blend_weights(cp[j]), "h"), (threshold_blend_weights(o[j]), "t")):
                comp = o[j] * (1 - w) + rec[j] * w
                e = float(((comp - clean[j]) ** 2).mean())
                if acc == "h":
                    se_head += e
                else:
                    se_thr += e
            n += 1
    return dict(composite_rmse_cloud_head=float(np.sqrt(se_head / n)), composite_rmse_threshold_rule=float(np.sqrt(se_thr / n)), n=n)


@torch.no_grad()
def blockiness(model, d, n=100, shift_ens=False):
    """Mean |horizontal+vertical step| of the reconstruction across 4-px patch boundaries divided by the mean
    step inside patches (1.0 = no grid artefact; the clean target is the reference)."""
    x = d["x"][:n]
    av = torch.ones(x.shape[0], 3, dtype=torch.bool)
    r = (model.reconstruct(x, av) if shift_ens else model.forward_all(x, av)["recon"])[:, :6]

    def ratio(img):
        dx = (img[..., 1:] - img[..., :-1]).abs()
        dy = (img[..., 1:, :] - img[..., :-1, :]).abs()
        bx = (torch.arange(dx.shape[-1]) % 4) == 3
        by = (torch.arange(dy.shape[-2]) % 4) == 3
        b = torch.cat([dx[..., bx].flatten(), dy[..., by, :].flatten()]).mean()
        i = torch.cat([dx[..., ~bx].flatten(), dy[..., ~by, :].flatten()]).mean()
        return float(b / i)

    return dict(recon=ratio(r), clean_target=ratio(d["target"][:n, :6]))


def mean_std(rows, keys=SEED_KEYS):
    return ({k: float(np.mean([r[k] for r in rows])) for k in keys},
            {k: float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else None for k in keys})


def main(labels=None):
    torch.set_num_threads(int(os.environ.get("TERRASCOPE_THREADS", "0")) or os.cpu_count() or 2)
    t0 = time.time()
    cal = os.path.join(ROOT, "artifacts", "calibration.json")
    if not os.path.exists(cal) or os.path.getmtime(cal) < os.path.getmtime(ck("finetuned", 0)):
        from .calibrate import main as calibrate
        calibrate()
    labels = labels or int(os.environ.get("LABELS", DEFAULTS["labels"]))
    ft, cfg = load_checkpoint(ck("finetuned", 0))
    sc, _ = load_checkpoint(ck("scratch", 0))
    norm = cfg.get("norm", "v1")
    raw64, raw128 = load_raw("test64"), load_raw("test128")
    te64, te128 = with_inputs(raw64, norm), with_inputs(raw128, norm)
    res = {"ablation": {}, "scratch": {}, "size128": {}}
    for k, av in SENSOR_SETS.items():
        res["ablation"][k] = evaluate(ft, te64, av, recon=(k == "All sensors"), shift_ens=True)
        res["scratch"][k] = evaluate(sc, te64, av)
    res["size128"]["finetuned"] = evaluate(ft, te128, [True, True, True], bs=20, recon=True, shift_ens=True)
    heavy = te64["cloud"].mean((1, 2)) > 0.3
    sub = {k: v[heavy] for k, v in te64.items() if hasattr(v, "shape") and v.shape[0] == heavy.shape[0]}
    res["heavy_cloud"] = {"n": int(heavy.sum()),
                          "finetuned_optical": evaluate(ft, sub, [True, False, False]),
                          "finetuned_all": evaluate(ft, sub, [True, True, True]),
                          "scratch_optical": evaluate(sc, sub, [True, False, False]),
                          "scratch_all": evaluate(sc, sub, [True, True, True])}
    res["curves"] = {"pretrain": read_log("pretrain"), "finetuned": read_log("finetuned"), "scratch": read_log("scratch")}

    # ---- reconstruction: single pass vs shift ensemble (raw MAE output on true-cloud pixels)
    r64 = evaluate(ft, te64, [True, True, True], recon=True, shift_ens=False)
    r128 = evaluate(ft, te128, [True, True, True], bs=20, recon=True, shift_ens=False)
    res["recon"] = dict(
        shifts=[[0, 0], [2, 0], [0, 2], [2, 2]],
        test64=dict(single_pass=r64["cloud_rmse"], shift_ensemble=res["ablation"]["All sensors"]["cloud_rmse"], cloudy_input=r64["cloud_rmse_input"]),
        test128=dict(single_pass=r128["cloud_rmse"], shift_ensemble=res["size128"]["finetuned"]["cloud_rmse"], cloudy_input=r128["cloud_rmse_input"]),
    )
    res["recon"]["blockiness"] = dict(single_pass=blockiness(ft, te64), shift_ensemble=blockiness(ft, te64, shift_ens=True),
                                      definition="boundary/interior mean abs step of recon, 100 test64 scenes, all sensors")
    # decide on the validation split (val64), report on test
    val = with_inputs(load_raw("val64"), norm)
    v_single = evaluate(ft, val, [True, True, True], recon=True, shift_ens=False)["cloud_rmse"]
    v_ens = evaluate(ft, val, [True, True, True], recon=True, shift_ens=True)["cloud_rmse"]
    res["recon"]["val64"] = dict(single_pass=v_single, shift_ensemble=v_ens, decision_split=True)
    use = bool(v_ens <= v_single)
    res["recon"]["shift_ensemble_used"] = use
    if not use:  # keep only if not worse: report the single-pass numbers under the legacy keys
        res["ablation"]["All sensors"]["cloud_rmse"] = r64["cloud_rmse"]
        res["size128"]["finetuned"]["cloud_rmse"] = r128["cloud_rmse"]
        print("[evaluate] shift ensemble is WORSE than single pass -> set TERRASCOPE_SHIFT_ENSEMBLE=0 for the app", flush=True)

    # ---- cloud head vs threshold rule
    a = res["ablation"]["All sensors"]
    res["cloud"] = dict(iou_cloud_head=a.get("cloud_iou"), iou_threshold_rule=a.get("cloud_iou_threshold_rule"),
                        truth="simulator cloud opacity > 0.3", **composite_scores(ft, te64))

    # ---- seeds
    seeds = seeds_available()
    per = []
    for s in seeds:
        f, _ = load_checkpoint(ck("finetuned", s))
        b, _ = load_checkpoint(ck("scratch", s))
        row = dict(seed=s)
        for name, mdl in (("finetuned", f), ("scratch", b)):
            for setname in ("All sensors", "Optical only", "SAR only"):
                r = evaluate(mdl, te64, SENSOR_SETS[setname])
                row[f"{name}|{setname}"] = {k: r[k] for k in SEED_KEYS}
        per.append(row)
    mean, std = {}, {}
    for key in [k for k in per[0] if k != "seed"]:
        m_, s_ = mean_std([r[key] for r in per])
        mean[key], std[key] = m_, s_
    res["seeds"] = dict(n=len(seeds), seeds=seeds, mean=mean, std=std, per_seed=per,
                        note="std is the sample std (ddof=1) over seeds; null when only one seed was trained")

    # ---- robustness
    rob = {"finetuned_v2": probe(ft, norm, raw64)}
    rob["scratch_v2"] = probe(sc, norm, raw64)
    v1 = os.path.join(CKPT, "v1", "finetuned.pt")
    if os.path.exists(v1):
        m1, c1 = load_checkpoint(v1)
        rob["finetuned_v1_before_fix"] = probe(m1, c1.get("norm", "v1"), raw64)
    res["robustness"] = rob

    meta = torch.load(ck("pretrained", 0), map_location="cpu", weights_only=True).get("meta", {}) if os.path.exists(ck("pretrained", 0)) else {}
    fmeta = torch.load(ck("finetuned", 0), map_location="cpu", weights_only=True).get("meta", {})
    res["config"] = dict(cfg, params=sum(p.numel() for p in ft.parameters()), labels=labels, train_scenes=3000,
                         pretrain_steps=meta.get("steps"), finetune_steps=fmeta.get("steps"), seeds_trained=seeds,
                         augmentation=dict(elev_offset_m=[-800, 1500], sar_offset_db=[-3, 3], optical_gain=[0.8, 1.25], p=0.85),
                         generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(os.path.join(ROOT, "artifacts", "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    s = res["scratch"]["All sensors"]
    print("TerraFM  mIoU %.3f flood IoU %.3f veg MAE %.3f" % (a["miou"], a["flood_iou"], a["veg_mae"]))
    print("Scratch  mIoU %.3f flood IoU %.3f veg MAE %.3f" % (s["miou"], s["flood_iou"], s["veg_mae"]))
    print("recon RMSE single %.4f  shift-ens %.4f" % (res["recon"]["test64"]["single_pass"], res["recon"]["test64"]["shift_ensemble"]))
    print("robustness gate:", "PASS" if rob["finetuned_v2"]["all_within_gate"] else "FAIL", f"({time.time() - t0:.0f}s)")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TerraScope simulator benchmark")
    ap.add_argument("--paper", action="store_true", help="paper grid -> artifacts/metrics_paper_sim.json (see core/paper_eval.py)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--limit", type=int, default=None, help="first N scenes of each test split (smoke tests)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true", help="read artifacts/dryrun/ckpt, 40 scenes, 50 resamples")
    a = ap.parse_args()
    if a.paper:
        from . import train as _T
        from .paper_eval import main as paper
        if a.dry_run:
            a.ckpt_dir = a.ckpt_dir or os.path.join(ROOT, "artifacts", "dryrun", "ckpt")
            a.limit, a.boot = a.limit or 40, min(a.boot, 50)
            a.out = a.out or os.path.join(ROOT, "artifacts", "dryrun", "metrics_paper_sim.json")
        _T.set_dirs(a.ckpt_dir, None)
        paper([int(s) for s in a.seeds.split(",") if s.strip()], a.limit, a.boot, a.out, a.device)
    else:
        main()
