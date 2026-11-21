"""Evaluate every fine-tuned (task, model, seed) checkpoint on the TEST split -> artifacts/metrics_real.json
(run by evaluate.bat). Schema: docs/PHASE3_CONTRACT.md ("metrics_real.json schema").

    python scripts/pc/evaluate_real.py [--tasks flood,burn] [--models ...] [--max-chips N] [--device cuda|cpu]

* metrics are pooled over all test pixels of a split (dataset-level IoU), then mean +- std (sample std) over seeds
* params are measured from the loaded model (encoder + task head); ``encoder_params`` is reported separately
* infer_ms_per_chip: wall time of the full tiled forward pass per test chip on the eval device (CUDA synchronised),
  pooled over chips of all seeds, first chip of each seed excluded as warm-up
* simple reference baselines are included where they are cheap and well defined (labelled as baselines)
* uncertainty (terrascope.fm.stats):
  - ``ci95_seeds``     95 % Student-t interval of the mean over seeds (null with one seed)
  - ``bootstrap_ci95`` 95 % percentile bootstrap over TEST CHIPS (1000 draws, chips resampled with replacement,
                       pooled IoU recomputed from the summed per-chip TP/FP/FN); per seed in ``per_seed`` and for
                       the seed-averaged metric at model level
  - ``paired``         per dataset: model minus terrafm / minus prithvi_tiny on common seeds; t-interval over
                       seeds + paired chip bootstrap (same resampled chips for both models) + two-sided bootstrap p
"""
from __future__ import annotations

import argparse, json, os, sys, time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

OUT = os.path.join(ROOT, "artifacts", "metrics_real.json")
METRIC_KEYS = {  # task -> {schema key: key in evaluate_model() result}
    "flood": {"flood_iou": "iou_pos", "accuracy": "accuracy"},
    "burn": {"burn_iou": "iou_pos", "accuracy": "accuracy"},
    "landcover": {"miou": "miou", "accuracy": "accuracy"},
    "cloud": {"cloud_rmse": "rmse", "cloud_ssim": "ssim"},
}


def _seed_ckpts(ckpt_root, task, model_id):
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(ckpt_root, task, f"{model_id}_s*.pt"))):
        if p.endswith(".last.pt"):
            continue
        js = os.path.splitext(p)[0] + ".json"
        try:
            with open(js) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if meta.get("complete"):
            out.append((int(meta["seed"]), p, meta))
    return out


def baselines(task, records, load_fn, max_chips=None, boot_draws=1000):
    from terrascope.fm.metrics import ReconMeter, SegMeter
    if task == "cloud":
        m = ReconMeter()
        for i, r in enumerate(records):
            if max_chips and i >= max_chips:
                break
            s = load_fn(r)
            m.update(s["optical"], s["target"])
        res = m.result()
        return {"identity_cloudy_input": dict(cloud_rmse=res["rmse"], cloud_ssim=res["ssim"],
                                              note="cloudy input compared with the clear target (no model)")}
    if task == "flood":
        from terrascope.fm.stats import bootstrap_ci95, draws
        m = SegMeter(2)
        any_sar, chips = False, []
        for i, r in enumerate(records):
            if max_chips and i >= max_chips:
                break
            s = load_fn(r)
            if s.get("sar") is None:
                continue
            any_sar = True
            before = m.conf.copy()
            m.update((s["sar"][0] < -15.0).astype(np.int64), s["mask"])
            chips.append(m.conf - before)
        if any_sar:
            res = m.result()
            idx = draws(len(chips), boot_draws)
            return {"sar_vv_threshold_-15dB": dict(
                flood_iou=res["iou_pos"], accuracy=res["accuracy"],
                bootstrap_ci95=dict(flood_iou=bootstrap_ci95(chips, "iou_pos", idx),
                                    accuracy=bootstrap_ci95(chips, "accuracy", idx)),
                note="water if Sentinel-1 VV < -15 dB (fixed heuristic, no training)")}
    if task == "landcover":
        cnt = np.zeros(6, np.int64)
        for i, r in enumerate(records):
            if max_chips and i >= max_chips:
                break
            y = load_fn(r)["mask"]
            cnt += np.bincount(y[y >= 0].ravel(), minlength=6)[:6]
        if cnt.sum():
            return {"majority_class": dict(accuracy=float(cnt.max() / cnt.sum()),
                                           note="always predict the most frequent test class")}
    return {}


def evaluate(tasks=None, models=None, max_chips=None, device=None, ckpt_root=None, data_roots=None, out=OUT,
             logger=None, records_override=None, boot_draws=1000, references=("terrafm", "prithvi_tiny")) -> dict:
    import logging
    import torch
    from terrascope.fm.finetune import amp_dtype_for, evaluate_model
    from terrascope.fm.metrics import mean_std
    from terrascope.fm import stats as st
    from terrascope.fm.model import CKPT_ROOT, load_checkpoint, model_sensors
    from terrascope.fm.specs import LEARNED_TASKS, MODEL_ORDER
    from terrascope.fm.weights import weights_path
    from terrascope.real.datasets import for_task
    lg = logger or logging.getLogger("evaluate_real")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = amp_dtype_for(dev)
    ckpt_root = ckpt_root or CKPT_ROOT
    res = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), device=str(dev),
               device_name=torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               torch=torch.__version__, datasets={}, missing=[],
               notes=["metrics pooled over all test pixels per split; mean/std over seeds (sample std)",
                      "params = encoder + task head, measured from the loaded model",
                      "infer_ms_per_chip pooled over test chips of all seeds; first chip per seed = warm-up, excluded",
                      f"ci95_seeds: Student-t over seeds; bootstrap_ci95: {boot_draws} chip-level resamples, pooled "
                      "metric recomputed from summed per-chip confusion; paired: model - reference"])
    for task in tasks or LEARNED_TASKS:
        spec = for_task(task)
        if records_override is not None:
            recs = records_override[task]
            load_fn = recs["load_fn"]
            split = {k: v for k, v in recs.items() if k != "load_fn"}
        else:
            root = (data_roots or {}).get(task) or spec.default_root()
            try:
                split = {s: spec.list_split(root, s) for s in ("train", "val", "test")}
            except FileNotFoundError as e:
                lg.warning("task %s: dataset not available (%s)", task, e)
                res["missing"].append(f"{task}: dataset not downloaded")
                continue
            load_fn = spec.load
        test = split["test"]
        if not test:
            res["missing"].append(f"{task}: empty test split")
            continue
        ds = dict(task=task, split_sizes={k: len(v) for k, v in split.items()}, models={},
                  baselines=baselines(task, test, load_fn, max_chips, boot_draws))
        n_chips = min(len(test), max_chips) if max_chips else len(test)
        idx = st.draws(n_chips, boot_draws, seed=0)  # same draws for every model -> paired comparisons
        chips_by_model: dict[str, dict] = {}
        if max_chips:
            ds["max_chips_evaluated"] = max_chips
        for mid in models or MODEL_ORDER:
            runs = _seed_ckpts(ckpt_root, task, mid)
            if not runs:
                res["missing"].append(f"{task}/{mid}: no completed checkpoint")
                continue
            per_seed, times, params, enc_params = [], [], None, None
            for seed, p, meta in runs:
                lg.info("evaluating %s / %s / seed %d on %d test chips (%s)", task, mid, seed, len(test), dev)
                m = load_checkpoint(p, weights_path(mid), device=dev)
                params = m.n_params()
                enc_params = int(sum(q.numel() for q in m.encoder.parameters()))
                sensors = (meta.get("sensors") if isinstance(meta.get("sensors"), list) else None) or model_sensors(mid, task)
                r = evaluate_model(m, task, test, load_fn, sensors, dev, amp, max_chips=max_chips, timing=True,
                                   per_chip=True)
                times += r.pop("ms_per_chip")[1:]
                chips = r.pop("chip_stats")
                chips_by_model.setdefault(mid, {})[seed] = chips
                r["bootstrap_ci95"] = {key: st.bootstrap_ci95(chips, src, idx) for key, src in METRIC_KEYS[task].items()}
                if task != "landcover":
                    r.pop("per_class_iou", None)
                per_seed.append(dict(seed=seed, **{k: v for k, v in r.items() if k != "score"}))
                del m
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            entry = dict(params=int(params), encoder_params=enc_params, seeds=[r["seed"] for r in per_seed])
            entry["ci95_seeds"], entry["bootstrap_ci95"] = {}, {}
            for key, src in METRIC_KEYS[task].items():
                entry[key] = mean_std([r.get(src) for r in per_seed])
                entry["ci95_seeds"][key] = st.t_ci95([r.get(src) for r in per_seed])
                entry["bootstrap_ci95"][key] = st.bootstrap_seed_mean_ci95(
                    [chips_by_model[mid][s] for s in entry["seeds"]], src, idx)
            entry["infer_ms_per_chip"] = mean_std(times) if times else dict(mean=None, std=None, n=0)
            entry["per_seed"] = per_seed
            ds["models"][mid] = entry
        ds["paired"] = {}
        for ref in references:
            if ref not in chips_by_model:
                continue
            comp = {}
            for mid, ch in chips_by_model.items():
                if mid == ref:
                    continue
                comp[mid] = {key: st.paired(ch, chips_by_model[ref], src, idx) for key, src in METRIC_KEYS[task].items()}
            ds["paired"][f"vs_{ref}"] = comp
        res["datasets"][spec.name] = ds
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    lg.info("wrote %s", out)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--models", default=None)
    ap.add_argument("--max-chips", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000, help="chip-level bootstrap draws")
    a = ap.parse_args(argv)
    from terrascope.utils.joblog import JobError, run_job
    def _fmt(k, v):
        if v.get("mean") is None:
            return f"{k}=n/a"
        sd = v.get("std")
        return f"{k}={v['mean']:.4f}" + (f"+-{sd:.4f}" if sd is not None else " (1 seed)")

    with run_job("evaluate_real") as log:
        r = evaluate(a.tasks.split(",") if a.tasks else None, a.models.split(",") if a.models else None,
                     a.max_chips, a.device, logger=log, boot_draws=a.bootstrap)
        for name, ds in r["datasets"].items():
            for mid, e in ds["models"].items():
                keys = [k for k in e if isinstance(e[k], dict) and "mean" in e[k]]
                log.info("%-15s %-17s %s", name, mid, "  ".join(
                    _fmt(k, e[k]) for k in keys))
            for ref, comp in (ds.get("paired") or {}).items():
                for mid, mets in comp.items():
                    k = next(iter(mets))
                    d = mets[k]
                    if d and d.get("mean_diff") is not None:
                        ci = d.get("bootstrap_ci95")
                        log.info("  paired %-16s %-17s d%s = %+.4f  boot95 %s  p=%s", ref, mid, k, d["mean_diff"],
                                 f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "n/a", d.get("p_bootstrap_two_sided"))
        if not any(ds["models"] for ds in r["datasets"].values()):
            raise JobError("no fine-tuned checkpoints were found to evaluate", "training has not been run (or failed)",
                           "Run train.bat quick first, then evaluate.bat.")


if __name__ == "__main__":
    main()
