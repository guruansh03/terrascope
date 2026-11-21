"""Fine-tune one (model, task, seed) on real data. Plain PyTorch loop.

* CUDA autocast (bf16 when supported, else fp16 + GradScaler), AdamW, warm-up + cosine LR
* encoder LR / head LR split, optional frozen encoder
* fixed number of random 224 px crops per epoch, early stopping on the validation metric (full chips, tiled)
* GPU-memory guard: CUDA OOM -> batch size halved and the epoch continues
* resumable: ``<ckpt>.last.pt`` is written every epoch and picked up on restart
* outputs: best checkpoint ``artifacts/ckpt/real/<task>/<model>_s<seed>.pt`` + ``.json`` run log

    python -m terrascope.fm.finetune --model prithvi_tiny --task flood --seed 0
"""
from __future__ import annotations

import argparse, json, logging, math, os, random, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from .data import ChipDataset, DiskChipCache, MemoryChipCache, sample_to_arrays
from .metrics import ReconMeter, SegMeter
from .model import build_task_model, ckpt_path, model_sensors, predict_tiled, save_checkpoint
from .specs import MODELS, TASK_OUT, TILE, TRAIN_DEFAULTS

log = logging.getLogger("terrascope.fm.finetune")


class MissingWeights(FileNotFoundError):
    pass


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_dtype_for(device: torch.device):
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def task_loss(task: str, out: torch.Tensor, batch: dict) -> torch.Tensor:
    if task in ("flood", "burn"):
        y = batch["mask"]
        valid = (y >= 0).float()
        logit = out[:, 0].float()
        t = y.clamp(min=0).float()
        bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
        p = torch.sigmoid(logit) * valid
        inter = (p * t).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + (t * valid).sum() + 1)
        return bce + 0.5 * dice
    if task == "landcover":
        return F.cross_entropy(out.float(), batch["mask"], ignore_index=-1)
    if task == "cloud":
        return F.l1_loss(out.float(), batch["target"].float())
    raise KeyError(task)


@torch.no_grad()
def evaluate_model(model, task: str, records, load_fn, sensors, device, amp_dtype=None, max_chips=None,
                   timing: bool = False, per_chip: bool = False) -> dict:
    """Full-chip tiled evaluation. Returns metrics (+ ms per chip when ``timing``; + ``chip_stats`` when
    ``per_chip``: one confusion matrix per chip for segmentation, (sse, n, ssim) per chip for 'cloud') so that
    chip-level bootstrap CIs can recompute the POOLED metric exactly."""
    model.eval()
    meter = ReconMeter() if task == "cloud" else SegMeter(TASK_OUT[task] if TASK_OUT[task] > 1 else 2)
    times, chips = [], []
    for i, rec in enumerate(records):
        if max_chips and i >= max_chips:
            break
        a = sample_to_arrays(load_fn(rec), sensors)
        o, s, d = (torch.from_numpy(a[k])[None].to(device) for k in ("optical", "sar", "dem"))
        av = torch.from_numpy(a["avail"])[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = predict_tiled({task: model}, o, s, d, av, want_feats=False, amp_dtype=amp_dtype)
        y = out[task]
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        if task == "cloud":
            pred = y.cpu().numpy()
            if per_chip:
                one = ReconMeter()
                one.update(pred, a["target"])
                chips.append((one.se, one.n, one.ssims[0]))
            meter.update(pred, a["target"])
        else:
            pred = (y[0] > 0).cpu().numpy().astype(np.int64) if TASK_OUT[task] == 1 else y.argmax(0).cpu().numpy()
            if per_chip:
                before = meter.conf.copy()
                meter.update(pred, a["mask"])
                chips.append(meter.conf - before)
            else:
                meter.update(pred, a["mask"])
    res = meter.result()
    if per_chip:
        res["chip_stats"] = chips
    if task in ("flood", "burn"):
        res["score"] = res.get("iou_pos")
    elif task == "landcover":
        res["score"] = res.get("miou")
    else:
        res["score"] = None if res["rmse"] is None else -res["rmse"]
    if timing:
        res["ms_per_chip"] = times
    return res


def _loader(ds, bs, samples, seed, workers):
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(ds, replacement=True, num_samples=samples, generator=g)
    return DataLoader(ds, batch_size=bs, sampler=sampler, num_workers=workers, drop_last=True,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def finetune(model_id: str, task: str, seed: int = 0, *, epochs: int = 30, patience: int = 6,
             samples_per_epoch: int = 800, batch_size: int | None = None, freeze_encoder: bool = False,
             device: str | None = None, weights_path: str | None = None, require_weights: bool = True,
             data_root: str | None = None, records: dict | None = None, load_fn=None, out_root: str | None = None,
             num_workers: int | None = None, crop: int = TILE, max_val_chips: int | None = None,
             logger=None, max_steps: int | None = None, cache_dir: str | None = None,
             val_cache_bytes: int = 1_500_000_000) -> dict:
    """Train and save the best checkpoint. ``records``/``load_fn`` override the real dataset (tests)."""
    lg = logger or log
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(seed)
    hp = dict(TRAIN_DEFAULTS[model_id])
    bs = int(batch_size or hp["batch_size"])
    sensors = model_sensors(model_id, task)
    if records is None:
        from ..real.datasets import for_task
        spec = for_task(task)
        root = data_root or spec.default_root()
        records = {s: spec.list_split(root, s) for s in ("train", "val", "test")}
        load_fn = spec.load
        dataset_name = spec.name
        if cache_dir is None:
            from ..real.common import data_root
            cache_dir = os.path.join(data_root(), "_cache", spec.name)
    else:
        dataset_name = "custom"
    if not records.get("train") or not records.get("val"):
        raise FileNotFoundError(f"no training/validation chips for task {task}; run get_data.bat first")
    if weights_path is None:
        from .weights import weights_path as wp
        weights_path = wp(model_id)
        if weights_path is None and require_weights and MODELS[model_id]["family"] != "terrafm" \
                and MODELS[model_id].get("needs_weights", True):
            raise MissingWeights(f"pretrained weights for {model_id} not found; run get_data.bat weights")
    path = ckpt_path(task, model_id, seed, out_root)
    last = path + ".last.pt"
    model = build_task_model(model_id, task, weights_path).to(dev)
    enc_params = [p for p in model.encoder.parameters()]
    if freeze_encoder:
        for p in enc_params:
            p.requires_grad_(False)
    groups = [dict(params=list(model.head.parameters()), lr=hp["lr_head"])]
    if not freeze_encoder:
        groups.append(dict(params=enc_params, lr=hp["lr_encoder"]))
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    amp_dtype = amp_dtype_for(dev)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    steps_per_epoch = max(1, samples_per_epoch // bs)
    total = epochs * steps_per_epoch
    warm = max(1, int(0.05 * total))
    state = dict(epoch=0, step=0, best=-math.inf, best_epoch=-1, bad=0, history=[], bs=bs)
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        bs = state["bs"]
        lg.info("resuming %s/%s seed %d from epoch %d (best %.4f)", model_id, task, seed, state["epoch"], state["best"])
    workers = num_workers if num_workers is not None else (2 if dev.type == "cuda" else 0)
    train_load = DiskChipCache(load_fn, cache_dir) if cache_dir else load_fn
    val_load = MemoryChipCache(train_load, val_cache_bytes)
    ds = ChipDataset(records["train"], train_load, sensors, crop=crop, train=True, seed=seed)
    # ONE loader for the whole run (persistent workers: no per-epoch process respawn on Windows)
    loader = _loader(ds, bs, steps_per_epoch * bs, seed * 1000 + state["epoch"], workers)
    lg.info("finetune %s on %s (%s) seed %d | device %s amp %s | bs %d | %d train / %d val chips | sensors %s | "
            "encoder %s | params %d", model_id, task, dataset_name, seed, dev, amp_dtype, bs, len(records["train"]),
            len(records["val"]), sensors, "frozen" if freeze_encoder else "trainable", model.n_params())
    t_start = time.time()
    while state["epoch"] < epochs:
        ep = state["epoch"]
        model.train()
        losses = []
        t0 = time.time()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        it = iter(loader)
        n_done = 0
        while n_done < steps_per_epoch:
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            out = loss = None
            lr_f = min(1.0, (state["step"] + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, state["step"] / total))))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * max(lr_f, 0.02)
            try:
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(batch["optical"], batch["sar"], batch["dem"], batch["avail"])
                    loss = task_loss(task, out, batch)
                opt.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            except torch.cuda.OutOfMemoryError:
                out = loss = batch = None  # drop references to the failed step's activations
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if bs <= 1:
                    raise
                bs = max(1, bs // 2)
                state["bs"] = bs
                lg.warning("CUDA out of memory -> batch size reduced to %d", bs)
                del it, loader
                loader = _loader(ds, bs, steps_per_epoch * bs, seed * 1000 + ep + 17, workers)
                it = iter(loader)
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep} step {state['step']}")
            losses.append(float(loss.detach()))
            state["step"] += 1
            n_done += 1
            if max_steps and n_done >= max_steps:
                break
        t_train = time.time() - t0
        val = evaluate_model(model, task, records["val"], val_load, sensors, dev, amp_dtype, max_chips=max_val_chips)
        t_val = time.time() - t0 - t_train
        score = val["score"] if val["score"] is not None else -math.inf
        improved = score > state["best"]
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
        state["history"].append(dict(epoch=ep, loss=float(np.mean(losses)) if losses else None, val=val["score"],
                                     lr=opt.param_groups[0]["lr"], secs=round(time.time() - t0, 1),
                                     gpu_peak_gb=round(mem, 2)))
        if improved:
            state.update(best=score, best_epoch=ep, bad=0)
            save_checkpoint(path, model, seed, freeze_encoder,
                            dict(val=val, epoch=ep, dataset=dataset_name, sensors=sensors, complete=False))
        else:
            state["bad"] += 1
        state["epoch"] = ep + 1
        if len(state["history"]) == 1:
            ep_s = time.time() - t0
            lg.info("  measured: epoch took %.0f s (train %.0f s incl. first-pass chip caching, val %.0f s) -> this run "
                    "<= %.1f min for %d epochs (early stopping usually ends sooner)", ep_s, t_train, t_val,
                    ep_s * (epochs - ep) / 60, epochs - ep)
        el = time.time() - t_start
        eta = el / max(1, len(state["history"])) * (epochs - state["epoch"])
        lg.info("  ep %2d loss %.4f val %.4f%s | %.0fs | gpu peak %.2f GB | ETA <= %.1f min", ep,
                state["history"][-1]["loss"] or float("nan"), score, " *" if improved else "",
                time.time() - t0, mem, eta / 60)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict() if scaler else None,
                        state=state), last + ".tmp")
        os.replace(last + ".tmp", last)
        if state["bad"] >= patience:
            lg.info("  early stop (no improvement for %d epochs; best epoch %d)", patience, state["best_epoch"])
            break
    if not os.path.exists(path):
        raise RuntimeError("no checkpoint was written (validation never produced a finite score)")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["complete"] = True
    ck["history"] = state["history"]
    torch.save(ck, path + ".tmp")
    os.replace(path + ".tmp", path)
    summary = dict(model_id=model_id, task=task, seed=seed, dataset=dataset_name, best_epoch=state["best_epoch"],
                   best_val=ck.get("val"), epochs_run=len(state["history"]), params=model.n_params(),
                   encoder_frozen=freeze_encoder, batch_size=bs, device=str(dev), amp=str(amp_dtype),
                   split_sizes={k: len(v) for k, v in records.items()}, history=state["history"],
                   weights=os.path.basename(weights_path) if weights_path else None, complete=True,
                   minutes=round((time.time() - t_start) / 60, 2))
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(summary, f, indent=1)
    try:
        os.remove(last)
    except OSError:
        pass
    return summary


def is_complete(task: str, model_id: str, seed: int, out_root: str | None = None) -> bool:
    p = os.path.splitext(ckpt_path(task, model_id, seed, out_root))[0] + ".json"
    try:
        with open(p) as f:
            return bool(json.load(f).get("complete"))
    except (OSError, ValueError):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--task", required=True, choices=["flood", "burn", "cloud", "landcover"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--freeze-encoder", action="store_true")
    a = ap.parse_args(argv)
    from ..utils.joblog import run_job
    with run_job(f"finetune_{a.model}_{a.task}_s{a.seed}") as lg:
        finetune(a.model, a.task, a.seed, epochs=a.epochs, patience=a.patience, batch_size=a.batch_size,
                 freeze_encoder=a.freeze_encoder, logger=lg)


if __name__ == "__main__":
    main()
