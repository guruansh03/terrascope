"""Training pipeline:  pretrain (self-supervised)  ->  finetune (few labels)  vs  scratch / U-Net baselines.

    python -m terrascope.core.train data
    python -m terrascope.core.train pretrain --steps 3000 --seed 0 [--no-sensor-dropout] [--no-cloud] [--no-mask]
    python -m terrascope.core.train finetune --steps 500 --labels 100 --seed 0 [--init-variant full|nodrop|nocloud|nomask]
    python -m terrascope.core.train scratch  --steps 500 --labels 100 --seed 0
    python -m terrascope.core.train unet     --steps 500 --labels 100 --seed 0   # supervised U-Net baseline
    python -m terrascope.core.train calibrate              # temperature scaling -> artifacts/calibration.json
    python -m terrascope.core.train sim_all --seeds 0,1,2  # full pretrain/finetune/scratch per seed + calibrate + evaluate
    python -m terrascope.core.train paper_sim --seeds 0,1,2  # paper grid (see below), resumable
    python -m terrascope.core.train paper_sim --dry-run    # same grid, tiny steps, separate dirs (smoke test)

Checkpoint names (``artifacts/ckpt``):
  full pretraining (default; unchanged)   pretrained.pt / pretrained_sN.pt   -> finetuned.pt / finetuned_sN.pt
  ablation variant V in nodrop|nocloud|nomask   pretrained_V_sN.pt  -> finetuned_V_sN.pt   (explicit _s0 for seed 0)
  scratch (same TerraFM, random init)     scratch.pt / scratch_sN.pt
  U-Net baseline                          unet_sN.pt
Pretraining ablations (each removes one component of the self-supervised recipe):
  nodrop   = --no-sensor-dropout : all three sensors always given to the encoder
  nocloud  = --no-cloud          : encoder input optical is the CLEAN image (no cloud-gap filling task)
  nomask   = --no-mask           : patch mask ratio 0 (no masked-token prediction)
All variants consume the random-number stream identically, so runs with the same seed see the same batches.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

from .data import ensure_data, load, load_raw, make_batch, augment
from .model import TerraFM, sample_avail
from .metrics import evaluate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.environ.get("TERRASCOPE_CKPT_DIR") or os.path.join(ROOT, "artifacts", "ckpt")
LOGS = os.environ.get("TERRASCOPE_LOG_DIR") or os.path.join(ROOT, "artifacts", "logs")
NORM = "v2"
CFG = dict(dim=96, depth=4, heads=4, dec_dim=64, dec_depth=2, patch=4, norm=NORM, cloud_head=True)
UNET_CFG = dict(arch="unet", base=32, norm=NORM, cloud_head=True)
DEFAULTS = dict(pretrain_steps=3000, ft_steps=500, labels=100)
VARIANTS = ("full", "nodrop", "nocloud", "nomask")
DRY = dict(pretrain_steps=4, ft_steps=4)


def set_dirs(ckpt=None, logs=None):
    global CKPT, LOGS
    if ckpt:
        CKPT = os.path.abspath(ckpt)
    if logs:
        LOGS = os.path.abspath(logs)
    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)


def suffix(seed):
    return "" if int(seed) == 0 else f"_s{int(seed)}"


def ckpt_path(name, seed=0):
    """Legacy helper: <name>[_sN].pt"""
    return os.path.join(CKPT, f"{name}{suffix(seed)}.pt")


def path_of(name):
    return os.path.join(CKPT, f"{name}.pt")


def pretrained_name(variant, seed):
    return f"pretrained{suffix(seed)}" if variant == "full" else f"pretrained_{variant}_s{int(seed)}"


def finetuned_name(variant, seed):
    return f"finetuned{suffix(seed)}" if variant == "full" else f"finetuned_{variant}_s{int(seed)}"


def scratch_name(seed):
    return f"scratch{suffix(seed)}"


def unet_name(seed):
    return f"unet_s{int(seed)}"


def variant_flags(variant):
    """'full' or '_'-joined subset of nodrop/nocloud/nomask -> dict(dropout, cloud, mask)."""
    parts = set() if variant == "full" else set(variant.split("_"))
    bad = parts - {"nodrop", "nocloud", "nomask"}
    if bad or not variant:
        raise ValueError(f"unknown pretraining variant {variant!r}")
    return dict(dropout="nodrop" not in parts, cloud="nocloud" not in parts, mask="nomask" not in parts)


def variant_from_args(no_drop, no_cloud, no_mask):
    parts = [p for p, on in (("nodrop", no_drop), ("nocloud", no_cloud), ("nomask", no_mask)) if on]
    return "_".join(parts) if parts else "full"


def pick_device(want="auto"):
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return want


def lr_at(step, total, base, warm=40):
    if step < warm:
        return base * (step + 1) / warm
    return base * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))) * 0.98 + base * 0.02


class Log:
    def __init__(self, name):
        os.makedirs(LOGS, exist_ok=True)
        self.path = os.path.join(LOGS, f"{name}.jsonl")
        open(self.path, "w").close()
        self.t0 = time.time()

    def __call__(self, **kw):
        kw["t"] = round(time.time() - self.t0, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        print(json.dumps(kw), flush=True)


def save(model, name, seed, extra=None, cfg=None):
    """``name`` is the full checkpoint stem (e.g. 'finetuned_s1'); legacy short names get the seed suffix."""
    os.makedirs(CKPT, exist_ok=True)
    blob = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "cfg": cfg or CFG,
            "meta": dict(seed=int(seed), norm=NORM, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))}
    p = path_of(name)
    tmp = p + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, p)


def pretrain_batch(d, idx, flags, bs, device=None):
    """One pretraining batch. RNG is consumed identically for every variant (paired ablations)."""
    x, tgt = make_batch(d, idx, NORM, aug=True, device=device)
    if not flags["cloud"]:
        x = tgt.clone()                      # encoder sees the clean optical image
    x, tgt = augment(x, tgt)
    avail = sample_avail(bs)
    if not flags["dropout"]:
        avail = torch.ones_like(avail)
    ratio = 0.0 if torch.rand(1) < 0.2 else float(torch.empty(1).uniform_(0.4, 0.8))
    if not flags["mask"]:
        ratio = 0.0
    return x, tgt, avail.to(x.device), ratio


def pretrain(steps, bs=48, seed=0, device="cpu", variant="full"):
    flags = variant_flags(variant)
    ensure_data()
    d = load_raw("train64")
    N = d["optical"].shape[0]
    torch.manual_seed(1000 * seed + 0)
    model = TerraFM(**CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05, betas=(0.9, 0.95))
    name = pretrained_name(variant, seed)
    log = Log(name.replace("pretrained", "pretrain", 1))
    print(f"[pretrain] {name} variant={variant} flags={flags} params {sum(p.numel() for p in model.parameters())}", flush=True)
    ema = None
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 1e-3)
        idx = torch.randint(0, N, (bs,))
        x, tgt, avail, ratio = pretrain_batch(d, idx, flags, bs, device)
        loss = model.recon_loss(x, avail, tgt, ratio)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        if step % 10 == 0 or step == steps - 1:
            log(step=step, loss=round(ema, 5))
    save(model, name, seed, dict(steps=steps, variant=variant, flags=flags, wall_s=round(time.time() - t0, 1), device=str(device)))
    return name


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum()
    return 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)


def finetune(name, init, steps, n_labels, bs=24, lr=1e-3, aux=0.5, seed=0, device="cpu", arch="terrafm"):
    """Supervised training on the first ``n_labels`` scenes. ``name``/``init`` are checkpoint stems
    (legacy short names 'finetuned'/'scratch'/'pretrained' get the seed suffix appended)."""
    if name in ("finetuned", "scratch"):
        name = name + suffix(seed)
    if init == "pretrained":
        init = pretrained_name("full", seed)
    ensure_data()
    d = load_raw("train64")
    val = load("val64", NORM)
    torch.manual_seed(1000 * seed + 1)
    if arch == "unet":
        from .unet import UNet
        model, cfg, aux = UNet(**UNET_CFG), UNET_CFG, 0.0
    else:
        model, cfg = TerraFM(**CFG), CFG
    if init:
        model.load_state_dict(torch.load(path_of(init), map_location="cpu", weights_only=True)["model"])
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
    log = Log(name)
    N = min(n_labels, d["optical"].shape[0])  # first n_labels scenes are the labelled subset
    pw = torch.tensor(3.0, device=device)
    t0 = time.time()
    for step in range(steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, 30)
        idx = torch.randint(0, N, (bs,))
        x, tgt = make_batch(d, idx, NORM, aug=True, device=device)
        x, tgt, lc, fl, veg, cl = augment(x, tgt, *(d[k][idx].to(device) for k in ("lc", "flood", "veg", "cloud")))
        avail = sample_avail(bs).to(device)
        out = model.forward_tasks(x, avail) if arch == "unet" else model.forward_all(x, avail)
        l_lc = F.cross_entropy(out["lc"], lc)
        l_fl = F.binary_cross_entropy_with_logits(out["flood"], fl, pos_weight=pw) + dice_loss(out["flood"], fl)
        l_vg = F.smooth_l1_loss(out["veg"], veg, beta=0.05) * 5
        l_rc = F.mse_loss(out["recon"], tgt) if aux > 0 else torch.zeros((), device=device)
        # cloud head: soft BCE against simulator opacity, only where optical was given to the model
        oa = avail[:, 0].float()
        l_cl = (F.binary_cross_entropy_with_logits(out["cloud"], cl, reduction="none").mean((1, 2)) * oa).sum() / oa.sum().clamp(min=1)
        loss = l_lc + l_fl + l_vg + aux * l_rc + l_cl
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            rec = dict(step=step, loss=round(loss.item(), 4), lc=round(l_lc.item(), 4), flood=round(l_fl.item(), 4),
                       veg=round(l_vg.item(), 4), cloud=round(l_cl.item(), 4))
            if step % 50 == 0 or step == steps - 1:
                m = evaluate(model, val, [True, True, True], limit=100)
                rec.update(val_miou=round(m["miou"], 4), val_flood_iou=round(m["flood_iou"], 4), val_veg_mae=round(m["veg_mae"], 4))
            log(**rec)
    save(model.eval(), name, seed, dict(steps=steps, labels=n_labels, init=init, arch=arch,
                                        wall_s=round(time.time() - t0, 1), device=str(device)), cfg=cfg)
    return name


def run_seed(seed, a, device):
    t = time.time()
    pretrain(a.pretrain_steps, seed=seed, device=device)
    finetune("finetuned", "pretrained", a.ft_steps, a.labels, seed=seed, device=device)
    finetune("scratch", None, a.ft_steps, a.labels, aux=0.0, seed=seed, device=device)
    print(f"[seed {seed}] done in {time.time() - t:.0f} s", flush=True)


def paper_runs(seeds, variants=VARIANTS):
    """The paper grid as an ordered list of (run name, kind, kwargs) with dependencies first."""
    runs = []
    for s in seeds:
        for v in variants:
            runs.append((pretrained_name(v, s), "pretrain", dict(seed=s, variant=v)))
            runs.append((finetuned_name(v, s), "finetune", dict(seed=s, init=pretrained_name(v, s))))
        runs.append((scratch_name(s), "scratch", dict(seed=s)))
        runs.append((unet_name(s), "unet", dict(seed=s)))
    return runs


def paper_sim(seeds, pretrain_steps, ft_steps, labels, device, force=False, variants=VARIANTS):
    """Resumable: a run is skipped when its checkpoint exists (unless ``force``). Times -> logs/paper_sim_times.json."""
    ensure_data()
    times_p = os.path.join(LOGS, "paper_sim_times.json")
    try:
        with open(times_p) as f:
            times = json.load(f)
    except (OSError, ValueError):
        times = {}
    t_all = time.time()
    for name, kind, kw in paper_runs(seeds, variants):
        if os.path.exists(path_of(name)) and not force:
            print(f"[paper_sim] skip {name} (checkpoint exists)", flush=True)
            continue
        t = time.time()
        s = kw["seed"]
        if kind == "pretrain":
            pretrain(pretrain_steps, seed=s, device=device, variant=kw["variant"])
        elif kind == "finetune":
            finetune(name, kw["init"], ft_steps, labels, seed=s, device=device)
        elif kind == "scratch":
            finetune(name, None, ft_steps, labels, aux=0.0, seed=s, device=device)
        else:
            finetune(name, None, ft_steps, labels, seed=s, device=device, arch="unet")
        dt = time.time() - t
        times[name] = dict(seconds=round(dt, 1), device=str(device), steps=pretrain_steps if kind == "pretrain" else ft_steps,
                           finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(times_p, "w") as f:
            json.dump(times, f, indent=1)
        print(f"[paper_sim] {name} ({kind}) took {dt:.1f} s on {device}", flush=True)
    print(f"[paper_sim] grid done in {time.time() - t_all:.1f} s", flush=True)
    return times


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["data", "pretrain", "finetune", "scratch", "unet", "seed", "calibrate", "evaluate",
                                      "sim_all", "paper_sim"])
    ap.add_argument("--steps", type=int, default=None, help="steps for pretrain/finetune/scratch/unet stages")
    ap.add_argument("--pretrain-steps", type=int, default=DEFAULTS["pretrain_steps"])
    ap.add_argument("--ft-steps", type=int, default=DEFAULTS["ft_steps"])
    ap.add_argument("--labels", type=int, default=DEFAULTS["labels"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2", help="comma list for sim_all / paper_sim")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--no-sensor-dropout", action="store_true", help="pretrain ablation: always all sensors")
    ap.add_argument("--no-cloud", action="store_true", help="pretrain ablation: encoder sees clean optical")
    ap.add_argument("--no-mask", action="store_true", help="pretrain ablation: mask ratio 0")
    ap.add_argument("--init-variant", default="full", help="finetune from pretrained variant full|nodrop|nocloud|nomask")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="paper_sim: pretraining variants")
    ap.add_argument("--force", action="store_true", help="paper_sim: retrain even if checkpoints exist")
    ap.add_argument("--dry-run", action="store_true",
                    help=f"tiny steps ({DRY['pretrain_steps']}/{DRY['ft_steps']}) into artifacts/dryrun/ (never touches real checkpoints)")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint directory (default artifacts/ckpt)")
    ap.add_argument("--log-dir", default=None, help="log directory (default artifacts/logs)")
    a = ap.parse_args(argv)
    if a.dry_run:
        a.pretrain_steps, a.ft_steps = (a.steps or DRY["pretrain_steps"]), (a.steps or DRY["ft_steps"])
        a.ckpt_dir = a.ckpt_dir or os.path.join(ROOT, "artifacts", "dryrun", "ckpt")
        a.log_dir = a.log_dir or os.path.join(ROOT, "artifacts", "dryrun", "logs")
    set_dirs(a.ckpt_dir, a.log_dir)
    torch.set_num_threads(a.threads or os.cpu_count() or 2)
    device = pick_device(a.device)
    print(f"[train] stage={a.stage} device={device} threads={torch.get_num_threads()} torch={torch.__version__} ckpt_dir={CKPT}", flush=True)
    seeds = [int(v) for v in a.seeds.split(",") if v.strip()]
    if a.stage == "data":
        ensure_data()
    elif a.stage == "pretrain":
        pretrain(a.steps or a.pretrain_steps, seed=a.seed, device=device,
                 variant=variant_from_args(a.no_sensor_dropout, a.no_cloud, a.no_mask))
    elif a.stage == "finetune":
        variant_flags(a.init_variant)
        finetune(finetuned_name(a.init_variant, a.seed), pretrained_name(a.init_variant, a.seed), a.steps or a.ft_steps,
                 a.labels, seed=a.seed, device=device)
    elif a.stage == "scratch":
        finetune(scratch_name(a.seed), None, a.steps or a.ft_steps, a.labels, aux=0.0, seed=a.seed, device=device)
    elif a.stage == "unet":
        finetune(unet_name(a.seed), None, a.steps or a.ft_steps, a.labels, seed=a.seed, device=device, arch="unet")
    elif a.stage == "seed":
        run_seed(a.seed, a, device)
    elif a.stage == "calibrate":
        from .calibrate import main as cal
        cal()
    elif a.stage == "evaluate":
        from .evaluate import main as ev
        ev(labels=a.labels)
    elif a.stage == "paper_sim":
        variants = [v.strip() for v in a.variants.split(",") if v.strip()]
        for v in variants:
            variant_flags(v)
        paper_sim(seeds, a.pretrain_steps, a.ft_steps, a.labels, device, force=a.force, variants=variants)
    else:  # sim_all
        ensure_data()
        for s in seeds:
            run_seed(s, a, device)
        from .calibrate import main as cal
        cal()
        from .evaluate import main as ev
        ev(labels=a.labels)


if __name__ == "__main__":
    sys.exit(main())
