"""Train the model x task x seed grid on the PC (run by train.bat).

    python scripts/pc/train_all.py quick   # 1 seed;  flood + burn;                      all 4 models
    python scripts/pc/train_all.py full    # 3 seeds; flood, burn, cloud, landcover;     all 4 models
    python scripts/pc/train_all.py sim     # TerraFM simulator pipeline (terrascope.core.train) for 3 seeds
    python scripts/pc/train_all.py paper   # 3 seeds; flood + burn; 4 FMs + 2 U-Net baselines; SAME hyper-
                                           # parameters as quick, so quick's seed-0 runs are reused (skipped)

Options: --models a,b  --tasks a,b  --seeds 0,1  --epochs N  --patience N  --freeze-encoder  --allow-cpu  --dry-run
Completed runs (``artifacts/ckpt/real/<task>/<model>_s<seed>.json`` with complete=true) are skipped, interrupted
runs resume from their ``.last.pt``. One failing run does not stop the grid; failures are listed at the end.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
ROOT_DIR_FOR_CLOUD = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # training never downloads; get_data.bat weights does

MODES = {
    "quick": dict(seeds=[0], tasks=["flood", "burn"], epochs=10, patience=4, samples_per_epoch=384, models="fm"),
    "paper": dict(seeds=[0, 1, 2], tasks=["flood", "burn"], epochs=10, patience=4, samples_per_epoch=384,
                  models="all"),
    "full": dict(seeds=[0, 1, 2], tasks=["flood", "burn", "landcover"] + (["cloud"] if os.path.isdir(os.path.join(ROOT_DIR_FOR_CLOUD, "data", "real", "sen12mscr", "patches")) else []), epochs=40, patience=8),
}
# ---- run-time ESTIMATE (not a measurement): FLOPs from the code's own shapes x an ASSUMED sustained throughput
# fwd FLOPs/224px tile ~= 2 * encoder_params * tokens (196 for patch 16; TerraFM: 3136 tokens of a 0.46M encoder
# ~= 2.9 GFLOPs) + ~10 GFLOPs for the UPerLite head; training step ~= 3x forward.
ASSUMED_TFLOPS = 15.0     # sustained bf16 on an RTX 5060 for these small models (assumption, likely pessimistic)
ASSUMED_DATA_S = 0.004    # s per training crop from the mmap chip cache (assumption)
VAL_TILES = {"flood": 89 * 9, "burn": 54 * 9, "cloud": 150, "landcover": 89 * 9}  # approx. val chips x tiles
ENC_GFLOP = {"terrafm": 2.9, "prithvi_tiny": 2.2, "terramind_tiny": 2.3 * 3, "prithvi_100m": 33.8}  # TM: 3 modalities
# U-Net ResNet-18 at 224 px: ~1.8 GFLOPs encoder + ~4 GFLOPs smp decoder (no UPerLite head) -> total ~6 GFLOPs
TOTAL_GFLOP = {"unet_r18_scratch": 6.0, "unet_r18_imagenet": 6.0}


def estimate_minutes(models, tasks, seeds, epochs, samples) -> float:
    tot = 0.0
    for m in models:
        f = (TOTAL_GFLOP[m] if m in TOTAL_GFLOP else ENC_GFLOP.get(m, 34.0) + 10.0) * 1e9
        for t in tasks:
            train_s = epochs * samples * (3 * f / (ASSUMED_TFLOPS * 1e12) + ASSUMED_DATA_S)
            val_s = epochs * VAL_TILES.get(t, 800) * (f / (ASSUMED_TFLOPS * 1e12) + 0.002)
            tot += (train_s + val_s + 60) * len(seeds)  # + ~1 min model build / first-epoch chip caching
    return tot / 60


def measured_minutes(ckpt_root=None) -> dict:
    """{(task, model): [minutes of completed runs]} read from the run logs finetune() writes."""
    import glob
    from terrascope.fm.model import CKPT_ROOT
    out = {}
    for js in glob.glob(os.path.join(ckpt_root or CKPT_ROOT, "*", "*_s*.json")):
        try:
            with open(js) as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if d.get("complete") and d.get("minutes") is not None:
            out.setdefault((d.get("task"), d.get("model_id")), []).append(float(d["minutes"]))
    return out


def remaining_estimate(todo, epochs, samples, ckpt_root=None):
    """Per remaining run: mean MEASURED minutes of finished runs of the same (task, model) when available,
    else the FLOP-based estimate. Returns (total_min, n_from_measured)."""
    meas = measured_minutes(ckpt_root)
    tot, n_meas = 0.0, 0
    for t, m, _ in todo:
        if meas.get((t, m)):
            tot += sum(meas[(t, m)]) / len(meas[(t, m)])
            n_meas += 1
        else:
            tot += estimate_minutes([m], [t], [0], epochs, samples)
    return tot, n_meas


DATA_BAT = {"flood": "get_data.bat floods", "burn": "get_data.bat burn", "cloud": "get_data.bat cloud",
            "landcover": "get_data.bat landcover"}
MIN_FREE_VRAM_GB = 1.5


def gpu_guard(log):
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    log.info("GPU memory free %.2f / %.2f GB", free / 2**30, total / 2**30)
    if free / 2**30 < MIN_FREE_VRAM_GB:
        log.warning("less than %.1f GB of GPU memory is free - close other GPU programs (games, browsers with "
                    "hardware acceleration); training will fall back to smaller batches if it runs out", MIN_FREE_VRAM_GB)
    return free


def run_sim(log, seeds):
    """Delegate to the ML agent's simulator pipeline, adapting to the CLI that is actually present."""
    py = sys.executable
    helptext = subprocess.run([py, "-m", "terrascope.core.train", "--help"], cwd=ROOT, capture_output=True,
                              text=True).stdout
    if "sim_all" in helptext:
        cmds = [[py, "-m", "terrascope.core.train", "sim_all", "--seeds", ",".join(map(str, seeds)), "--device", "auto"]]
    elif "--seed" in helptext:
        cmds = []
        for s in seeds:
            cmds += [[py, "-m", "terrascope.core.train", st, "--seed", str(s)] for st in ("pretrain", "finetune", "scratch")]
    else:
        log.warning("terrascope.core.train has no --seed option: running the single-seed stages")
        cmds = [[py, "-m", "terrascope.core.train", st] for st in ("pretrain", "finetune", "scratch")]
    for c in cmds:
        log.info("$ %s", " ".join(c[1:]))
        p = subprocess.Popen(c, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.info("  | %s", line.rstrip())
        if p.wait() != 0:
            from terrascope.utils.joblog import JobError
            raise JobError(f"simulator stage failed: {' '.join(c[2:])} (exit {p.returncode})",
                           "see the lines marked '|' above", "Send this log file to Claude.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["quick", "full", "sim", "paper"])
    ap.add_argument("--models", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--samples-per-epoch", type=int, default=None)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from terrascope.utils.joblog import JobError, MissingData, run_job
    with run_job(f"train_{a.mode}") as log:
        if a.mode == "sim":
            seeds = [int(s) for s in (a.seeds or "0,1,2").split(",")]
            run_sim(log, seeds)
            return
        import torch
        from terrascope.fm.finetune import MissingWeights, finetune, is_complete
        from terrascope.fm.specs import FOUNDATION_MODELS, MODEL_ORDER
        from terrascope.real.datasets import for_task
        cfg = dict(MODES[a.mode])
        models = a.models.split(",") if a.models else \
            list(MODEL_ORDER if cfg.get("models") == "all" else FOUNDATION_MODELS)
        tasks = a.tasks.split(",") if a.tasks else cfg["tasks"]
        seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
        epochs, patience = a.epochs or cfg["epochs"], a.patience or cfg["patience"]
        samples = a.samples_per_epoch or cfg.get("samples_per_epoch", 800)
        if not torch.cuda.is_available() and not a.allow_cpu and not a.dry_run:
            raise JobError("no CUDA GPU visible to PyTorch", "CPU-only torch was installed, or the NVIDIA driver is "
                           "missing/old (RTX 50xx needs CUDA 12.8 wheels)",
                           "Run gpu_check.bat and send its log; re-run setup.bat if torch is CPU-only.")
        # data availability up front (fail fast, with the .bat to run)
        for t in tasks:
            spec = for_task(t)
            try:
                n = len(spec.list_split(spec.default_root(), "train"))
            except FileNotFoundError:
                n = 0
            if n == 0:
                raise MissingData(spec.default_root(), bat=DATA_BAT[t], what=f"no training data for task '{t}'")
            log.info("task %-9s dataset %-14s train chips: %d", t, spec.name, n)
        grid = [(t, m, s) for s in seeds for t in tasks for m in models]
        todo = [g for g in grid if not is_complete(*g)]
        log.info("grid: %d runs (%d already complete, %d to do) | epochs %d patience %d | %d crops/epoch", len(grid),
                 len(grid) - len(todo), len(todo), epochs, patience, samples)
        est, n_meas = remaining_estimate(todo, epochs, samples)
        log.info("ESTIMATE for the %d remaining runs: ~%.0f min (%d runs from MEASURED times of finished runs of the "
                 "same model/task, %d from FLOPs at an assumed %.0f TFLOPS); the log prints the measured epoch time "
                 "after the first epoch of each run", len(todo), est, n_meas, len(todo) - n_meas, ASSUMED_TFLOPS)
        if a.dry_run:
            for g in todo:
                log.info("  would run task=%s model=%s seed=%d", *g)
            return
        done_minutes, failures, results = [], {}, []
        for i, (t, m, s) in enumerate(todo, 1):
            eta = (sum(done_minutes) / len(done_minutes) * (len(todo) - i + 1)) if done_minutes else None
            log.info("==== run %d/%d: task=%s model=%s seed=%d%s", i, len(todo), t, m, s,
                     f" | ETA for the rest ~{eta:.0f} min" if eta else "")
            gpu_guard(log)
            t0 = time.time()
            try:
                res = finetune(m, t, s, epochs=epochs, patience=patience, samples_per_epoch=samples,
                               freeze_encoder=a.freeze_encoder, logger=log)
                results.append({k: res[k] for k in ("model_id", "task", "seed", "best_epoch", "best_val", "minutes")})
                log.info("run OK: best val %s at epoch %s (%.1f min)", (res.get("best_val") or {}).get("score"),
                         res["best_epoch"], res["minutes"])
            except MissingWeights as e:
                failures[f"{t}/{m}/s{s}"] = f"{e}"
                log.error("skipping %s: %s", m, e)
            except Exception as e:
                failures[f"{t}/{m}/s{s}"] = f"{type(e).__name__}: {e}"
                log.exception("run %s/%s/s%d FAILED", t, m, s)
            finally:
                done_minutes.append((time.time() - t0) / 60)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        log.info("MEASURED: %d runs took %.1f min in total (mean %.1f min per run)", len(done_minutes),
                 sum(done_minutes), sum(done_minutes) / max(1, len(done_minutes)))
        out = os.path.join(ROOT, "artifacts", "ckpt", "real", f"train_{a.mode}_summary.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(dict(mode=a.mode, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"), results=results,
                           failures=failures, measured_minutes_total=round(sum(done_minutes), 2),
                           hyper=dict(epochs=epochs, patience=patience, samples_per_epoch=samples)), f, indent=1)
        log.info("summary written to %s", out)
        if failures:
            raise JobError(f"{len(failures)} of {len(todo)} training runs failed: {', '.join(failures)}",
                           "; ".join(f"{k}: {v}"[:200] for k, v in failures.items()),
                           "If weights are missing run get_data.bat weights; otherwise send this log to Claude. "
                           "Re-running train.bat skips the runs that already finished.")


if __name__ == "__main__":
    main()
