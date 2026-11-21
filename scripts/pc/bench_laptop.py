"""CPU speed benchmark per model (OPS) -> artifacts/bench_laptop.json.

Run it on the demo laptop (bench_laptop.bat) for laptop numbers; on the PC it is forced to CPU and the JSON
records which machine produced the numbers, so a PC run is never mistaken for a laptop run.

For every runner the model registry offers (terrascope.fm.registry; falls back to the TerraFM runner alone):
  * runtime "torch", and runtime "onnx" when onnxruntime is installed (env TERRASCOPE_RUNTIME=onnx)
  * plus every raw ONNX file in artifacts/onnx (fp32 and *int8* variants) through a bare onnxruntime session
measures, each in its own subprocess (so peak RAM is per variant and a crash cannot take the benchmark down):
  load time, seconds per scene at 128 / 256 / 512 px (1 warm-up + --reps timed calls), peak RAM of the process.
Models that cannot run get an honest {"status": "skipped", "reason": ...} entry. Inputs are synthetic arrays with
realistic value ranges (speed does not depend on content for these models).

  python scripts/pc/bench_laptop.py [--sizes 128,256,512] [--reps 3] [--models terrafm,...] [--out PATH] [--quick]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrascope.utils.joblog import job_done, job_failed, start_job  # noqa: E402

RESULT_TAG = "BENCH_RESULT="
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "TERRASCOPE_DEVICE": "cpu"}


# ----------------------------------------------------------------------------- helpers (worker side)
def peak_ram_mb() -> float | None:
    try:
        import psutil  # type: ignore
        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows: true peak working set
        if peak:
            return round(peak / 2**20, 1)
    except Exception:
        pass
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
        return round(kb / 1024 if sys.platform != "darwin" else kb / 2**20, 1)
    except Exception:
        return None


def synthetic_scene(size: int, seed: int = 0) -> dict:
    import numpy as np
    r = np.random.default_rng(seed)
    f = lambda lo, hi, c: r.uniform(lo, hi, (c, size, size)).astype(np.float32)  # noqa: E731
    dem = np.stack([r.uniform(0, 500, (size, size)), r.uniform(0, 30, (size, size))]).astype(np.float32)
    return {"optical": f(0.02, 0.4, 6), "sar": f(-25, -5, 2), "dem": dem, "s2_full": f(0.02, 0.4, 13),
            "meta": {"data_source": "synthetic-benchmark", "bounds": None, "gsd_m": 10.0, "acquisition_date": None}}


def discover(runtime: str):
    """Return (models: list[dict], get_runner) for the given runtime. Never raises for a missing registry."""
    os.environ["TERRASCOPE_RUNTIME"] = runtime
    try:
        from terrascope.fm.registry import get_runner, list_models  # type: ignore
        return list_models(), get_runner, "terrascope.fm.registry"
    except ImportError:
        pass
    from terrascope.core.engine.terrafm_runner import TerraFMRunner  # type: ignore

    def _get(mid):
        if mid != "terrafm":
            raise KeyError(mid)
        return TerraFMRunner()
    return [{"id": "terrafm", "name": "TerraFM", "family": "terrafm", "status": "ready", "params": None}], _get, \
        "terrascope.core.engine.terrafm_runner (registry not available)"


def _time_calls(fn, reps: int) -> dict:
    fn()  # warm-up (allocations, lazy init)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"s_per_scene_mean": round(sum(ts) / len(ts), 4), "s_per_scene_min": round(min(ts), 4), "reps": reps}


def worker_runner(spec: dict) -> dict:
    out = {"kind": "runner", "model_id": spec["model"], "runtime": spec["runtime"], "variant": spec["runtime"]}
    with _no_grad():
        return _worker_runner(spec, out)


def _no_grad():
    """torch.no_grad() when torch is importable (scoped: never leaks global grad mode into the caller)."""
    try:
        import torch
        return torch.no_grad()
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def _worker_runner(spec: dict, out: dict) -> dict:
    models, get_runner, src = discover(spec["runtime"])
    out["registry"] = src
    if spec["runtime"] != "torch" and "registry not available" in src:
        return {**out, "status": "skipped", "reason": "runtime selection needs terrascope.fm.registry (not available)"}
    t0 = time.perf_counter()
    try:
        r = get_runner(spec["model"])
    except Exception as e:
        return {**out, "status": "skipped", "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    out["construct_s"] = round(time.perf_counter() - t0, 3)
    actual = getattr(r, "runtime", None) or ("onnx" if "onnx" in type(r).__name__.lower() else "torch")
    out["runner_runtime"], out["runner_class"] = actual, type(r).__name__
    if spec["runtime"] == "onnx" and actual != "onnx":
        return {**out, "status": "skipped",
                "reason": f"{type(r).__name__} runs in {actual} even with TERRASCOPE_RUNTIME=onnx (same as the torch row)"}
    out["name"] = getattr(r, "name", spec["model"])
    out["params"] = getattr(r, "params", None)
    out["device"] = getattr(r, "device", "cpu")
    try:
        import torch
        out["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    per = {}
    # cold start = construction + first predict (runners may load weights lazily)
    try:
        sc0 = synthetic_scene(spec["sizes"][0])
        t1 = time.perf_counter()
        r.predict(sc0, [True, True, True])
        out["first_call_s"] = round(time.perf_counter() - t1, 3)
        out["load_s"] = round(out["construct_s"] + out["first_call_s"], 3)
    except Exception as e:
        out["first_call_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    for s in spec["sizes"]:
        sc = synthetic_scene(s)
        try:
            per[str(s)] = _time_calls(lambda: r.predict(sc, [True, True, True]), spec["reps"])
        except Exception as e:
            per[str(s)] = {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}
    out["per_size"] = per
    ok = any("s_per_scene_mean" in v for v in per.values())
    out["status"] = "ok" if ok else "error"
    if not ok:
        out["reason"] = "no size could be run: " + "; ".join(v.get("skipped", "") for v in per.values())[:400]
    return out


def worker_onnx(spec: dict) -> dict:
    path = spec["path"]
    rel = os.path.relpath(path, ROOT)
    out = {"kind": "onnx_file", "file": rel.replace("\\", "/"), "variant": "onnx-int8" if "int8" in os.path.basename(path).lower() else "onnx-fp32",
           "runtime": "onnxruntime", "file_mb": round(os.path.getsize(path) / 2**20, 1)}
    try:
        import numpy as np
        import onnxruntime as ort  # type: ignore
    except Exception as e:
        return {**out, "status": "skipped", "reason": f"onnxruntime not installed ({e})"}
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        return {**out, "status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    out["load_s"] = round(time.perf_counter() - t0, 3)  # session creation (file load + graph optimisation)
    out["inputs"] = [{"name": i.name, "shape": [d if isinstance(d, int) else str(d) for d in i.shape], "type": i.type}
                     for i in sess.get_inputs()]
    per = {}
    for s in spec["sizes"]:
        feeds, why = {}, None
        for i in sess.get_inputs():
            shp = list(i.shape)
            if len(shp) == 4 and all(isinstance(d, int) for d in shp[2:]) and tuple(shp[2:]) != (s, s):
                why = f"model input '{i.name}' has a fixed size {shp[2]}x{shp[3]}"
                break
            dims = []
            for k, d in enumerate(shp):
                if isinstance(d, int) and d > 0:
                    dims.append(d)
                elif k == 0:
                    dims.append(1)
                elif len(shp) == 4 and k >= 2:
                    dims.append(s)
                else:
                    why = f"cannot infer dynamic dim {d!r} of input '{i.name}'"
                    break
            if why:
                break
            dt_ = np.int64 if "int64" in i.type else (np.float16 if "float16" in i.type else np.float32)
            feeds[i.name] = (np.random.default_rng(0).uniform(0, 0.3, dims)).astype(dt_)
        if why:
            per[str(s)] = {"skipped": why}
            continue
        try:
            per[str(s)] = _time_calls(lambda: sess.run(None, feeds), spec["reps"])
        except Exception as e:
            per[str(s)] = {"skipped": f"{type(e).__name__}: {str(e)[:200]}"}
    out["per_size"] = per
    ok = any("s_per_scene_mean" in v for v in per.values())
    out["status"] = "ok" if ok else "error"
    if not ok:
        out["reason"] = "no size could be run"
    out["note"] = "bare onnxruntime session on random input (model only; excludes TerraScope pre/post-processing)"
    return out


def worker_main(spec_json: str) -> int:
    spec = json.loads(spec_json)
    try:
        res = worker_runner(spec) if spec["kind"] == "runner" else worker_onnx(spec)
    except Exception as e:
        res = {"kind": spec.get("kind"), "model_id": spec.get("model"), "file": spec.get("path"),
               "runtime": spec.get("runtime"), "status": "error", "reason": f"{type(e).__name__}: {str(e)[:300]}"}
    res["peak_ram_mb"] = peak_ram_mb()
    print(RESULT_TAG + json.dumps(res), flush=True)
    return 0


# ----------------------------------------------------------------------------- driver side
def machine_info() -> dict:
    info = {"os": platform.platform(), "python": platform.python_version(), "logical_cores": os.cpu_count(),
            "host": platform.node()}
    try:
        sys.path.insert(0, str(ROOT / "scripts" / "pc"))
        from gpu_check import cpu_name, ram_gb  # type: ignore
        info["cpu"] = cpu_name()
        info["ram_total_gb"] = ram_gb().get("total_gb")
    except Exception:
        info["cpu"] = platform.processor() or "unknown"
    for mod in ("torch", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def run_worker(spec: dict, timeout: float, log) -> dict:
    env = {**os.environ, **CPU_ENV, "PYTHONUTF8": "1"}
    if spec["kind"] == "runner":
        env["TERRASCOPE_RUNTIME"] = spec["runtime"]
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(spec)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(ROOT),
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"status": "error", "reason": f"timed out after {timeout:.0f} s"}
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith(RESULT_TAG)), None)
    if line is None:
        tail = (p.stderr or p.stdout).strip().splitlines()[-5:]
        log.warning("worker output:\n%s", "\n".join(tail))
        return {"status": "error", "reason": f"worker crashed (exit {p.returncode}): {' | '.join(tail)[-300:]}"}
    return json.loads(line[len(RESULT_TAG):])


def plan(models_filter: list[str] | None, log) -> list[dict]:
    specs = []
    try:
        models, _, src = discover("torch")
    except Exception as e:
        log.error("no model registry or TerraFM runner importable: %s", e)
        models, src = [], "none"
    log.info("model source: %s (%d models)", src, len(models))
    runtimes = ["torch"]
    try:
        import onnxruntime  # noqa: F401
        runtimes.append("onnx")
    except Exception:
        log.info("onnxruntime not installed - runner-level ONNX variants skipped")
    skipped = []
    for m in models:
        mid = m.get("id")
        if models_filter and mid not in models_filter:
            continue
        if m.get("status", "ready") != "ready":
            skipped.append({"kind": "runner", "model_id": mid, "name": m.get("name"), "runtime": "torch",
                            "variant": "torch", "status": "skipped", "reason": f"model status: {m.get('status')}"})
            # an ONNX export can make a torch-unready model runnable on the laptop: still try onnx
            if "onnx" in runtimes:
                specs.append({"kind": "runner", "model": mid, "runtime": "onnx"})
            continue
        for rt in runtimes:
            specs.append({"kind": "runner", "model": mid, "runtime": rt})
    for p in sorted(glob.glob(str(ROOT / "artifacts" / "onnx" / "**" / "*.onnx"), recursive=True)):
        if models_filter and not any(f in os.path.basename(p) for f in models_filter):
            continue
        specs.append({"kind": "onnx_file", "path": p})
    return specs, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--sizes", default="128,256,512")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--models", default="", help="comma-separated model ids (default: all)")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "bench_laptop.json"))
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds per model variant")
    ap.add_argument("--quick", action="store_true", help="128 px only, 1 rep (smoke test)")
    a = ap.parse_args(argv)
    if a.worker:
        return worker_main(a.worker)

    log = start_job("bench_laptop", probe_torch=False)
    try:
        sizes = [128] if a.quick else [int(s) for s in a.sizes.split(",") if s.strip()]
        reps = 1 if a.quick else max(1, a.reps)
        mf = [m.strip() for m in a.models.split(",") if m.strip()] or None
        mach = machine_info()
        log.info("machine: %s | %s logical cores | RAM %s GB | torch %s | onnxruntime %s",
                 mach.get("cpu"), mach.get("logical_cores"), mach.get("ram_total_gb"), mach.get("torch"), mach.get("onnxruntime"))
        specs, results = plan(mf, log)
        for spec in specs:
            label = spec.get("model") or os.path.basename(spec.get("path", ""))
            log.info("benchmarking %s [%s] sizes=%s reps=%d ...", label, spec.get("runtime", "onnx file"), sizes, reps)
            spec.update(sizes=sizes, reps=reps)
            r = run_worker(spec, a.timeout, log)
            r.setdefault("model_id", spec.get("model"))
            if spec["kind"] == "onnx_file":
                r.setdefault("file", os.path.relpath(spec["path"], ROOT).replace("\\", "/"))
            r.setdefault("runtime", spec.get("runtime"))
            results.append(r)
            if r.get("status") == "ok":
                cells = ", ".join(f"{k}px {v['s_per_scene_mean']:.3f}s" if "s_per_scene_mean" in v else f"{k}px skipped"
                                  for k, v in r.get("per_size", {}).items())
                log.info("  -> load %ss | %s | peak RAM %s MB", r.get("load_s"), cells, r.get("peak_ram_mb"))
            else:
                log.info("  -> %s: %s", r.get("status"), r.get("reason"))
        doc = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "machine": mach,
               "settings": {"sizes": sizes, "reps": reps, "warmup": 1, "device": "cpu",
                            "input": "synthetic arrays (optical 6, sar 2, dem 2, s2_full 13 bands)"},
               "note": "Measured on the machine named in 'machine'. Only laptop runs represent laptop speed.",
               "results": results}
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        n_ok = sum(r.get("status") == "ok" for r in results)
        log.info("wrote %s (%d measured, %d skipped/error)", out, n_ok, len(results) - n_ok)
        if n_ok == 0:
            log.warning("nothing could be measured - see the reasons above")
        job_done()
        return 0
    except Exception as e:
        job_failed(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
