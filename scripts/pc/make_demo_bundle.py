"""Build the offline demo bundle for the CPU laptop (OPS) -> dist/terrascope_demo.zip.

Steps (each logged to logs/make_demo_bundle_<ts>.txt; a missing piece is skipped with a reason, never faked):
  1. demo scenes -> data/demo/: ~4 test-split chips per real dataset found in data/real (centre-cropped to
     256 px, ground truth kept, sidecar .json with bounds/gsd/provenance) + the simulator presets of the UI
  2. ONNX export of every ready foundation model (fp32 + int8) via terrascope.fm.export.export_onnx
     -> artifacts/onnx/  (the laptop runs them with onnxruntime, TERRASCOPE_RUNTIME=onnx)
  3. stage only what the laptop needs: code + web, TerraFM checkpoints, ONNX files (or FM checkpoints when
     --include-terratorch), metrics JSONs, demo scenes, start_local.bat, bench_laptop.bat, requirements-demo.txt
  4. precompute: start the staged app on 127.0.0.1:<port> and POST /api/analyze for every demo scene x ready model
     and every sim preset x ready model; the disk cache (cache/) and artifacts/demo_precomputed.json go in the bundle
  5. optional --with-wheels: every win_amd64 wheel for requirements-demo.txt -> wheels/ (laptop installs offline);
     optional --with-python: the python.org Windows installer -> python-installer/
  6. BUNDLE_MANIFEST.json, zip (top folder terrascope_demo/), structure check, size printout

  python scripts/pc/make_demo_bundle.py [--with-wheels] [--with-python] [--dry-run] [--no-precompute] ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrascope.utils.joblog import JobError, MissingData, job_done, job_failed, start_job  # noqa: E402

BUNDLE_NAME = "terrascope_demo"
# Same presets as the Explorer (terrascope/web/app.js PRESETS)
SIM_PRESETS = [
    {"slug": "clear_day", "name": "Clear day", "seed": 12, "flood": 0.0, "cloud": 0.0, "drought": 0.0},
    {"slug": "flash_flood", "name": "Flash flood", "seed": 7, "flood": 0.8, "cloud": 0.05, "drought": 0.0},
    {"slug": "monsoon_flood_cloud", "name": "Monsoon flood under cloud", "seed": 7, "flood": 0.8, "cloud": 0.7, "drought": 0.1},
    {"slug": "drought_haze", "name": "Drought in haze", "seed": 31, "flood": 0.0, "cloud": 0.3, "drought": 0.9},
    {"slug": "river_delta", "name": "River delta", "seed": 52, "flood": 0.55, "cloud": 0.4, "drought": 0.3},
]
# dataset id, module, loader fn, truth key, main image key in the record, list_split kwargs, task
REAL_DATASETS = [
    ("sen1floods11", "terrascope.real.sen1floods11", "load", "flood", "s2", {"split": "test"}, "flood"),
    ("worldcover", "terrascope.real.sen1floods11", "load_landcover", "lc", "s2",
     {"split": "test", "require": ("s2", "label", "wc")}, "landcover"),
    ("hls_burn_scars", "terrascope.real.burnscars", "load", "burn", "image", {"split": "test"}, "burn"),
    ("sen12mscr", "terrascope.real.sen12mscr", "load", "optical_clean", "s2_cloudy", {"split": "test"}, "cloud"),
]
CODE_EXCLUDE = {"__pycache__", ".pytest_cache"}
STORED_EXT = {".whl", ".zip", ".png", ".jpg", ".exe", ".woff2", ".gz"}
REQUIRED_MEMBERS = ["start_local.bat", "bench_laptop.bat", "requirements-demo.txt", "BUNDLE_MANIFEST.json",
                    "terrascope/server/app.py", "terrascope/web/index.html", "terrascope/utils/joblog.py",
                    "terrascope/utils/open_when_ready.py", "scripts/pc/bench_laptop.py"]
# TerraFM checkpoint: current one, or the archived v1 one the runner falls back to (terrafm_runner.resolve_ckpt)
TERRAFM_CKPTS = ["artifacts/ckpt/finetuned.pt", "artifacts/ckpt/v1/finetuned.pt"]


def terrafm_ckpt() -> Path | None:
    for c in TERRAFM_CKPTS:
        if (ROOT / c).exists():
            return ROOT / c
    return None
PY_INSTALLER_URL = "https://www.python.org/ftp/python/{v}/python-{v}-amd64.exe"


# ============================================================================ 1. demo scenes
def _np():
    import numpy as np
    return np


def _crop(a, y0, x0, n):
    return None if a is None else a[..., y0:y0 + n, x0:x0 + n]


def _geo(path: str, y0: int, x0: int, h: int, w: int) -> dict:
    """bounds (EPSG:4326) + gsd of the crop window; {} when the file has no georeference."""
    try:
        import rasterio
        from rasterio.windows import Window, transform as wtransform
        from rasterio.transform import array_bounds
        from terrascope.real.geotiff import bounds_4326, gsd_metres
        with rasterio.open(path) as src:
            t = wtransform(Window(x0, y0, w, h), src.transform)
            b = array_bounds(h, w, t)  # (south, west, north, east)
            return {"bounds": bounds_4326(src.crs, (b[1], b[0], b[3], b[2])), "gsd_m": gsd_metres(t, src.crs, h, w)}
    except Exception as e:
        return {"geo_error": f"{type(e).__name__}: {str(e)[:120]}"}


def _score(truth_key, truth, invalid_frac, s):
    np = _np()
    if truth_key in ("flood", "burn"):
        p = float(truth.mean())
        sc = -abs(p - 0.25)          # clearly visible but not saturated
    elif truth_key == "lc":
        counts = np.bincount(truth.ravel(), minlength=6)[:6] / truth.size
        sc = float((counts > 0.02).sum())
    else:  # cloud removal: prefer visibly cloudy inputs
        sc = float(np.abs(s["optical"] - truth).mean())
    if s.get("sar") is not None:
        sc += 0.05
    return sc - 5.0 * invalid_frac


def convert_real(ds: str, truth_key: str, rec: dict, s: dict, img_key: str, crop: int) -> dict:
    """RS loader output -> TerraScope scene arrays (centre crop to <= crop px)."""
    np = _np()
    H, W = s["optical"].shape[1:]
    n = min(crop, H, W)
    n -= n % 4
    y0, x0 = (H - n) // 2, (W - n) // 2
    sc = {"optical": _crop(s["optical"], y0, x0, n).astype(np.float32)}
    for k in ("sar", "dem"):
        if s.get(k) is not None:
            sc[k] = _crop(s[k], y0, x0, n).astype(np.float32)
    invalid = 0.0
    if truth_key == "optical_clean":
        sc["optical_clean"] = _crop(s["target"], y0, x0, n).astype(np.float32)
    else:
        m = _crop(s["mask"], y0, x0, n)
        bad = m < 0
        invalid = float(bad.mean())
        if truth_key == "lc":
            sc["lc"] = np.where(bad, 0, m).astype(np.uint8)
        else:
            sc[truth_key] = (m == 1).astype(np.uint8)
    if img_key in ("s2", "s2_cloudy") and rec.get(img_key):
        try:
            from terrascope.real.bands import to_reflectance
            from terrascope.real.common import fill_nan, read_tif
            full = read_tif(rec[img_key])
            if full.shape[0] == 13:
                sc["s2_full"] = np.clip(fill_nan(to_reflectance(full), 0.0), 0, 1)[:, y0:y0 + n, x0:x0 + n].astype(np.float32)
        except Exception:
            pass
    geo = _geo(rec.get(img_key, ""), y0, x0, n, n) if rec.get(img_key) else {}
    return {"arrays": sc, "invalid_frac": invalid, "crop": [int(y0), int(x0), int(n)], "src_size": [int(H), int(W)], **geo}


def _safe_id(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:70]


def write_demo_scene(demo_dir: Path, stem: str, arrays: dict, side: dict) -> Path:
    np = _np()
    demo_dir.mkdir(parents=True, exist_ok=True)
    p = demo_dir / f"{stem}.npz"
    tmp = demo_dir / f"{stem}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, p)
    (demo_dir / f"{stem}.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    return p


def select_real(demo_dir: Path, per_dataset: int, scan: int, crop: int, log) -> tuple[list, list]:
    scenes, skipped = [], []
    for ds, modname, fn, truth_key, img_key, kw, task in REAL_DATASETS:
        try:
            mod = importlib.import_module(modname)
            recs = mod.list_split(**kw)
        except Exception as e:
            skipped.append({"dataset": ds, "reason": f"{type(e).__name__}: {str(e)[:200]}"})
            log.info("real dataset %-14s skipped: %s", ds, skipped[-1]["reason"])
            continue
        if not recs:
            skipped.append({"dataset": ds, "reason": "test split is empty"})
            continue
        recs = sorted(recs, key=lambda r: str(r.get("id")))
        step = max(1, len(recs) // scan)
        cands = []
        for rec in recs[::step][:scan]:
            try:
                s = getattr(mod, fn)(dict(rec))
                c = convert_real(ds, truth_key, rec, s, img_key, crop)
                a = c["arrays"]
                if not _np().isfinite(a["optical"]).all():
                    continue
                truth = a[truth_key]
                c["score"] = _score(truth_key, truth, c["invalid_frac"], {**a, "sar": a.get("sar")})
                c["rec_id"] = str(rec.get("id"))
                cands.append(c)
            except Exception as e:
                log.warning("  %s %s unreadable: %s", ds, rec.get("id"), e)
        cands.sort(key=lambda c: -c["score"])
        for c in cands[:per_dataset]:
            stem = _safe_id(f"{ds}_{c['rec_id']}")
            side = {"name": f"{ds} {c['rec_id']}", "data_source": f"dataset:{ds}", "dataset": ds,
                    "tasks": [task], "acquisition_date": None, "bounds": c.get("bounds"), "gsd_m": c.get("gsd_m"),
                    "description": (f"test split chip {c['rec_id']}, centre crop {c['crop'][2]} px of "
                                    f"{c['src_size'][0]}x{c['src_size'][1]}"
                                    + (f"; {100 * c['invalid_frac']:.1f}% of label pixels are no-data (set to 0)"
                                       if c["invalid_frac"] > 0 else "")),
                    "truth_invalid_frac": round(c["invalid_frac"], 4), "crop_yx_size": c["crop"]}
            write_demo_scene(demo_dir, stem, c["arrays"], side)
            scenes.append({"id": "demo-" + stem, "file": f"data/demo/{stem}.npz", "dataset": ds, "task": task,
                           "invalid_frac": round(c["invalid_frac"], 4), "bands": sorted(c["arrays"])})
        log.info("real dataset %-14s %d candidates scanned, %d written", ds, len(cands), min(per_dataset, len(cands)))
        if not cands:
            skipped.append({"dataset": ds, "reason": "no readable test chip"})
    return scenes, skipped


def write_sim_presets(demo_dir: Path, size: int, log) -> list:
    from terrascope.core.simulator import generate_scene
    out = []
    for p in SIM_PRESETS:
        sc = generate_scene(p["seed"], size, flood=p["flood"], cloud=p["cloud"], drought=p["drought"])
        arrays = {k: sc[k] for k in ("optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud") if k in sc}
        stem = f"sim_{p['slug']}"
        side = {"name": f"{p['name']} (simulated)", "data_source": "simulated", "tasks": ["landcover", "flood", "veg"],
                "acquisition_date": None, "bounds": None, "gsd_m": None,
                "description": f"simulator preset seed={p['seed']} flood={p['flood']} cloud={p['cloud']} drought={p['drought']}",
                "preset": p}
        write_demo_scene(demo_dir, stem, arrays, side)
        out.append({"id": "demo-" + stem, "file": f"data/demo/{stem}.npz", "dataset": "simulated", "preset": p["slug"]})
    log.info("simulated presets written: %d", len(out))
    return out


# ============================================================================ 2. models + ONNX
def list_models(log) -> tuple[list, str]:
    try:
        from terrascope.fm.registry import list_models as lm  # type: ignore
        return lm(), "terrascope.fm.registry"
    except ImportError as e:
        log.info("model registry not available (%s) - TerraFM only", e)
        return [{"id": "terrafm", "name": "TerraFM", "family": "terrafm", "status": "ready", "tasks": []}], "fallback"


def export_fms(models: list, onnx_dir: Path, precisions: list[str], log) -> list:
    rows = []
    fms = [m for m in models if m.get("family") != "terrafm" and m.get("id") != "terrafm"]
    try:
        from terrascope.fm.export import export_onnx  # type: ignore
    except ImportError as e:
        for m in fms:
            rows.append({"model": m.get("id"), "status": "skipped", "reason": f"terrascope.fm.export not available ({e})"})
        return rows
    for m in fms:
        if m.get("status") != "ready":
            rows.append({"model": m["id"], "status": "skipped", "reason": f"model status {m.get('status')}"})
            continue
        for task in m.get("tasks") or []:
            for prec in precisions:
                t0 = time.time()
                try:
                    res = export_onnx(m["id"], task, str(onnx_dir), int8=(prec == "int8"))
                    rows.append({"model": m["id"], "task": task, "precision": prec, "status": "ok",
                                 "result": res if isinstance(res, dict) else (str(res) if res is not None else None), "seconds": round(time.time() - t0, 1)})
                    log.info("ONNX %s/%s %s ok (%.0f s)", m["id"], task, prec, time.time() - t0)
                except Exception as e:
                    rows.append({"model": m["id"], "task": task, "precision": prec, "status": "error",
                                 "reason": f"{type(e).__name__}: {str(e)[:300]}"})
                    log.warning("ONNX %s/%s %s FAILED: %s", m["id"], task, prec, e)
    return rows


def onnx_coverage(models: list, exports: list) -> dict:
    """model id -> True when every task of a ready FM has at least one successful ONNX export."""
    cov = {}
    for m in models:
        if m.get("family") == "terrafm" or m.get("id") == "terrafm" or m.get("status") != "ready":
            continue
        derived = set(m.get("derived_outputs") or []) | {"veg"}  # computed at runtime, nothing to export
        tasks = [t for t in (m.get("tasks_ready") or m.get("tasks") or []) if t not in derived]
        ok = {(r["model"], r.get("task")) for r in exports if r.get("status") == "ok"}
        cov[m["id"]] = bool(tasks) and all((m["id"], t) in ok for t in tasks)
    return cov


# ============================================================================ 3. staging
def _copytree(src: Path, dst: Path, files: list):
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d not in CODE_EXCLUDE]
        for fn in fns:
            if fn.endswith((".pyc", ".pyo")):
                continue
            s = Path(dp) / fn
            d = dst / s.relative_to(src)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            files.append(d)


def _copy(src: Path, dst: Path, files: list, required=False):
    if not src.exists():
        if required:
            raise MissingData(src, "train.bat (choose S = sim)")
        return False
    if src.is_dir():
        _copytree(src, dst, files)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(dst)
    return True


def pick_ckpts(ckpt_root: Path, models_needing_torch: set[str]) -> list[Path]:
    """TerraFM: every real fine-tune (tiny). Other models: lowest seed per task, only if they must run in torch."""
    out = []
    real = ckpt_root / "real"
    if not real.is_dir():
        return out
    for task_dir in sorted(p for p in real.iterdir() if p.is_dir()):
        by_model: dict[str, list[Path]] = {}
        for f in sorted(task_dir.glob("*.pt")):
            mid = f.stem.rsplit("_s", 1)[0]
            by_model.setdefault(mid, []).append(f)
        for mid, fs in by_model.items():
            if mid == "terrafm":
                out += fs
            elif mid in models_needing_torch:
                out.append(sorted(fs, key=lambda p: p.stem)[0])
    return out


def stage_bundle(stage: Path, demo_dir: Path, onnx_dir: Path, include_onnx: bool, ckpts: list[Path],
                 extra_reqs: list[str], log) -> list[Path]:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    files: list[Path] = []
    _copy(ROOT / "terrascope", stage / "terrascope", files, required=True)
    for f in ("start_local.bat", "bench_laptop.bat", "README.md", "LOCAL_GUIDE.md"):
        _copy(ROOT / f, stage / f, files)
    for f in ("bench_laptop.py", "gpu_check.py"):
        _copy(ROOT / "scripts" / "pc" / f, stage / "scripts" / "pc" / f, files)
    reqs = (ROOT / "requirements-demo.txt").read_text(encoding="utf-8")
    if extra_reqs:
        reqs += "\n# added by make_demo_bundle.py (some foundation models could not be exported to ONNX)\n" + "\n".join(extra_reqs) + "\n"
    (stage / "requirements-demo.txt").write_text(reqs, encoding="utf-8")
    files.append(stage / "requirements-demo.txt")
    for f in sorted((ROOT / "artifacts").glob("*.json")):
        if f.name not in ("pc_info.json",):
            _copy(f, stage / "artifacts" / f.name, files)
    ck = terrafm_ckpt()
    if ck is None:
        raise MissingData(ROOT / TERRAFM_CKPTS[0], "train.bat (choose S = sim)")
    _copy(ck, stage / ck.relative_to(ROOT), files, required=True)
    for f in sorted(ck.parent.glob("*.json")):  # calibration / config files next to the checkpoint, if any
        _copy(f, stage / f.relative_to(ROOT), files)
    for c in ckpts:
        _copy(c, stage / c.relative_to(ROOT), files)
    if include_onnx and onnx_dir.is_dir():
        _copy(onnx_dir, stage / "artifacts" / "onnx", files)
    _copy(demo_dir, stage / "data" / "demo", files)
    log.info("staged %d files (%.1f MB) in %s", len(files), sum(f.stat().st_size for f in files) / 2**20, stage)
    return files


# ============================================================================ 4. precompute via the app
def _http(method: str, url: str, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"null")
        except Exception:
            return e.code, None


def precompute(stage: Path, port: int, runtime: str, log, startup_timeout=600) -> dict:
    env = {**os.environ, "PYTHONPATH": str(stage), "TERRASCOPE_RUNTIME": runtime, "CUDA_VISIBLE_DEVICES": "",
           "TERRASCOPE_DEVICE": "cpu", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "TERRASCOPE_CACHE_MB": "100000",  # never evict while precomputing
           "TERRASCOPE_RATE_LIMIT": "0"}  # one local client sending ~90 requests in a row
    for k in ("TERRASCOPE_DEMO_DIR", "TERRASCOPE_SCENES_DIR", "TERRASCOPE_CKPT", "TERRASCOPE_CACHE_DIR", "TERRASCOPE_ONNX_DIR"):
        env.pop(k, None)
    base = f"http://127.0.0.1:{port}"
    srv_log = open(stage.parent / f"precompute_server_{port}.txt", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "terrascope.server.app:app", "--host", "127.0.0.1",
                             "--port", str(port)], cwd=str(stage), env=env, stdout=srv_log, stderr=subprocess.STDOUT)
    done: list[dict] = []
    try:
        t0 = time.time()
        while True:
            if proc.poll() is not None:
                raise JobError("The app did not start for precomputing (exit code %s)." % proc.returncode,
                               "See the server output file next to the bundle (dist/precompute_server_*.txt).",
                               "Fix the error shown there or rerun with --no-precompute.")
            try:
                st, h = _http("GET", base + "/api/health", timeout=3)
                if st == 200:
                    break
            except Exception:
                pass
            if time.time() - t0 > startup_timeout:
                raise JobError("The app did not answer /api/health in time.", "Slow start or a hang.", "Rerun with --no-precompute.")
            time.sleep(1)
        log.info("app ready for precompute (%.0f s): %s", time.time() - t0, h)
        st, models = _http("GET", base + "/api/models")
        if st != 200 or not isinstance(models, list):
            models = [{"id": "terrafm", "status": "ready"}]
            log.info("/api/models not available (HTTP %s) - precomputing TerraFM only", st)
        ready = [m["id"] for m in models if m.get("status") == "ready"]
        st, scenes = _http("GET", base + "/api/scenes")
        demo = [s["id"] for s in scenes if s.get("source") == "demo"] if st == 200 and isinstance(scenes, list) else []
        if st != 200:
            log.info("/api/scenes not available (HTTP %s) - only simulator presets are precomputed", st)
        reqs = [{"source": "scene", "scene_id": sid} for sid in demo]
        reqs += [{"source": "sim", "seed": p["seed"], "flood": p["flood"], "cloud": p["cloud"], "drought": p["drought"],
                  "size": 128} for p in SIM_PRESETS]
        for r in reqs:
            for mid in ready:
                body = {**r, "model": mid, "sensors": [True, True, True]}
                t1 = time.time()
                st, res = _http("POST", base + "/api/analyze", body)
                row = {"request": body, "http": st, "seconds": round(time.time() - t1, 2)}
                if st == 200 and isinstance(res, dict):
                    row["analysis_id"] = res.get("id")
                else:
                    row["detail"] = (res or {}).get("detail") if isinstance(res, dict) else None
                done.append(row)
                log.info("precompute %-28s %-22s HTTP %s %.1fs", r.get("scene_id") or f"sim seed={r.get('seed')}", mid,
                         st, row["seconds"])
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        srv_log.close()
    ok = [d for d in done if d["http"] == 200]
    doc = {"generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "runtime": runtime,
           "device": "cpu", "entries": done}
    (stage / "artifacts").mkdir(exist_ok=True)
    (stage / "artifacts" / "demo_precomputed.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    cache = stage / "cache"
    cache_mb = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 2**20 if cache.is_dir() else 0.0
    log.info("precomputed %d/%d analyses, cache %.1f MB", len(ok), len(done), cache_mb)
    if cache_mb > 400:
        log.warning("precomputed cache is %.0f MB; start_local.bat raises TERRASCOPE_CACHE_MB to 3000 in demo mode so "
                    "it is not evicted (app default 500 MB)", cache_mb)
    return {"requests": len(done), "ok": len(ok), "cache_mb": round(cache_mb, 1), "runtime": runtime}


# ============================================================================ 5. wheels / python
def download_wheels(stage: Path, py: str, log) -> dict:
    wd = stage / "wheels"
    wd.mkdir(exist_ok=True)
    if sys.platform != "win32":
        log.warning("not running on Windows: environment markers are evaluated for %s, the wheel set may miss "
                    "Windows-only dependencies - build the bundle on the PC", sys.platform)
    reqs = stage / "requirements-demo.txt"
    if "terratorch" in reqs.read_text(encoding="utf-8"):
        # antlr4-python3-runtime 4.9.3 (via terratorch) is published as a pure-python sdist only: build its wheel
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(wd), "antlr4-python3-runtime==4.9.3"],
                       check=False)
    cmd = [sys.executable, "-m", "pip", "download", "-r", str(reqs), "-d", str(wd), "--only-binary=:all:",
           "--platform", "win_amd64", "--python-version", py, "--implementation", "cp", "--find-links", str(wd),
           "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    log.info("$ %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (p.stdout + p.stderr).splitlines()[-15:]:
        log.info("  %s", line)
    if p.returncode:
        from terrascope.utils.joblog import diagnose_text
        hit = diagnose_text(p.stdout + p.stderr) or ("pip download failed", "Read the pip lines above; rerun without --with-wheels if needed.")
        raise JobError("Downloading Windows wheels for the laptop failed.", *hit)
    (wd / "PYTHON_VERSION.txt").write_text(py, encoding="ascii")
    n = len(list(wd.glob("*.whl")))
    mb = sum(f.stat().st_size for f in wd.glob("*")) / 2**20
    log.info("wheels: %d files, %.0f MB, for CPython %s win_amd64", n, mb, py)
    return {"python": py, "count": n, "mb": round(mb, 1)}


def download_python(stage: Path, version: str, log) -> dict:
    url = PY_INSTALLER_URL.format(v=version)
    d = stage / "python-installer"
    d.mkdir(exist_ok=True)
    dst = d / url.rsplit("/", 1)[1]
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        log.info("python installer: %s (%.1f MB)", dst.name, dst.stat().st_size / 2**20)
        return {"file": f"python-installer/{dst.name}", "url": url}
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        log.warning("python installer download failed (%s) - the laptop must already have Python", e)
        return {"error": f"{type(e).__name__}: {e}", "url": url}


# ============================================================================ 6. zip + check
def make_zip(stage: Path, zpath: Path, log) -> int:
    zpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = zpath.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for f in sorted(p for p in stage.rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")):
            arc = f"{BUNDLE_NAME}/{f.relative_to(stage).as_posix()}"
            comp = zipfile.ZIP_STORED if f.suffix.lower() in STORED_EXT else zipfile.ZIP_DEFLATED
            z.write(f, arc, compress_type=comp)
    os.replace(tmp, zpath)
    return zpath.stat().st_size


def validate_zip(zpath: Path) -> list[str]:
    """Structural check of a bundle zip -> list of problems (empty = OK)."""
    probs = []
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        bad = z.testzip()
        if bad:
            probs.append(f"corrupt member {bad}")
        for m in REQUIRED_MEMBERS:
            if f"{BUNDLE_NAME}/{m}" not in names:
                probs.append(f"missing {m}")
        if not any(f"{BUNDLE_NAME}/{c}" in names for c in TERRAFM_CKPTS):
            probs.append("missing TerraFM checkpoint (artifacts/ckpt/finetuned.pt)")
        if not any(n.startswith(f"{BUNDLE_NAME}/data/demo/") and n.endswith(".npz") for n in names):
            probs.append("no demo scene in data/demo/")
        for n in names:
            if not n.startswith(BUNDLE_NAME + "/") or ".." in n.split("/"):
                probs.append(f"unsafe path {n}")
            if any(part in (".venv", ".git", "__pycache__", "logs") or part.startswith(".venv") for part in n.split("/")):
                probs.append(f"should not be bundled: {n}")
            if "/data/real/" in n:
                probs.append(f"raw dataset file bundled: {n}")
        try:
            man = json.loads(z.read(f"{BUNDLE_NAME}/BUNDLE_MANIFEST.json"))
            for k in ("created_at", "scenes", "options"):
                if k not in man:
                    probs.append(f"manifest lacks {k}")
        except KeyError:
            pass
        bat = z.read(f"{BUNDLE_NAME}/start_local.bat") if f"{BUNDLE_NAME}/start_local.bat" in names else b""
        if bat and b"\r\n" not in bat:
            probs.append("start_local.bat lost its CRLF line endings")
    return probs


def boot_smoke(zpath: Path, port: int, log, timeout: float = 300) -> dict:
    """Unzip the finished bundle to a temp folder (with a space in its name), start its app on 127.0.0.1,
    require /api/health == 200 and the index page, stop it. Raises JobError (loudly) on any problem."""
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="boot smoke ", dir=str(zpath.parent)))
    try:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp)
        app_dir = tmp / BUNDLE_NAME
        env = {k: v for k, v in os.environ.items() if not k.startswith("TERRASCOPE_") or k == "TERRASCOPE_LOG_DIR"}
        env.update(PYTHONPATH=str(app_dir), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", PYTHONUTF8="1",
                   TERRASCOPE_RUNTIME="onnx" if (app_dir / "artifacts" / "onnx").is_dir() else "torch")
        imp = subprocess.run([sys.executable, "-c", "import terrascope.server.app"], cwd=str(app_dir), env=env,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if imp.returncode:
            raise JobError("The bundled app cannot be imported: " + (imp.stderr.strip().splitlines() or ["?"])[-1],
                           "A file the app needs was not put into the bundle, or a code error.",
                           "Send this log to Claude; do not copy this zip to the laptop.")
        out = open(tmp / "server.txt", "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "terrascope.server.app:app", "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=str(app_dir), env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            t0, health = time.time(), None
            while time.time() - t0 < timeout and proc.poll() is None:
                try:
                    st, h = _http("GET", f"http://127.0.0.1:{port}/api/health", timeout=3)
                    if st == 200:
                        health = h
                        break
                except Exception:
                    pass
                time.sleep(1)
            st_index = _http_status(f"http://127.0.0.1:{port}/") if health is not None else None
        finally:
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            out.close()
        if health is None or st_index != 200 or not (isinstance(health, dict) and health.get("ready", True)):
            tail = (tmp / "server.txt").read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            for l in tail:
                log.error("  server: %s", l)
            raise JobError(f"The app inside the bundle did not start correctly (health={health}, index HTTP {st_index}).",
                           "See the server lines above.", "Send this log to Claude; do not copy this zip to the laptop.")
        log.info("boot smoke OK: bundled app answered /api/health %s and / (HTTP 200)", health)
        return health
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _http_status(url: str) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = ROOT / ".git" / head[5:]
            return ref.read_text().strip() if ref.exists() else None
        return head
    except Exception:
        return None


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "dist"))
    ap.add_argument("--demo-dir", default=str(ROOT / "data" / "demo"))
    ap.add_argument("--onnx-dir", default=str(ROOT / "artifacts" / "onnx"))
    ap.add_argument("--per-dataset", type=int, default=4)
    ap.add_argument("--scan", type=int, default=40, help="test chips examined per dataset when choosing demo scenes")
    ap.add_argument("--crop", type=int, default=256, help="demo scene size (the app accepts up to 256 px)")
    ap.add_argument("--sim-size", type=int, default=128)
    ap.add_argument("--onnx-precision", choices=["both", "fp32", "int8"], default="both",
                    help="both = bundle fp32 (active) + int8 files; int8 = int8 active (smaller/faster, accuracy unmeasured)")
    ap.add_argument("--include-terratorch", action="store_true",
                    help="bundle FM checkpoints + terratorch for models whose ONNX export failed")
    ap.add_argument("--with-wheels", action="store_true", default=True,
                    help="(default) bundle win_amd64 wheels -> laptop installs offline")
    ap.add_argument("--no-wheels", dest="with_wheels", action="store_false", help="smaller zip; laptop needs internet once")
    ap.add_argument("--wheels-python", default="3.12")
    ap.add_argument("--with-python", action="store_true", default=True, help="(default) bundle the python.org Windows installer")
    ap.add_argument("--no-python", dest="with_python", action="store_false")
    ap.add_argument("--no-boot-smoke", action="store_true", help="skip starting the app from the finished zip")
    ap.add_argument("--python-installer-version", default="3.12.10")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-precompute", action="store_true")
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--port", type=int, default=8104)
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="scenes + staging + zip only (no ONNX export, no precompute, no downloads)")
    a = ap.parse_args(argv)
    log = start_job("make_demo_bundle", probe_torch=not a.dry_run)
    try:
        out_dir, demo_dir, onnx_dir = Path(a.out_dir), Path(a.demo_dir), Path(a.onnx_dir)
        stage = out_dir / BUNDLE_NAME
        zpath = out_dir / f"{BUNDLE_NAME}.zip"
        if terrafm_ckpt() is None:
            raise MissingData(ROOT / TERRAFM_CKPTS[0], "train.bat (choose S = sim)")
        free_gb = shutil.disk_usage(out_dir if out_dir.exists() else out_dir.parent).free / 1e9
        log.info("free disk for dist/: %.1f GB", free_gb)
        if free_gb < 2 * a.max_gb + 1 and not a.dry_run:
            log.warning("less than %.0f GB free: staging + zip may not fit", 2 * a.max_gb + 1)
        if a.dry_run:
            log.info("DRY RUN: no ONNX export, no precompute, no wheel / installer downloads")

        # 1. scenes
        if demo_dir.exists():  # remove only scenes this script generated earlier
            ours = tuple(["sim_"] + [d[0] + "_" for d in REAL_DATASETS])
            for f in demo_dir.glob("*"):
                if f.is_file() and f.name.startswith(ours) and f.suffix in (".npz", ".json", ".tmp"):
                    f.unlink()
        scenes, skipped_ds = ([], [{"dataset": "all", "reason": "--no-real"}]) if a.no_real else \
            select_real(demo_dir, a.per_dataset, a.scan, a.crop, log)
        scenes += write_sim_presets(demo_dir, a.sim_size, log)

        # 2. models + ONNX
        models, msrc = list_models(log)
        # export_onnx writes <task>.json pointing at the files of the LAST export, and the ONNX runner loads those:
        # "both" exports int8 first and fp32 last, so the laptop runs fp32 (the precision the metrics were measured
        # with) and the int8 files are still bundled for bench_laptop. --onnx-precision int8 makes int8 active.
        precisions = ["int8", "fp32"] if a.onnx_precision == "both" else [a.onnx_precision]
        exports = [] if (a.dry_run or a.no_export) else export_fms(models, onnx_dir, precisions, log)
        cov = onnx_coverage(models, exports)
        need_torch = {m for m, ok in cov.items() if not ok}
        extra_reqs = []
        if need_torch:
            if a.include_terratorch:
                extra_reqs = ["terratorch==1.2.13"]
                log.warning("no complete ONNX export for %s: bundling their checkpoints + terratorch", sorted(need_torch))
            else:
                log.warning("no complete ONNX export for %s: they will show as unavailable on the laptop "
                            "(use --include-terratorch to ship them in PyTorch form)", sorted(need_torch))
                need_torch = set()
        ckpts = pick_ckpts(ROOT / "artifacts" / "ckpt", need_torch)

        # 3. stage
        include_onnx = not a.dry_run and onnx_dir.is_dir() and any(onnx_dir.rglob("*.onnx"))
        stage_bundle(stage, demo_dir, onnx_dir, include_onnx, ckpts, extra_reqs, log)

        # 4. precompute
        runtime = "onnx" if (stage / "artifacts" / "onnx").is_dir() else "torch"
        pre = {"skipped": "dry run" if a.dry_run else "--no-precompute"}
        if not (a.dry_run or a.no_precompute):
            pre = precompute(stage, a.port, runtime, log)

        # 5. downloads
        wheels = py_inst = None
        if a.with_wheels and not a.dry_run:
            wheels = download_wheels(stage, a.wheels_python, log)
        if a.with_python and not a.dry_run:
            py_inst = download_python(stage, a.python_installer_version, log)

        # 6. manifest + zip
        manifest = {
            "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "built_on": {"host": platform.node(), "os": platform.platform(), "python": platform.python_version()},
            "git_commit": _git_commit(), "dry_run": a.dry_run,
            "options": {k: v for k, v in vars(a).items() if k not in ("out_dir", "demo_dir", "onnx_dir")},
            "models": [{k: m.get(k) for k in ("id", "name", "family", "status", "params", "tasks")} for m in models],
            "model_source": msrc, "onnx_exports": exports, "onnx_coverage": cov,
            "runtime_on_laptop": runtime, "torch_checkpoints": [str(c.relative_to(ROOT)).replace("\\", "/") for c in ckpts],
            "scenes": scenes, "datasets_skipped": skipped_ds, "precompute": pre, "wheels": wheels, "python_installer": py_inst,
            "requirements_extra": extra_reqs,
        }
        (stage / "BUNDLE_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        size = make_zip(stage, zpath, log)
        probs = validate_zip(zpath)
        if probs:
            raise JobError("The bundle zip failed its structure check: " + "; ".join(probs[:8]),
                           "A required file was not staged.", "Send this log to Claude.")
        if not a.keep_stage:
            shutil.rmtree(stage, ignore_errors=True)
        if not a.no_boot_smoke:
            boot_smoke(zpath, a.port, log)
        gb = size / 1e9
        log.info("BUNDLE: %s  size %.2f GB (%d MB)", zpath, gb, size // 2**20)
        if gb > a.max_gb:
            log.warning("bundle is larger than %.1f GB - consider --onnx-precision int8 or no --with-wheels", a.max_gb)
        job_done(f"demo bundle ready: {zpath} ({size / 2**20:.0f} MB) - copy it to the laptop, unzip, double-click start_local.bat")
        return 0
    except Exception as e:
        job_failed(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
