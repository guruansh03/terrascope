"""FastAPI backend + static frontend.

Run:  python -m terrascope.server.app            (binds 127.0.0.1:8000; HOST / PORT env to change)
  or  python -m uvicorn terrascope.server.app:app --host 127.0.0.1 --port 8000
Env:  TERRASCOPE_RATE_LIMIT (analyses / min / IP, default 30, 0 = off), TERRASCOPE_CACHE_DIR, TERRASCOPE_CACHE_MB (500),
      MAX_UPLOAD_MB (20), ENABLE_API_DOCS=1, TERRASCOPE_SHIFT_ENSEMBLE (1), TERRASCOPE_DEVICE (cpu|cuda)
"""
from __future__ import annotations

import io, json, logging, os, threading, time
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..core.engine import Engine, ChangeError, JobManager, RunnerUnavailable, SceneError, VERSION, ROOT, is_unavailable
from ..core.simulator import generate_scene

log = logging.getLogger("terrascope")
WEB = os.path.join(os.path.dirname(__file__), "..", "web")
ART = os.path.join(ROOT, "artifacts")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
ENABLE_DOCS = os.environ.get("ENABLE_API_DOCS", "0") == "1"

app = FastAPI(title="TerraScope", version=VERSION,
              docs_url="/docs" if ENABLE_DOCS else None, redoc_url=None, openapi_url="/openapi.json" if ENABLE_DOCS else None)
engine = Engine()
jobs = JobManager(workers=1)

# RS-owned routes (/api/models, /api/scenes, …) are optional: the app must run without them.
RS_ROUTES = False
try:
    from .routes_rs import router as _rs_router
    app.include_router(_rs_router)
    RS_ROUTES = True
except ImportError as _e:
    log.warning("routes_rs not available (%s); /api/models and /api/scenes fall back to TerraFM-only", _e)
except Exception:  # a bug in routes_rs must not take the whole app down, but must be visible
    log.exception("routes_rs failed to load")


# ------------------------------------------------------------------ scene deletion -> drop derived state
@app.middleware("http")
async def _forget_deleted_scenes(request: Request, call_next):
    """DELETE /api/scenes/{id} is served by RS's router; afterwards drop the engine's in-memory upload copy and
    every cached analysis computed from that scene (so deleted data is not served from the cache)."""
    resp = await call_next(request)
    path = request.url.path
    if request.method == "DELETE" and path.startswith("/api/scenes/") and resp.status_code < 300:
        sid = path[len("/api/scenes/"):].strip("/")
        if sid and "/" not in sid:
            await run_in_threadpool(engine.forget_scene, sid)
    return resp


# ------------------------------------------------------------------ rate limit (token bucket per client IP)
class RateLimiter:
    def __init__(self, per_min: float):
        self.per_min = float(per_min)
        self.buckets: dict = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.per_min <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.per_min, now))
            tokens = min(self.per_min, tokens + (now - last) * self.per_min / 60.0)
            ok = tokens >= 1
            self.buckets[key] = (tokens - 1 if ok else tokens, now)
            if len(self.buckets) > 10_000:
                cutoff = now - 120
                self.buckets = {k: v for k, v in self.buckets.items() if v[1] > cutoff}
            return ok

    def reset(self):
        with self.lock:
            self.buckets.clear()


limiter = RateLimiter(float(os.environ.get("TERRASCOPE_RATE_LIMIT", "30")))
TRUST_PROXY = os.environ.get("TERRASCOPE_TRUST_PROXY", "0") == "1"


def client_key(request: Request) -> str:
    """Rate-limit key. Default: the direct peer address only; X-Forwarded-For is ignored, so clients cannot
    spoof their way past the limiter. With TERRASCOPE_TRUST_PROXY=1 (app behind exactly one trusted reverse
    proxy) the RIGHT-most X-Forwarded-For entry is used — the one appended by that proxy, not by the client.
    NB: uvicorn's --proxy-headers/--forwarded-allow-ips='*' rewrites the peer from X-Forwarded-For before the
    app sees it; do not combine that with an untrusted network."""
    peer = request.client.host if request.client else "unknown"
    if TRUST_PROXY:
        xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if xff:
            return xff[-1]
    return peer


def _rate(request: Request):
    ip = client_key(request)
    if not limiter.allow(ip):
        raise HTTPException(429, f"Too many analyses — limit is {limiter.per_min:g} per minute. Try again shortly.",
                            headers={"Retry-After": "5"})


# ------------------------------------------------------------------ requests
def _check_size(v: int) -> int:
    if v % 4:
        raise ValueError("size must be a multiple of 4")
    return v


class AnalyzeReq(BaseModel):
    source: Literal["sim", "upload", "scene"] = "sim"
    seed: int = Field(7, ge=0, le=2**31 - 1)
    size: int = Field(128, ge=32, le=256)
    flood: float = Field(0.0, ge=0, le=1)
    cloud: float = Field(0.0, ge=0, le=1)
    drought: float = Field(0.0, ge=0, le=1)
    sensors: list[bool] = [True, True, True]
    scene_id: str | None = Field(None, max_length=96)
    model: str = Field("terrafm", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")

    @field_validator("size")
    @classmethod
    def _size(cls, v: int) -> int:
        return _check_size(v)


def _precheck(r: AnalyzeReq):
    if len(r.sensors) != 3 or not any(r.sensors):
        raise HTTPException(400, "Enable at least one sensor (optical, SAR, DEM).")
    if r.source in ("upload", "scene") and not r.scene_id:
        raise HTTPException(404, "Scene not found — upload or pick a scene first.")
    try:
        engine.get_runner(r.model)
    except RunnerUnavailable as e:
        raise HTTPException(409, str(e))
    except Exception as e:
        if is_unavailable(e):
            raise HTTPException(409, str(e))
        log.exception("runner lookup failed")
        raise HTTPException(500, f"Model '{r.model}' failed to load.")


def _user_error(e: Exception) -> tuple[int, str]:
    if isinstance(e, SceneError):
        return 400, f"Could not analyse scene: {e}"
    if isinstance(e, KeyError):
        return 404, str(e.args[0]) if e.args else "Not found."
    if is_unavailable(e):
        return 409, str(e)
    return 500, "Analysis failed for this scene."


@app.get("/api/health")
def health():
    models = engine.list_models()
    return dict(ready=engine.ready, params=engine.n_params, models=sum(1 for m in models if m.get("status") == "ready"),
                device=engine.device, version=VERSION, norm=getattr(engine.runner, "norm", None),
                weights_file=getattr(engine.runner, "weights_file", None), rs_routes=RS_ROUTES)


if not RS_ROUTES:
    @app.get("/api/models")
    def models_fallback():
        return engine.list_models()


@app.post("/api/analyze")
def analyze(r: AnalyzeReq, request: Request):
    _precheck(r)
    _rate(request)
    try:
        return engine.run(r.model_dump())
    except Exception as e:
        code, msg = _user_error(e)
        if code == 500:
            log.exception("analysis failed")
        raise HTTPException(code, msg)


@app.post("/api/analyze_async")
def analyze_async(r: AnalyzeReq, request: Request):
    _precheck(r)
    _rate(request)
    req = r.model_dump()

    def err(e):
        code, msg = _user_error(e)
        if code == 500:
            log.error("async analysis failed", exc_info=e)
        return msg

    job = jobs.submit(lambda on_stage: engine.run(req, on_stage), error_message=err)
    return dict(job_id=job.id)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "Unknown or expired job.")
    return j.public()


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str):
    if jobs.get(job_id) is None:
        raise HTTPException(404, "Unknown or expired job.")

    def gen():
        for ev in jobs.stream(job_id):
            yield ": keep-alive\n\n" if ev is None else f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _read_capped(f: UploadFile) -> bytes:
    raw = await f.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // 2**20} MB).")
    return raw


def _is_tif(f: UploadFile | None) -> bool:
    return f is not None and (f.filename or "").lower().endswith((".tif", ".tiff"))


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile | None = File(None), optical: UploadFile | None = File(None),
                 sar: UploadFile | None = File(None), dem: UploadFile | None = File(None), name: str | None = Form(None)):
    _rate(request)
    parts = {k: v for k, v in (("optical", optical), ("sar", sar), ("dem", dem)) if v is not None}
    if file is None and not parts:
        raise HTTPException(400, "No file uploaded (field 'file', or GeoTIFF fields 'optical', 'sar', 'dem').")
    if parts or _is_tif(file):
        try:
            from ..real.geotiff import register_geotiff, GeoTiffError
        except ImportError:
            raise HTTPException(415, "GeoTIFF upload is not available in this installation (rasterio missing); upload .npz instead.")
        data = {k: await _read_capped(v) for k, v in parts.items()} if parts else await _read_capped(file)
        try:
            sid, sc = await run_in_threadpool(register_geotiff, data,
                                              name=(name or (file.filename if file else None) or "GeoTIFF upload")[:120])
        except GeoTiffError as e:
            raise HTTPException(400, f"Could not read GeoTIFF: {e}")
        except Exception:
            log.exception("GeoTIFF upload failed")
            raise HTTPException(400, "Could not read GeoTIFF.")
        meta = sc.get("meta") or {}
        return dict(scene_id=sid, size=list(np.asarray(sc["optical"]).shape[1:]), source="geotiff",
                    bounds=meta.get("bounds"), gsd_m=meta.get("gsd_m"))
    raw = await _read_capped(file)
    try:
        sid, sc = await run_in_threadpool(engine.register_upload_bytes, raw)
    except SceneError as e:
        raise HTTPException(400, f"Could not read scene: {e}")
    except Exception:
        log.exception("upload parse failed")
        raise HTTPException(400, "Could not read scene: not a valid .npz file.")
    return dict(scene_id=sid, size=list(sc["optical"].shape[1:]), source="upload", bounds=None, gsd_m=sc["meta"]["gsd_m"],
                gsd_assumed=sc["meta"]["gsd_m"] is None)


@app.get("/api/layer/{aid}/{name}.png")
def layer(aid: str, name: str):
    data = engine.layer(aid, name)
    if data is None:
        raise HTTPException(404)
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.get("/api/similarity/{aid}.png")
def similarity(aid: str, x: int = Query(0, ge=0, le=4096), y: int = Query(0, ge=0, le=4096)):
    if not engine.has(aid):
        raise HTTPException(404)
    try:
        return Response(engine.similarity_png(aid, x, y), media_type="image/png")
    except KeyError:
        raise HTTPException(404, "No embeddings stored for this result.")


@app.get("/api/export/{aid}.npz")
def export(aid: str):
    arrs = engine.export_arrays(aid)
    if not arrs:
        raise HTTPException(404)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename=terrascope_{aid}.npz"})


@app.get("/api/change")
def change(a: str = Query(..., max_length=64), b: str = Query(..., max_length=64)):
    try:
        return engine.change(a, b)
    except ChangeError as e:
        raise HTTPException(409, str(e))
    except KeyError:
        raise HTTPException(404, "Analysis not found (expired?) — run both analyses again.")


@app.get("/api/sample.npz")
def sample(seed: int = Query(21, ge=0, le=2**31 - 1), flood: float = Query(0.8, ge=0, le=1),
           cloud: float = Query(0.4, ge=0, le=1), drought: float = Query(0.3, ge=0, le=1),
           size: int = Query(128, ge=32, le=256, multiple_of=4)):
    """A ready-made scene in the upload format (try the 'Bring your own data' flow)."""
    sc = generate_scene(seed, size, flood=flood, cloud=cloud, drought=drought)
    buf = io.BytesIO()
    np.savez_compressed(buf, optical=sc["optical"], sar=sc["sar"], dem=sc["dem"])
    return Response(buf.getvalue(), media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=terrascope_sample.npz"})


# ------------------------------------------------------------------ benchmarks
_BENCH: dict = {}
SIM_KEYS = ("ablation", "scratch", "size128", "heavy_cloud", "curves", "config", "seeds", "recon", "cloud")


def _json_or_none(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _bench():
    paths = {k: os.path.join(ART, f) for k, f in (("sim", "metrics.json"), ("cal", "calibration.json"),
                                                  ("real", "metrics_real.json"), ("laptop", "bench_laptop.json"))}
    stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else None for p in paths.values())
    if _BENCH.get("stamp") == stamp:
        return _BENCH["data"]
    m = _json_or_none(paths["sim"])
    if m is None:
        return None
    out = {k: m[k] for k in SIM_KEYS if k in m}                     # legacy top-level keys stay
    out["sim"] = {k: m[k] for k in SIM_KEYS if k in m}
    seeds = m.get("seeds")
    out["sim"]["seeds"] = None if seeds is None else dict(n=seeds.get("n"), std=seeds.get("std"), mean=seeds.get("mean"), seeds=seeds.get("seeds"))
    out["robustness"] = m.get("robustness")
    cal = _json_or_none(paths["cal"])
    out["calibration"] = None if cal is None else {k: cal.get(k) for k in ("method", "T", "apply", "decision", "val", "bins", "ece_before", "ece_after", "nll_before",
                                                                         "nll_after", "fit_split", "eval_split", "n_bins", "note")}
    out["real"] = _json_or_none(paths["real"])
    out["laptop_bench"] = _json_or_none(paths["laptop"])
    _BENCH.update(stamp=stamp, data=out)
    return out


@app.get("/api/benchmarks")
def benchmarks():
    b = _bench()
    if b is None:
        raise HTTPException(404, "metrics.json missing — run terrascope.core.evaluate")
    return JSONResponse(b)


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/static", StaticFiles(directory=WEB), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")), workers=1)
