"""Analysis engine: scene preparation, runner dispatch, rendering, stats, caching, change detection."""
from __future__ import annotations

import base64, datetime as _dt, hashlib, json, logging, os, secrets, threading, time

import numpy as np

from ..simulator import CLASS_NAMES, CLASS_COLORS, GSD_M, generate_scene
from . import render, stats as stats_mod
from .cache import ResultCache
from .runner import RunnerUnavailable, is_unavailable
from .terrafm_runner import TerraFMRunner, ROOT
from .validate import SceneError, arrays_from_npz_bytes, scene_from_arrays

log = logging.getLogger("terrascope.engine")
TRUTH_KEYS = ("lc", "flood", "veg", "burn", "cloud")
SCENE_KEYS = ("optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud", "burn")
VERSION = "3.0.0"


def _noop(stage, detail=None):
    pass


def b64u8(a):
    return base64.b64encode(np.ascontiguousarray(np.asarray(a).astype(np.uint8)).tobytes()).decode()


def _registry():
    try:
        from ...fm import registry  # RS-owned; optional
        return registry
    except ImportError:
        return None


def _scene_store():
    try:
        from ...real.scenes import STORE  # RS-owned; optional
        return STORE
    except ImportError:
        return None


def request_signature(req):
    """Canonical, JSON-able form of an analysis request (what determines the result besides the weights)."""
    src = req.get("source", "sim")
    sig = dict(source=src, model=req.get("model") or "terrafm", sensors=[bool(v) for v in req.get("sensors", [True, True, True])],
               runtime=os.environ.get("TERRASCOPE_RUNTIME", "torch"))
    if src == "sim":
        sig.update(seed=int(req.get("seed", 7)), size=int(req.get("size", 128)),
                   **{k: round(float(req.get(k, 0.0)), 6) for k in ("flood", "cloud", "drought")})
    else:
        sig.update(scene_id=req.get("scene_id"))
    return sig


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:32]


class ChangeError(ValueError):
    """Two analyses cannot be compared (e.g. different sizes) -> 409."""


class Engine:
    def __init__(self, ckpt="finetuned.pt", cache_dir=None, cache_mb=None, max_cache=14):
        self.uploads: dict = {}
        self.lock = threading.Lock()
        self.runner, self.runner_error = None, None
        try:
            self.runner = TerraFMRunner(ckpt)
        except RunnerUnavailable as e:
            self.runner_error = str(e)
        cache_dir = cache_dir or os.environ.get("TERRASCOPE_CACHE_DIR") or os.path.join(ROOT, "cache")
        cache_mb = float(cache_mb if cache_mb is not None else os.environ.get("TERRASCOPE_CACHE_MB", "500"))
        self.cache = ResultCache(cache_dir, int(cache_mb * 2**20), mem_items=max_cache)
        self.precomputed = self._load_precomputed()

    def _load_precomputed(self):
        """artifacts/demo_precomputed.json: [{request, analysis_id}] written by the demo-bundle builder.
        Entries are trusted to match the bundled weights; they are used only if the id is in the cache."""
        p = os.environ.get("TERRASCOPE_PRECOMPUTED") or os.path.join(ROOT, "artifacts", "demo_precomputed.json")
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return {}
        rows = doc.get("entries", []) if isinstance(doc, dict) else doc   # {"generated_at",…,"entries":[…]} or [...]
        out = {}
        if not isinstance(rows, list):
            log.warning("demo_precomputed.json: unexpected format, ignored")
            return out
        for r in rows:
            try:
                if isinstance(r, dict) and r.get("analysis_id") and isinstance(r.get("request"), dict):
                    out[_sha(request_signature(r["request"]))] = str(r["analysis_id"])
            except (AttributeError, ValueError, TypeError, KeyError):
                continue   # one bad row must not stop the app from booting
        return out

    def request_key(self, req, runner):
        fp = getattr(runner, "fingerprint", None) or f"{runner.id}:{runner.weights_file}:{runner.params}"
        return _sha(dict(request_signature(req), weights=fp))

    def cached_result(self, req, runner):
        """Response of an identical earlier request (same scene/sim params, model weights, sensors, runtime) or None."""
        if req.get("source", "sim") != "sim" and not req.get("scene_id"):
            return None
        aid = self.cache.lookup(self.request_key(req, runner))
        if aid is None:
            pre = self.precomputed.get(_sha(request_signature(req)))
            aid = pre if pre and pre in self.cache else None
        if aid is None:
            return None
        m = self.cache.meta(aid) or {}
        res = m.get("response")
        if not res:
            return None
        return dict(res, cached=True)

    # ------------------------------------------------------------------ status
    @property
    def ready(self):
        return self.runner is not None

    @property
    def model(self):
        return self.runner.model if self.runner else None

    @property
    def n_params(self):
        return self.runner.params if self.runner else 0

    @property
    def device(self):
        return self.runner.device if self.runner else "cpu"

    def list_models(self):
        reg = _registry()
        if reg is not None:
            try:
                return reg.list_models()
            except Exception:  # registry problems must not take the app down
                log.exception("fm.registry.list_models failed")
        if self.runner:
            return [self.runner.info()]
        return [dict(id="terrafm", name="TerraFM (ours)", family="terrafm", params=None, sensors=["optical", "sar", "dem"],
                     tasks=["landcover", "flood", "veg", "cloud"], status="weights_missing", weights_file=None,
                     license="project-internal", source_url=None)]

    def get_runner(self, model_id="terrafm"):
        """-> Runner, or raises RunnerUnavailable (API: 409) for unknown/unready models."""
        if model_id in (None, "", "terrafm"):
            if self.runner is None:
                raise RunnerUnavailable(self.runner_error or "TerraFM checkpoint not found.")
            return self.runner
        reg = _registry()
        if reg is None:
            raise RunnerUnavailable(f"Model '{model_id}' is not available in this build (only 'terrafm').")
        try:
            return reg.get_runner(model_id)
        except Exception as e:
            if is_unavailable(e):
                raise RunnerUnavailable(str(e)) from e
            if isinstance(e, KeyError):
                raise RunnerUnavailable(f"Unknown model '{model_id}'.") from e
            raise

    # ------------------------------------------------------------------ scenes
    def sim_scene(self, seed, size, flood, cloud, drought):
        sc = generate_scene(int(seed), int(size), flood=float(flood), cloud=float(cloud), drought=float(drought))
        out = {k: sc[k] for k in ["optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud"]}
        out["meta"] = dict(data_source="simulated", gsd_m=GSD_M, bounds=None, acquisition_date=None)
        return out

    def register_upload_bytes(self, raw):
        """Validate an uploaded .npz (headers first, then contents) and register it."""
        return self.register_upload(arrays_from_npz_bytes(raw))

    def register_upload(self, arrays):
        sc = scene_from_arrays(arrays)
        sc["meta"] = dict(data_source="upload", gsd_m=None, bounds=None, acquisition_date=None)  # .npz carries no GSD
        sid = "up" + secrets.token_urlsafe(12)
        with self.lock:
            self.uploads[sid] = sc
            while len(self.uploads) > 6:
                self.uploads.pop(next(iter(self.uploads)))
        store = _scene_store()
        if store is not None:  # persist + list in /api/scenes (RS); in-memory copy stays authoritative
            try:
                store.add({k: v for k, v in sc.items()}, name="upload", source="upload", sid=sid)
            except Exception:
                log.exception("could not persist upload to the scene store")
        return sid, sc

    def get_upload(self, sid):
        sc = self.uploads.get(sid or "")
        if sc is None and sid:
            store = _scene_store()
            if store is not None:
                try:
                    sc = store.get(sid)
                except Exception:
                    sc = None
        return sc

    def get_scene(self, scene_id):
        store = _scene_store()
        if store is None:
            raise KeyError("scene store not available")
        return store.get(scene_id)

    def prepare(self, req):
        """req: dict(source, seed, size, flood, cloud, drought, scene_id). -> (sc, meta)."""
        src = req.get("source", "sim")
        if src == "upload":
            sc = self.get_upload(req.get("scene_id"))
            if sc is None:
                raise KeyError("Uploaded scene expired — upload again.")
            meta = dict(source="upload", scene_id=req.get("scene_id"))
        elif src == "scene":
            try:
                sc = self.get_scene(req.get("scene_id") or "")
            except KeyError:
                raise KeyError("Scene not found.")
            meta = dict(source="scene", scene_id=req.get("scene_id"))
        else:
            sc = self.sim_scene(req.get("seed", 7), req.get("size", 128), req.get("flood", 0.0), req.get("cloud", 0.0), req.get("drought", 0.0))
            meta = {k: req.get(k) for k in ("seed", "size", "flood", "cloud", "drought")}
            meta["source"] = "sim"
        return sc, meta

    def run(self, req, on_stage=_noop):
        """prepare + analyze, emitting real stage events."""
        runner = self.get_runner(req.get("model", "terrafm"))
        on_stage("preparing")
        hit = self.cached_result(req, runner)
        if hit is not None:
            on_stage("rendering", "cached result")
            return hit
        sc, meta = self.prepare(req)
        key = self.request_key(req, runner) if (req.get("source", "sim") == "sim" or req.get("scene_id")) else None
        return self.analyze(sc, req.get("sensors", [True, True, True]), meta, runner=runner, on_stage=on_stage, key=key)

    # ------------------------------------------------------------------ analysis
    def analyze(self, sc, sensors, meta, model="terrafm", runner=None, on_stage=_noop, key=None):
        t0 = time.perf_counter()
        runner = runner or self.get_runner(model)
        smeta = dict(sc.get("meta") or {})
        avail = smeta.get("available") or {}
        sensors = [bool(s) and bool(avail.get(k, True)) for s, k in zip(sensors, ("optical", "sar", "dem"))]
        if not any(sensors):
            raise SceneError("none of the requested sensors is present in this scene")
        o = np.asarray(sc["optical"], np.float32)
        H, W = o.shape[1:]
        if H % 4 or W % 4:
            raise SceneError("scene height and width must be multiples of 4")
        arrays = {k: np.asarray(sc[k]) for k in SCENE_KEYS if k in sc}
        arrays["optical"] = o
        pred = runner.predict(arrays, sensors, on_stage)
        pred = {k: (np.asarray(v, np.float32) if isinstance(v, np.ndarray) else v) for k, v in pred.items()}
        runtime_ms = (time.perf_counter() - t0) * 1000
        on_stage("rendering")
        gsd = float(smeta["gsd_m"]) if smeta.get("gsd_m") else None   # None -> areas are not reported
        L = render.input_layers(arrays, gsd or GSD_M)                  # display-only hillshade scale
        out_L, derived = render.output_layers(arrays, pred, sensors)
        L.update(out_L)
        truth_keys = [k for k in ("lc", "flood", "veg") if k in arrays]
        truth = len(truth_keys) > 0
        if truth:
            L.update(render.truth_layers(arrays))
        cal = getattr(runner, "calibration", None)
        st = stats_mod.compute(arrays, pred, derived, sensors, gsd, cal)
        st.update(timing_ms=round(runtime_ms, 1), size=[int(H), int(W)])
        aid = "a" + secrets.token_urlsafe(12)
        data_source = smeta.get("data_source") or ("simulated" if meta.get("source") == "sim" else meta.get("source", "upload"))
        prov = dict(model_id=runner.id, model_name=runner.name, params=runner.params, weights_file=runner.weights_file,
                    data_source=data_source, acquisition_date=smeta.get("acquisition_date"),
                    processed_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    runtime_ms=round(runtime_ms, 1), device=getattr(runner, "device", "cpu"))
        lc = derived.get("lc")
        fp, veg = pred.get("flood_prob"), pred.get("veg")
        pix = dict(
            lc=b64u8(lc) if lc is not None else None,
            flood=b64u8(fp * 255) if fp is not None else None,
            veg=b64u8(np.clip(veg, 0, 1) * 255) if veg is not None else None,
            unc=b64u8(derived["ent"] * 255) if "ent" in derived else None,
            cloud_mask=b64u8(derived["cloud_mask"]) if "cloud_mask" in derived else None,
            burn=b64u8(pred["burn_prob"] * 255) if "burn_prob" in pred else None,
            lc_true=b64u8(arrays["lc"]) if "lc" in arrays else None,
            flood_true=b64u8(arrays["flood"]) if "flood" in arrays else None,
            veg_true=b64u8(np.clip(arrays["veg"], 0, 1) * 255) if "veg" in arrays else None,
        )
        bounds = smeta.get("bounds")
        res = dict(id=aid, cached=False, model=runner.id, layers=sorted(L.keys()), stats=st, pix=pix, truth=truth, meta=meta,
                   sensors=sensors, provenance=prov, bounds=bounds, gsd_m=gsd, gsd_assumed=gsd is None,
                   composite_method=derived.get("composite_method"),
                   classes=[dict(name=n, color="#%02x%02x%02x" % c) for n, c in zip(CLASS_NAMES, CLASS_COLORS)])
        store = {f"sc_{k}": v for k, v in arrays.items()}
        store.update(pred_lc=lc, pred_flood=fp, pred_veg=veg, pred_cloud=pred.get("cloud_prob"), pred_burn=pred.get("burn_prob"),
                     feats=pred["feats"].astype(np.float16) if "feats" in pred else None,
                     feat_hw=np.array(pred["feat_hw"]) if "feat_hw" in pred else None, size=np.array([H, W]))
        self.cache.put(aid, L, store, dict(kind="analysis", response=res, gsd_m=gsd, bounds=bounds), key=key,
                       scene_id=meta.get("scene_id") if meta.get("source") in ("upload", "scene") else None)
        return res

    def forget_scene(self, scene_id):
        """A scene/upload was deleted: drop the in-memory copy and every cached result computed from it."""
        with self.lock:
            self.uploads.pop(scene_id, None)
        return self.cache.remove_scene(scene_id)

    # ------------------------------------------------------------------ cached results
    def layer(self, aid, name):
        return self.cache.layer(aid, name)

    def has(self, aid):
        return aid in self.cache

    def similarity_png(self, aid, x, y):
        arr = self.cache.arrays(aid)
        if arr is None or "feats" not in arr:
            raise KeyError(aid)
        meta = self.cache.meta(aid) or {}
        base = self.cache.layer(aid, "rgb_clean") or self.cache.layer(aid, "rgb")
        return render.similarity_png(arr["feats"].astype(np.float32), tuple(arr["feat_hw"]), tuple(arr["size"]), base, x, y)

    def export_arrays(self, aid):
        arr = self.cache.arrays(aid)
        if arr is None:
            return None
        out = {k[3:]: v for k, v in arr.items() if k.startswith("sc_")}
        for src, dst in (("pred_lc", "pred_lc"), ("pred_flood", "pred_flood_prob"), ("pred_veg", "pred_veg"),
                         ("pred_cloud", "pred_cloud_prob"), ("pred_burn", "pred_burn_prob")):
            if src in arr:
                out[dst] = arr[src]
        return out

    # ------------------------------------------------------------------ change detection
    def change(self, a, b):
        ma, mb = self.cache.meta(a), self.cache.meta(b)
        if ma is None or mb is None:
            raise KeyError("analysis not found (expired?)")
        if ma.get("kind") != "analysis" or mb.get("kind") != "analysis":
            raise ChangeError("change detection needs two analysis ids (not change results)")
        ba, bb = ma.get("bounds"), mb.get("bounds")
        if ba and bb and any(abs(float(ba[k]) - float(bb[k])) > 1e-4 for k in ("west", "south", "east", "north")):
            raise ChangeError("the two analyses cover different areas (bounds differ)")
        A, B = self.cache.arrays(a), self.cache.arrays(b)
        if A is None or B is None:
            raise KeyError("analysis not found (expired?)")
        if "pred_lc" not in A and "pred_flood" not in A:
            raise ChangeError("these results cannot be compared (not analyses)")
        if tuple(A["size"]) != tuple(B["size"]):
            raise ChangeError(f"scene sizes differ ({tuple(A['size'])} vs {tuple(B['size'])}); change detection needs the same grid")
        gsd = mb.get("gsd_m")
        px_km2 = None if not gsd else (float(gsd) / 1000.0) ** 2

        def km2(m):
            return None if px_km2 is None else round(float(m.sum() * px_km2), 4)
        masks, st = {}, dict(flood_onset_km2=None, veg_loss_km2=None, urban_growth_km2=None, lc_changed_frac=None)
        if "pred_flood" in A and "pred_flood" in B:
            masks["flood_onset"] = (B["pred_flood"] > 0.5) & ~(A["pred_flood"] > 0.5)
            st["flood_onset_km2"] = km2(masks["flood_onset"])
        if "pred_veg" in A and "pred_veg" in B:
            vegm = np.isin(A["pred_lc"], (1, 2, 4)) if "pred_lc" in A else np.ones_like(A["pred_veg"], bool)
            masks["veg_loss"] = vegm & ((A["pred_veg"] - B["pred_veg"]) > 0.2)
            st["veg_loss_km2"] = km2(masks["veg_loss"])
        if "pred_lc" in A and "pred_lc" in B:
            masks["urban_growth"] = (B["pred_lc"] == 3) & (A["pred_lc"] != 3)
            masks["lc_change"] = B["pred_lc"] != A["pred_lc"]
            st["urban_growth_km2"] = km2(masks["urban_growth"])
            st["lc_changed_frac"] = round(float(masks["lc_change"].mean()), 5)
        o = B["sc_optical"]
        base = render.stretch_rgb(o[2], o[1], o[0])
        L = render.change_layers(base, masks, B.get("pred_lc"))
        cid = "c" + secrets.token_urlsafe(12)
        res = dict(id=cid, a=a, b=b, layers=sorted(L), stats=st, gsd_m=gsd, gsd_assumed=gsd is None,
                   rule=dict(flood_onset="flood prob > 0.5 in b and not in a", veg_loss="vegetated in a and veg index drop > 0.2",
                             urban_growth="urban in b and not in a", lc_change="predicted class differs"))
        self.cache.put(cid, L, None, dict(kind="change", response=res))
        return res
