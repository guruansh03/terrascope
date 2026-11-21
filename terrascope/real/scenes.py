"""Scene store behind GET/PATCH/DELETE /api/scenes.

Holds three kinds of scenes:
* ``upload``   - .npz uploads (the ML agent's upload route can call ``STORE.add(sc, name, "upload")``)
* ``geotiff``  - GeoTIFF imports via :func:`terrascope.real.geotiff.register_geotiff`
* ``demo``     - read-only files found in ``data/demo/*.npz`` / ``data/demo/*.tif`` with an optional
                 sidecar ``<stem>.json`` ({name, source, acquisition_date, bounds, gsd_m, tasks, description})

User scenes are persisted to ``data/scenes/<id>.npz`` + ``<id>.json`` so they survive a restart.
"""
from __future__ import annotations

import json, os, re, secrets, threading, time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
ARRAY_KEYS = ("optical", "sar", "dem", "optical_clean", "lc", "flood", "veg", "burn", "cloud", "s2_full")
TRUTH_TASK = {"lc": "landcover", "flood": "flood", "veg": "veg", "burn": "burn", "cloud": "cloud"}
OPT_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12"]


class SceneNotFound(KeyError):
    pass


class SceneReadOnly(PermissionError):
    pass


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _summary(sid, sc, info):
    meta = sc.get("meta") or {}
    avail = meta.get("available") or {k: k in sc for k in ("optical", "sar", "dem")}
    bands = []
    if avail.get("optical"):
        bands += OPT_NAMES
    if avail.get("sar"):
        bands += ["VV", "VH"]
    if avail.get("dem"):
        bands += ["elevation", "slope"]
    truth = [k for k in TRUTH_TASK if k in sc]
    tasks = info.get("tasks") or [TRUTH_TASK[k] for k in truth]
    H, W = sc["optical"].shape[1:] if "optical" in sc else (None, None)
    return dict(id=sid, name=info.get("name") or sid, source=info.get("source", "upload"),
                acquisition_date=info.get("acquisition_date", meta.get("acquisition_date")),
                bounds=info.get("bounds", meta.get("bounds")), size=[H, W], bands=bands,
                has_truth=bool(truth), tasks=tasks, gsd_m=info.get("gsd_m", meta.get("gsd_m")),
                created_at=info.get("created_at"), read_only=info.get("source") == "demo")


class SceneStore:
    def __init__(self, scenes_dir=None, demo_dir=None, max_user_scenes: int = 40):
        self.scenes_dir = scenes_dir or os.environ.get("TERRASCOPE_SCENES_DIR", os.path.join(ROOT, "data", "scenes"))
        self.demo_dir = demo_dir or os.environ.get("TERRASCOPE_DEMO_DIR", os.path.join(ROOT, "data", "demo"))
        self.max_user = max_user_scenes
        self.lock = threading.RLock()
        self._mem: dict[str, dict] = {}     # id -> scene (small LRU of loaded scenes)

    # ------------------------------------------------------------------ paths
    def _p(self, sid, ext):
        if not ID_RE.match(sid or ""):
            raise SceneNotFound(sid)
        return os.path.join(self.scenes_dir, f"{sid}{ext}")

    def _demo_files(self):
        out = {}
        if os.path.isdir(self.demo_dir):
            for fn in sorted(os.listdir(self.demo_dir)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() in (".npz", ".tif", ".tiff") and ID_RE.match(stem):
                    out.setdefault("demo-" + stem, os.path.join(self.demo_dir, fn))
        return out

    # ------------------------------------------------------------------ CRUD
    def add(self, sc: dict, name: str = "upload", source: str = "upload", sid: str | None = None) -> str:
        sid = sid or ("sc" + secrets.token_urlsafe(9).replace("-", "x").replace("_", "y"))
        meta = dict(sc.get("meta") or {})
        info = dict(name=str(name)[:120], source=source, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    acquisition_date=meta.get("acquisition_date"), bounds=meta.get("bounds"), gsd_m=meta.get("gsd_m"),
                    meta=_jsonable(meta))
        with self.lock:
            os.makedirs(self.scenes_dir, exist_ok=True)
            arrays = {k: np.asarray(sc[k]) for k in ARRAY_KEYS if k in sc and sc[k] is not None}
            tmp = self._p(sid, ".npz.tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, self._p(sid, ".npz"))
            with open(self._p(sid, ".json"), "w") as f:
                json.dump(info, f)
            self._mem[sid] = sc
            self._evict()
        return sid

    def _evict(self):
        ids = self._user_ids()
        if len(ids) > self.max_user:
            ids.sort(key=lambda i: os.path.getmtime(self._p(i, ".json")))
            for i in ids[: len(ids) - self.max_user]:
                self.delete(i)
        while len(self._mem) > 8:
            self._mem.pop(next(iter(self._mem)))

    def _user_ids(self):
        if not os.path.isdir(self.scenes_dir):
            return []
        return [fn[:-5] for fn in os.listdir(self.scenes_dir) if fn.endswith(".json") and ID_RE.match(fn[:-5])
                and os.path.exists(os.path.join(self.scenes_dir, fn[:-5] + ".npz"))]

    def _info(self, sid):
        if sid.startswith("demo-"):
            path = self._demo_files().get(sid)
            if not path:
                raise SceneNotFound(sid)
            side = os.path.splitext(path)[0] + ".json"
            info = {}
            if os.path.exists(side):
                try:
                    with open(side) as f:
                        info = json.load(f)
                except Exception:
                    info = {}
            info = dict(info)
            info["source"] = "demo"
            info.setdefault("name", sid[5:].replace("_", " "))
            return info, path
        jp = self._p(sid, ".json")
        if not os.path.exists(jp):
            raise SceneNotFound(sid)
        with open(jp) as f:
            return json.load(f), self._p(sid, ".npz")

    def get(self, sid: str) -> dict:
        with self.lock:
            if sid in self._mem:
                return self._mem[sid]
            info, path = self._info(sid)
            if path.lower().endswith((".tif", ".tiff")):
                from .geotiff import scene_from_geotiffs
                sc = scene_from_geotiffs(single=path)
                sc["meta"]["data_source"] = info.get("data_source", "geotiff")
            else:
                with np.load(path, allow_pickle=False) as z:
                    sc = {k: z[k] for k in z.files if k in ARRAY_KEYS}
                meta = dict(info.get("meta") or {})
                for k in ("bounds", "acquisition_date", "gsd_m"):
                    if info.get(k) is not None:
                        meta[k] = info[k]
                meta.setdefault("data_source", info.get("data_source") or
                                ("upload" if info.get("source") == "upload" else info.get("source", "upload")))
                if info.get("source") == "demo" and info.get("data_source"):
                    meta["data_source"] = info["data_source"]
                meta.setdefault("available", {k: k in sc for k in ("optical", "sar", "dem")})
                # optical-only datasets (e.g. HLS Burn Scars) have no SAR/DEM arrays: zero-fill them like the
                # GeoTIFF path does; meta.available keeps them switched off for every model
                ref = next((sc[k] for k in ("optical", "sar", "dem") if k in sc), None)
                if ref is not None:
                    h, w = np.asarray(ref).shape[-2:]
                    for k, c in (("optical", 6), ("sar", 2), ("dem", 2)):
                        if k not in sc:
                            sc[k] = np.zeros((c, h, w), np.float32)
                sc["meta"] = meta
            self._mem[sid] = sc
            self._evict()
            return sc

    def summary(self, sid):
        info, _ = self._info(sid)
        return _summary(sid, self.get(sid), info)

    def list(self) -> list[dict]:
        out = []
        with self.lock:
            ids = sorted(self._user_ids(), key=lambda i: -os.path.getmtime(self._p(i, ".json")))
            for sid in ids + list(self._demo_files()):
                try:
                    out.append(self.summary(sid))
                except Exception as e:  # a broken file must not break the listing
                    out.append(dict(id=sid, name=sid, source="error", acquisition_date=None, bounds=None,
                                    size=[None, None], bands=[], has_truth=False, tasks=[], error=str(e)[:200]))
        return out

    def rename(self, sid: str, name: str) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("name must not be empty")
        with self.lock:
            if sid.startswith("demo-"):
                self._info(sid)
                raise SceneReadOnly(sid)
            info, _ = self._info(sid)
            info["name"] = name[:120]
            with open(self._p(sid, ".json"), "w") as f:
                json.dump(info, f)
        return self.summary(sid)

    def delete(self, sid: str) -> None:
        with self.lock:
            if sid.startswith("demo-"):
                self._info(sid)
                raise SceneReadOnly(sid)
            self._info(sid)
            for ext in (".npz", ".json"):
                try:
                    os.remove(self._p(sid, ext))
                except FileNotFoundError:
                    pass
            self._mem.pop(sid, None)


STORE = SceneStore()


def get_scene(sid: str) -> dict:
    """Convenience for the analyze route (source:"scene")."""
    return STORE.get(sid)
