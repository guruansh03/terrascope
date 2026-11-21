"""Analysis result cache: small in-memory LRU in front of a disk store (``cache/``) bounded by total bytes.

Layout per entry:  cache/<id>/meta.json  (response JSON + bookkeeping)
                   cache/<id>/layers/<name>.png
                   cache/<id>/arrays.npz  (scene + predictions + features; no pickles)
Results therefore survive a server restart. Eviction is least-recently-used by total size.
"""
from __future__ import annotations

import io, json, os, re, shutil, threading, time
from collections import OrderedDict

import numpy as np

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,40}$")


def _dir_size(p):
    t = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t


class ResultCache:
    def __init__(self, root, max_bytes=500 * 2**20, mem_items=14):
        self.root = os.path.abspath(root)
        self.max_bytes = int(max_bytes)
        self.mem_items = mem_items
        self.mem: "OrderedDict[str, dict]" = OrderedDict()
        self.lock = threading.RLock()
        os.makedirs(self.root, exist_ok=True)
        self._sizes = {}
        self._scene_of = {}  # id -> scene id it was computed from (uploads / stored scenes)
        self._keys = {}   # request key -> id (request-keyed reuse, e.g. precomputed demo analyses)
        for d in os.listdir(self.root):
            p = os.path.join(self.root, d)
            if ID_RE.match(d) and os.path.isdir(p):
                if os.path.exists(os.path.join(p, "meta.json")):
                    self._sizes[d] = _dir_size(p)
                    try:
                        with open(os.path.join(p, "key.txt")) as f:
                            self._keys[f.read().strip()] = d
                    except OSError:
                        pass
                    try:
                        with open(os.path.join(p, "scene.txt")) as f:
                            self._scene_of[d] = f.read().strip()
                    except OSError:
                        pass
                else:  # interrupted write
                    shutil.rmtree(p, ignore_errors=True)
            elif d.startswith(".tmp-"):
                shutil.rmtree(p, ignore_errors=True)

    @staticmethod
    def valid_id(aid):
        return bool(aid) and bool(ID_RE.match(aid))

    def _path(self, aid):
        if not self.valid_id(aid):
            raise KeyError(aid)
        return os.path.join(self.root, aid)

    def total_bytes(self):
        return sum(self._sizes.values())

    # ------------------------------------------------------------------ write
    def put(self, aid, layers, arrays=None, meta=None, key=None, scene_id=None):
        """layers: name -> PNG bytes, arrays: name -> ndarray, meta: JSON-able dict, key: request key for reuse."""
        path = self._path(aid)
        tmp = os.path.join(self.root, f".tmp-{aid}")
        os.makedirs(os.path.join(tmp, "layers"), exist_ok=True)
        for name, data in layers.items():
            if not NAME_RE.match(name):
                raise ValueError(f"bad layer name {name!r}")
            with open(os.path.join(tmp, "layers", f"{name}.png"), "wb") as f:
                f.write(data)
        if arrays:
            np.savez(os.path.join(tmp, "arrays.npz"), **{k: np.asarray(v) for k, v in arrays.items() if v is not None})
        meta = dict(meta or {}, id=aid, layers=sorted(layers), created=time.time())
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f)
        if key:
            with open(os.path.join(tmp, "key.txt"), "w") as f:
                f.write(key)
        if scene_id:
            with open(os.path.join(tmp, "scene.txt"), "w") as f:
                f.write(str(scene_id))
        with self.lock:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
            os.replace(tmp, path)
            self._sizes[aid] = _dir_size(path)
            if key:
                self._keys[key] = aid
            if scene_id:
                self._scene_of[aid] = str(scene_id)
            self.mem[aid] = dict(meta=meta, layers=dict(layers), arrays=arrays)
            self.mem.move_to_end(aid)
            while len(self.mem) > self.mem_items:
                self.mem.popitem(last=False)
            self._evict(keep=aid)

    def _evict(self, keep=None):
        while self.total_bytes() > self.max_bytes and len(self._sizes) > 1:
            order = sorted((k for k in self._sizes if k != keep), key=self._atime)
            if not order:
                break
            self.remove(order[0])

    def remove(self, aid):
        with self.lock:
            shutil.rmtree(os.path.join(self.root, aid), ignore_errors=True)
            self._sizes.pop(aid, None)
            self.mem.pop(aid, None)
            self._scene_of.pop(aid, None)
            self._keys = {k: v for k, v in self._keys.items() if v != aid}

    def remove_scene(self, scene_id):
        with self.lock:
            victims = [a for a, s in self._scene_of.items() if s == scene_id]
            for a in victims:
                self.remove(a)
        return victims

    def _atime(self, aid):
        try:
            return os.path.getmtime(os.path.join(self.root, aid, "meta.json"))
        except OSError:
            return 0.0

    def _touch(self, aid):
        try:
            os.utime(os.path.join(self.root, aid, "meta.json"))
        except OSError:
            pass

    # ------------------------------------------------------------------ read
    def lookup(self, key):
        """id of a cached result produced by an identical request, or None."""
        with self.lock:
            aid = self._keys.get(key)
        return aid if aid is not None and aid in self else None

    def __contains__(self, aid):
        return self.valid_id(aid) and (aid in self.mem or aid in self._sizes)

    def meta(self, aid):
        if aid not in self:
            return None
        with self.lock:
            if aid in self.mem:
                self.mem.move_to_end(aid)
                self._touch(aid)
                return self.mem[aid]["meta"]
        try:
            with open(os.path.join(self._path(aid), "meta.json")) as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
        self._touch(aid)
        return m

    def layer(self, aid, name):
        if aid not in self or not NAME_RE.match(name or ""):
            return None
        with self.lock:
            e = self.mem.get(aid)
            if e is not None and name in e["layers"]:
                return e["layers"][name]
        p = os.path.join(self._path(aid), "layers", f"{name}.png")
        try:
            with open(p, "rb") as f:
                return f.read()
        except OSError:
            return None

    def arrays(self, aid):
        if aid not in self:
            return None
        with self.lock:
            e = self.mem.get(aid)
            if e is not None and e.get("arrays") is not None:
                return e["arrays"]
        p = os.path.join(self._path(aid), "arrays.npz")
        if not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=False) as z:
            arr = {k: z[k] for k in z.files}
        with self.lock:
            if aid in self.mem:
                self.mem[aid]["arrays"] = arr
        return arr

    def get(self, aid):
        """Backward-compatible accessor: dict(meta, layers, arrays) or None."""
        m = self.meta(aid)
        if m is None:
            return None
        return dict(meta=m, layers={n: self.layer(aid, n) for n in m.get("layers", [])}, arrays=self.arrays(aid))
