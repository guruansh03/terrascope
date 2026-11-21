"""PyTorch dataset wrapper around the real-data loaders (random crops + flips/rot90 for training), plus chip
caches so fine-tuning does not re-decode full GeoTIFF chips for every 224 px crop:

* :class:`DiskChipCache` - decodes each chip ONCE into ``<cache>/<id>-<sig>/<key>.npy`` (float16 / int16) and
  afterwards returns ``np.load(mmap_mode="r")`` views, so a training crop only reads the bytes it needs
  (works with DataLoader worker processes on Windows: the object is small and picklable).
  The signature hashes the source paths + sizes + mtimes, so e.g. a DEM built later invalidates the entry.
* :class:`MemoryChipCache` - keeps fully decoded chips in RAM up to a byte budget (used for the validation set).
"""
from __future__ import annotations

import hashlib, os, re
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from .specs import TILE

KEYS = ("optical", "sar", "dem", "mask", "target")


def sample_to_arrays(s: dict, sensors) -> dict:
    """Standard numpy sample -> float32 arrays with zeros for absent sensors + availability vector."""
    H, W = s["optical"].shape[1:]
    use = set(sensors)
    av = np.array([("optical" in use) and s.get("optical") is not None,
                   ("sar" in use) and s.get("sar") is not None,
                   ("dem" in use) and s.get("dem") is not None], np.float32)
    out = dict(
        optical=np.asarray(s["optical"], np.float32),
        sar=np.asarray(s["sar"], np.float32) if s.get("sar") is not None else np.zeros((2, H, W), np.float32),
        dem=np.asarray(s["dem"], np.float32) if s.get("dem") is not None else np.zeros((2, H, W), np.float32),
        avail=av,
    )
    if s.get("mask") is not None:
        out["mask"] = np.asarray(s["mask"], np.int64)
    if s.get("target") is not None:
        out["target"] = np.asarray(s["target"], np.float32)
    return out


def _signature(rec: dict) -> str:
    parts = []
    for k in sorted(rec):
        v = rec[k]
        if isinstance(v, str) and os.path.exists(v):
            st = os.stat(v)
            parts.append(f"{k}:{os.path.abspath(v)}:{st.st_size}:{int(st.st_mtime)}")
        else:
            parts.append(f"{k}:{v}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


class DiskChipCache:
    def __init__(self, load_fn, cache_dir: str):
        self.load_fn, self.dir = load_fn, cache_dir

    def path(self, rec) -> str:
        rid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rec.get("id", "chip")))[:80]
        return os.path.join(self.dir, f"{rid}-{_signature(rec)}")

    def __call__(self, rec) -> dict:
        d = self.path(rec)
        done = os.path.join(d, "_complete")
        if not os.path.exists(done):
            s = self.load_fn(rec)
            os.makedirs(d, exist_ok=True)
            for k in KEYS:
                v = s.get(k)
                if v is None:
                    continue
                v = np.asarray(v)
                v = v.astype(np.int16) if k == "mask" else v.astype(np.float16)
                tmp = os.path.join(d, f"{k}.tmp{os.getpid()}.npy")
                np.save(tmp, v)
                os.replace(tmp, os.path.join(d, f"{k}.npy"))
            with open(done, "w") as f:
                f.write(",".join(k for k in KEYS if s.get(k) is not None))
        with open(done) as f:
            present = [k for k in f.read().split(",") if k]
        out = {k: None for k in KEYS}
        out["id"] = rec.get("id")
        for k in present:
            out[k] = np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r")
        return out


class MemoryChipCache:
    def __init__(self, load_fn, max_bytes: int = 1_500_000_000):
        self.load_fn, self.max = load_fn, max_bytes
        self.mem: OrderedDict = OrderedDict()
        self.bytes = 0

    def __call__(self, rec) -> dict:
        key = (rec.get("id"), _signature(rec))
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        s = self.load_fn(rec)
        s = {k: (np.ascontiguousarray(v, np.int16 if k == "mask" else np.float16) if k in KEYS and v is not None else v)
             for k, v in s.items()}
        n = sum(v.nbytes for k, v in s.items() if k in KEYS and v is not None)
        if n <= self.max:
            self.mem[key] = s
            self.bytes += n
            while self.bytes > self.max and self.mem:
                _, old = self.mem.popitem(last=False)
                self.bytes -= sum(v.nbytes for k, v in old.items() if k in KEYS and v is not None)
        return s


class ChipDataset(Dataset):
    """Random ``crop`` px windows (+flips/rot90). The crop is taken BEFORE converting to float32, so with a
    :class:`DiskChipCache` only the window's bytes are read."""

    def __init__(self, records, load_fn, sensors, crop: int = TILE, train: bool = True, seed: int = 0):
        self.records, self.load_fn, self.sensors = list(records), load_fn, list(sensors)
        self.crop, self.train = crop, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        s = self.load_fn(self.records[i])
        if self.train:
            rng = np.random.default_rng(self.rng.integers(1 << 31) + i)
            H, W = s["optical"].shape[1:]
            c = self.crop
            y0 = int(rng.integers(0, max(0, H - c) + 1))
            x0 = int(rng.integers(0, max(0, W - c) + 1))
            s = {k: (v[..., y0:y0 + c, x0:x0 + c] if k in KEYS and v is not None else v) for k, v in s.items()}
        a = sample_to_arrays(s, self.sensors)
        keys = [k for k in ("optical", "sar", "dem", "mask", "target") if k in a]
        if self.train:
            H, W = a["optical"].shape[1:]
            c = self.crop
            if H < c or W < c:  # pad small chips
                ph, pw = max(0, c - H), max(0, c - W)
                for k in keys:
                    pad = [(0, 0)] * (a[k].ndim - 2) + [(0, ph), (0, pw)]
                    a[k] = np.pad(a[k], pad, mode="constant", constant_values=-1 if k == "mask" else 0)
            if rng.random() < 0.5:
                for k in keys:
                    a[k] = a[k][..., :, ::-1]
            if rng.random() < 0.5:
                for k in keys:
                    a[k] = a[k][..., ::-1, :]
            r = int(rng.integers(4))
            if r:
                for k in keys:
                    a[k] = np.rot90(a[k], r, axes=(-2, -1))
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in a.items()}
