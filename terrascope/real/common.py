"""Shared bits for the real-data dataset modules (paths, file indexing, split helpers, sample format).

Standard sample (numpy) returned by every dataset ``load(record)``:
    optical (6,H,W) float32 reflectance [B2,B3,B4,B8,B11,B12]  (cloudy input for SEN12MS-CR)
    sar     (2,H,W) float32 dB [VV,VH]                          or None
    dem     (2,H,W) float32 [elevation m, slope deg]           or None
    mask    (H,W)   int64 class ids, -1 = ignore                 (flood / burn / landcover)
    target  (6,H,W) float32 reflectance                         (cloud reconstruction only)
    id      str
"""
from __future__ import annotations

import hashlib, os, re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def data_root() -> str:
    return os.environ.get("TERRASCOPE_DATA", os.path.join(ROOT, "data", "real"))


def index_files(root: str, suffixes: tuple[str, ...]) -> dict[str, str]:
    """Walk ``root`` and return {basename: path} for files ending with any suffix (case-insensitive)."""
    out = {}
    if not os.path.isdir(root):
        return out
    suf = tuple(s.lower() for s in suffixes)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(suf) and not fn.endswith(".part"):
                out.setdefault(fn, os.path.join(dp, fn))
    return out


def stable_fraction(key: str) -> float:
    """Deterministic pseudo-random number in [0,1) from a string (for reproducible splits)."""
    h = hashlib.sha1(key.encode()).hexdigest()
    return int(h[:8], 16) / 2**32


def read_tif(path: str, window=None) -> np.ndarray:
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(window=window).astype(np.float32)
        nd = ds.nodata
    if nd is not None and not np.isnan(nd):
        a[a == np.float32(nd)] = np.nan
    return a


def fill_nan(a: np.ndarray, value=None) -> np.ndarray:
    a = np.asarray(a, np.float32)
    bad = ~np.isfinite(a)
    if not bad.any():
        return a
    a = a.copy()
    for c in range(a.shape[0]):
        b = bad[c]
        if b.any():
            if value is not None:
                a[c][b] = value
            else:
                a[c][b] = np.nanmedian(a[c]) if (~b).any() else 0.0
    return a


CHIP_RE = re.compile(r"^(.+?_\d+)(?:_[A-Za-z0-9]+)?(?:\.tif)?$")


def chip_id(token: str) -> str | None:
    """'Bolivia_103757_S1Hand.tif' / 'Bolivia_103757' -> 'Bolivia_103757'."""
    t = os.path.basename(token.strip().strip('"').strip("'"))
    if not t:
        return None
    m = CHIP_RE.match(t)
    return m.group(1) if m else None


def read_split_ids(path: str) -> list[str]:
    """Parse a split file (.txt one id/filename per line, or .csv with filenames in columns)."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = chip_id(line.split(",")[0])  # header rows have no '<name>_<digits>' token -> None
            if c and c not in ids:
                ids.append(c)
    return ids
