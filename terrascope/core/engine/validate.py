"""Validation of user-supplied scenes (.npz uploads and in-memory arrays)."""
from __future__ import annotations

import io, zipfile
import numpy as np

N_CLASSES = 6
MAX_PX = 256
MIN_PX = 32
_EXPECT = {"optical": 6, "sar": 2, "dem": 2}


class SceneError(ValueError):
    """User-facing problem with an uploaded scene."""


def npz_headers(raw):
    """Read (shape, dtype) of every member WITHOUT decompressing data (guards against zip bombs)."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for info in zf.infolist():
            name = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            with zf.open(info) as fh:
                ver = np.lib.format.read_magic(fh)
                rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = rd(fh)
            if dtype.hasobject:
                raise SceneError(f"array '{name}' holds Python objects; only numeric arrays are accepted")
            out[name] = (shape, dtype)
    return out


def arrays_from_npz_bytes(raw):
    """Header-checked arrays from an uploaded .npz (optical, sar, dem + optional lc/flood/veg)."""
    try:
        heads = npz_headers(raw)
    except SceneError:
        raise
    except Exception:
        raise SceneError("not a valid .npz archive")
    for k, c in _EXPECT.items():
        if k not in heads:
            raise SceneError(f"missing array '{k}' (need optical, sar, dem)")
        shp = heads[k][0]
        if len(shp) != 3 or shp[0] != c:
            raise SceneError(f"'{k}' must be shaped ({c}, H, W); got {tuple(shp)}")
        if max(shp[1:]) > MAX_PX or min(shp[1:]) < MIN_PX:
            raise SceneError(f"scene must be between {MIN_PX} and {MAX_PX} px per side; got {shp[1]}x{shp[2]}")
    for k in ("lc", "flood", "veg"):
        if k in heads and len(heads[k][0]) != 2:
            raise SceneError(f"'{k}' must be shaped (H, W)")
    for k, (shp, _) in heads.items():
        if int(np.prod(shp)) > 6 * MAX_PX * MAX_PX:
            raise SceneError(f"array '{k}' is too large")
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    return {k: z[k] for k in z.files if k in ("optical", "sar", "dem", "lc", "flood", "veg")}


def scene_from_arrays(arrays):
    o, s, d = (np.asarray(arrays[k], np.float32) for k in ("optical", "sar", "dem"))
    if o.ndim != 3 or s.ndim != 3 or d.ndim != 3 or o.shape[0] != 6 or s.shape[0] != 2 or d.shape[0] != 2:
        raise SceneError("expected optical(6,H,W) sar(2,H,W) dem(2,H,W)")
    H, W = o.shape[1:]
    if s.shape[1:] != (H, W) or d.shape[1:] != (H, W):
        raise SceneError("all modalities must share H,W")
    H4, W4 = H // 4 * 4, W // 4 * 4
    if min(H4, W4) < MIN_PX or max(H4, W4) > MAX_PX:
        raise SceneError(f"scene must be between {MIN_PX} and {MAX_PX} px")
    for name, a in (("optical", o), ("sar", s), ("dem", d)):
        if not np.isfinite(a).all():
            raise SceneError(f"'{name}' contains NaN or infinite values")
    sc = dict(optical=o[:, :H4, :W4], sar=s[:, :H4, :W4], dem=d[:, :H4, :W4])
    # ground truth is only used when ALL three label layers are present and valid
    if all(k in arrays for k in ("lc", "flood", "veg")):
        lc, fl, vg = (np.asarray(arrays[k]) for k in ("lc", "flood", "veg"))
        if any(a.shape[:2] != (H, W) for a in (lc, fl, vg)):
            raise SceneError("label arrays must match the scene's H,W")
        if not np.isfinite(lc.astype(np.float64)).all() or lc.min() < 0 or lc.max() >= N_CLASSES:
            raise SceneError(f"'lc' must contain class ids 0..{N_CLASSES - 1}")
        if not np.isfinite(vg.astype(np.float64)).all():
            raise SceneError("'veg' contains NaN or infinite values")
        sc["lc"] = lc[:H4, :W4].astype(np.int64)
        sc["flood"] = (fl[:H4, :W4] > 0).astype(np.uint8)
        sc["veg"] = np.clip(vg[:H4, :W4].astype(np.float32), 0, 1)
    return sc
