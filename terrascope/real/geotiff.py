"""GeoTIFF / COG reading -> TerraScope scene arrays (+ geo-referencing).

Public API
----------
read_raster(src, max_px=256, mode="downsample")  -> RasterInfo  (array + bounds + gsd + date)
scene_from_geotiffs(single=None, optical=None, sar=None, dem=None, ...) -> scene dict
register_geotiff(data, name=None, store=None, ...) -> (scene_id, scene)       # used by POST /api/upload

Accepted layouts (contract):
* one GeoTIFF with 10 bands  [B2,B3,B4,B8,B11,B12,VV,VH,elevation,slope]
* one GeoTIFF with 6 bands   [B2,B3,B4,B8,B11,B12]                  (optical only)
* one GeoTIFF with 13 bands  Sentinel-2 L1C order B1..B12 incl. B8A  (optical only, reduced to our 6)
* separate files ``optical`` (6 or 13 bands), ``sar`` (2 bands VV,VH), ``dem`` (1 band elevation or 2 bands
  elevation+slope) on the SAME grid (same CRS, transform, size)

Guards (equivalent of the .npz zip-bomb guard): file bytes, declared raster size (bands x width x height)
checked from the header BEFORE any pixel is decoded; reads are decimated with ``out_shape`` so at most
``max_px`` x ``max_px`` pixels per band are materialised; NaN / nodata fraction limit.
"""
from __future__ import annotations

import io, os, re
from dataclasses import dataclass, field

import numpy as np

from .bands import l1c_to_ours, to_reflectance, sar_to_db

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_SRC_SIDE = int(os.environ.get("MAX_GEOTIFF_SIDE", "12000"))      # declared pixels per side
MAX_SRC_VALUES = int(os.environ.get("MAX_GEOTIFF_VALUES", str(13 * 12000 * 12000)))
MAX_BANDS = 16
MIN_PX = 32
MAX_PX = 256
MAX_NODATA_FRAC = 0.5


class GeoTiffError(ValueError):
    """User-facing problem with a GeoTIFF (message is safe to show)."""


@dataclass
class RasterInfo:
    array: np.ndarray                 # (C,H,W) float32, NaN where nodata
    crs: str | None
    bounds: dict | None               # {west,south,east,north,crs:"EPSG:4326"}
    gsd_m: float | None
    acquisition_date: str | None
    src_shape: tuple                  # (C,H,W) declared
    scale: float = 1.0                # downsample factor applied (>=1)
    transform: tuple | None = None    # affine (a,b,c,d,e,f) of the returned grid
    tags: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------- helpers
def _rasterio():
    try:
        import rasterio  # noqa: F401
        import rasterio.io, rasterio.windows, rasterio.warp  # noqa: F401
        return rasterio
    except Exception as e:  # pragma: no cover
        raise GeoTiffError("GeoTIFF support needs the 'rasterio' package (pip install rasterio)") from e


def _open(src):
    """Return a context manager yielding an open rasterio dataset for bytes / path / file-like."""
    rio = _rasterio()
    if isinstance(src, (bytes, bytearray, memoryview)):
        b = bytes(src)
        if len(b) > MAX_UPLOAD_BYTES:
            raise GeoTiffError(f"file too large (max {MAX_UPLOAD_BYTES // 2**20} MB)")
        if not (b[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")):
            raise GeoTiffError("not a TIFF file")
        mf = rio.io.MemoryFile(b)
        return _MemCtx(mf)
    if hasattr(src, "read"):
        return _open(src.read(MAX_UPLOAD_BYTES + 1))
    return rio.open(str(src))


class _MemCtx:
    def __init__(self, mf):
        self.mf = mf

    def __enter__(self):
        try:
            self.ds = self.mf.open()
        except Exception as e:
            self.mf.close()
            raise GeoTiffError("could not decode GeoTIFF") from e
        return self.ds

    def __exit__(self, *a):
        try:
            self.ds.close()
        finally:
            self.mf.close()


def gsd_metres(transform, crs, height: int, width: int) -> float | None:
    """Ground sampling distance in metres (mean of |x| and |y| pixel size)."""
    a, e = abs(transform.a), abs(transform.e)
    if crs is None:
        return None
    if crs.is_geographic:
        # degrees -> metres at the raster centre latitude
        cy = transform.f + transform.e * height / 2.0
        mx = a * 111320.0 * np.cos(np.radians(cy))
        my = e * 110574.0
        return float((mx + my) / 2.0)
    try:
        f = crs.linear_units_factor[1]
    except Exception:
        f = 1.0
    return float((a + e) / 2.0 * f)


def bounds_4326(crs, bounds) -> dict | None:
    if crs is None:
        return None
    from rasterio.warp import transform_bounds
    w, s, e, n = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    return dict(west=round(float(w), 7), south=round(float(s), 7), east=round(float(e), 7),
                north=round(float(n), 7), crs="EPSG:4326")


_DATE_RE = re.compile(r"(\d{4})[-:_/]?(\d{2})[-:_/]?(\d{2})")


def _acq_date(tags: dict) -> str | None:
    for k in ("ACQUISITION_DATE", "acquisition_date", "DATE_ACQUIRED", "SENSING_TIME", "datetime",
              "TIFFTAG_DATETIME", "DATE"):
        v = tags.get(k)
        if v:
            m = _DATE_RE.search(str(v))
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def read_raster(src, max_px: int = MAX_PX, mode: str = "downsample", resampling: str = "average",
                window=None) -> RasterInfo:
    """Read a GeoTIFF with header-first guards. ``mode``: 'downsample' (keep the whole extent) or 'crop'
    (central max_px x max_px window at native resolution)."""
    rio = _rasterio()
    from rasterio.enums import Resampling
    from rasterio.windows import Window
    with _open(src) as ds:
        C, H, W = ds.count, ds.height, ds.width
        if C < 1 or C > MAX_BANDS:
            raise GeoTiffError(f"unsupported band count {C} (1..{MAX_BANDS})")
        if max(H, W) > MAX_SRC_SIDE or C * H * W > MAX_SRC_VALUES:
            raise GeoTiffError(f"raster too large ({C}x{H}x{W}); max {MAX_SRC_SIDE} px per side")
        if min(H, W) < MIN_PX:
            raise GeoTiffError(f"raster too small ({H}x{W}); need at least {MIN_PX} px per side")
        tags = {**ds.tags(), **{k: v for k, v in (ds.tags(ns="IMAGE_STRUCTURE") or {}).items()}}
        if window is None and mode == "crop" and max(H, W) > max_px:
            h, w = min(H, max_px), min(W, max_px)
            window = Window((W - w) // 2, (H - h) // 2, w, h)
        if window is not None:
            H0, W0 = int(window.height), int(window.width)
            transform = ds.window_transform(window)
            bnds = rio.windows.bounds(window, ds.transform)
        else:
            H0, W0 = H, W
            transform = ds.transform
            bnds = ds.bounds
        f = max(1.0, max(H0, W0) / float(max_px))
        oh, ow = max(1, int(round(H0 / f))), max(1, int(round(W0 / f)))
        rs = getattr(Resampling, resampling)
        arr = ds.read(window=window, out_shape=(C, oh, ow), resampling=rs, masked=True)
        data = np.asarray(arr.filled(np.nan) if np.ma.isMaskedArray(arr) else arr, dtype=np.float32)
        if ds.nodata is not None and not np.isnan(ds.nodata):
            data[data == np.float32(ds.nodata)] = np.nan
        from affine import Affine
        t = transform @ Affine.scale(W0 / ow, H0 / oh)
        info = RasterInfo(
            array=data, crs=ds.crs.to_string() if ds.crs else None,
            bounds=bounds_4326(ds.crs, bnds), gsd_m=gsd_metres(t, ds.crs, oh, ow),
            acquisition_date=_acq_date(tags), src_shape=(C, H, W), scale=f, transform=tuple(t)[:6], tags=tags,
        )
    return info


# ---------------------------------------------------------------------------------------- scene assembly
def _fill_nodata(a: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    bad = ~np.isfinite(a)
    frac = float(bad.any(0).mean())
    if frac > MAX_NODATA_FRAC:
        raise GeoTiffError(f"'{name}' is {frac * 100:.0f}% nodata/NaN (max {MAX_NODATA_FRAC * 100:.0f}%)")
    if bad.any():
        a = a.copy()
        for c in range(a.shape[0]):
            b = bad[c]
            if b.any():
                med = np.nanmedian(a[c]) if (~b).any() else 0.0
                a[c][b] = med
    return a.astype(np.float32), frac


def slope_degrees(elev_m: np.ndarray, gsd_m: float) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(elev_m, np.float64), max(float(gsd_m or 30.0), 1e-3))
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _crop4(a, H4, W4):
    return a[..., :H4, :W4]


def scene_from_geotiffs(single=None, optical=None, sar=None, dem=None, max_px: int = MAX_PX,
                        mode: str = "downsample") -> dict:
    """Build a TerraScope scene dict from one stacked GeoTIFF or separate optical/sar/dem GeoTIFFs."""
    if single is None and optical is None and sar is None and dem is None:
        raise GeoTiffError("no GeoTIFF supplied")
    avail = dict(optical=False, sar=False, dem=False)
    parts: dict[str, np.ndarray] = {}
    ref: RasterInfo | None = None
    if single is not None:
        ref = read_raster(single, max_px, mode)
        a, C = ref.array, ref.array.shape[0]
        if C == 10:
            parts["optical"], parts["sar"], parts["dem"] = to_reflectance(a[:6]), sar_to_db(a[6:8]), a[8:10]
        elif C == 6:
            parts["optical"] = to_reflectance(a)
        elif C == 13:
            parts["optical"] = l1c_to_ours(a)
        else:
            raise GeoTiffError(f"single GeoTIFF must have 10 [B2,B3,B4,B8,B11,B12,VV,VH,elev,slope], 6 [B2..B12] "
                               f"or 13 (Sentinel-2 L1C) bands; got {C}")
    else:
        infos = {}
        for k, src in (("optical", optical), ("sar", sar), ("dem", dem)):
            if src is None:
                continue
            infos[k] = read_raster(src, max_px, mode, resampling="average" if k != "dem" else "bilinear")
        ref = infos.get("optical") or infos.get("sar") or infos.get("dem")
        for k, inf in infos.items():
            if inf.src_shape[1:] != ref.src_shape[1:] or inf.array.shape[1:] != ref.array.shape[1:]:
                raise GeoTiffError(f"'{k}' grid {inf.src_shape[1:]} differs from {ref.src_shape[1:]}; "
                                   "optical, sar and dem must share one grid")
            if inf.crs != ref.crs:
                raise GeoTiffError(f"'{k}' CRS {inf.crs} differs from {ref.crs}")
            if inf.transform and ref.transform and not np.allclose(inf.transform, ref.transform, rtol=1e-6, atol=1e-9):
                raise GeoTiffError(f"'{k}' geotransform differs from the other files (not co-registered)")
        if "optical" in infos:
            o = infos["optical"].array
            if o.shape[0] == 6:
                parts["optical"] = to_reflectance(o)
            elif o.shape[0] == 13:
                parts["optical"] = l1c_to_ours(o)
            else:
                raise GeoTiffError(f"optical GeoTIFF must have 6 or 13 bands; got {o.shape[0]}")
        if "sar" in infos:
            s = infos["sar"].array
            if s.shape[0] != 2:
                raise GeoTiffError(f"sar GeoTIFF must have 2 bands [VV,VH]; got {s.shape[0]}")
            parts["sar"] = sar_to_db(s)
        if "dem" in infos:
            d = infos["dem"].array
            if d.shape[0] not in (1, 2):
                raise GeoTiffError(f"dem GeoTIFF must have 1 (elevation) or 2 (elevation, slope) bands; got {d.shape[0]}")
            parts["dem"] = d
    H, W = ref.array.shape[1:]
    H4, W4 = H // 4 * 4, W // 4 * 4
    if min(H4, W4) < MIN_PX:
        raise GeoTiffError(f"scene too small after resampling ({H}x{W}); need >= {MIN_PX} px")
    nodata = {}
    for k in list(parts):
        parts[k], nodata[k] = _fill_nodata(parts[k], k)
    if "dem" in parts and parts["dem"].shape[0] == 1:
        e = parts["dem"][0]
        parts["dem"] = np.stack([e, slope_degrees(e, ref.gsd_m or 30.0)]).astype(np.float32)
    sc: dict = {}
    for k, c in (("optical", 6), ("sar", 2), ("dem", 2)):
        if k in parts:
            sc[k] = np.ascontiguousarray(_crop4(parts[k], H4, W4), dtype=np.float32)
            avail[k] = True
        else:
            # placeholder so legacy code paths that index sc[k] keep working; meta.available says it is absent
            sc[k] = np.zeros((c, H4, W4), np.float32)
    if not any(avail.values()):
        raise GeoTiffError("no usable sensor data found")
    sc["meta"] = dict(
        bounds=ref.bounds, gsd_m=None if ref.gsd_m is None else round(float(ref.gsd_m), 3),
        acquisition_date=ref.acquisition_date, data_source="geotiff", crs=ref.crs,
        available=avail, sensors=[avail["optical"], avail["sar"], avail["dem"]],
        nodata_frac={k: round(v, 4) for k, v in nodata.items()}, src_shape=list(ref.src_shape),
        downsample=round(ref.scale, 4),
    )
    # bounds are for the full read extent; trimming to a multiple of 4 removes < 4 px (noted, not re-computed)
    return sc


def _is_tiff_name(name: str) -> bool:
    return str(name).lower().endswith((".tif", ".tiff"))


def register_geotiff(data, name: str | None = None, store=None, max_px: int = MAX_PX, mode: str = "downsample",
                     source: str = "geotiff"):
    """Parse GeoTIFF upload(s) and register the scene in the RS scene store.

    ``data``: bytes / path of one GeoTIFF, or a dict with any of ``optical``, ``sar``, ``dem`` (bytes or paths).
    Returns ``(scene_id, scene)``. Raises :class:`GeoTiffError` with a user-facing message.
    """
    try:
        if isinstance(data, dict):
            unknown = set(data) - {"optical", "sar", "dem"}
            if unknown:
                raise GeoTiffError(f"unknown field(s) {sorted(unknown)}; use optical, sar, dem")
            sc = scene_from_geotiffs(optical=data.get("optical"), sar=data.get("sar"), dem=data.get("dem"),
                                     max_px=max_px, mode=mode)
        else:
            sc = scene_from_geotiffs(single=data, max_px=max_px, mode=mode)
    except GeoTiffError:
        raise
    except Exception as e:  # rasterio errors etc. -> user-facing
        raise GeoTiffError(f"could not read GeoTIFF ({type(e).__name__})") from e
    sc["meta"]["data_source"] = source
    if store is None:
        from .scenes import STORE as store
    sid = store.add(sc, name=name or "GeoTIFF upload", source=source)
    return sid, sc


def write_geotiff(path, array, transform, crs="EPSG:4326", nodata=None, dtype=None, tags=None):
    """Small helper used by dataset builders and tests: write (C,H,W) array as a tiled, deflate GeoTIFF."""
    rio = _rasterio()
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    dtype = dtype or a.dtype
    prof = dict(driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0], dtype=str(np.dtype(dtype)),
                crs=crs, transform=transform, compress="deflate")
    if a.shape[1] >= 16 and a.shape[2] >= 16 and a.shape[1] % 16 == 0 and a.shape[2] % 16 == 0:
        prof.update(tiled=True, blockxsize=min(256, a.shape[2]), blockysize=min(256, a.shape[1]))
    if nodata is not None:
        prof["nodata"] = nodata
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}.tif"  # atomic: an interrupted write never leaves a valid-looking file
    try:
        with rio.open(tmp, "w", **prof) as ds:
            ds.write(a.astype(dtype))
            if tags:
                ds.update_tags(**tags)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def geotiff_bytes(array, transform, crs="EPSG:4326", nodata=None, dtype=None, tags=None) -> bytes:
    rio = _rasterio()
    a = np.asarray(array)
    if a.ndim == 2:
        a = a[None]
    dtype = dtype or a.dtype
    with rio.io.MemoryFile() as mf:
        with mf.open(driver="GTiff", height=a.shape[1], width=a.shape[2], count=a.shape[0],
                     dtype=str(np.dtype(dtype)), crs=crs, transform=transform, nodata=nodata) as ds:
            ds.write(a.astype(dtype))
            if tags:
                ds.update_tags(**tags)
        return bytes(mf.read())
