"""Copernicus GLO-30 DEM: tile naming, windowed read + reprojection onto a chip grid, slope.

Source: public AWS Open Data bucket ``copernicus-dem-30m`` (Cloud-Optimised GeoTIFFs, 1 x 1 degree tiles).
Object layout (ASSUMPTION, could not be verified from the build machine - S3 is blocked there):
    https://copernicus-dem-30m.s3.amazonaws.com/<NAME>/<NAME>.tif
    NAME = Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM
where (lat, lon) is the integer SOUTH-WEST corner of the tile (floor of the coordinates).  The code does not
trust this blindly: after reprojection it checks coverage and logs a loud warning if a tile did not cover the
area it was expected to cover. Tiles over open ocean do not exist (HTTP 404) -> elevation 0 m is used there.

License: Copernicus DEM GLO-30 is distributed free of charge under the Copernicus DEM licence (attribution
"produced using Copernicus WorldDEM-30 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018
provided under COPERNICUS by the European Union and ESA; all rights reserved").
"""
from __future__ import annotations

import logging, math, os

import numpy as np

log = logging.getLogger("terrascope.real.dem")

COP_DEM_BASE = os.environ.get("COP_DEM_BASE", "https://copernicus-dem-30m.s3.amazonaws.com")


def cop_tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(int(lat)):02d}_00_{ew}{abs(int(lon)):03d}_00_DEM"


def cop_tile_url(lat: int, lon: int, base: str = COP_DEM_BASE) -> str:
    n = cop_tile_name(lat, lon)
    return f"{base}/{n}/{n}.tif"


def tiles_for_bounds(west: float, south: float, east: float, north: float) -> list[tuple[int, int]]:
    """Integer SW corners of all 1-degree tiles intersecting the lon/lat box."""
    eps = 1e-9
    lats = range(math.floor(south), math.floor(north - eps) + 1)
    lons = range(math.floor(west), math.floor(east - eps) + 1)
    return [(la, lo) for la in lats for lo in lons]


def slope_deg(elev: np.ndarray, transform, crs) -> np.ndarray:
    """Slope in degrees. Handles geographic CRS (degree pixels -> metres per row)."""
    e = np.asarray(elev, np.float64)
    a, ey = float(transform[0]), float(transform[4])
    if crs is not None and getattr(crs, "is_geographic", False):
        H = e.shape[0]
        lat = float(transform[5]) + ey * (np.arange(H) + 0.5)
        dx = np.abs(a) * 111320.0 * np.cos(np.radians(lat))[:, None]
        dy = abs(ey) * 110574.0
        gy = np.gradient(e, axis=0) / dy
        gx = np.gradient(e, axis=1) / np.maximum(dx, 1e-3)
    else:
        gy, gx = np.gradient(e, abs(ey) or 1.0, abs(a) or 1.0)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


def _tile_source(lat, lon, cache_dir, session, logger):
    """Return a path/URL rasterio can open for the tile, or None if it does not exist (ocean)."""
    name = cop_tile_name(lat, lon)
    if cache_dir:
        from .download import http_download, DownloadError
        dest = os.path.join(cache_dir, name + ".tif")
        missing = dest + ".missing"
        if os.path.exists(missing):
            return None
        try:
            http_download(cop_tile_url(lat, lon), dest, session=session, logger=logger)
            return dest
        except DownloadError as e:
            if "404" in str(e):
                os.makedirs(cache_dir, exist_ok=True)
                open(missing, "w").close()
                logger.info("  DEM tile %s does not exist (ocean?) -> 0 m", name)
                return None
            raise
    return "/vsicurl/" + cop_tile_url(lat, lon)


def dem_for_grid(crs, transform, height: int, width: int, cache_dir: str | None = None, session=None,
                 logger=None, tile_opener=None) -> np.ndarray:
    """Copernicus DEM elevation + slope resampled (bilinear) onto the given grid -> (2,H,W) float32.

    ``tile_opener(lat, lon)`` may be injected (tests): returns a path/dataset-opener or None.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, transform_bounds, Resampling
    from affine import Affine

    lg = logger or log
    dst_crs = CRS.from_user_input(crs)
    T = transform if isinstance(transform, Affine) else Affine(*tuple(transform)[:6])
    b = array_bounds(height, width, T)  # (west, south, east, north)
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *b, densify_pts=21)
    out = np.full((height, width), np.nan, np.float32)
    tiles = tiles_for_bounds(west, south, east, north)
    for lat, lon in tiles:
        src_path = tile_opener(lat, lon) if tile_opener else _tile_source(lat, lon, cache_dir, session, lg)
        if src_path is None:
            continue
        with rasterio.open(src_path) as src:
            tb = src.bounds
            if not (tb.left - 1e-6 <= lon and tb.right + 1e-6 >= lon + 1 and tb.bottom - 1e-6 <= lat and tb.top + 1e-6 >= lat + 1):
                lg.warning("DEM tile %s covers %s, not [%d,%d]x[%d,%d] as the naming assumption expects",
                           cop_tile_name(lat, lon), tuple(tb), lon, lon + 1, lat, lat + 1)
            # read only the window that overlaps the chip (+ 2 px margin) -> cheap over /vsicurl/
            sb = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
            win = src.window(*sb).round_offsets().round_lengths()
            from rasterio.windows import Window
            win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
            win = win.intersection(Window(0, 0, src.width, src.height))
            arr = src.read(1, window=win, masked=True).astype(np.float32).filled(np.nan)
            tmp = np.full((height, width), np.nan, np.float32)
            reproject(arr, tmp, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=T, dst_crs=dst_crs, dst_nodata=np.nan, resampling=Resampling.bilinear)
            m = np.isfinite(tmp) & ~np.isfinite(out)
            out[m] = tmp[m]
    miss = ~np.isfinite(out)
    if miss.any():
        frac = float(miss.mean())
        if frac > 0.5 and tiles:
            lg.warning("DEM: %.0f%% of the chip had no DEM coverage (ocean, or wrong tile-naming assumption)", frac * 100)
        out[miss] = 0.0
    return np.stack([out, slope_deg(out, T, dst_crs)]).astype(np.float32)


def dem_for_geotiff(path: str, cache_dir: str | None = None, session=None, logger=None, tile_opener=None):
    """DEM on the grid of an existing GeoTIFF (e.g. a Sen1Floods11 chip)."""
    import rasterio
    with rasterio.open(path) as ds:
        crs, T, H, W = ds.crs, ds.transform, ds.height, ds.width
    return dem_for_grid(crs, T, H, W, cache_dir, session, logger, tile_opener), crs, T
