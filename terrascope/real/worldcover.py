"""ESA WorldCover 10 m v200 (2021) -> TerraScope's 6 land-cover classes, and label building for chips.

Source: AWS Open Data bucket ``esa-worldcover`` (public COGs, 3 x 3 degree tiles). Layout (ASSUMPTION - could not
be fetched from the build machine; matches ESA's documented naming):
    https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{TILE}_Map.tif
    TILE = {N|S}{lat:02d}{E|W}{lon:03d}, (lat, lon) = SOUTH-WEST corner, multiples of 3 (e.g. N48E006, S03W060)
License: CC BY 4.0 ("(c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021)
processed by ESA WorldCover consortium").

WorldCover classes (product user manual)            -> TerraScope class
  10 Tree cover                                     -> 1 Forest
  20 Shrubland                                      -> 4 Grassland   (no shrub class; closest low vegetation)
  30 Grassland                                      -> 4 Grassland
  40 Cropland                                       -> 2 Cropland
  50 Built-up                                       -> 3 Urban
  60 Bare / sparse vegetation                       -> 5 Barren
  70 Snow and ice                                   -> 5 Barren      (non-vegetated; no snow class)
  80 Permanent water bodies                         -> 0 Water
  90 Herbaceous wetland                             -> 4 Grassland   (vegetated; NOT water - documented choice)
  95 Mangroves                                      -> 1 Forest
 100 Moss and lichen                                -> 4 Grassland   (low vegetation)
   0 / anything else (no data)                      -> 255 ignore
Label caveat: WorldCover 2021 is itself a classifier output with imperfect accuracy (see ESA's product
validation report for the figures) and differs in date from the imagery it is paired with, so it is a NOISY reference, not ground truth.
"""
from __future__ import annotations

import logging, math, os

import numpy as np

log = logging.getLogger("terrascope.real.worldcover")

WC_BASE = os.environ.get("WORLDCOVER_BASE", "https://esa-worldcover.s3.eu-central-1.amazonaws.com")
WC_CLASSES = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
              60: "Bare / sparse vegetation", 70: "Snow and ice", 80: "Permanent water bodies",
              90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen"}
WC_TO_OURS = {10: 1, 20: 4, 30: 4, 40: 2, 50: 3, 60: 5, 70: 5, 80: 0, 90: 4, 95: 1, 100: 4}
IGNORE = 255
OUR_CLASSES = ["Water", "Forest", "Cropland", "Urban", "Grassland", "Barren"]

_LUT = np.full(256, IGNORE, np.uint8)
for _k, _v in WC_TO_OURS.items():
    _LUT[_k] = _v


def map_worldcover(wc) -> np.ndarray:
    """WorldCover class codes (any int array) -> our class ids (uint8, 255 = ignore)."""
    a = np.asarray(wc)
    a = np.where((a >= 0) & (a <= 255), a, 0).astype(np.int64)
    return _LUT[a]


def wc_tile_name(lat: float, lon: float) -> str:
    """Tile containing (lat, lon): SW corner snapped down to a multiple of 3 degrees."""
    la = int(math.floor(lat / 3.0) * 3)
    lo = int(math.floor(lon / 3.0) * 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def wc_tile_url(tile: str, base: str = WC_BASE) -> str:
    return f"{base}/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def tiles_for_bounds(west, south, east, north) -> list[str]:
    eps = 1e-9
    out = []
    la = math.floor(south / 3.0) * 3
    while la <= north - eps:
        lo = math.floor(west / 3.0) * 3
        while lo <= east - eps:
            out.append(wc_tile_name(la + 0.5, lo + 0.5))
            lo += 3
        la += 3
    return out


def labels_for_grid(crs, transform, height, width, tile_opener=None, logger=None) -> np.ndarray:
    """WorldCover mapped to our 6 classes on the chip grid (nearest-neighbour / mode). (H,W) uint8, 255=ignore.

    ``tile_opener(tile) -> path | None`` lets tests inject local files; default opens the COG via /vsicurl/
    (only the needed window is fetched).
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import array_bounds
    from rasterio.warp import reproject, transform_bounds, Resampling
    from rasterio.windows import Window
    from affine import Affine

    lg = logger or log
    dst_crs = CRS.from_user_input(crs)
    T = transform if isinstance(transform, Affine) else Affine(*tuple(transform)[:6])
    west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *array_bounds(height, width, T), densify_pts=21)
    out = np.zeros((height, width), np.uint8)  # 0 = WorldCover "no data"
    for tile in tiles_for_bounds(west, south, east, north):
        try:
            src_path = tile_opener(tile) if tile_opener else "/vsicurl/" + wc_tile_url(tile)
            if src_path is None:
                continue
            with rasterio.open(src_path) as src:
                sb = transform_bounds("EPSG:4326", src.crs, west, south, east, north, densify_pts=21)
                win = src.window(*sb).round_offsets().round_lengths()
                win = Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
                win = win.intersection(Window(0, 0, src.width, src.height))
                arr = src.read(1, window=win)
                tmp = np.zeros((height, width), np.uint8)
                # WorldCover (10 m) onto ~10 m chips: 'mode' avoids inventing class codes by interpolation
                reproject(arr, tmp, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=0,
                          dst_transform=T, dst_crs=dst_crs, dst_nodata=0, resampling=Resampling.mode)
                m = (tmp > 0) & (out == 0)
                out[m] = tmp[m]
        except Exception as e:
            if not tile_missing_error(e):
                raise  # network / SSL / auth problems must not silently become all-ignore labels
            lg.warning("WorldCover tile %s does not exist (open ocean?) -> pixels ignored", tile)
    return map_worldcover(out)


def tile_missing_error(e: BaseException) -> bool:
    """True only for 'this object does not exist' (HTTP 404 / file not found), not for network failures."""
    if isinstance(e, FileNotFoundError):
        return True
    msg = str(e).lower()
    return ("404" in msg or "no such file" in msg or "does not exist" in msg) and "ssl" not in msg
