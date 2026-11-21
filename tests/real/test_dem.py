import numpy as np
import pytest

from terrascope.real import dem


def test_tile_naming():
    assert dem.cop_tile_name(48, 6) == "Copernicus_DSM_COG_10_N48_00_E006_00_DEM"
    assert dem.cop_tile_name(-3, -60) == "Copernicus_DSM_COG_10_S03_00_W060_00_DEM"
    assert dem.cop_tile_url(48, 6).endswith("/Copernicus_DSM_COG_10_N48_00_E006_00_DEM/Copernicus_DSM_COG_10_N48_00_E006_00_DEM.tif")
    assert dem.tiles_for_bounds(5.5, 47.9, 6.2, 48.1) == [(47, 5), (47, 6), (48, 5), (48, 6)]
    assert dem.tiles_for_bounds(6.0, 48.0, 7.0, 49.0) == [(48, 6)]


def test_slope_projected_and_geographic():
    from affine import Affine
    from rasterio.crs import CRS
    e = np.tile(np.arange(20, dtype=np.float32) * 10.0, (20, 1))  # 10 m rise per 10 m pixel
    s = dem.slope_deg(e, Affine(10, 0, 0, 0, -10, 0), CRS.from_epsg(32633))
    assert np.allclose(s, 45.0, atol=1e-3)
    # geographic: at the equator 1e-4 deg ~ 11.13 m
    s2 = dem.slope_deg(e, Affine(1e-4, 0, 0, 0, -1e-4, 0.001), CRS.from_epsg(4326))
    assert np.degrees(np.arctan(10 / 11.132)) == pytest.approx(float(np.median(s2)), abs=0.2)


def test_dem_for_grid_reprojects_local_tile(tmp_path, utm_transform):
    from affine import Affine
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from terrascope.real.geotiff import write_geotiff
    w, s, e, n = transform_bounds("EPSG:32633", "EPSG:4326", *array_bounds(64, 64, utm_transform))
    lat, lon = int(np.floor(s)), int(np.floor(w))
    res = 1 / 3600.0  # 1 arc-second like GLO-30
    size = 3600
    # elevation = 1000 m * (lat fraction) -> smooth north-south ramp; build a coarse tile and let GDAL resample
    T = Affine(res * 10, 0, lon, 0, -res * 10, lat + 1)
    yy = np.linspace(lat + 1, lat, size // 10, dtype=np.float64)[:, None].repeat(size // 10, 1)
    elev = ((yy - lat) * 1000.0).astype(np.float32)
    path = write_geotiff(str(tmp_path / "tile.tif"), elev, T, crs="EPSG:4326", dtype="float32")
    calls = []

    def opener(la, lo):
        calls.append((la, lo))
        return path if (la, lo) == (lat, lon) else None

    out = dem.dem_for_grid("EPSG:32633", utm_transform, 64, 64, tile_opener=opener)
    assert out.shape == (2, 64, 64) and out.dtype == np.float32
    assert (lat, lon) in calls
    expect_top = (n - lat) * 1000.0
    assert out[0, 0].mean() == pytest.approx(expect_top, abs=2.0)
    assert out[0, 0].mean() > out[0, -1].mean()  # north higher than south
    assert np.isfinite(out).all() and (out[1] >= 0).all()


def test_dem_missing_tile_gives_zero(utm_transform):
    out = dem.dem_for_grid("EPSG:32633", utm_transform, 16, 16, tile_opener=lambda la, lo: None)
    assert (out == 0).all()
