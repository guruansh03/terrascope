import numpy as np
import pytest

from rs_helpers import fake_s1_db, fake_s2_l1c
from terrascope.real import geotiff as gt
from terrascope.real.scenes import SceneStore


@pytest.fixture
def store(tmp_path):
    return SceneStore(scenes_dir=str(tmp_path / "scenes"), demo_dir=str(tmp_path / "demo"))


def _stack10(rng, H, W):
    opt = (0.05 + 0.3 * rng.random((6, H, W))).astype(np.float32)
    sar = fake_s1_db(rng, H, W)
    elev = (100 + np.linspace(0, 50, W)[None].repeat(H, 0)).astype(np.float32)
    slope = np.full((H, W), 4.0, np.float32)
    return np.concatenate([opt, sar, elev[None], slope[None]])


def test_bounds_and_gsd_utm(rng, utm_transform):
    from rasterio.warp import transform_bounds
    b = gt.geotiff_bytes(_stack10(rng, 64, 96), utm_transform, crs="EPSG:32633", dtype="float32",
                         tags={"ACQUISITION_DATE": "2021-07-14T10:20:00"})
    info = gt.read_raster(b)
    assert info.array.shape == (10, 64, 96)
    assert info.gsd_m == pytest.approx(10.0)
    w, s, e, n = transform_bounds("EPSG:32633", "EPSG:4326", 400000, 5300000 - 640, 400960, 5300000, densify_pts=21)
    assert info.bounds["crs"] == "EPSG:4326"
    assert info.bounds["west"] == pytest.approx(w, abs=1e-6) and info.bounds["north"] == pytest.approx(n, abs=1e-6)
    assert 13 < info.bounds["west"] < 14 and 47 < info.bounds["south"] < 48.5
    assert info.acquisition_date == "2021-07-14"


def test_gsd_geographic_and_downsample(rng):
    from affine import Affine
    T = Affine(0.0001, 0, 10.0, 0, -0.0001, 45.0)  # ~7.9 m x 11.1 m at 45 N
    b = gt.geotiff_bytes(_stack10(rng, 512, 512), T, crs="EPSG:4326", dtype="float32")
    info = gt.read_raster(b, max_px=256)
    assert info.array.shape == (10, 256, 256) and info.scale == pytest.approx(2.0)
    expect = (0.0001 * 111320 * np.cos(np.radians(45 - 0.0256)) + 0.0001 * 110574) / 2 * 2
    assert info.gsd_m == pytest.approx(expect, rel=1e-3)
    crop = gt.read_raster(b, max_px=128, mode="crop")
    assert crop.array.shape == (10, 128, 128) and crop.scale == 1.0


def test_register_10band(rng, utm_transform, store):
    b = gt.geotiff_bytes(_stack10(rng, 100, 130), utm_transform, crs="EPSG:32633", dtype="float32")
    sid, sc = gt.register_geotiff(b, name="t", store=store)
    assert sc["optical"].shape == (6, 100, 128) and sc["sar"].shape == (2, 100, 128)  # W trimmed to /4
    assert sc["meta"]["available"] == dict(optical=True, sar=True, dem=True)
    assert sc["meta"]["data_source"] == "geotiff" and sc["meta"]["gsd_m"] == pytest.approx(10.0)
    assert store.get(sid)["optical"].shape == (6, 100, 128)
    summ = store.summary(sid)
    assert summ["bands"][:6] == ["B2", "B3", "B4", "B8", "B11", "B12"] and "VV" in summ["bands"]
    assert summ["bounds"]["crs"] == "EPSG:4326" and summ["source"] == "geotiff"


def test_register_optical_only_dn_and_13band(rng, utm_transform, store):
    dn = (rng.random((6, 64, 64)) * 3000).astype(np.uint16)
    _, sc = gt.register_geotiff(gt.geotiff_bytes(dn, utm_transform, crs="EPSG:32633"), store=store)
    assert sc["meta"]["available"] == dict(optical=True, sar=False, dem=False)
    assert sc["meta"]["sensors"] == [True, False, False]
    assert sc["optical"].max() <= 0.31 and not sc["sar"].any()
    s2 = fake_s2_l1c(rng, 64, 64)
    _, sc13 = gt.register_geotiff(gt.geotiff_bytes(s2, utm_transform, crs="EPSG:32633"), store=store)
    np.testing.assert_allclose(sc13["optical"][3], s2[7].astype(np.float32) / 10000, rtol=1e-5)


def test_register_separate_files_and_slope(rng, utm_transform, store):
    opt = gt.geotiff_bytes((rng.random((6, 64, 64)) * 0.3).astype(np.float32), utm_transform, crs="EPSG:32633")
    sar = gt.geotiff_bytes(fake_s1_db(rng, 64, 64), utm_transform, crs="EPSG:32633")
    elev = np.tile(np.arange(64, dtype=np.float32) * 10.0, (64, 1))  # +10 m per 10 m pixel -> 45 deg
    dem = gt.geotiff_bytes(elev, utm_transform, crs="EPSG:32633")
    _, sc = gt.register_geotiff(dict(optical=opt, sar=sar, dem=dem), store=store)
    assert sc["dem"].shape == (2, 64, 64)
    assert np.median(sc["dem"][1]) == pytest.approx(45.0, abs=0.5)
    assert all(sc["meta"]["available"].values())


def test_guards(rng, utm_transform, store, monkeypatch):
    from affine import Affine
    with pytest.raises(gt.GeoTiffError, match="not a TIFF"):
        gt.register_geotiff(b"PK\x03\x04garbage", store=store)
    with pytest.raises(gt.GeoTiffError, match="band"):
        gt.register_geotiff(gt.geotiff_bytes(np.zeros((4, 64, 64), np.float32), utm_transform, crs="EPSG:32633"),
                            store=store)
    with pytest.raises(gt.GeoTiffError, match="too small"):
        gt.register_geotiff(gt.geotiff_bytes(np.zeros((6, 16, 16), np.float32), utm_transform, crs="EPSG:32633"),
                            store=store)
    nanimg = np.full((6, 64, 64), np.nan, np.float32)
    nanimg[:, :10] = 0.1
    with pytest.raises(gt.GeoTiffError, match="nodata"):
        gt.register_geotiff(gt.geotiff_bytes(nanimg, utm_transform, crs="EPSG:32633"), store=store)
    # a few NaNs are filled, not rejected
    some = (rng.random((6, 64, 64)) * 0.2).astype(np.float32)
    some[:, :3, :3] = np.nan
    _, sc = gt.register_geotiff(gt.geotiff_bytes(some, utm_transform, crs="EPSG:32633"), store=store)
    assert np.isfinite(sc["optical"]).all() and sc["meta"]["nodata_frac"]["optical"] > 0
    # declared size guard (zip-bomb equivalent): checked from the header before decoding pixels
    monkeypatch.setattr(gt, "MAX_SRC_SIDE", 100)
    big = gt.geotiff_bytes(np.zeros((6, 128, 128), np.uint8), utm_transform, crs="EPSG:32633")
    with pytest.raises(gt.GeoTiffError, match="too large"):
        gt.register_geotiff(big, store=store)
    monkeypatch.setattr(gt, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(gt.GeoTiffError, match="too large"):
        gt.register_geotiff(big, store=store)


def test_mismatched_grids_rejected(rng, utm_transform, store):
    from affine import Affine
    opt = gt.geotiff_bytes((rng.random((6, 64, 64)) * 0.3).astype(np.float32), utm_transform, crs="EPSG:32633")
    shifted = Affine(10.0, 0, 400500.0, 0, -10.0, 5300000.0)
    sar = gt.geotiff_bytes(fake_s1_db(rng, 64, 64), shifted, crs="EPSG:32633")
    with pytest.raises(gt.GeoTiffError, match="co-registered"):
        gt.register_geotiff(dict(optical=opt, sar=sar), store=store)
    sar2 = gt.geotiff_bytes(fake_s1_db(rng, 64, 64), utm_transform, crs="EPSG:32632")
    with pytest.raises(gt.GeoTiffError, match="CRS"):
        gt.register_geotiff(dict(optical=opt, sar=sar2), store=store)
    with pytest.raises(gt.GeoTiffError, match="unknown field"):
        gt.register_geotiff(dict(optical=opt, lidar=opt), store=store)
