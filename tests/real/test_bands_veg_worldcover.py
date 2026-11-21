import numpy as np
import pytest

from terrascope.real import bands, vegetation, worldcover


def test_l1c_indices_and_scaling():
    assert bands.OUR_FROM_L1C == [1, 2, 3, 7, 11, 12]
    assert [bands.S2_L1C_BANDS[i] for i in bands.OUR_FROM_L1C] == ["B2", "B3", "B4", "B8", "B11", "B12"]
    s2 = np.zeros((13, 4, 4), np.float32)
    for i in range(13):
        s2[i] = (i + 1) * 100.0  # DN
    o = bands.l1c_to_ours(s2)
    assert o.shape == (6, 4, 4) and o.dtype == np.float32
    np.testing.assert_allclose(o[:, 0, 0], np.array([2, 3, 4, 8, 12, 13]) * 100 / 10000.0, rtol=1e-6)


def test_reflectance_passthrough_and_errors():
    refl = np.full((6, 3, 3), 0.25, np.float32)
    np.testing.assert_allclose(bands.hls_to_ours(refl), refl)          # already reflectance: unchanged
    np.testing.assert_allclose(bands.hls_to_ours(refl * 10000), refl)  # DN: scaled
    with pytest.raises(ValueError):
        bands.l1c_to_ours(np.zeros((6, 2, 2)))
    assert bands.l2a_to_ours(np.ones((12, 2, 2)) * 5000).shape == (6, 2, 2)


def test_sar_db_conversion():
    lin = np.full((2, 3, 3), 0.1, np.float32)
    np.testing.assert_allclose(bands.sar_to_db(lin), -10.0, atol=1e-5)
    db = np.full((2, 3, 3), -12.0, np.float32)
    np.testing.assert_allclose(bands.sar_to_db(db), db)


def test_vegetation_index_is_documented_and_bounded():
    o = np.zeros((6, 2, 2), np.float32)
    o[2], o[3], o[4] = 0.04, 0.45, 0.2  # healthy canopy
    h = vegetation.health_index(o)
    assert h.shape == (2, 2) and (h > 0.8).all() and (h <= 1).all()
    o[3] = 0.05  # bare / non-vegetated
    assert (vegetation.health_index(o) < 0.2).all()
    d = vegetation.describe()
    assert d["ground_truth"] is False and "NDVI" in d["formula"]


def test_worldcover_mapping_table():
    codes = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100, 0, 255, 7])
    got = worldcover.map_worldcover(codes)
    assert got.tolist() == [1, 4, 4, 2, 3, 5, 5, 0, 4, 1, 4, 255, 255, 255]
    assert set(worldcover.WC_TO_OURS) == set(worldcover.WC_CLASSES)


def test_worldcover_tile_naming():
    assert worldcover.wc_tile_name(49.5, 7.2) == "N48E006"
    assert worldcover.wc_tile_name(-1.0, -58.0) == "S03W060"
    assert worldcover.wc_tile_name(0.5, -0.5) == "N00W003"
    assert worldcover.wc_tile_url("N48E006").endswith("/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E006_Map.tif")
    assert worldcover.tiles_for_bounds(5.5, 47.5, 6.5, 48.5) == ["N45E003", "N45E006", "N48E003", "N48E006"]


def test_worldcover_labels_on_grid(tmp_path, utm_transform):
    from affine import Affine
    from terrascope.real.geotiff import write_geotiff
    from rasterio.warp import transform_bounds
    # a local 'tile' in EPSG:4326 covering the chip: left half tree cover, right half water
    W, S, E, N = transform_bounds("EPSG:32633", "EPSG:4326", 400000, 5300000 - 640, 400640, 5300000)
    res = 0.0001
    cols, rows = int((E - W + 0.02) / res), int((N - S + 0.02) / res)
    T = Affine(res, 0, W - 0.01, 0, -res, N + 0.01)
    wc = np.full((rows, cols), 10, np.uint8)
    wc[:, cols // 2:] = 80
    tile = write_geotiff(str(tmp_path / "tile.tif"), wc, T, crs="EPSG:4326", dtype="uint8", nodata=0)
    lab = worldcover.labels_for_grid("EPSG:32633", utm_transform, 64, 64, tile_opener=lambda t: tile)
    assert lab.shape == (64, 64) and lab.dtype == np.uint8
    assert set(np.unique(lab)) <= {0, 1}
    assert (lab == 1).mean() > 0.3 and (lab == 0).mean() > 0.3
    # missing tile -> all ignore
    lab2 = worldcover.labels_for_grid("EPSG:32633", utm_transform, 16, 16, tile_opener=lambda t: None)
    assert (lab2 == 255).all()
