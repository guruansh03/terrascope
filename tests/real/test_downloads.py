"""Download code paths with a fake HTTP session / fake Hugging Face API (no network)."""
import base64, hashlib, io, json, os, tarfile, urllib.parse

import numpy as np
import pytest

from rs_helpers import fake_s1_db, fake_s2_l1c
from terrascope.real import download as dl
from terrascope.real.geotiff import geotiff_bytes


# ------------------------------------------------------------------------------------------ fake HTTP
class FakeResp:
    def __init__(self, status, body=b"", js=None, headers=None):
        self.status_code, self._body, self._js = status, body, js
        self.headers = headers or {"Content-Length": str(len(body))}
        self.raw = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def json(self):
        return self._js


class FakeBucket:
    """Serves objects of a fake public GCS bucket via the JSON API + media URLs, honouring Range."""

    def __init__(self, objects: dict[str, bytes], page=3):
        self.objects, self.page, self.calls = objects, page, []

    def get(self, url, params=None, headers=None, stream=False, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if "/storage/v1/b/" in url and url.endswith("/o"):
            names = sorted(n for n in self.objects if n.startswith(params["prefix"]))
            start = int(params.get("pageToken", 0))
            chunk = names[start:start + self.page]
            js = {"items": [dict(name=n, size=str(len(self.objects[n])),
                                 md5Hash=base64.b64encode(hashlib.md5(self.objects[n]).digest()).decode()) for n in chunk]}
            if start + self.page < len(names):
                js["nextPageToken"] = str(start + self.page)
            return FakeResp(200, js=js)
        if "/download/storage/v1/b/" in url:
            name = urllib.parse.unquote(url.split("/o/")[1].split("?")[0])
            if name not in self.objects:
                return FakeResp(404)
            body = self.objects[name]
            rng = (headers or {}).get("Range")
            if rng:
                a = int(rng.split("=")[1].split("-")[0])
                return FakeResp(206, body[a:])
            return FakeResp(200, body)
        return FakeResp(404)


def test_http_download_resume_and_md5(tmp_path):
    body = os.urandom(5000)
    b = FakeBucket({"x/file.bin": body})
    url = dl.gcs_media_url("bkt", "x/file.bin")
    dest = str(tmp_path / "file.bin")
    with open(dest + ".part", "wb") as f:
        f.write(body[:1234])
    md5 = base64.b64encode(hashlib.md5(body).digest()).decode()
    dl.http_download(url, dest, size=len(body), md5=md5, session=b)
    assert open(dest, "rb").read() == body
    assert b.calls[-1][2]["Range"] == "bytes=1234-"
    n = len(b.calls)
    dl.http_download(url, dest, size=len(body), md5=md5, session=b)  # already complete -> no request
    assert len(b.calls) == n
    with pytest.raises(dl.DownloadError):
        dl.http_download(url, str(tmp_path / "bad.bin"), size=len(body), md5="AAAA", session=b, retries=1)
    with pytest.raises(dl.DownloadError, match="404"):
        dl.http_download(dl.gcs_media_url("bkt", "nope"), str(tmp_path / "n.bin"), session=b, retries=1)


def test_disk_check(tmp_path):
    dl.check_disk(str(tmp_path), 1)
    with pytest.raises(dl.DownloadError, match="disk"):
        dl.check_disk(str(tmp_path), 10 ** 18)


def test_safe_extract_refuses_traversal(tmp_path):
    p = tmp_path / "evil.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        data = b"x"
        ti = tarfile.TarInfo("../../escape.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    with pytest.raises(dl.DownloadError, match="outside"):
        dl.safe_extract(str(p), str(tmp_path / "out"))
    assert not (tmp_path.parent / "escape.txt").exists()


# ------------------------------------------------------------------------------------------ Sen1Floods11
def _s1f11_bucket(rng, utm_transform):
    chips = {"Bolivia_101": "train", "Ghana_202": "train", "Sri-Lanka_303": "valid", "USA_404": "test"}
    base = "v1.1/data/flood_events/HandLabeled/"
    objs = {}
    for c in chips:
        objs[f"{base}S2Hand/{c}_S2Hand.tif"] = geotiff_bytes(fake_s2_l1c(rng, 32, 32), utm_transform, crs="EPSG:32633")
        s1 = fake_s1_db(rng, 32, 32)
        s1[0, 0, 0] = np.nan
        objs[f"{base}S1Hand/{c}_S1Hand.tif"] = geotiff_bytes(s1, utm_transform, crs="EPSG:32633")
        lab = (rng.random((32, 32)) > 0.7).astype(np.int16)
        lab[:2] = -1
        objs[f"{base}LabelHand/{c}_LabelHand.tif"] = geotiff_bytes(lab, utm_transform, crs="EPSG:32633")
    for sp in ("train", "valid", "test"):
        rows = "".join(f"{c}_S1Hand.tif,{c}_LabelHand.tif\n" for c, s in chips.items() if s == sp)
        objs[f"v1.1/splits/flood_handlabeled/flood_{sp}_data.csv"] = rows.encode()
    objs["v1.1/Sen1Floods11_Metadata.geojson"] = b'{"type":"FeatureCollection","features":[]}'
    return objs


def test_sen1floods11_download_list_load(tmp_path, rng, utm_transform, monkeypatch):
    from terrascope.real import sen1floods11 as s
    objs = _s1f11_bucket(rng, utm_transform)
    bucket = FakeBucket(objs)
    root = str(tmp_path / "s1f11")
    summ = s.download(root, session=bucket)
    assert summ["source"] == "gs://sen1floods11"
    assert summ["chips"] == 4 and summ["split_sizes"] == {"train": 2, "val": 1, "test": 1}
    assert any("pageToken" in c[1] for c in bucket.calls)  # pagination exercised
    recs = s.list_split(root, "val")
    assert [r["id"] for r in recs] == ["Sri-Lanka_303"]
    smp = s.load(recs[0])
    assert smp["optical"].shape == (6, 32, 32) and smp["optical"].max() < 1.0
    assert smp["sar"].shape == (2, 32, 32) and np.isfinite(smp["sar"]).all()
    assert smp["dem"] is None
    assert smp["mask"].dtype == np.int64 and (smp["mask"][:2] == -1).all() and set(np.unique(smp["mask"])) <= {-1, 0, 1}
    # re-run: everything verified -> no media downloads
    n_media = sum("/download/" in c[0] for c in bucket.calls)
    s.download(root, session=bucket)
    assert sum("/download/" in c[0] for c in bucket.calls) == n_media


def test_sen1floods11_hf_fallback(tmp_path, rng, utm_transform, monkeypatch):
    from terrascope.real import sen1floods11 as s
    objs = _s1f11_bucket(rng, utm_transform)

    class Broken:
        def get(self, *a, **k):
            return FakeResp(403)

    class FakeApi:
        def list_datasets(self, search, limit):
            class D:
                id = "someone/sen1floods11-mirror"
            return [D()]

        def list_repo_files(self, repo, repo_type):
            return ["README.md"] + [k.replace("v1.1/", "") for k in objs]

    def fake_hf_download(repo, f, local_dir, **kw):
        p = os.path.join(local_dir, f)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(objs["v1.1/" + f])
        return p

    monkeypatch.setattr(dl, "hf_api", lambda: FakeApi())
    monkeypatch.setattr(dl, "hf_download", fake_hf_download)
    root = str(tmp_path / "s1f11")
    summ = s.download(root, session=Broken())
    assert summ["source"] == "hf:someone/sen1floods11-mirror" and summ["chips"] == 4


def test_build_dem_and_worldcover_for_chips(tmp_path, rng, utm_transform):
    from terrascope.real import sen1floods11 as s
    from terrascope.real.datasets import DATASETS
    from terrascope.real.geotiff import write_geotiff
    from affine import Affine
    root = str(tmp_path / "s1f11")
    s.download(root, session=FakeBucket(_s1f11_bucket(rng, utm_transform)))
    # fake global DEM / WorldCover tiles in EPSG:4326 covering the chips
    T = Affine(0.001, 0, 13.0, 0, -0.001, 48.5)
    dem_t = write_geotiff(str(tmp_path / "dem.tif"), np.full((1000, 2000), 321.0, np.float32), T, dtype="float32")
    wc = np.full((1000, 2000), 40, np.uint8)
    wc_t = write_geotiff(str(tmp_path / "wc.tif"), wc, T, dtype="uint8", nodata=0)
    assert s.build_dem(root, tile_opener=lambda la, lo: dem_t) == 4
    assert s.build_worldcover(root, tile_opener=lambda t: wc_t) == 4
    assert s.build_dem(root, tile_opener=lambda la, lo: dem_t) == 0  # resumable: nothing new
    rec = s.list_split(root, "train")[0]
    smp = s.load(rec)
    assert smp["dem"].shape == (2, 32, 32) and np.allclose(smp["dem"][0], 321.0, atol=0.5)
    lc = DATASETS["worldcover"]
    recs = lc.list_split(root, "train")
    m = lc.load(recs[0])["mask"]
    assert (m == 2).all()  # WorldCover cropland (40) -> our Cropland (2)


# ------------------------------------------------------------------------------------------ burn scars
def test_burnscars_download_tar_and_splits(tmp_path, rng, utm_transform, monkeypatch):
    from terrascope.real import burnscars as b
    tarp = tmp_path / "hls_burn_scars.tar.gz"
    with tarfile.open(tarp, "w:gz") as tf:
        for grp, n in (("training", 12), ("validation", 3)):
            for i in range(n):
                stem = f"subsetted_512x512_HLS.S30.T10SEH.2018{i:03d}.v1.4"
                img = (rng.random((6, 32, 32)) * 0.3).astype(np.float32)
                m = (rng.random((32, 32)) > 0.8).astype(np.int16)
                for name, data in ((f"{grp}/{stem}_merged.tif", geotiff_bytes(img, utm_transform, crs="EPSG:32610")),
                                   (f"{grp}/{stem}.mask.tif", geotiff_bytes(m, utm_transform, crs="EPSG:32610"))):
                    ti = tarfile.TarInfo(f"hls_burn_scars/{name}")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))

    class FakeApi:
        def list_repo_files(self, repo, repo_type):
            assert repo == "ibm-nasa-geospatial/hls_burn_scars"
            return [".gitattributes", "README.md", "hls_burn_scars.tar.gz"]

        def list_repo_tree(self, repo, repo_type, recursive):
            class It:
                path, size = "hls_burn_scars.tar.gz", os.path.getsize(tarp)
            return [It()]

    monkeypatch.setattr(dl, "hf_download", lambda repo, f, local_dir, **kw: str(tarp))
    root = str(tmp_path / "burn")
    res = b.download(root, api=FakeApi())
    sz = res["split_sizes"]
    assert sz["test"] == 3 and sz["train"] + sz["val"] == 12 and sz["train"] > sz["val"]
    smp = b.load(b.list_split(root, "test")[0])
    assert smp["optical"].shape == (6, 32, 32) and smp["sar"] is None and smp["mask"].dtype == np.int64


# ------------------------------------------------------------------------------------------ SEN12MS-CR
def test_sen12mscr_subset_from_local_archives(tmp_path, rng, utm_transform):
    from terrascope.real import sen12mscr as s
    root = tmp_path / "sen12"
    raw = root / "raw"
    raw.mkdir(parents=True)
    roi = "ROIs1158_spring"
    for kind in ("s1", "s2", "s2_cloudy"):
        with tarfile.open(raw / f"{roi}_{kind}.tar.gz", "w:gz") as tf:
            for scene in (1, 2, 3):
                for p in range(5):
                    if kind == "s1":
                        arr = fake_s1_db(rng, 32, 32)
                    else:
                        arr = fake_s2_l1c(rng, 32, 32)
                        if kind == "s2_cloudy":
                            arr[:, :10] = 8000
                    data = geotiff_bytes(arr, utm_transform, crs="EPSG:32633")
                    ti = tarfile.TarInfo(f"{roi}/{kind}_{scene}/{roi}_{kind}_{scene}_p{p}.tif")
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
    res = s.download(str(root), max_patches=8, per_scene=3)
    tot = sum(res["split_sizes"].values())
    assert tot == 8  # 3 + 3 + 2 patches, all with s1/s2/s2_cloudy
    allrecs = [r for sp in ("train", "val", "test") for r in s.list_split(str(root), sp)]
    smp = s.load(allrecs[0])
    assert smp["optical"].shape == smp["target"].shape == (6, 32, 32)
    assert smp["optical"][:, :10].mean() > smp["target"][:, :10].mean()  # cloudy input brighter
    assert smp["sar"].shape == (2, 32, 32)
    # re-run is a no-op (already complete)
    assert sum(s.download(str(root), max_patches=8, per_scene=3)["split_sizes"].values()) == 8


def test_split_id_parsing(tmp_path):
    from terrascope.real.common import chip_id, read_split_ids
    assert chip_id("Bolivia_103757_S1Hand.tif") == "Bolivia_103757"
    assert chip_id("Sri-Lanka_12_LabelHand.tif") == "Sri-Lanka_12"
    assert chip_id("Mekong_333") == "Mekong_333"
    p = tmp_path / "flood_test_data.txt"
    p.write_text("Ghana_1\nUSA_22\n\n")
    assert read_split_ids(str(p)) == ["Ghana_1", "USA_22"]
    p2 = tmp_path / "flood_train_data.csv"
    p2.write_text("S1,Label\nGhana_1_S1Hand.tif,Ghana_1_LabelHand.tif\n")
    assert read_split_ids(str(p2)) == ["Ghana_1"]


# ------------------------------------------------------------------------------------------ robustness fixes
def test_http_download_416_on_complete_part_without_size(tmp_path):
    body = os.urandom(3000)

    class S416:
        def __init__(self):
            self.calls = []

        def get(self, url, headers=None, **k):
            self.calls.append(dict(headers or {}))
            if headers and "Range" in headers:
                return FakeResp(416, b"", headers={})
            return FakeResp(200, body)

    dest = str(tmp_path / "f.bin")
    with open(dest + ".part", "wb") as f:
        f.write(body)  # complete download whose final rename was interrupted
    s = S416()
    dl.http_download("https://x/f.bin", dest, size=None, session=s, retries=1)
    assert open(dest, "rb").read() == body and not os.path.exists(dest + ".part")


def test_sen1floods11_media_errors_do_not_switch_to_hf(tmp_path, rng, utm_transform, monkeypatch):
    from terrascope.real import sen1floods11 as s
    objs = _s1f11_bucket(rng, utm_transform)

    class MediaBroken(FakeBucket):
        def get(self, url, **k):
            if "/download/" in url:
                return FakeResp(500)
            return super().get(url, **k)

    monkeypatch.setattr(s, "_download_hf", lambda *a, **k: pytest.fail("must not fall back to HF"))
    with pytest.raises(dl.DownloadError, match="HTTP 500"):
        s.download(str(tmp_path / "r"), session=MediaBroken(objs))


def test_hf_download_retries_on_503(tmp_path, monkeypatch):
    import huggingface_hub

    class E(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.response = type("R", (), {"status_code": code})()

    n = {"k": 0}

    def flaky(**kw):
        n["k"] += 1
        if n["k"] < 3:
            raise E(503)
        return "ok-path"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", flaky)
    assert dl.hf_download("r", "f", str(tmp_path)) == "ok-path" and n["k"] == 3
    n["k"] = -100

    def notfound(**kw):
        n["k"] += 1
        raise E(404)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", notfound)
    with pytest.raises(E):
        dl.hf_download("r", "f", str(tmp_path))
    assert n["k"] == -99  # 404: no retry


def test_worldcover_network_errors_are_not_silent(tmp_path, rng, utm_transform):
    from terrascope.real import sen1floods11 as s
    from terrascope.real.worldcover import labels_for_grid
    with pytest.raises(RuntimeError, match="SSL"):
        labels_for_grid("EPSG:32633", utm_transform, 16, 16,
                        tile_opener=lambda t: (_ for _ in ()).throw(RuntimeError("CURL error: SSL connect error")))

    def missing(t):
        raise FileNotFoundError("HTTP response code: 404")

    assert (labels_for_grid("EPSG:32633", utm_transform, 16, 16, tile_opener=missing) == 255).all()
    root = str(tmp_path / "s1f11")
    s.download(root, session=FakeBucket(_s1f11_bucket(rng, utm_transform)))

    def boom(t):
        raise RuntimeError("CURL error: Could not resolve host")

    with pytest.raises(RuntimeError, match="4 chips"):
        s.build_worldcover(root, tile_opener=boom)
    assert not os.path.isdir(os.path.join(root, "derived", "WorldCover")) or \
        not os.listdir(os.path.join(root, "derived", "WorldCover"))  # nothing written -> retried next run
    assert s.build_worldcover(root, tile_opener=missing) == 4


def test_write_geotiff_is_atomic(tmp_path, utm_transform):
    from terrascope.real.geotiff import write_geotiff
    p = str(tmp_path / "a.tif")
    write_geotiff(p, np.zeros((1, 32, 32), np.float32), utm_transform, crs="EPSG:32633")
    assert os.listdir(tmp_path) == ["a.tif"]
    with pytest.raises(Exception):
        write_geotiff(str(tmp_path / "b.tif"), np.zeros((1, 32, 32), np.float32), utm_transform, crs="NOT-A-CRS")
    assert os.listdir(tmp_path) == ["a.tif"]  # no partial b.tif, no temp file left behind
