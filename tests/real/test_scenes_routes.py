import json

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from terrascope.real import scenes as scenes_mod
from terrascope.real.scenes import SceneStore, SceneReadOnly


def _sc(H=32, W=32, truth=False):
    sc = dict(optical=np.full((6, H, W), 0.1, np.float32), sar=np.full((2, H, W), -12, np.float32),
              dem=np.zeros((2, H, W), np.float32), meta=dict(data_source="upload", gsd_m=30.0))
    if truth:
        sc["flood"] = np.zeros((H, W), np.uint8)
    return sc


@pytest.fixture
def store(tmp_path):
    demo = tmp_path / "demo"
    demo.mkdir()
    np.savez_compressed(demo / "valencia_flood.npz", optical=np.zeros((6, 64, 64), np.float32),
                        sar=np.zeros((2, 64, 64), np.float32), dem=np.zeros((2, 64, 64), np.float32),
                        flood=np.zeros((64, 64), np.uint8))
    (demo / "valencia_flood.json").write_text(json.dumps(dict(
        name="Valencia flood (demo)", acquisition_date="2024-10-30", data_source="dataset:sen1floods11",
        bounds=dict(west=-0.5, south=39.3, east=-0.3, north=39.5, crs="EPSG:4326"), tasks=["flood"])))
    (demo / "../bad name.npz").write_bytes(b"x")  # outside demo dir / invalid id: ignored
    return SceneStore(scenes_dir=str(tmp_path / "scenes"), demo_dir=str(demo))


def test_store_crud_and_persistence(store, tmp_path):
    sid = store.add(_sc(truth=True), name="my scene", source="upload")
    lst = store.list()
    ids = [s["id"] for s in lst]
    assert sid in ids and "demo-valencia_flood" in ids
    me = next(s for s in lst if s["id"] == sid)
    assert me["size"] == [32, 32] and me["has_truth"] and me["tasks"] == ["flood"] and me["source"] == "upload"
    assert store.rename(sid, "renamed")["name"] == "renamed"
    # a new store instance (= restart) sees the persisted scene
    st2 = SceneStore(scenes_dir=store.scenes_dir, demo_dir=store.demo_dir)
    assert st2.get(sid)["optical"].shape == (6, 32, 32)
    assert st2.summary(sid)["name"] == "renamed"
    st2.delete(sid)
    with pytest.raises(KeyError):
        st2.get(sid)


def test_demo_scene_metadata_and_read_only(store):
    d = store.summary("demo-valencia_flood")
    assert d["name"] == "Valencia flood (demo)" and d["source"] == "demo" and d["read_only"]
    assert d["acquisition_date"] == "2024-10-30" and d["bounds"]["crs"] == "EPSG:4326"
    sc = store.get("demo-valencia_flood")
    assert sc["meta"]["data_source"] == "dataset:sen1floods11"
    with pytest.raises(SceneReadOnly):
        store.rename("demo-valencia_flood", "x")
    with pytest.raises(SceneReadOnly):
        store.delete("demo-valencia_flood")


def test_store_rejects_bad_ids(store):
    for bad in ("../etc/passwd", "", "a/b", "x" * 200):
        with pytest.raises(KeyError):
            store.get(bad)


@pytest.fixture
def client(store, monkeypatch, tmp_path):
    from terrascope.server import routes_rs
    monkeypatch.setattr(routes_rs, "_store", lambda: store)
    monkeypatch.setenv("TERRASCOPE_REAL_CKPT", str(tmp_path / "ckpt"))
    monkeypatch.setenv("TERRASCOPE_ONNX_DIR", str(tmp_path / "onnx"))
    from terrascope.fm import registry, weights
    monkeypatch.setattr(weights, "LOCAL_WEIGHTS", str(tmp_path / "w"))
    monkeypatch.setattr(weights, "HUB_CACHE", str(tmp_path / "hub"))
    registry.reset()
    app = FastAPI()
    app.include_router(routes_rs.router)
    return TestClient(app)


def test_routes_scenes(client, store):
    sid = store.add(_sc(), name="up", source="upload")
    r = client.get("/api/scenes")
    assert r.status_code == 200
    rows = r.json()
    assert {"id", "name", "source", "acquisition_date", "bounds", "size", "bands", "has_truth", "tasks"} <= set(rows[0])
    assert client.get(f"/api/scenes/{sid}").json()["name"] == "up"
    assert client.patch(f"/api/scenes/{sid}", json={"name": "new"}).json()["name"] == "new"
    assert client.patch(f"/api/scenes/{sid}", json={"name": ""}).status_code == 422
    assert client.patch("/api/scenes/demo-valencia_flood", json={"name": "x"}).status_code == 403
    assert client.delete("/api/scenes/demo-valencia_flood").status_code == 403
    assert client.delete(f"/api/scenes/{sid}").status_code == 200
    assert client.delete(f"/api/scenes/{sid}").status_code == 404
    assert client.get("/api/scenes/nope").status_code == 404


def test_routes_models_schema(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    ms = r.json()
    assert [m["id"] for m in ms] == ["terrafm", "prithvi_tiny", "terramind_tiny", "prithvi_100m",
                                     "unet_r18_scratch", "unet_r18_imagenet"]
    keys = {"id", "name", "family", "params", "sensors", "tasks", "status", "weights_file", "license", "source_url"}
    for m in ms:
        assert keys <= set(m), m
        assert m["status"] in ("ready", "weights_missing", "not_finetuned")
        assert m["family"] in ("terrafm", "prithvi", "terramind", "unet")
    by = {m["id"]: m for m in ms}
    # nothing downloaded in this sandbox -> FMs honestly report missing weights
    for k in ("prithvi_tiny", "terramind_tiny", "prithvi_100m", "unet_r18_imagenet"):
        assert by[k]["status"] == "weights_missing"
    assert by["unet_r18_scratch"]["status"] == "not_finetuned"  # needs no pretrained file
    assert by["unet_r18_scratch"]["params"] == 14_350_161  # whole U-Net, flood input (10 channels), measured
    assert by["prithvi_100m"]["params"] == 86_237_186  # measured from the built encoder
    assert by["prithvi_tiny"]["params"] == 5_634_050
    assert by["terramind_tiny"]["params"] == 5_776_896  # 6-band S2L1C subset + S1GRD + DEM embeddings
    assert by["prithvi_tiny"]["sensors"] == ["optical"]


def test_optical_only_demo_scene_analyses(tmp_path, monkeypatch):
    """HLS-style demo scenes carry no SAR/DEM arrays; analysing them must not 404 (was KeyError 'sar')."""
    import json
    import numpy as np
    from fastapi.testclient import TestClient
    d = tmp_path / "demo"; d.mkdir()
    np.savez(d / "hls_burn_scars_x.v1.4.npz", optical=(np.random.rand(6, 64, 64) * 0.3).astype("float32"),
             burn=np.zeros((64, 64), "uint8"))
    (d / "hls_burn_scars_x.v1.4.json").write_text(json.dumps({"name": "b", "data_source": "dataset:hls_burn_scars"}))
    from terrascope.real import scenes
    monkeypatch.setattr(scenes.STORE, "demo_dir", str(d))
    monkeypatch.setattr(scenes.STORE, "_mem", {})
    from terrascope.server.app import app
    r = TestClient(app).post("/api/analyze", json={"source": "scene", "scene_id": "demo-hls_burn_scars_x.v1.4",
                                                   "model": "terrafm", "sensors": [True, True, True]})
    assert r.status_code == 200, r.text
    assert r.json()["sensors"] == [True, False, False]
