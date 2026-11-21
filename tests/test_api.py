"""API regression tests.  Run:  python -m pytest -q tests"""
import io, json, os, tempfile
os.environ.setdefault("TERRASCOPE_CACHE_DIR", tempfile.mkdtemp(prefix="terrascope-test-cache-"))
os.environ.setdefault("TERRASCOPE_SCENES_DIR", tempfile.mkdtemp(prefix="terrascope-test-scenes-"))  # RS scene store
import numpy as np
import pytest
from fastapi.testclient import TestClient

import terrascope.server.app as appmod
from terrascope.server.app import app, engine

c = TestClient(app, raise_server_exceptions=False)
pytestmark = pytest.mark.skipif(not engine.ready, reason="checkpoint missing")
appmod.limiter.per_min = 0  # rate limiting has its own test below
# Isolation even when another test module imported the app first (env vars above would then be too late):
from terrascope.core.engine import ResultCache as _RC
engine.cache = _RC(tempfile.mkdtemp(prefix="terrascope-test-cache-"))
try:
    from terrascope.real import scenes as _scenes
    _scenes.STORE.scenes_dir = tempfile.mkdtemp(prefix="terrascope-test-scenes-")
except ImportError:
    pass
STAGES = ["queued", "preparing", "encoding", "decoding", "rendering", "done"]


def npz(**arrs):
    b = io.BytesIO(); np.savez(b, **arrs); return b.getvalue()


def npzc(**arrs):
    b = io.BytesIO(); np.savez_compressed(b, **arrs); return b.getvalue()


def scene(n=64, **extra):
    return dict(optical=np.full((6, n, n), 0.1, np.float32), sar=np.full((2, n, n), -12, np.float32),
                dem=np.stack([np.full((n, n), 200.0), np.full((n, n), 5.0)]).astype(np.float32), **extra)


def analyze(**kw):
    return c.post("/api/analyze", json=kw)


def test_health_and_index():
    h = c.get("/api/health").json()
    assert h["ready"] and h["params"] > 0 and h["models"] >= 1 and h["device"] in ("cpu", "cuda") and h["version"]
    r = c.get("/")
    assert r.status_code == 200
    import re
    assets = re.findall(r'(?:src|href)="(/static/[^"]+)"', r.text)
    assert assets, "index.html references no /static assets"
    # static mount works (individual asset completeness is the UI's e2e suite's job)
    assert any(c.get(a).status_code == 200 for a in assets)


def test_analyze_layers_export_similarity():
    r = analyze(seed=7, flood=0.8, cloud=0.5)
    assert r.status_code == 200
    d = r.json()
    assert d["truth"] and "accuracy" in d["stats"]
    for L in d["layers"]:
        assert c.get(f"/api/layer/{d['id']}/{L}.png").headers["content-type"] == "image/png"
    assert c.get(f"/api/similarity/{d['id']}.png?x=10&y=10").status_code == 200
    assert c.get(f"/api/similarity/{d['id']}.png?x=99999&y=1").status_code == 422
    assert c.get(f"/api/similarity/{d['id']}.png?x=-5&y=1").status_code == 422
    f = np.load(io.BytesIO(c.get(f"/api/export/{d['id']}.npz").content)).files
    assert {"pred_lc", "pred_flood_prob", "pred_veg"} <= set(f)


def test_ids_unguessable():
    a, b = analyze(seed=101).json()["id"], analyze(seed=102).json()["id"]
    assert a != b and len(a) > 12 and not a[1:].isdigit()


def test_identical_request_reuses_cached_result():
    body = dict(seed=4242, size=64, flood=0.3, cloud=0.2)
    first = analyze(**body).json()
    again = analyze(**body).json()
    assert first["cached"] is False and again["cached"] is True and again["id"] == first["id"]
    assert again["stats"] == first["stats"]
    assert analyze(**body, sensors=[True, False, True]).json()["id"] != first["id"]    # different request -> new run
    assert analyze(**{**body, "flood": 0.31}).json()["id"] != first["id"]
    from terrascope.core.engine import Engine                                        # survives a restart
    hit = Engine(cache_dir=engine.cache.root).run(dict(body, source="sim", sensors=[True, True, True], model="terrafm"))
    assert hit["id"] == first["id"] and hit["cached"]
    j = c.post("/api/analyze_async", json=body).json()
    ev = _events(j["job_id"])
    assert [e["stage"] for e in ev] == ["queued", "preparing", "rendering", "done"] and ev[2]["detail"] == "cached result"
    assert c.get(f"/api/jobs/{j['job_id']}").json()["result"]["id"] == first["id"]


def test_precomputed_index_is_honoured(tmp_path, monkeypatch):
    from terrascope.core.engine import Engine
    body = dict(source="sim", seed=777, size=32, flood=0.0, cloud=0.0, drought=0.0, sensors=[True, True, True], model="terrafm")
    aid = engine.run(body)["id"]
    e2 = Engine(cache_dir=engine.cache.root)
    e2.cache._keys.clear()                                   # pretend the cache has no request keys (built elsewhere)
    assert e2.cached_result(body, e2.runner) is None
    p = tmp_path / "pre.json"
    p.write_text(json.dumps([{"request": body, "analysis_id": aid}]))
    monkeypatch.setenv("TERRASCOPE_PRECOMPUTED", str(p))
    e3 = Engine(cache_dir=engine.cache.root)
    e3.cache._keys.clear()
    assert e3.cached_result(body, e3.runner)["id"] == aid


@pytest.mark.parametrize("sensors", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
def test_sensor_subsets(sensors):
    assert analyze(seed=3, sensors=[bool(s) for s in sensors]).status_code == 200


@pytest.mark.parametrize("body,code", [
    (dict(sensors=[False, False, False]), 400), (dict(sensors=[True, True]), 400),
    (dict(size=33), 422), (dict(size=300), 422), (dict(flood=2), 422), (dict(seed=-1), 422),
    (dict(source="bogus"), 422), (dict(source="upload", scene_id="nope"), 404)])
def test_analyze_validation(body, code):
    assert analyze(**body).status_code == code


@pytest.mark.parametrize("q", ["size=4096", "size=3", "size=33", "flood=5", "seed=-1"])
def test_sample_validation(q):
    assert c.get(f"/api/sample.npz?{q}").status_code == 422


def test_sample_roundtrip_upload():
    raw = c.get("/api/sample.npz").content
    u = c.post("/api/upload", files={"file": ("s.npz", raw)}).json()
    d = analyze(source="upload", scene_id=u["scene_id"]).json()
    assert d["truth"] is False and "accuracy" not in d["stats"]


@pytest.mark.parametrize("payload", ids=["garbage", "missing_dem", "nan", "bad_classes", "too_big", "pickled"], argvalues=[
    b"hello",
    npz(optical=np.zeros((6, 64, 64)), sar=np.zeros((2, 64, 64))),                  # missing dem
    npz(**{**scene(), "optical": np.full((6, 64, 64), np.nan, np.float32)}),         # NaN
    npz(**scene(), lc=np.full((64, 64), 99), flood=np.zeros((64, 64)), veg=np.zeros((64, 64))),  # bad classes
    # tiny on the wire, 1.4 GB decompressed: must be refused from the header alone
    npzc(optical=np.zeros((6, 6000, 6000), np.float32), sar=np.zeros((2, 64, 64)), dem=np.zeros((2, 64, 64))),
    npz(optical=np.array([object()], dtype=object)),
])
def test_upload_rejects_bad_files(payload):
    assert c.post("/api/upload", files={"file": ("x.npz", payload)}).status_code == 400


def test_upload_partial_labels_ignored():
    u = c.post("/api/upload", files={"file": ("x.npz", npz(**scene(), lc=np.zeros((64, 64))))}).json()
    assert analyze(source="upload", scene_id=u["scene_id"]).json()["truth"] is False


def test_upload_full_labels_scored():
    u = c.post("/api/upload", files={"file": ("x.npz", npz(**scene(), lc=np.zeros((64, 64)), flood=np.zeros((64, 64)), veg=np.zeros((64, 64))))}).json()
    assert "accuracy" in analyze(source="upload", scene_id=u["scene_id"]).json()["stats"]


def test_upload_size_cap(monkeypatch):
    import terrascope.server.app as m
    monkeypatch.setattr(m, "MAX_UPLOAD_BYTES", 1000)
    assert c.post("/api/upload", files={"file": ("x.npz", b"0" * 2000)}).status_code == 413


def test_benchmarks_and_docs_off():
    b = c.get("/api/benchmarks").json()
    assert {"ablation", "scratch", "size128", "heavy_cloud", "curves", "config"} <= set(b)   # legacy keys
    assert {"sim", "robustness", "calibration", "real", "laptop_bench"} <= set(b)
    assert "ablation" in b["sim"] and "seeds" in b["sim"]
    assert c.get("/docs").status_code == 404


# ------------------------------------------------------------------ phase 3 contract
def test_analyze_new_fields_and_risk_shape():
    d = analyze(seed=7, flood=0.8, cloud=0.5).json()
    st = d["stats"]
    for k in ("flood", "drought", "urban", "composite"):
        r = st["risk"][k]
        assert r["kind"] == "heuristic" and isinstance(r["formula"], str) and isinstance(r["inputs"], dict)
        assert 0 <= r["value"] <= 100
    assert "confidence" in st and "confidence_calibrated" in st and "calibration" in st
    p = d["provenance"]
    assert {"model_id", "model_name", "params", "weights_file", "data_source", "acquisition_date", "processed_at", "runtime_ms", "device"} <= set(p)
    assert p["model_id"] == "terrafm" and p["data_source"] == "simulated" and p["processed_at"].endswith("Z")
    assert d["bounds"] is None and d["gsd_m"] == 30.0
    n = 128 * 128
    import base64
    cm = np.frombuffer(base64.b64decode(d["pix"]["cloud_mask"]), np.uint8)
    assert cm.size == n and set(np.unique(cm)) <= {0, 255} and cm.max() == 255   # cloudy scene: something rebuilt
    assert "lc_prob" not in d["pix"]


def test_unknown_model_409():
    assert analyze(seed=1, model="no_such_model").status_code == 409
    assert c.post("/api/analyze_async", json=dict(seed=1, model="no_such_model")).status_code == 409
    assert analyze(seed=1, model="../etc").status_code == 422


def _events(job_id):
    r = c.get(f"/api/jobs/{job_id}/events")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    return [json.loads(l[5:]) for l in r.text.splitlines() if l.startswith("data:")]


def test_async_job_sse_stage_order():
    j = c.post("/api/analyze_async", json=dict(seed=11, size=64, cloud=0.4)).json()
    ev = _events(j["job_id"])
    assert [e["stage"] for e in ev] == STAGES
    t = [e["t_ms"] for e in ev]
    assert t == sorted(t) and all(isinstance(v, int) for v in t)
    st = c.get(f"/api/jobs/{j['job_id']}").json()
    assert st["status"] == "done" and st["result"]["stats"]["size"] == [64, 64]
    assert c.get(f"/api/layer/{st['result']['id']}/lc.png").status_code == 200
    assert c.get("/api/jobs/nope").status_code == 404 and c.get("/api/jobs/nope/events").status_code == 404


def test_async_job_error_event():
    j = c.post("/api/analyze_async", json=dict(source="upload", scene_id="upnope")).json()
    ev = _events(j["job_id"])
    assert ev[0]["stage"] == "queued" and ev[-1]["stage"] == "error" and ev[-1]["detail"]
    assert c.get(f"/api/jobs/{j['job_id']}").json()["status"] == "error"


def test_change_endpoint():
    a = analyze(seed=7, size=64, flood=0.0).json()["id"]
    b = analyze(seed=7, size=64, flood=1.0).json()["id"]
    r = c.get(f"/api/change?a={a}&b={b}")
    assert r.status_code == 200
    d = r.json()
    assert set(d["layers"]) == {"flood_onset", "veg_loss", "urban_growth", "lc_change"}
    assert {"flood_onset_km2", "veg_loss_km2", "urban_growth_km2", "lc_changed_frac"} <= set(d["stats"])
    assert d["stats"]["flood_onset_km2"] > 0
    for L in d["layers"]:
        assert c.get(f"/api/layer/{d['id']}/{L}.png").headers["content-type"] == "image/png"
    big = analyze(seed=7, size=128).json()["id"]
    assert c.get(f"/api/change?a={a}&b={big}").status_code == 409
    assert c.get(f"/api/change?a={a}&b=anope").status_code == 404


def test_results_survive_new_engine_instance():
    from terrascope.core.engine import Engine
    d = analyze(seed=9, size=64).json()
    e2 = Engine(cache_dir=engine.cache.root)          # simulates a restart: fresh process state, same disk cache
    assert e2.layer(d["id"], "rgb") == c.get(f"/api/layer/{d['id']}/rgb.png").content
    assert e2.similarity_png(d["id"], 5, 5)[:4] == b"\x89PNG"
    assert "pred_lc" in e2.export_arrays(d["id"])


def test_layer_path_traversal_rejected():
    assert c.get("/api/layer/..%2F..%2Fartifacts/rgb.png").status_code == 404
    assert c.get("/api/layer/a1/..%2Fmeta.png").status_code == 404


def test_rate_limit_429(monkeypatch):
    lim = appmod.RateLimiter(2)
    monkeypatch.setattr(appmod, "limiter", lim)
    codes = [analyze(seed=1, size=32).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429
    assert c.post("/api/analyze_async", json=dict(seed=1)).status_code == 429
    assert c.get("/api/health").status_code == 200        # only analyses are limited


def test_scene_source_unknown_404():
    assert analyze(source="scene", scene_id="does-not-exist").status_code == 404
    assert analyze(source="scene").status_code == 404


def test_tif_upload_delegates_or_415():
    r = c.post("/api/upload", files={"file": ("x.tif", b"not a tiff")})
    assert r.status_code in (400, 415)


def test_upload_response_fields():
    raw = c.get("/api/sample.npz?size=64").content
    u = c.post("/api/upload", files={"file": ("s.npz", raw)}).json()
    assert u["source"] == "upload" and u["gsd_m"] is None and u["gsd_assumed"] and u["size"] == [64, 64]
    d = analyze(source="upload", scene_id=u["scene_id"]).json()
    assert d["provenance"]["data_source"] == "upload"
    # .npz carries no pixel size -> no areas, only fractions
    assert d["gsd_m"] is None and d["gsd_assumed"] is True and d["stats"]["area_km2"] is None
    assert d["stats"]["flood"]["km2"] is None and d["stats"]["flood"]["frac"] >= 0
    assert all(c_["km2"] is None for c_ in d["stats"]["composition"])
    sim = analyze(seed=3, size=64).json()
    assert sim["gsd_m"] == 30.0 and sim["gsd_assumed"] is False and sim["stats"]["area_km2"] > 0



def test_precomputed_dict_format_boots_and_is_honoured(tmp_path, monkeypatch):
    """make_demo_bundle writes {"generated_at", "runtime", "device", "entries": [...]}."""
    from terrascope.core.engine import Engine
    body = dict(source="sim", seed=778, size=32, flood=0.0, cloud=0.0, drought=0.0, sensors=[True, True, True], model="terrafm")
    aid = engine.run(body)["id"]
    p = tmp_path / "pre.json"
    p.write_text(json.dumps({"generated_at": "2026-11-04T10:00:00+01:00", "runtime": "torch", "device": "cpu",
                             "entries": [{"request": {"seed": 778, "size": 32}, "http": 200, "analysis_id": aid},
                                         {"request": "garbage"}, {"request": {"seed": "x"}, "analysis_id": "a1"}, 7]}))
    monkeypatch.setenv("TERRASCOPE_PRECOMPUTED", str(p))
    e = Engine(cache_dir=engine.cache.root)
    e.cache._keys.clear()
    assert e.cached_result(body, e.runner)["id"] == aid      # partial request body uses API defaults
    for bad in ('{"entries": 5}', '"just a string"', "{not json", "[1, 2]"):
        p.write_text(bad)
        assert Engine(cache_dir=engine.cache.root).precomputed == {}


def test_upload_is_rate_limited(monkeypatch):
    monkeypatch.setattr(appmod, "limiter", appmod.RateLimiter(1))
    raw = c.get("/api/sample.npz?size=32").content
    assert c.post("/api/upload", files={"file": ("s.npz", raw)}).status_code == 200
    assert c.post("/api/upload", files={"file": ("s.npz", raw)}).status_code == 429


def test_x_forwarded_for_does_not_bypass_rate_limit(monkeypatch):
    monkeypatch.setattr(appmod, "limiter", appmod.RateLimiter(2))
    monkeypatch.setattr(appmod, "TRUST_PROXY", False)
    codes = [c.post("/api/analyze", json=dict(seed=i, size=32), headers={"X-Forwarded-For": f"10.0.0.{i}"}).status_code
             for i in range(4)]
    assert codes[2:] == [429, 429]
    monkeypatch.setattr(appmod, "TRUST_PROXY", True)                       # behind one trusted proxy: right-most hop
    from starlette.requests import Request
    req = Request({"type": "http", "headers": [(b"x-forwarded-for", b"6.6.6.6, 1.2.3.4")], "client": ("127.0.0.1", 1)})
    assert appmod.client_key(req) == "1.2.3.4"


def test_confidence_metrics_consistent_and_calibration_flag():
    d = analyze(seed=8, size=64).json()["stats"]
    assert 1 / 6 <= d["confidence"] <= 1                                   # mean max-prob, same metric as calibrated
    cal = d["calibration"]
    if cal is not None and not cal["applied"]:
        assert d["confidence_calibrated"] is None and cal["ece_after"] >= cal["ece_before"]
    if cal is not None and cal["applied"]:
        assert 0 <= d["confidence_calibrated"] <= 1


def test_delete_scene_drops_upload_and_cached_results():
    pytest.importorskip("terrascope.real.scenes")
    raw = c.get("/api/sample.npz?size=32&seed=5").content
    sid = c.post("/api/upload", files={"file": ("s.npz", raw)}).json()["scene_id"]
    d = analyze(source="upload", scene_id=sid).json()
    assert c.get(f"/api/layer/{d['id']}/rgb.png").status_code == 200
    r = c.delete(f"/api/scenes/{sid}")
    assert r.status_code < 300, r.text
    assert sid not in engine.uploads
    assert c.get(f"/api/layer/{d['id']}/rgb.png").status_code == 404
    assert analyze(source="upload", scene_id=sid).status_code == 404


def test_change_rejects_change_ids_and_mismatched_bounds():
    a = analyze(seed=7, size=64, flood=0.0).json()["id"]
    b = analyze(seed=7, size=64, flood=0.9).json()["id"]
    cid = c.get(f"/api/change?a={a}&b={b}").json()["id"]
    assert c.get(f"/api/change?a={cid}&b={b}").status_code == 409
    # same grid size but different footprints -> 409
    for aid, w in ((a, 10.0), (b, 11.0)):
        meta = engine.cache.meta(aid)
        meta["bounds"] = dict(west=w, south=50.0, east=w + 0.1, north=50.1, crs="EPSG:4326")
    assert c.get(f"/api/change?a={a}&b={b}").status_code == 409
