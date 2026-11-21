import json
import sys
import types

import numpy as np
import pytest

import bench_laptop


class FakeRunner:
    id, name, params, device = "fake", "Fake FM", 123, "cpu"

    def predict(self, sc, sensors, on_stage=lambda s: None):
        on_stage("encoding")
        if sc["optical"].shape[-1] > 64:
            raise ValueError("tile too large")
        return {"flood_prob": np.zeros(sc["optical"].shape[1:], np.float32)}


@pytest.fixture
def fake_registry(monkeypatch):
    def unavailable(mid):
        raise RuntimeError(f"{mid}: weights missing")
    reg = types.ModuleType("terrascope.fm.registry")
    reg.list_models = lambda: [{"id": "fake", "status": "ready"}, {"id": "big", "status": "weights_missing"}]
    reg.get_runner = lambda mid: FakeRunner() if mid == "fake" else unavailable(mid)
    monkeypatch.setitem(sys.modules, "terrascope.fm.registry", reg)
    import terrascope
    if "terrascope.fm" not in sys.modules:
        monkeypatch.setitem(sys.modules, "terrascope.fm", types.ModuleType("terrascope.fm"))
    return reg


def test_worker_runner_measures_and_skips_sizes(fake_registry):
    r = bench_laptop.worker_runner({"kind": "runner", "model": "fake", "runtime": "torch", "sizes": [32, 128], "reps": 2})
    assert r["status"] == "ok" and r["params"] == 123
    assert r["per_size"]["32"]["reps"] == 2 and r["per_size"]["32"]["s_per_scene_mean"] >= 0
    assert "tile too large" in r["per_size"]["128"]["skipped"]
    assert "load_s" in r and "construct_s" in r


def test_worker_runner_unavailable_model_is_skipped_with_reason(fake_registry):
    r = bench_laptop.worker_runner({"kind": "runner", "model": "big", "runtime": "torch", "sizes": [32], "reps": 1})
    assert r["status"] == "skipped" and "weights missing" in r["reason"]


def test_plan_lists_not_ready_models_as_skipped(fake_registry):
    import logging
    specs, skipped = bench_laptop.plan(None, logging.getLogger("t"))
    assert {"kind": "runner", "model": "fake", "runtime": "torch"} in specs
    assert any(s["model_id"] == "big" and s["status"] == "skipped" for s in skipped)


def test_onnx_file_worker_fp32_and_int8(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper
    x = helper.make_tensor_value_info("optical", TensorProto.FLOAT, ["N", 6, "H", "W"])
    y = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 6, "H", "W"])
    g = helper.make_graph([helper.make_node("Relu", ["optical"], ["out"])], "g", [x], [y])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    dyn = tmp_path / "flood.encoder.int8.onnx"
    onnx.save(m, dyn)
    xf = helper.make_tensor_value_info("optical", TensorProto.FLOAT, [1, 6, 64, 64])
    yf = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 6, 64, 64])
    mf = helper.make_model(helper.make_graph([helper.make_node("Relu", ["optical"], ["out"])], "g", [xf], [yf]),
                           opset_imports=[helper.make_opsetid("", 17)])
    mf.ir_version = 8
    fixed = tmp_path / "flood.encoder.onnx"
    onnx.save(mf, fixed)
    r = bench_laptop.worker_onnx({"path": str(dyn), "sizes": [32, 64], "reps": 1})
    assert r["status"] == "ok" and r["variant"] == "onnx-int8"
    assert set(r["per_size"]) == {"32", "64"} and all("s_per_scene_mean" in v for v in r["per_size"].values())
    r = bench_laptop.worker_onnx({"path": str(fixed), "sizes": [32, 64], "reps": 1})
    assert r["variant"] == "onnx-fp32" and "fixed size" in r["per_size"]["32"]["skipped"]
    assert "s_per_scene_mean" in r["per_size"]["64"]


def test_end_to_end_with_terrafm_only(tmp_path):
    """Real subprocess run: whatever the registry offers, TerraFM must be measured; nothing crashes."""
    out = tmp_path / "bench.json"
    assert bench_laptop.main(["--quick", "--models", "terrafm", "--out", str(out), "--timeout", "600"]) == 0
    d = json.loads(out.read_text())
    assert d["machine"]["logical_cores"] >= 1 and d["settings"]["device"] == "cpu"
    tf = [r for r in d["results"] if r.get("model_id") == "terrafm" and r.get("runtime") == "torch"]
    assert tf and tf[0]["status"] == "ok", tf
    assert tf[0]["per_size"]["128"]["s_per_scene_mean"] > 0 and tf[0]["peak_ram_mb"] > 0
    for r in d["results"]:
        assert r["status"] in ("ok", "skipped", "error")
        if r["status"] != "ok":
            assert r.get("reason")


def test_worker_does_not_leak_grad_mode(fake_registry):
    torch = pytest.importorskip("torch")
    assert torch.is_grad_enabled()
    bench_laptop.worker_runner({"kind": "runner", "model": "fake", "runtime": "torch", "sizes": [32], "reps": 1})
    assert torch.is_grad_enabled(), "bench must not switch off autograd globally"
