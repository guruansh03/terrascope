import os

import numpy as np
import pytest
import torch

from fm_helpers import write_ckpt
from terrascope.fm import registry
from terrascope.fm.export import OnnxTask, export_onnx
from terrascope.fm.model import build_task_model

ort = pytest.importorskip("onnxruntime")


def _sessions(d, meta):
    mk = lambda f: ort.InferenceSession(os.path.join(d, f), providers=["CPUExecutionProvider"])
    return mk(meta["encoder"]), mk(meta["head"])


def _inputs(seed=0, n=224):
    g = torch.Generator().manual_seed(seed)
    o = torch.rand(1, 6, n, n, generator=g) * 0.4
    s = torch.randn(1, 2, n, n, generator=g) * 3 - 14
    d = torch.stack([torch.rand(1, n, n, generator=g) * 300, torch.rand(1, n, n, generator=g) * 20], 1)
    return o, s, d, torch.ones(1, 3)


@pytest.mark.parametrize("model_id,task", [("terrafm", "landcover"), ("prithvi_tiny", "flood"),
                                           ("terramind_tiny", "cloud")])
def test_onnx_parity_fp32(fm_env, model_id, task):
    m = build_task_model(model_id, task).eval()
    meta = export_onnx(model_id, task, fm_env["onnx"], int8=False, model=m)
    d = os.path.join(fm_env["onnx"], model_id)
    assert os.path.exists(os.path.join(d, f"{task}.json")) and meta["tile"] == 224 and not meta["int8"]
    ot = OnnxTask(*_sessions(d, meta), meta)
    o, s, dd, a = _inputs()
    with torch.no_grad():
        ref = m(o, s, dd, a)
    lv, raw = ot.encode(o, s, dd, a)
    y = ot.decode(lv, raw, o, a)
    assert y.shape == ref.shape
    np.testing.assert_allclose(y.numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


def test_onnx_int8_export_runs(fm_env):
    m = build_task_model("prithvi_tiny", "flood").eval()
    fp = export_onnx("prithvi_tiny", "flood", fm_env["onnx"], int8=False, model=m)
    q = export_onnx("prithvi_tiny", "flood", fm_env["onnx"], int8=True, model=m)
    d = os.path.join(fm_env["onnx"], "prithvi_tiny")
    assert q["int8"] and q["encoder"].endswith(".int8.onnx")
    assert os.path.getsize(os.path.join(d, q["encoder"])) < 0.5 * os.path.getsize(os.path.join(d, fp["encoder"]))
    ot = OnnxTask(*_sessions(d, q), q)
    o, s, dd, a = _inputs(1)
    y = ot.decode(*ot.encode(o, s, dd, a), o, a)
    with torch.no_grad():
        ref = m(o, s, dd, a)
    assert np.isfinite(y.numpy()).all()
    assert np.corrcoef(y.numpy().ravel(), ref.numpy().ravel())[0, 1] > 0.9  # quantised: close, not identical


def test_export_rejects_derived_and_missing(fm_env):
    with pytest.raises(ValueError, match="derived"):
        export_onnx("prithvi_tiny", "veg", fm_env["onnx"])
    with pytest.raises(FileNotFoundError, match="train.bat"):
        export_onnx("prithvi_tiny", "flood", fm_env["onnx"], ckpt_root=fm_env["ckpt"])


def test_onnx_runner_matches_torch_runner(fm_env, monkeypatch):
    """Same fine-tuned checkpoint served by torch and by ONNX through the registry (incl. pad-to-224 tiling)."""
    m = build_task_model("prithvi_tiny", "flood").eval()
    write_ckpt(m, fm_env["ckpt"])
    export_onnx("prithvi_tiny", "flood", fm_env["onnx"], ckpt_root=fm_env["ckpt"])  # from the checkpoint on disk
    r = np.random.default_rng(3)
    sc = dict(optical=(0.05 + 0.3 * r.random((6, 96, 80))).astype(np.float32),
              sar=np.full((2, 96, 80), -12, np.float32), dem=np.zeros((2, 96, 80), np.float32))
    torch_out = registry.get_runner("prithvi_tiny").predict(sc, [True, False, False])
    monkeypatch.setenv("TERRASCOPE_RUNTIME", "onnx")
    registry.reset()
    entry = {x["id"]: x for x in registry.list_models()}["prithvi_tiny"]
    assert entry["status"] == "ready" and entry["runtime"] == "onnx"
    rr = registry.get_runner("prithvi_tiny")
    assert type(rr).__name__ == "OnnxFMRunner"
    stages = []
    onnx_out = rr.predict(sc, [True, False, False], on_stage=stages.append)
    assert stages == ["encoding", "decoding"]
    assert onnx_out["flood_prob"].shape == (96, 80)
    # both runtimes pad the 96x80 scene to one 224 tile -> same numbers up to float error
    np.testing.assert_allclose(onnx_out["flood_prob"], torch_out["flood_prob"], atol=1e-4)
    np.testing.assert_allclose(onnx_out["feats"], torch_out["feats"], atol=1e-3)
    assert onnx_out["feat_hw"] == torch_out["feat_hw"] == (6, 5)
    with pytest.raises(Exception, match="optical"):
        rr.predict(sc, [False, True, False])
