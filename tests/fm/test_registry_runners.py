import numpy as np
import pytest
import torch

from fm_helpers import write_ckpt
from terrascope.fm import registry
from terrascope.fm.errors import RunnerUnavailable
from terrascope.fm.model import build_task_model


def _scene(H=64, W=64, seed=0):
    r = np.random.default_rng(seed)
    return dict(optical=(0.05 + 0.3 * r.random((6, H, W))).astype(np.float32),
                sar=(-12 + 3 * r.standard_normal((2, H, W))).astype(np.float32),
                dem=np.stack([np.full((H, W), 150.0), np.full((H, W), 3.0)]).astype(np.float32))


def _check(out, H, W):
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            assert v.dtype == np.float32, k
            assert np.isfinite(v).all(), k
    if "flood_prob" in out:
        assert out["flood_prob"].shape == (H, W) and 0 <= out["flood_prob"].min() and out["flood_prob"].max() <= 1
    if "burn_prob" in out:
        assert out["burn_prob"].shape == (H, W)
    if "lc_prob" in out:
        assert out["lc_prob"].shape == (6, H, W)
        np.testing.assert_allclose(out["lc_prob"].sum(0), 1.0, atol=1e-4)
    if "recon" in out:
        assert out["recon"].shape == (6, H, W)
    h, w = out["feat_hw"]
    assert out["feats"].shape[0] == h * w
    assert out["veg"].shape == (H, W)


def test_list_models_without_files(fm_env):
    ms = registry.list_models()
    assert ms[0]["id"] == "terrafm"
    st = {m["id"]: m["status"] for m in ms}
    assert st["prithvi_tiny"] == st["terramind_tiny"] == st["prithvi_100m"] == "weights_missing"
    with pytest.raises(RunnerUnavailable, match="get_data.bat weights"):
        registry.get_runner("prithvi_tiny")
    with pytest.raises(KeyError):
        registry.get_runner("nope")


def test_not_finetuned_when_only_weights(fm_env):
    import os
    os.makedirs(fm_env["weights"], exist_ok=True)
    open(os.path.join(fm_env["weights"], "Prithvi_EO_V2_tiny_TL.pt"), "wb").close()
    st = {m["id"]: m for m in registry.list_models()}
    assert st["prithvi_tiny"]["status"] == "not_finetuned"
    assert st["prithvi_tiny"]["weights_file"].endswith("Prithvi_EO_V2_tiny_TL.pt")
    with pytest.raises(RunnerUnavailable, match="train.bat"):
        registry.get_runner("prithvi_tiny")


def test_terratorch_import_does_not_leak_precision(fm_env):
    torch.set_float32_matmul_precision("highest")
    build_task_model("prithvi_tiny", "flood")
    assert torch.get_float32_matmul_precision() == "highest"


@pytest.mark.parametrize("model_id,tasks", [
    ("prithvi_tiny", ["flood", "landcover"]),
    ("terramind_tiny", ["flood", "burn", "cloud"]),
    ("prithvi_100m", ["burn"]),
])
def test_runner_contract_outputs(fm_env, model_id, tasks):
    for i, t in enumerate(tasks):
        write_ckpt(build_task_model(model_id, t).eval(), fm_env["ckpt"], seed=0)
    m = {x["id"]: x for x in registry.list_models()}[model_id]
    assert m["status"] == "ready" and set(tasks) <= set(m["tasks_ready"])
    r = registry.get_runner(model_id)
    stages = []
    out = r.predict(_scene(), [True, True, True], on_stage=stages.append)
    assert stages == ["encoding", "decoding"]
    _check(out, 64, 64)
    keymap = dict(flood="flood_prob", burn="burn_prob", landcover="lc_prob", cloud="recon")
    for t in tasks:
        assert keymap[t] in out
    assert r.derived_outputs["veg"].startswith("health")


def test_sensor_rules_and_tiling(fm_env):
    write_ckpt(build_task_model("prithvi_tiny", "flood").eval(), fm_env["ckpt"])
    write_ckpt(build_task_model("terramind_tiny", "flood").eval(), fm_env["ckpt"])
    write_ckpt(build_task_model("terramind_tiny", "burn").eval(), fm_env["ckpt"])
    p = registry.get_runner("prithvi_tiny")
    with pytest.raises(RunnerUnavailable, match="optical"):
        p.predict(_scene(), [False, True, True])
    tm = registry.get_runner("terramind_tiny")
    out = tm.predict(_scene(), [False, True, False])  # SAR only: flood works, burn (optical) skipped, no veg
    assert "flood_prob" in out and "burn_prob" not in out and "veg" not in out
    big = tm.predict(_scene(256, 240, 1), [True, True, True])  # > 224 px: overlapping 224 tiles
    _check(big, 256, 240)
    assert big["feat_hw"] == (16, 15)


def test_best_seed_selected(fm_env):
    from terrascope.fm.runners import best_checkpoints
    write_ckpt(build_task_model("prithvi_tiny", "flood").eval(), fm_env["ckpt"], seed=0, score=0.4)
    write_ckpt(build_task_model("prithvi_tiny", "flood").eval(), fm_env["ckpt"], seed=1, score=0.6)
    assert best_checkpoints("prithvi_tiny", fm_env["ckpt"])["flood"].endswith("prithvi_tiny_s1.pt")
