"""Core/ML tests: normalisation v2 invariance, cloud head, shift ensemble, checkpoint compatibility, calibration, cache."""
import json, os, tempfile

import numpy as np
import pytest
import torch

from terrascope.core.model import TerraFM, normalize, load_checkpoint, SHIFTS
from terrascope.core.engine import Engine, TerraFMRunner, ResultCache
from terrascope.core.engine.render import blend_weights, composite
from terrascope.core.simulator import generate_scene
from terrascope.core.calibrate import fit_temperature, reliability

ROOT = os.path.join(os.path.dirname(__file__), "..")
CK = os.path.join(ROOT, "artifacts", "ckpt")
V2_CFG = dict(dim=32, depth=1, heads=2, dec_dim=32, dec_depth=1, patch=4, norm="v2", cloud_head=True)


def _scene(size=64, seed=3):
    sc = generate_scene(seed, size, flood=0.6, cloud=0.5, drought=0.2)
    return {k: sc[k] for k in ("optical", "optical_clean", "sar", "dem", "lc", "flood", "veg", "cloud")}


def _tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    m = TerraFM(**V2_CFG)
    p = os.path.join(tmp_path, "tiny_v2.pt")
    torch.save({"model": m.state_dict(), "cfg": V2_CFG}, p)
    return p


def test_v2_normalisation_invariant_to_elevation_offset():
    sc = _scene()
    t = {k: torch.from_numpy(np.asarray(sc[k], np.float32))[None] for k in ("optical", "sar", "dem")}
    d2 = t["dem"].clone(); d2[:, 0] += 1500
    a = normalize(t["optical"], t["sar"], t["dem"], "v2")
    b = normalize(t["optical"], t["sar"], d2, "v2")
    assert torch.allclose(a, b, atol=1e-3)
    v1a, v1b = normalize(t["optical"], t["sar"], t["dem"], "v1"), normalize(t["optical"], t["sar"], d2, "v1")
    assert (v1a - v1b).abs().max() > 5            # the legacy scheme is not invariant (the shortcut)
    assert a[:, 9].min() >= 0                       # HAND-like channel is non-negative


def test_cloud_head_and_shift_ensemble_outputs(tmp_path):
    r = TerraFMRunner(ckpt=_tiny_ckpt(tmp_path))
    assert r.norm == "v2" and "cloud" in r.tasks and len(r.shifts) == len(SHIFTS)
    stages = []
    out = r.predict(_scene(), [True, True, True], stages.append)
    assert stages == ["encoding", "decoding"]
    assert out["cloud_prob"].shape == (64, 64) and 0 <= out["cloud_prob"].min() and out["cloud_prob"].max() <= 1
    assert out["recon"].shape == (6, 64, 64) and out["lc_prob"].shape == (6, 64, 64)
    assert np.allclose(out["lc_prob"].sum(0), 1, atol=1e-4)
    assert "cloud_prob" not in r.predict(_scene(), [False, True, True])   # no optical -> no cloud estimate


def test_shift_ensemble_matches_single_pass_when_disabled(tmp_path):
    torch.manual_seed(0)
    m = TerraFM(**V2_CFG).eval()
    x = torch.randn(1, 10, 32, 32)
    av = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        single = m.forward_all(x, av)["recon"]
        assert torch.allclose(m.reconstruct(x, av, shifts=((0, 0),)), single, atol=1e-5)
        assert m.reconstruct(x, av).shape == single.shape


def test_composite_uses_cloud_probability():
    o = np.full((6, 32, 32), 0.1, np.float32)
    rec = np.full((6, 32, 32), 0.3, np.float32)
    cp = np.zeros((32, 32), np.float32); cp[10:20, 10:20] = 0.95
    out, a, method = composite(o, rec, cp)
    assert method == "cloud_head" and a[15, 15] > 0.9 and a[0, 0] < 0.05
    assert abs(out[0, 15, 15] - 0.3) < 0.02 and abs(out[0, 0, 0] - 0.1) < 0.02
    _, a2, m2 = composite(o, rec, None, optical_on=False)
    assert m2 == "no_optical_input" and a2.min() == 1
    assert blend_weights(cp).shape == (1, 32, 32)


@pytest.mark.skipif(not os.path.exists(os.path.join(CK, "v1", "finetuned.pt")), reason="v1 checkpoint not present")
def test_v1_checkpoint_still_loads_and_runs():
    m, cfg = load_checkpoint(os.path.join(CK, "v1", "finetuned.pt"))
    assert m.norm_version == "v1" and not m.has_cloud_head
    assert sum(p.numel() for p in m.parameters()) == 686248
    r = TerraFMRunner(ckpt=os.path.join(CK, "v1", "finetuned.pt"))
    out = r.predict(_scene(), [True, True, True])
    assert "cloud_prob" not in out and out["lc_prob"].shape == (6, 64, 64)
    e = Engine(ckpt=os.path.join(CK, "v1", "finetuned.pt"), cache_dir=tempfile.mkdtemp())
    res = e.analyze(_scene(), [True, True, True], dict(source="sim"))
    assert res["composite_method"] == "threshold_rule" and res["stats"]["cloud_method"] == "threshold_rule"


@pytest.mark.skipif(not os.path.exists(os.path.join(CK, "finetuned.pt")), reason="v2 checkpoint not trained yet")
def test_current_checkpoint_is_v2_with_cloud_head():
    m, cfg = load_checkpoint(os.path.join(CK, "finetuned.pt"))
    assert cfg["norm"] == "v2" and cfg["cloud_head"] is True


def test_temperature_scaling_recovers_known_temperature():
    torch.manual_seed(0)
    n, C, T_true = 20000, 6, 2.5
    z = torch.randn(n, C) * 3
    y = torch.distributions.Categorical(logits=z / T_true).sample()
    T = fit_temperature(z, y)
    assert abs(T - T_true) < 0.2
    before, after = reliability(z, y, 1.0), reliability(z, y, T)
    assert after["ece"] < before["ece"] and len(after["bins"]) == 15


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "artifacts", "calibration.json")), reason="calibration not run yet")
def test_calibration_json_schema():
    cal = json.load(open(os.path.join(ROOT, "artifacts", "calibration.json")))
    assert cal["method"] == "temperature" and cal["T"] > 0
    assert 0 <= cal["ece_after"] <= 1 and 0 <= cal["ece_before"] <= 1
    assert cal["fit_split"]["name"] == "val64" and cal["eval_split"]["name"] == "test64"
    assert sum(b["count_before"] for b in cal["bins"]) == cal["eval_split"]["pixels"]


def test_result_cache_lru_by_bytes(tmp_path):
    rc = ResultCache(str(tmp_path), max_bytes=3000)
    for i in range(5):
        rc.put(f"a{i}", {"rgb": b"x" * 1000}, None, {"i": i})
    assert "a4" in rc and "a0" not in rc and rc.total_bytes() <= 3000 + 1500
    rc2 = ResultCache(str(tmp_path), max_bytes=3000)          # reopen from disk
    assert rc2.layer("a4", "rgb") == b"x" * 1000 and rc2.meta("a4")["i"] == 4
    assert rc2.layer("../a4", "rgb") is None and rc2.layer("a4", "../meta") is None


def test_stats_calibrated_confidence_only_when_validation_ece_improves():
    from terrascope.core.engine import stats
    sc = _scene(32)
    p = np.random.default_rng(0).dirichlet(np.ones(6), (32, 32)).transpose(2, 0, 1).astype(np.float32)
    pc = (p ** 2 / (p ** 2).sum(0)).astype(np.float32)
    pred = dict(lc_prob=p, lc_prob_calibrated=pc, flood_prob=np.zeros((32, 32), np.float32), veg=np.zeros((32, 32), np.float32))
    der = dict(lc=p.argmax(0).astype(np.uint8))
    # decision uses VALIDATION ECE only; test ECE (ece_before/ece_after) is an outcome and must not decide
    good = dict(method="temperature", T=0.5, ece_before=0.01, ece_after=0.05, val=dict(ece_before=0.05, ece_after=0.01))
    st = stats.compute(sc, pred, der, [True, True, True], 30.0, good)
    assert abs(st["confidence"] - p.max(0).mean()) < 1e-6 and abs(st["confidence_calibrated"] - pc.max(0).mean()) < 1e-6
    assert st["calibration"]["applied"] is True and st["calibration"]["decision_split"] == "val64"
    bad = dict(good, ece_before=0.05, ece_after=0.01, val=dict(ece_before=0.01, ece_after=0.05))   # test improves, val does not
    st = stats.compute(sc, pred, der, [True, True, True], 30.0, bad)
    assert st["confidence_calibrated"] is None and st["calibration"]["applied"] is False
    assert stats.calibration_improves(dict(ece_before=0.05, ece_after=0.01)) is False      # test-only numbers never apply
    assert stats.calibration_improves(dict(good, apply=False)) is False                     # explicit decision wins
    st = stats.compute(sc, pred, der, [True, True, True], None, None)
    assert st["area_km2"] is None and st["gsd_assumed"] and st["calibration"] is None


def test_metrics_robustness_has_gated_rows_and_stress():
    p = os.path.join(ROOT, "artifacts", "metrics.json")
    if not os.path.exists(p):
        pytest.skip("metrics.json missing")
    r = json.load(open(p))["robustness"]["finetuned_v2"]
    names = {x["name"] for x in r["stress"]}
    assert {"elevation +3000 m", "SAR +6 dB", "SAR -6 dB", "optical x0.6", "optical x1.5"} <= names
    assert all(x["kind"] == "in_range" for x in r["rows"]) and all(x["kind"] == "stress" for x in r["stress"])
    assert r["all_within_gate"] == all(x["within_gate"] for x in r["rows"])


def test_cache_remove_scene(tmp_path):
    rc = ResultCache(str(tmp_path))
    rc.put("a1", {"rgb": b"x"}, None, {}, key="k1", scene_id="upX")
    rc.put("a2", {"rgb": b"y"}, None, {}, key="k2", scene_id="upY")
    assert rc.remove_scene("upX") == ["a1"]
    assert "a1" not in rc and rc.lookup("k1") is None and "a2" in rc
    assert "a2" in ResultCache(str(tmp_path)) and ResultCache(str(tmp_path)).remove_scene("upY") == ["a2"]
