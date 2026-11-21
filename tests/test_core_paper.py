"""Paper-revision pipeline: pretraining ablation flags, U-Net baseline, resumable paper_sim grid, paper evaluation,
validation-based calibration decision, risk sensitivity. Runs on a tiny private dataset with a few steps."""
import json, os

import numpy as np
import pytest
import torch

import terrascope.core.data as D
import terrascope.core.train as T
from terrascope.core.model import load_checkpoint


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("tinydata")
    mp = pytest.MonkeyPatch()
    mp.setattr(D, "DATA_DIR", str(root))
    for k, (rng, size) in {"train64": (range(10_000, 10_016), 64), "val64": (range(800_000, 800_008), 64),
                           "test64": (range(900_000, 900_012), 64), "test128": (range(950_000, 950_008), 128)}.items():
        mp.setitem(D.SPLITS, k, (rng, size))
    D.ensure_data()
    yield root
    mp.undo()


@pytest.fixture
def dirs(tmp_path, monkeypatch, tiny_data):
    monkeypatch.setattr(D, "DATA_DIR", str(tiny_data))
    for k, v in {"train64": (range(10_000, 10_016), 64), "val64": (range(800_000, 800_008), 64),
                 "test64": (range(900_000, 900_012), 64), "test128": (range(950_000, 950_008), 128)}.items():
        monkeypatch.setitem(D.SPLITS, k, v)
    monkeypatch.setattr(T, "CKPT", str(tmp_path / "ckpt"))
    monkeypatch.setattr(T, "LOGS", str(tmp_path / "logs"))
    T.set_dirs(str(tmp_path / "ckpt"), str(tmp_path / "logs"))
    return tmp_path


def test_names_are_backward_compatible():
    assert T.pretrained_name("full", 0) == "pretrained" and T.pretrained_name("full", 2) == "pretrained_s2"
    assert T.finetuned_name("full", 0) == "finetuned" and T.finetuned_name("nodrop", 0) == "finetuned_nodrop_s0"
    assert T.pretrained_name("nomask", 1) == "pretrained_nomask_s1" and T.unet_name(0) == "unet_s0"
    assert T.variant_from_args(False, False, False) == "full"
    assert T.variant_from_args(True, False, False) == "nodrop" and T.variant_from_args(False, True, True) == "nocloud_nomask"
    with pytest.raises(ValueError):
        T.variant_flags("bogus")


def test_pretrain_batch_flags_change_only_their_component(tiny_data, monkeypatch):
    monkeypatch.setattr(D, "DATA_DIR", str(tiny_data))
    d = D.load_raw("train64")
    idx = torch.arange(8)
    out = {}
    for v in ("full", "nodrop", "nocloud", "nomask"):
        torch.manual_seed(0)
        ratios = [T.pretrain_batch(d, idx, T.variant_flags(v), 8)[3] for _ in range(20)]
        torch.manual_seed(0)
        out[v] = T.pretrain_batch(d, idx, T.variant_flags(v), 8) + (ratios,)
    x, tgt, av, _, r = out["full"]
    assert not torch.equal(x, tgt) and not av.all() and max(r) > 0          # clouds in input, dropout, masking
    assert torch.equal(out["nocloud"][0], out["nocloud"][1])                # encoder input = clean target
    assert torch.equal(out["nocloud"][1], tgt)                              # same batch (paired RNG)
    assert out["nodrop"][2].all() and torch.equal(out["nodrop"][0], x)
    assert max(out["nomask"][4]) == 0.0 and torch.equal(out["nomask"][0], x)


def test_unet_baseline_param_count_and_outputs():
    from terrascope.core.unet import UNet
    m = UNet(**{k: v for k, v in T.UNET_CFG.items()})
    n = sum(p.numel() for p in m.parameters())
    assert n == 1_931_561 and n > 686_281                                   # larger than TerraFM
    o = m.forward_tasks(torch.randn(2, 10, 64, 64), torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool))
    assert o["lc"].shape == (2, 6, 64, 64) and o["cloud"].shape == (2, 64, 64) and (o["veg"] >= 0).all()
    # missing sensors are zeroed: changing a disabled sensor's data must not change the output
    x = torch.randn(1, 10, 64, 64)
    av = torch.tensor([[True, False, True]])
    x2 = x.clone(); x2[:, 6:8] += 5
    m.eval()
    assert torch.allclose(m.forward_tasks(x, av)["lc"], m.forward_tasks(x2, av)["lc"])


def test_paper_sim_grid_resumes_and_paper_eval(dirs):
    argv = ["paper_sim", "--seeds", "0", "--pretrain-steps", "2", "--ft-steps", "2", "--labels", "8", "--threads", "2",
            "--device", "cpu", "--ckpt-dir", str(dirs / "ckpt"), "--log-dir", str(dirs / "logs")]
    T.main(argv)
    expect = ["pretrained", "finetuned", "pretrained_nodrop_s0", "finetuned_nodrop_s0", "pretrained_nocloud_s0",
              "finetuned_nocloud_s0", "pretrained_nomask_s0", "finetuned_nomask_s0", "scratch", "unet_s0"]
    for n in expect:
        assert os.path.exists(dirs / "ckpt" / f"{n}.pt"), n
    meta = torch.load(dirs / "ckpt" / "pretrained_nocloud_s0.pt", weights_only=True)["meta"]
    assert meta["variant"] == "nocloud" and meta["flags"] == dict(dropout=True, cloud=False, mask=True)
    um, ucfg = load_checkpoint(str(dirs / "ckpt" / "unet_s0.pt"))
    assert ucfg["arch"] == "unet" and type(um).__name__ == "UNet"
    times = json.load(open(dirs / "logs" / "paper_sim_times.json"))
    assert set(expect) <= set(times) and all(v["seconds"] >= 0 for v in times.values())
    mt = {n: os.path.getmtime(dirs / "ckpt" / f"{n}.pt") for n in expect}
    T.main(argv)                                                            # resume: nothing retrained
    assert all(os.path.getmtime(dirs / "ckpt" / f"{n}.pt") == t for n, t in mt.items())

    from terrascope.core.paper_eval import main as paper
    doc = paper(seeds=[0], n_boot=20, out=str(dirs / "paper.json"), device="cpu")
    assert doc["missing_checkpoints"] == [] and set(doc["aggregate"]) == {
        "finetuned_full", "finetuned_nodrop", "finetuned_nocloud", "finetuned_nomask", "scratch", "unet"}
    r = doc["per_seed"]["unet"]["0"]
    assert set(r["test64"]) == {"All sensors", "Optical only", "SAR only", "DEM only", "SAR + DEM", "Optical + SAR"}
    assert r["cloud_rmse"] is None and r["test128"]["n_scenes"] == 8                 # U-Net: no recon decoder
    assert doc["per_seed"]["scratch"]["0"]["test128"]["miou"] is not None          # 128 px for scratch too
    assert doc["per_seed"]["finetuned_full"]["0"]["cloud_rmse"] is not None
    b = doc["per_seed"]["finetuned_full"]["0"]["bootstrap"]["test64|All sensors"]["miou"]
    assert b["lo"] <= doc["per_seed"]["finetuned_full"]["0"]["test64"]["All sensors"]["miou"] <= b["hi"]
    a = doc["aggregate"]["finetuned_full"]["test64|All sensors|miou"]
    assert a["n"] == 1 and a["std"] is None and a["ci95"] is None                  # one seed: no spread claimed
    pc = doc["paired"]["finetuned_full_vs_scratch"]["metrics"]["test64|All sensors|miou"]
    assert pc["n"] == 1 and pc["n_positive"] in (0, 1)
    assert "stress" in doc["robustness"]["per_seed"]["0"]


def test_paper_eval_aggregation_math():
    from terrascope.core.paper_eval import agg, metrics_from_sums
    g = agg([0.70, 0.72, 0.74])
    assert abs(g["mean"] - 0.72) < 1e-12 and abs(g["std"] - 0.02) < 1e-12
    from scipy.stats import t
    h = t.ppf(0.975, 2) * 0.02 / np.sqrt(3)
    assert np.allclose(g["ci95"], [0.72 - h, 0.72 + h])
    conf = np.zeros(36); conf[0] = 5; conf[7] = 5                              # perfect on classes 0,1
    m = metrics_from_sums(conf, np.array(3.0), np.array(4.0), np.array(1.0), np.array(10.0))
    assert m["oa"] == 1 and abs(m["miou"] - 2 / 6) < 1e-12 and m["flood_iou"] == 0.75 and m["veg_mae"] == 0.1


def test_calibration_decision_is_made_on_validation(tiny_data, monkeypatch, tmp_path):
    import terrascope.core.calibrate as C
    monkeypatch.setattr(D, "DATA_DIR", str(tiny_data))
    torch.manual_seed(0)
    from terrascope.core.model import TerraFM
    m = TerraFM(**T.CFG)
    (tmp_path / "ckpt").mkdir()
    torch.save({"model": m.state_dict(), "cfg": T.CFG}, tmp_path / "ckpt" / "finetuned.pt")
    monkeypatch.setattr(C, "ROOT", str(tmp_path))
    monkeypatch.setattr(C, "OUT", str(tmp_path / "cal.json"))
    os.makedirs(tmp_path / "artifacts" / "ckpt")
    os.replace(tmp_path / "ckpt" / "finetuned.pt", tmp_path / "artifacts" / "ckpt" / "finetuned.pt")
    res = C.main()
    assert res["apply"] == (res["val"]["ece_after"] < res["val"]["ece_before"])
    assert res["decision"]["split"] == "val64"


def test_risk_sensitivity_consistent_with_app_formulas():
    from terrascope.core.engine.stats import RISK_COEF, risk_indices
    from terrascope.core.risk_sensitivity import analyse
    # default coefficients reproduce the documented formulas
    fl, dr, ur, comp = risk_indices(0.1, 0.01, 0.02, 0.3, 0.64, 0.05)
    assert abs(fl - 100 * (0.25 + 0.12 + 0.08)) < 1e-9 and abs(dr - 100 * 1.6 * 0.3 * 0.8) < 1e-9 and abs(ur - 25) < 1e-9
    assert abs(comp - (0.45 * fl + 0.40 * dr + 0.15 * ur)) < 1e-9
    rng = np.random.default_rng(0)
    inp = dict(flood_frac=rng.uniform(0, .4, 60), urban_fl=rng.uniform(0, .03, 60), crop_fl=rng.uniform(0, .05, 60),
               stress=rng.uniform(0, 1, 60), veg_frac=rng.uniform(0, 1, 60), urban_frac=rng.uniform(0, .3, 60))
    res = analyse(inp, draws=40)
    assert set(res["oat"]) == set(RISK_COEF)
    for v in res["oat"].values():
        assert -1 <= v["min_rho"] <= 1 and 0 <= v["max_band_change"] <= 1
    j = res["joint"]
    assert j["draws"] == 40 and j["rho"]["p5"] <= j["rho"]["median"] <= 1 and 0 <= j["band_change"]["median"] <= 1
    assert sum(res["default_band_counts"].values()) == 60
