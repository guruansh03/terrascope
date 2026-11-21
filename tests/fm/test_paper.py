"""Paper revision: U-Net baselines, paper training mode, seed/bootstrap/paired statistics (fixtures only)."""
import json, os

import numpy as np
import pytest
import torch

from fm_helpers import fake_records, load_fake, write_ckpt
from terrascope.fm import finetune as ft
from terrascope.fm import registry, stats
from terrascope.fm.model import build_task_model, ckpt_path, load_checkpoint
from test_finetune_eval import _ft, _load_script


# ------------------------------------------------------------------------------------------ U-Net baselines
@pytest.mark.parametrize("task,channels", [("flood", 10), ("burn", 6)])
def test_unet_input_channels_match_terrafm(task, channels):
    m = build_task_model("unet_r18_scratch", task).eval()
    assert m.encoder.enc.conv1.weight.shape[1] == channels
    o, s, d = torch.rand(1, 6, 64, 64), torch.randn(1, 2, 64, 64) - 12, torch.rand(1, 2, 64, 64) * 50
    with torch.no_grad():
        y = m(o, s, d, torch.ones(1, 3))
    assert y.shape == (1, 1, 64, 64)
    assert m.n_params() == m.encoder.encoder_params()  # head-free: smp decoder IS the head
    assert m.n_params() == {"flood": 14_350_161, "burn": 14_337_617}[task]


def test_imagenet_first_conv_adaptation(tmp_path):
    import segmentation_models_pytorch as smp
    from safetensors.torch import save_file
    from terrascope.fm.baselines import adapt_first_conv
    ref = smp.Unet("resnet18", encoder_weights=None, in_channels=3, classes=1).encoder  # stand-in for ImageNet
    sd = {k: v.clone().contiguous() for k, v in ref.state_dict().items()}
    sd["fc.weight"], sd["fc.bias"] = torch.zeros(1000, 512), torch.zeros(1000)  # like the real file
    f = str(tmp_path / "model.safetensors")
    save_file(sd, f)
    m = build_task_model("unet_r18_imagenet", "flood", weights_path=f)
    w = m.encoder.enc.conv1.weight.detach()
    rgb = sd["conv1.weight"]
    C = 10
    np.testing.assert_allclose(w[:, 2], rgb[:, 0] * 3 / C, rtol=1e-6)   # B4 (red)   <- R
    np.testing.assert_allclose(w[:, 1], rgb[:, 1] * 3 / C, rtol=1e-6)   # B3 (green) <- G
    np.testing.assert_allclose(w[:, 0], rgb[:, 2] * 3 / C, rtol=1e-6)   # B2 (blue)  <- B
    np.testing.assert_allclose(w[:, 6], rgb.mean(1) * 3 / C, rtol=1e-5)  # VV <- mean RGB
    torch.testing.assert_close(m.encoder.enc.layer4[1].conv2.weight, sd["layer4.1.conv2.weight"])
    assert adapt_first_conv(rgb, ["B4", "B3", "B2"]).allclose(rgb)  # identity for a true RGB input


def test_unet_registry_finetune_runner_onnx(fm_env):
    st = {m["id"]: m for m in registry.list_models()}
    assert st["unet_r18_scratch"]["status"] == "not_finetuned" and st["unet_r18_imagenet"]["status"] == "weights_missing"
    s = _ft("unet_r18_scratch", "flood", fm_env["ckpt"], epochs=2)  # no weights needed
    assert s["complete"] and s["params"] == 14_350_161
    with pytest.raises(ft.MissingWeights):
        ft.finetune("unet_r18_imagenet", "flood", 0, epochs=1, records=fake_records("flood"), load_fn=load_fake,
                    out_root=fm_env["ckpt"], num_workers=0, device="cpu")
    registry.reset()
    r = registry.get_runner("unet_r18_scratch")
    stages = []
    sc = load_fake(dict(task="flood", i=3, H=96, W=80))
    out = r.predict(sc, [True, True, True], on_stage=stages.append)
    assert stages == ["encoding", "decoding"] and out["flood_prob"].shape == (96, 80)
    assert out["feats"].shape[0] == out["feat_hw"][0] * out["feat_hw"][1]
    # ONNX: 6 encoder levels instead of 4 -> generic export path
    from terrascope.fm.export import OnnxTask, export_onnx
    import onnxruntime as ort
    meta = export_onnx("unet_r18_scratch", "flood", fm_env["onnx"], ckpt_root=fm_env["ckpt"])
    assert meta["n_levels"] == 6 and meta["required_sensors"] == ["optical"]
    d = os.path.join(fm_env["onnx"], "unet_r18_scratch")
    sess = [ort.InferenceSession(os.path.join(d, meta[k]), providers=["CPUExecutionProvider"]) for k in ("encoder", "head")]
    ot = OnnxTask(*sess, meta)
    m = load_checkpoint(ckpt_path("flood", "unet_r18_scratch", 0, fm_env["ckpt"]))
    o, s_, dd = torch.rand(1, 6, 224, 224) * 0.3, torch.randn(1, 2, 224, 224) - 12, torch.rand(1, 2, 224, 224) * 80
    a = torch.ones(1, 3)
    with torch.no_grad():
        ref = m(o, s_, dd, a)
    np.testing.assert_allclose(ot.decode(*ot.encode(o, s_, dd, a), o, a).numpy(), ref.numpy(), atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------------------------------------ statistics
def test_t_ci_and_single_seed():
    assert stats.t_ci95([0.5]) is None
    lo, hi = stats.t_ci95([0.4, 0.5, 0.6])
    assert lo == pytest.approx(0.5 - 4.302653 * 0.1 / 3 ** 0.5, abs=1e-5) and hi > 0.5


def test_bootstrap_recomputes_pooled_iou():
    # chip A: TP=90 FP=10 FN=0 ; chip B: TP=10 FP=0 FN=90 -> pooled IoU = 100/200 = 0.5 (mean of chip IoUs = 0.5)
    A = np.array([[100, 10], [0, 90]])
    B = np.array([[100, 0], [90, 10]])
    chips = [A, B]
    assert stats.metric_from_conf(A + B, "iou_pos") == pytest.approx(0.5)
    idx = stats.draws(2, 500, seed=0)
    lo, hi = stats.bootstrap_ci95(chips, "iou_pos", idx)
    assert lo == pytest.approx(0.1, abs=1e-6) and hi == pytest.approx(0.9, abs=1e-6)  # AA -> 0.9, BB -> 0.1
    same = stats.paired({0: chips}, {0: chips}, "iou_pos", idx)
    assert same["mean_diff"] == 0 and same["bootstrap_ci95"] == [0.0, 0.0] and same["p_bootstrap_two_sided"] == 1.0


def test_evaluate_real_paper_statistics(fm_env):
    for seed in (0, 1, 2):
        _ft("terrafm", "flood", fm_env["ckpt"], seed=seed, epochs=1)
        _ft("unet_r18_scratch", "flood", fm_env["ckpt"], seed=seed, epochs=1)
    _ft("prithvi_tiny", "flood", fm_env["ckpt"], seed=0, epochs=1)  # single seed must still work
    ev = _load_script("evaluate_real")
    recs = dict(fake_records("flood", n_test=4), load_fn=load_fake)
    out = os.path.join(fm_env["tmp"], "metrics_real.json")
    ev.evaluate(tasks=["flood"], ckpt_root=fm_env["ckpt"], records_override={"flood": recs}, out=out, device="cpu",
                boot_draws=200)
    ds = json.load(open(out))["datasets"]["sen1floods11"]
    t = ds["models"]["terrafm"]
    for k in ("params", "seeds", "flood_iou", "accuracy", "infer_ms_per_chip", "per_seed", "ci95_seeds", "bootstrap_ci95"):
        assert k in t
    assert t["seeds"] == [0, 1, 2] and t["flood_iou"]["n"] == 3 and t["flood_iou"]["std"] is not None
    lo, hi = t["ci95_seeds"]["flood_iou"]
    assert lo <= t["flood_iou"]["mean"] <= hi
    assert len(t["per_seed"]) == 3 and all(len(r["bootstrap_ci95"]["flood_iou"]) == 2 for r in t["per_seed"])
    p = ds["models"]["prithvi_tiny"]
    assert p["flood_iou"]["std"] is None and p["ci95_seeds"]["flood_iou"] is None
    assert p["per_seed"][0]["bootstrap_ci95"]["flood_iou"] is not None
    pr = ds["paired"]["vs_terrafm"]["unet_r18_scratch"]["flood_iou"]
    assert pr["seeds"] == [0, 1, 2] and len(pr["diff_per_seed"]) == 3 and pr["ci95_seeds"] is not None
    assert pr["mean_diff"] == pytest.approx(ds["models"]["unet_r18_scratch"]["flood_iou"]["mean"] - t["flood_iou"]["mean"])
    assert 0 <= pr["p_bootstrap_two_sided"] <= 1 and pr["bootstrap_ci95"][0] <= pr["bootstrap_ci95"][1]
    assert ds["paired"]["vs_prithvi_tiny"]["terrafm"]["flood_iou"]["seeds"] == [0]
    assert "bootstrap_ci95" in ds["baselines"]["sar_vv_threshold_-15dB"]


# ------------------------------------------------------------------------------------------ paper training mode
def test_paper_mode_reuses_quick_hyperparameters(fm_env, tmp_path):
    ta = _load_script("train_all")
    q, p = ta.MODES["quick"], ta.MODES["paper"]
    for k in ("epochs", "patience", "samples_per_epoch", "tasks"):
        assert q[k] == p[k]
    assert p["seeds"] == [0, 1, 2] and p["models"] == "all"
    # seed-0 run of quick is the SAME checkpoint path, so paper skips it
    write_ckpt(build_task_model("prithvi_tiny", "flood").eval(), fm_env["ckpt"], seed=0)
    assert ft.is_complete("flood", "prithvi_tiny", 0, fm_env["ckpt"])
    js = ckpt_path("flood", "prithvi_tiny", 0, fm_env["ckpt"]).replace(".pt", ".json")
    d = json.load(open(js))
    d["minutes"] = 7.5
    json.dump(d, open(js, "w"))
    todo = [("flood", "prithvi_tiny", 1), ("flood", "prithvi_tiny", 2), ("burn", "unet_r18_scratch", 0)]
    tot, n_meas = ta.remaining_estimate(todo, 10, 384, fm_env["ckpt"])
    assert n_meas == 2 and tot > 15.0  # 2 x measured 7.5 min + FLOP estimate for the U-Net run
    from terrascope.fm.specs import MODEL_ORDER
    est = ta.estimate_minutes(MODEL_ORDER, p["tasks"], p["seeds"], p["epochs"], p["samples_per_epoch"])
    assert est < 6 * 60


def test_unet_normalisation_equals_terrafm_v2():
    from terrascope.core.model import normalize
    from terrascope.fm.baselines import normalize_v2
    g = torch.Generator().manual_seed(0)
    o, s, d = torch.rand(2, 6, 64, 48, generator=g), torch.randn(2, 2, 64, 48, generator=g) - 12, \
        torch.rand(2, 2, 64, 48, generator=g) * 900
    torch.testing.assert_close(normalize_v2(o, s, d), normalize(o, s, d, "v2"))
