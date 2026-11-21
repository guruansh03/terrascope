import importlib.util, json, os

import numpy as np
import pytest
import torch

from fm_helpers import fake_records, load_fake, ROOT
from terrascope.fm import finetune as ft
from terrascope.fm.model import ckpt_path, load_checkpoint


def _ft(model_id, task, root, seed=0, epochs=2, **kw):
    return ft.finetune(model_id, task, seed, epochs=epochs, patience=5, samples_per_epoch=4, batch_size=2,
                       records=fake_records(task), load_fn=load_fake, out_root=root, require_weights=False,
                       num_workers=0, crop=32, device="cpu", **kw)


@pytest.mark.parametrize("model_id,task", [("terramind_tiny", "flood"), ("prithvi_tiny", "cloud"),
                                           ("terrafm", "landcover")])
def test_two_step_finetune_writes_checkpoint_and_log(fm_env, model_id, task):
    s = _ft(model_id, task, fm_env["ckpt"])
    p = ckpt_path(task, model_id, 0, fm_env["ckpt"])
    assert os.path.exists(p) and not os.path.exists(p + ".last.pt")
    js = json.load(open(os.path.splitext(p)[0] + ".json"))
    assert js["complete"] and js["epochs_run"] == 2 and len(js["history"]) == 2
    assert js["split_sizes"] == {"train": 4, "val": 2, "test": 2}
    assert ft.is_complete(task, model_id, 0, fm_env["ckpt"])
    m = load_checkpoint(p)
    assert m.task == task and m.n_params() == s["params"]


def test_frozen_encoder_checkpoint_needs_weights(fm_env):
    _ft("prithvi_tiny", "flood", fm_env["ckpt"], freeze_encoder=True, epochs=1)
    p = ckpt_path("flood", "prithvi_tiny", 0, fm_env["ckpt"])
    ck = torch.load(p, weights_only=False)
    assert ck["encoder"] is None and ck["encoder_frozen"]
    with pytest.raises(FileNotFoundError):
        load_checkpoint(p)


def test_missing_weights_is_explicit(fm_env):
    with pytest.raises(ft.MissingWeights, match="get_data.bat weights"):
        ft.finetune("prithvi_tiny", "flood", 0, epochs=1, records=fake_records("flood"), load_fn=load_fake,
                    out_root=fm_env["ckpt"], num_workers=0, device="cpu")


def test_resume_after_interruption(fm_env, monkeypatch):
    calls = {"n": 0}
    real = ft.evaluate_model

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("simulated power cut")
        return real(*a, **k)

    monkeypatch.setattr(ft, "evaluate_model", flaky)
    with pytest.raises(KeyboardInterrupt):
        _ft("terramind_tiny", "burn", fm_env["ckpt"], epochs=3)
    p = ckpt_path("burn", "terramind_tiny", 0, fm_env["ckpt"])
    assert os.path.exists(p + ".last.pt") and not ft.is_complete("burn", "terramind_tiny", 0, fm_env["ckpt"])
    monkeypatch.setattr(ft, "evaluate_model", real)
    s = _ft("terramind_tiny", "burn", fm_env["ckpt"], epochs=3)
    assert s["epochs_run"] == 3 and [h["epoch"] for h in s["history"]] == [0, 1, 2]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "pc", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_evaluate_real_schema(fm_env):
    for seed in (0, 1):
        _ft("prithvi_tiny", "flood", fm_env["ckpt"], seed=seed, epochs=1)
    _ft("terramind_tiny", "flood", fm_env["ckpt"], seed=0, epochs=1)
    ev = _load_script("evaluate_real")
    recs = dict(fake_records("flood"), load_fn=load_fake)
    out = os.path.join(fm_env["tmp"], "metrics_real.json")
    r = ev.evaluate(tasks=["flood"], ckpt_root=fm_env["ckpt"], records_override={"flood": recs}, out=out, device="cpu")
    js = json.load(open(out))
    assert js == json.loads(json.dumps(r))
    assert {"generated_at", "device", "datasets"} <= set(js) and js["device"] == "cpu"
    ds = js["datasets"]["sen1floods11"]
    assert ds["task"] == "flood" and ds["split_sizes"] == {"train": 4, "val": 2, "test": 2}
    pm = ds["models"]["prithvi_tiny"]
    assert isinstance(pm["params"], int) and pm["params"] > pm["encoder_params"] > 5_000_000
    assert pm["seeds"] == [0, 1]
    for k in ("flood_iou", "accuracy", "infer_ms_per_chip"):
        assert set(pm[k]) >= {"mean", "std"}
        assert pm[k]["mean"] is not None and pm[k]["std"] is not None  # 2 seeds (or many chips) -> real std
    assert ds["models"]["terramind_tiny"]["seeds"] == [0] and ds["models"]["terramind_tiny"]["flood_iou"]["std"] is None
    assert "sar_vv_threshold_-15dB" in ds["baselines"]
    assert any("prithvi_100m" in m for m in js["missing"])


def test_train_all_reports_missing_data(fm_env, monkeypatch, tmp_path):
    monkeypatch.setenv("TERRASCOPE_DATA", str(tmp_path / "nodata"))
    monkeypatch.setenv("TERRASCOPE_LOG_DIR", str(tmp_path / "logs"))
    ta = _load_script("train_all")
    with pytest.raises(SystemExit) as e:
        ta.main(["quick", "--dry-run"])
    assert e.value.code == 1
    log = open(next((tmp_path / "logs").glob("train_quick_*.txt"))).read()
    assert "get_data.bat floods" in log and "JOB FAILED" in log


def test_get_data_weights_step_and_manifest(fm_env, monkeypatch, tmp_path):
    monkeypatch.setenv("TERRASCOPE_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("TERRASCOPE_LOG_DIR", str(tmp_path / "logs"))
    from terrascope.fm import weights

    def fake_fetch(mid, logger=None):
        if mid == "unet_r18_scratch":
            return None  # mirrors the real fetch_weights: nothing to download
        p = tmp_path / "w" / f"{mid}.pt"
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x" * 10)
        return str(p)

    monkeypatch.setattr(weights, "fetch_weights", fake_fetch)
    gd = _load_script("get_data")
    gd.main(["weights"])
    man = json.load(open(tmp_path / "data" / "manifest.json"))
    assert man["weights"]["ok"] and set(man["weights"]["result"]) == {"prithvi_tiny", "terramind_tiny", "prithvi_100m",
                                                                  "unet_r18_imagenet"}
    log = open(next((tmp_path / "logs").glob("get_data_weights_*.txt"))).read()
    assert "JOB OK" in log


# ------------------------------------------------------------------------------------------ speed / robustness fixes
def test_disk_and_memory_chip_caches(tmp_path):
    from terrascope.fm.data import ChipDataset, DiskChipCache, MemoryChipCache
    calls = []

    def counting(rec):
        calls.append(rec["id"])
        return load_fake(rec)

    recs = fake_records("flood")["train"]
    dc = DiskChipCache(counting, str(tmp_path / "cache"))
    a = dc(recs[0])
    b = dc(recs[0])
    assert calls == ["c0"] and isinstance(b["optical"], np.memmap) and b["sar"] is not None
    np.testing.assert_allclose(np.asarray(b["optical"], np.float32), load_fake(recs[0])["optical"], atol=1e-3)
    ds = ChipDataset(recs, dc, ["optical", "sar", "dem"], crop=32, train=True)
    item = ds[1]
    assert item["optical"].shape == (6, 32, 32) and item["mask"].dtype == torch.int64
    mc = MemoryChipCache(counting)
    mc(recs[2]); mc(recs[2])
    assert calls.count("c2") == 1


def test_loader_built_once_and_oom_halves_batch(fm_env, monkeypatch, tmp_path):
    n_loaders = {"k": 0}
    real_loader = ft._loader

    def counting_loader(*a, **k):
        n_loaders["k"] += 1
        return real_loader(*a, **k)

    real_loss = ft.task_loss
    state = {"raised": False}

    def oom_once(task, out, batch):
        if not state["raised"]:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_loss(task, out, batch)

    monkeypatch.setattr(ft, "_loader", counting_loader)
    monkeypatch.setattr(ft, "task_loss", oom_once)
    s = _ft("prithvi_tiny", "burn", fm_env["ckpt"], epochs=3, cache_dir=str(tmp_path / "cc"))
    assert s["batch_size"] == 1 and s["epochs_run"] == 3
    assert n_loaders["k"] == 2  # once for the run + once after the OOM batch-size change (not once per epoch)
    assert os.listdir(tmp_path / "cc")  # chips decoded once into the disk cache


def test_mean_std_single_value_has_null_std():
    from terrascope.fm.metrics import mean_std
    assert mean_std([0.5]) == dict(mean=0.5, std=None, n=1)
    assert mean_std([None]) == dict(mean=None, std=None, n=0)
    assert mean_std([1.0, 3.0])["std"] == pytest.approx(2 ** 0.5)


def test_quick_mode_estimate_within_budget():
    ta = _load_script("train_all")
    from terrascope.fm.specs import MODEL_ORDER
    q = ta.MODES["quick"]
    est = ta.estimate_minutes(MODEL_ORDER, q["tasks"], q["seeds"], q["epochs"], q["samples_per_epoch"])
    assert est <= 90, est
