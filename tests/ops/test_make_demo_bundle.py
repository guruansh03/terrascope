import json
import zipfile

import numpy as np
import pytest

import make_demo_bundle as mdb


@pytest.fixture(scope="module")
def dry_bundle(tmp_path_factory):
    import os
    t = tmp_path_factory.mktemp("bundle")
    os.environ["TERRASCOPE_LOG_DIR"] = str(t / "logs")
    rc = mdb.main(["--dry-run", "--no-real", "--out-dir", str(t / "dist"), "--demo-dir", str(t / "demo"),
                   "--onnx-dir", str(t / "onnx"), "--keep-stage"])
    return rc, t


def test_dry_run_produces_valid_zip(dry_bundle):
    rc, t = dry_bundle
    assert rc == 0
    z = t / "dist" / "terrascope_demo.zip"
    assert z.exists() and mdb.validate_zip(z) == []
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        man = json.loads(f.read("terrascope_demo/BUNDLE_MANIFEST.json"))
        bat = f.read("terrascope_demo/start_local.bat")
    assert all(n.startswith("terrascope_demo/") for n in names)
    assert sum(n.startswith("terrascope_demo/data/demo/sim_") and n.endswith(".npz") for n in names) == len(mdb.SIM_PRESETS)
    assert not any("/.venv" in n or "__pycache__" in n or "/data/real/" in n or "/logs/" in n for n in names)
    assert not any(n.endswith(".onnx") for n in names), "dry run exports nothing"
    assert b"\r\n" in bat
    assert man["dry_run"] is True and man["precompute"] == {"skipped": "dry run"}
    assert man["options"]["with_wheels"] is True and man["options"]["with_python"] is True  # default on
    assert len(man["scenes"]) == len(mdb.SIM_PRESETS)


def test_demo_scene_files_match_scene_store_format(dry_bundle):
    _, t = dry_bundle
    side = json.loads((t / "demo" / "sim_flash_flood.json").read_text())
    assert side["data_source"] == "simulated" and "name" in side
    with np.load(t / "demo" / "sim_flash_flood.npz", allow_pickle=False) as z:
        assert z["optical"].shape[0] == 6 and z["sar"].shape[0] == 2 and z["dem"].shape[0] == 2
        assert set(z.files) <= {"optical", "sar", "dem", "optical_clean", "lc", "flood", "veg", "burn", "cloud", "s2_full"}


def test_demo_scenes_are_listed_by_the_scene_store(dry_bundle):
    _, t = dry_bundle
    pytest.importorskip("terrascope.real.scenes")
    from terrascope.real.scenes import SceneStore
    st = SceneStore(scenes_dir=str(t / "scenes"), demo_dir=str(t / "demo"))
    ls = st.list()
    assert {s["id"] for s in ls} >= {"demo-sim_flash_flood", "demo-sim_clear_day"}
    assert all(s["source"] == "demo" for s in ls)


def test_validate_zip_reports_problems(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("terrascope_demo/start_local.bat", "@echo off\nREM lf only\n")
        f.writestr("terrascope_demo/.venv/x.txt", "x")
        f.writestr("evil/../x", "x")
    probs = mdb.validate_zip(z)
    text = " ".join(probs)
    assert "missing requirements-demo.txt" in text and "no demo scene" in text
    assert "CRLF" in text and ".venv" in text and "unsafe path" in text and "TerraFM checkpoint" in text


def test_convert_real_crop_and_truth():
    rng = np.random.default_rng(0)
    s = {"optical": rng.random((6, 512, 512), np.float32), "sar": rng.random((2, 512, 512), np.float32), "dem": None,
         "mask": np.where(rng.random((512, 512)) < 0.3, 1, 0)}
    s["mask"][0, :] = -1  # no-data row outside the centre crop
    c = mdb.convert_real("sen1floods11", "flood", {"id": "x"}, s, "s2", 256)
    a = c["arrays"]
    assert a["optical"].shape == (6, 256, 256) and "dem" not in a and a["flood"].dtype == np.uint8
    assert set(np.unique(a["flood"])) <= {0, 1} and c["invalid_frac"] == 0.0 and c["crop"] == [128, 128, 256]
    s["mask"][256, :] = -1
    c = mdb.convert_real("worldcover", "lc", {"id": "x"}, s, "s2", 256)
    assert c["invalid_frac"] == pytest.approx(1 / 256) and c["arrays"]["lc"].min() >= 0


def test_onnx_coverage():
    models = [{"id": "terrafm", "family": "terrafm", "status": "ready"},
              {"id": "a", "family": "prithvi", "status": "ready", "tasks": ["flood", "burn"]},
              {"id": "b", "family": "terramind", "status": "ready", "tasks": ["flood"]}]
    ex = [{"model": "a", "task": "flood", "status": "ok"}, {"model": "a", "task": "burn", "status": "error"},
          {"model": "b", "task": "flood", "status": "ok"}]
    assert mdb.onnx_coverage(models, ex) == {"a": False, "b": True}


def test_boot_smoke_ran_and_passed(dry_bundle):
    _, t = dry_bundle
    log = next((t / "logs").glob("make_demo_bundle_*.txt")).read_text(encoding="utf-8")
    assert "boot smoke OK" in log and "JOB OK" in log
    assert not list((t / "dist").glob("boot smoke*")), "temp extraction removed"


def test_boot_smoke_fails_loudly_on_broken_bundle(tmp_path):
    import logging
    z = tmp_path / "broken.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("terrascope_demo/README.md", "no code here")
    with pytest.raises(mdb.JobError, match="cannot be imported"):
        mdb.boot_smoke(z, 8104, logging.getLogger("t"), timeout=60)


def test_wheels_and_python_default_on_and_can_be_disabled():
    import argparse
    src = (mdb.ROOT / "scripts" / "pc" / "make_demo_bundle.py").read_text()
    assert '"--no-wheels", dest="with_wheels", action="store_false"' in src
    assert '"--no-python", dest="with_python", action="store_false"' in src
