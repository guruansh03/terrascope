import errno
import re
import os
import subprocess
import sys
import textwrap
import urllib.error
from pathlib import Path

import pytest

from terrascope.utils import joblog
from terrascope.utils.joblog import JobError, MissingData, diagnose, diagnose_text, job_done, job_failed, start_job

ROOT = Path(__file__).resolve().parents[2]


def _log_text():
    p = joblog.current_log_path()
    for h in joblog._S.logger.handlers:
        h.flush()
    return p, p.read_text(encoding="utf-8")


def test_start_job_writes_named_log_with_env_header(tmp_path):
    start_job("unit_x", install_excepthook=False)
    p, t = _log_text()
    assert p.parent == tmp_path / "logs"
    assert p.name.startswith("unit_x_") and p.suffix == ".txt"
    assert re.fullmatch(r"unit_x_\d{8}-\d{6}", p.stem)  # <name>_YYYYmmdd-HHMMSS
    for key in ("JOB START: unit_x", "python", "os", "disk_free_gb", "torch", "cuda_available"):
        assert key in t
    job_done("all fine")
    _, t = _log_text()
    assert "JOB OK: unit_x" in t and "all fine" in t


def test_two_jobs_same_second_do_not_overwrite():
    start_job("dup", install_excepthook=False)
    a = joblog.current_log_path()
    start_job("dup", install_excepthook=False)
    b = joblog.current_log_path()
    assert a != b and a.exists() and b.exists()


@pytest.mark.parametrize("exc, cause_word, todo_word", [
    (RuntimeError("Torch not compiled with CUDA enabled"), "GPU", "setup.bat"),
    (AssertionError("CUDA is not available"), "GPU", "gpu_check.bat"),
    (RuntimeError("CUDA error: no kernel image is available for execution on the device"), "Blackwell", "cu128"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "GPU ran out", "quick"),
    (OSError(errno.ENOSPC, "No space left on device"), "disk is full", "Free space"),
    (urllib.error.URLError("[Errno 11001] getaddrinfo failed"), "network", "online"),
    (ConnectionResetError("Connection reset by peer"), "network", "resume"),
    (FileNotFoundError(2, "No such file", "C:/x/data/real/sen1floods11/split.csv"), "not been run", "get_data.bat"),
    (FileNotFoundError(2, "No such file", "artifacts/ckpt/real/flood/prithvi_s0.pt"), "not been run", "train.bat"),
    (ModuleNotFoundError("No module named 'terratorch'", name="terratorch"), "terratorch", "setup.bat"),
    (MemoryError(), "RAM", "Close"),
    (PermissionError(13, "Permission denied", "x.json"), "refused access", "Close"),
    (KeyboardInterrupt(), "Stopped", "again"),
    (ValueError("weird"), "Not recognised", "Send this log"),
])
def test_diagnose_gives_cause_and_todo(exc, cause_word, todo_word):
    d = diagnose(exc)
    assert set(d) == {"what", "cause", "todo"}
    assert cause_word.lower() in d["cause"].lower(), d
    assert todo_word.lower() in d["todo"].lower(), d


def test_missing_data_names_the_bat():
    d = diagnose(MissingData("data/real/sen12mscr", "get_data.bat cloud"))
    assert "data/real/sen12mscr" in d["what"] and "get_data.bat cloud" in d["todo"]
    assert isinstance(MissingData("x"), FileNotFoundError)


def test_diagnose_text_for_pip_output():
    assert "Python" in diagnose_text("ERROR: Could not find a version that satisfies the requirement terratorch")[1]
    assert "disk" in diagnose_text("OSError: [Errno 28] No space left on device")[0]
    assert diagnose_text("Successfully installed numpy") is None


def test_job_failed_block_written_to_log():
    start_job("failing", install_excepthook=False)
    try:
        raise JobError("download broke", "server said 503", "run get_data.bat again")
    except JobError as e:
        block = job_failed(e)
    _, t = _log_text()
    for s in ("JOB FAILED: failing", "WHAT FAILED : download broke", "LIKELY CAUSE: server said 503",
              "WHAT TO DO  : run get_data.bat again", "Traceback"):
        assert s in t
    assert "LOG FILE" in block


def _run_py(code, tmp_path):
    env = {**os.environ, "TERRASCOPE_LOG_DIR": str(tmp_path / "sublogs"), "PYTHONPATH": str(ROOT)}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                          timeout=120)


def test_run_job_exit_code_and_log(tmp_path):
    r = _run_py("""
        from terrascope.utils.joblog import run_job
        with run_job("ctx", probe_torch=False) as log:
            log.info("working")
            raise FileNotFoundError(2, "missing", "data/real/hls_burn_scars")
    """, tmp_path)
    assert r.returncode == 1
    logs = list((tmp_path / "sublogs").glob("ctx_*.txt"))
    assert len(logs) == 1
    t = logs[0].read_text(encoding="utf-8")
    assert "JOB FAILED: ctx" in t and "get_data.bat" in t


def test_run_job_ok(tmp_path):
    r = _run_py("""
        from terrascope.utils.joblog import run_job
        with run_job("okjob", probe_torch=False) as log:
            log.info("x")
    """, tmp_path)
    assert r.returncode == 0
    t = next((tmp_path / "sublogs").glob("okjob_*.txt")).read_text(encoding="utf-8")
    assert "JOB OK: okjob" in t


def test_uncaught_exception_hook_writes_failure_block(tmp_path):
    r = _run_py("""
        from terrascope.utils.joblog import start_job
        start_job("hooked", probe_torch=False)
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
    """, tmp_path)
    assert r.returncode == 1
    t = next((tmp_path / "sublogs").glob("hooked_*.txt")).read_text(encoding="utf-8")
    assert "JOB FAILED: hooked" in t and "cu128" in t
