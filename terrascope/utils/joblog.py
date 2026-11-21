"""Job logging for the PC / laptop entry points (OPS).

Every long job (download, training, evaluation, bundle, tests) runs unattended on
the owner's Windows PC. Claude cannot see the terminal there, only the files the
job leaves behind, so every job must:

* print to the console AND write ``logs/<name>_<YYYYmmdd-HHMMSS>.txt``;
* start with an environment header (python / torch / CUDA / GPU / OS / disk);
* end with ``JOB OK`` or a ``JOB FAILED`` block that says, in plain words,
  WHAT failed, the LIKELY CAUSE and WHAT TO DO (which .bat to run next).

Usage in a script::

    from terrascope.utils.joblog import start_job, job_failed, job_done, run_job

    with run_job("train_all") as log:      # start_job + JOB OK / JOB FAILED + exit code 1
        log.info("training ...")

    # or manually
    log = start_job("get_data")
    try:
        ...
        job_done()
    except Exception as e:
        job_failed(e)
        sys.exit(1)

Scripts can raise :class:`JobError` (or :class:`MissingData`) to give their own
"what / cause / todo" text. Only the standard library is required; torch,
psutil and nvidia-smi are used when present.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import errno
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
TS_FORMAT = "%Y%m%d-%H%M%S"

__all__ = ["start_job", "job_failed", "job_done", "run_job", "diagnose", "diagnose_text",
           "env_info", "JobError", "MissingData", "current_log_path", "ROOT"]


class JobError(RuntimeError):
    """An error that already knows how to explain itself to the owner."""

    def __init__(self, what: str, cause: str = "", todo: str = ""):
        super().__init__(what)
        self.what, self.cause, self.todo = what, cause, todo


class MissingData(JobError, FileNotFoundError):
    """A required input file/folder is missing. ``bat`` = the .bat that creates it."""

    def __init__(self, path, bat: str = "", what: str = ""):
        self.path = str(path)
        todo = (f"Double-click {bat} first, wait for 'JOB OK' at the end of its log, then run this job again."
                if bat else "Create the missing input, then run this job again.")
        JobError.__init__(self, what or f"Required input not found: {self.path}",
                          "The step that produces this input has not been run yet or did not finish.", todo)


# --------------------------------------------------------------------------- state
class _State:
    logger: logging.Logger | None = None
    name: str | None = None
    path: Path | None = None
    t0: float = 0.0
    finished: bool = False


_S = _State()


def current_log_path() -> Path | None:
    return _S.path


def _log_dir() -> Path:
    d = Path(os.environ.get("TERRASCOPE_LOG_DIR") or (ROOT / "logs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_console():
    """Windows consoles may be cp1252: never crash on a character, replace it."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


# --------------------------------------------------------------------------- env info
def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def env_info(probe_torch: bool = True) -> dict:
    """Machine facts for the log header (cheap; never raises)."""
    info: dict = {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "cwd": os.getcwd(),
        "argv": sys.argv,
        "cpu_count": os.cpu_count(),
    }
    try:
        du = shutil.disk_usage(ROOT)
        info["disk_free_gb"] = round(du.free / 1e9, 1)
        info["disk_total_gb"] = round(du.total / 1e9, 1)
    except Exception:
        pass
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1e9, 1)
        info["ram_available_gb"] = round(vm.available / 1e9, 1)
    except Exception:
        pass
    if probe_torch:
        try:
            import torch  # type: ignore
            info["torch"] = torch.__version__
            info["torch_cuda_build"] = torch.version.cuda
            ok = bool(torch.cuda.is_available())
            info["cuda_available"] = ok
            if ok:
                i = torch.cuda.current_device()
                p = torch.cuda.get_device_properties(i)
                info["gpu"] = p.name
                info["gpu_vram_gb"] = round(p.total_memory / 1e9, 2)
                info["gpu_compute_capability"] = f"{p.major}.{p.minor}"
                info["torch_arch_list"] = torch.cuda.get_arch_list()
        except Exception as e:  # torch missing or broken: that is information too
            info["torch"] = f"not importable ({type(e).__name__}: {e})"
    smi = shutil.which("nvidia-smi")
    if smi:
        q = _run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        if q:
            info["nvidia_smi"] = q.splitlines()[0]
    return info


# --------------------------------------------------------------------------- start / end
def start_job(name: str, install_excepthook: bool = True, probe_torch: bool = True) -> logging.Logger:
    """Create the job logger (console + ``logs/<name>_<ts>.txt``) and write the env header."""
    _safe_console()
    d = _log_dir()
    ts = _dt.datetime.now().strftime(TS_FORMAT)
    path = d / f"{name}_{ts}.txt"
    k = 1
    while path.exists():
        path = d / f"{name}_{ts}_{k}.txt"
        k += 1

    lg = logging.getLogger(f"terrascope.job.{name}")
    for h in list(lg.handlers):
        lg.removeHandler(h)
        h.close()
    lg.setLevel(logging.INFO)
    lg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    lg.addHandler(fh)
    lg.addHandler(ch)

    _S.logger, _S.name, _S.path, _S.t0, _S.finished = lg, name, path, time.time(), False

    lg.info("=" * 72)
    lg.info("JOB START: %s", name)
    lg.info("log file : %s", path)
    for key, val in env_info(probe_torch).items():
        lg.info("%-17s: %s", key, val)
    lg.info("=" * 72)

    if install_excepthook:
        prev = sys.excepthook

        def _hook(tp, val, tb):
            if _S.logger is not None and not _S.finished and not issubclass(tp, SystemExit):
                job_failed(val, tb=tb)
            else:
                prev(tp, val, tb)

        sys.excepthook = _hook
    return lg


def _elapsed() -> str:
    s = int(time.time() - _S.t0) if _S.t0 else 0
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def job_done(summary: str = "") -> None:
    lg = _S.logger or logging.getLogger("terrascope.job")
    if summary:
        lg.info(summary)
    lg.info("=" * 72)
    lg.info("JOB OK: %s  (elapsed %s)", _S.name, _elapsed())
    if _S.path:
        lg.info("log saved to %s", _S.path)
    lg.info("=" * 72)
    _S.finished = True
    for h in lg.handlers:
        h.flush()


# --------------------------------------------------------------------------- diagnosis
_NET_WORDS = ("getaddrinfo failed", "name or service not known", "temporary failure in name resolution",
              "failed to establish a new connection", "connection refused", "connection reset",
              "connection aborted", "timed out", "read timed out", "max retries exceeded",
              "proxyerror", "sslerror", "certificate verify failed", "remote end closed connection",
              "network is unreachable", "no route to host", "http error 403", "403 forbidden",
              "http error 429", "too many requests", "couldn't connect to server", "nodename nor servname")
_DISK_WORDS = ("no space left on device", "not enough space on the disk", "errno 28", "winerror 112",
               "disk quota exceeded")


def _bat_for_path(p: str) -> str:
    q = p.replace("\\", "/").lower()
    if "data/real" in q or "/datasets" in q:
        return "get_data.bat"
    if "data/demo" in q:
        return "make_demo_bundle.bat"
    if "ckpt/real" in q or "metrics_real" in q:
        return "train.bat (then evaluate.bat)"
    if "artifacts/onnx" in q:
        return "make_demo_bundle.bat"
    if "ckpt" in q:
        return "train.bat"
    if ".venv" in q:
        return "setup.bat"
    return ""


def diagnose_text(text: str) -> tuple[str, str] | None:
    """Classify free text (e.g. pip / subprocess output). Returns (cause, todo) or None."""
    t = (text or "").lower()
    if any(w in t for w in _DISK_WORDS):
        return ("The disk is full.",
                f"Free space on the drive that holds {ROOT} (empty the Recycle Bin, delete old files in "
                "data\\real\\_tmp or dist\\), then run the same .bat again; downloads resume.")
    if "cuda out of memory" in t or "outofmemoryerror" in t:
        return ("The GPU ran out of its 8 GB of memory.",
                "Close other programs that use the GPU (games, browsers with video), then run again. If it "
                "repeats, run 'train.bat quick' or lower the batch size in the training log's config.")
    if "no kernel image is available" in t or "sm_120 is not compatible" in t:
        return ("The installed PyTorch was not built for the RTX 50xx (Blackwell, sm_120) GPU.",
                "Run setup.bat again: it installs PyTorch from the CUDA 12.8 index "
                "(https://download.pytorch.org/whl/cu128), which supports this GPU.")
    if "driver version is insufficient" in t or "found no nvidia driver" in t or "nvidia driver on your system is too old" in t:
        return ("The NVIDIA driver is missing or too old for CUDA 12.8.",
                "Install the latest NVIDIA Game Ready / Studio driver (R570 or newer) from nvidia.com, reboot, "
                "then run gpu_check.bat.")
    if "torch not compiled with cuda" in t or ("cuda" in t and "not available" in t):
        return ("PyTorch cannot see the GPU (CPU-only PyTorch was installed, or the NVIDIA driver is missing).",
                "Run gpu_check.bat and read artifacts\\pc_info.json. If 'torch_cuda_build' is null, run "
                "setup.bat again (it reinstalls the CUDA 12.8 build). If nvidia-smi is missing, install the "
                "NVIDIA driver (R570 or newer) and reboot.")
    if "could not find a version that satisfies" in t or "no matching distribution" in t:
        return ("pip found no package build for this Python version (or the package index was unreachable).",
                "Use 64-bit Python 3.12 from python.org (setup.bat picks it via 'py -3.12'), check the "
                "internet connection, then run setup.bat again.")
    if any(w in t for w in _NET_WORDS):
        return ("A network download failed (no internet, firewall / antivirus / VPN / proxy blocking, or the "
                "server refused or rate-limited the request).",
                "Check that this PC is online (open the URL from the log in a browser), pause VPN, then run "
                "the same .bat again - downloads resume where they stopped.")
    if "access is denied" in t or "permission denied" in t or "winerror 5" in t or "winerror 32" in t:
        return ("Windows refused access to a file (open in another program, antivirus scan, or a OneDrive "
                "folder that is still syncing).",
                "Close programs that may hold the file (Excel, Explorer preview, another TerraScope window), "
                "then run again. Avoid running the project from inside a syncing OneDrive folder.")
    return None


def diagnose(exc: BaseException) -> dict:
    """Map an exception to {what, cause, todo} in plain language."""
    name = type(exc).__name__
    msg = str(exc)
    what = f"{name}: {msg}" if msg else name
    if isinstance(exc, JobError):
        return {"what": exc.what, "cause": exc.cause or "see the traceback above",
                "todo": exc.todo or "Send this log file to Claude."}
    if isinstance(exc, KeyboardInterrupt):
        return {"what": "The job was stopped (Ctrl+C or the window was closed).",
                "cause": "Stopped by the user.", "todo": "Run the same .bat again when ready; finished parts are kept."}
    if isinstance(exc, MemoryError):
        return {"what": what, "cause": "The computer ran out of RAM.",
                "todo": "Close other programs (browsers, games) and run again; use the 'quick' mode if available."}
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        mod = getattr(exc, "name", None) or msg
        return {"what": what, "cause": f"The Python package '{mod}' is not installed in the virtual environment.",
                "todo": "Double-click setup.bat (PC) or delete the .venv-demo folder and run start_local.bat "
                        "(laptop) to reinstall the packages, then run this job again."}
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112:
        c, t = diagnose_text("no space left on device")  # type: ignore[misc]
        return {"what": what, "cause": c, "todo": t}
    if isinstance(exc, FileNotFoundError):
        p = exc.filename or msg
        bat = _bat_for_path(str(p))
        return {"what": f"File or folder not found: {p}",
                "cause": "An earlier step has not been run yet, did not finish, or the file was moved.",
                "todo": (f"Run {bat} first (wait for 'JOB OK' in its log), then run this job again."
                         if bat else "Check the path above; run the previous step in LOCAL_GUIDE.md first.")}
    text = f"{name} {msg}"
    cls_net = name in ("URLError", "HTTPError", "ConnectionError", "ConnectTimeout", "ReadTimeout", "ProxyError",
                       "SSLError", "ConnectError", "RemoteDisconnected", "gaierror", "TimeoutError",
                       "IncompleteRead", "ChunkedEncodingError")
    hit = diagnose_text(text)
    if hit is None and cls_net:
        hit = diagnose_text("connection refused")
    if hit is not None:
        return {"what": what, "cause": hit[0], "todo": hit[1]}
    if isinstance(exc, PermissionError):
        c, t = diagnose_text("permission denied")  # type: ignore[misc]
        return {"what": what, "cause": c, "todo": t}
    return {"what": what, "cause": "Not recognised automatically (a bug or an unexpected input).",
            "todo": "Send this log file to Claude; do not delete data or checkpoints."}


def job_failed(exc: BaseException, hint: str | None = None, tb=None) -> str:
    """Log the traceback and a human-readable WHAT / LIKELY CAUSE / WHAT TO DO block. Returns the block."""
    lg = _S.logger or logging.getLogger("terrascope.job")
    tb = tb if tb is not None else exc.__traceback__
    tb_text = "".join(traceback.format_exception(type(exc), exc, tb)).rstrip()
    d = diagnose(exc)
    if hint:
        d["todo"] = f"{hint} {d['todo']}"
    block = "\n".join([
        "#" * 72,
        f"JOB FAILED: {_S.name or '?'}  (after {_elapsed()})",
        f"WHAT FAILED : {d['what']}",
        f"LIKELY CAUSE: {d['cause']}",
        f"WHAT TO DO  : {d['todo']}",
        f"LOG FILE    : {_S.path}" if _S.path else "LOG FILE    : (console only)",
        "#" * 72,
    ])
    try:
        lg.error("traceback:\n%s", tb_text)
        lg.error("\n%s", block)
        for h in lg.handlers:
            h.flush()
    except Exception:
        print(tb_text, block, sep="\n", file=sys.stderr)
    _S.finished = True
    return block


@contextlib.contextmanager
def run_job(name: str, exit_on_error: bool = True, **kw) -> Iterator[logging.Logger]:
    """``with run_job("x") as log:`` -> JOB OK on success; JOB FAILED + exit code 1 on error."""
    lg = start_job(name, **kw)
    try:
        yield lg
    except SystemExit as e:
        if e.code not in (0, None):
            job_failed(JobError(f"the job exited with code {e.code}", "see the messages above", "Send this log file to Claude."))
        else:
            job_done()
        raise
    except BaseException as e:  # noqa: BLE001 - we want KeyboardInterrupt too
        job_failed(e)
        if exit_on_error:
            sys.exit(1)
        raise
    else:
        if not _S.finished:
            job_done()
