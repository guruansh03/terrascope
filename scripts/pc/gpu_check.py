"""PC readiness check (OPS): GPU, VRAM, CUDA, compute capability, disk, RAM, CPU -> artifacts/pc_info.json.

Run:  python scripts/pc/gpu_check.py [--strict] [--out artifacts/pc_info.json] [--need-disk-gb 30]
Works on CPU-only machines too (GPU fields are null and the check says why).
--strict: exit code 1 when CUDA training is not possible (used by gpu_check.bat on the PC).
"""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrascope.utils.joblog import JobError, job_done, job_failed, start_job  # noqa: E402

# Default free-disk requirement is the planning assumption from LOCAL_GUIDE.md (an estimate, not a measurement).
DEFAULT_NEED_DISK_GB = 30.0


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg  # type: ignore
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> dict:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {"total_gb": round(vm.total / 1e9, 2), "available_gb": round(vm.available / 1e9, 2), "source": "psutil"}
    except Exception:
        pass
    if sys.platform == "win32":
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):  # type: ignore[attr-defined]
            return {"total_gb": round(m.ullTotalPhys / 1e9, 2), "available_gb": round(m.ullAvailPhys / 1e9, 2), "source": "win32"}
    try:
        kv = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.split()[1:2]}
        return {"total_gb": round(kv["MemTotal"] * 1024 / 1e9, 2),
                "available_gb": round(kv.get("MemAvailable", 0) * 1024 / 1e9, 2), "source": "/proc/meminfo"}
    except Exception:
        return {"total_gb": None, "available_gb": None, "source": "unknown"}


def nvidia_smi() -> dict | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total,memory.free",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return {"error": (r.stderr or r.stdout).strip()[:300]}
        f = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": f[0], "driver_version": f[1], "memory_total_mb": float(f[2]), "memory_free_mb": float(f[3])}
    except Exception as e:  # pragma: no cover - depends on the driver
        return {"error": f"{type(e).__name__}: {e}"}


def torch_info() -> dict:
    out: dict = {"importable": False}
    try:
        import torch  # type: ignore
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out.update(importable=True, version=torch.__version__, cuda_build=torch.version.cuda,
               cudnn=(torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None),
               cuda_available=bool(torch.cuda.is_available()), gpus=[])
    if not out["cuda_available"]:
        return out
    out["arch_list"] = torch.cuda.get_arch_list()
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        cc = f"{p.major}.{p.minor}"
        g = {"index": i, "name": p.name, "vram_total_gb": round(p.total_memory / 1e9, 2),
             "compute_capability": cc, "sm_in_torch_build": f"sm_{p.major}{p.minor}" in out["arch_list"]}
        try:
            free, total = torch.cuda.mem_get_info(i)
            g["vram_free_gb"] = round(free / 1e9, 2)
        except Exception:
            g["vram_free_gb"] = None
        # Run a real kernel: catches "no kernel image is available" (wheel built without sm_120).
        try:
            a = torch.randn(512, 512, device=f"cuda:{i}")
            v = float((a @ a).sum().item())
            torch.cuda.synchronize(i)
            g["kernel_test"] = "ok" if v == v else "nan"
        except Exception as e:
            g["kernel_test"] = f"FAILED {type(e).__name__}: {str(e)[:200]}"
        out["gpus"].append(g)
    return out


def collect(need_disk_gb: float = DEFAULT_NEED_DISK_GB) -> dict:
    du = shutil.disk_usage(ROOT)
    info = {
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "os": platform.platform(),
        "python": {"version": platform.python_version(), "executable": sys.executable,
                   "bits": 64 if sys.maxsize > 2**32 else 32},
        "cpu": {"name": cpu_name(), "logical_cores": os.cpu_count()},
        "ram": ram_gb(),
        "disk": {"path": str(ROOT), "free_gb": round(du.free / 1e9, 1), "total_gb": round(du.total / 1e9, 1),
                 "need_gb_estimate": need_disk_gb},
        "nvidia_smi": nvidia_smi(),
        "torch": torch_info(),
    }
    try:
        import psutil  # type: ignore
        info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
    except Exception:
        info["cpu"]["physical_cores"] = None
    checks = []
    t = info["torch"]
    checks.append({"name": "python_64bit", "ok": info["python"]["bits"] == 64, "detail": info["python"]["version"]})
    checks.append({"name": "torch_importable", "ok": t["importable"], "detail": t.get("version") or t.get("error")})
    checks.append({"name": "torch_cuda_build", "ok": bool(t.get("cuda_build")),
                   "detail": t.get("cuda_build") or "CPU-only PyTorch build"})
    checks.append({"name": "nvidia_driver", "ok": bool(info["nvidia_smi"] and "driver_version" in info["nvidia_smi"]),
                   "detail": (info["nvidia_smi"] or {}).get("driver_version", "nvidia-smi not found")})
    checks.append({"name": "cuda_available", "ok": bool(t.get("cuda_available")), "detail": str(t.get("cuda_available"))})
    g0 = (t.get("gpus") or [None])[0]
    checks.append({"name": "gpu_kernel_runs", "ok": bool(g0 and g0.get("kernel_test") == "ok"),
                   "detail": g0.get("kernel_test") if g0 else "no CUDA GPU"})
    checks.append({"name": "disk_free", "ok": info["disk"]["free_gb"] >= need_disk_gb,
                   "detail": f"{info['disk']['free_gb']} GB free, ~{need_disk_gb} GB needed (estimate)"})
    info["checks"] = checks
    info["cuda_training_ready"] = all(c["ok"] for c in checks if c["name"] in
                                      ("torch_importable", "torch_cuda_build", "cuda_available", "gpu_kernel_runs"))
    return info


def explain(info: dict) -> JobError | None:
    """Turn a failed check into owner-facing guidance (None = ready)."""
    by = {c["name"]: c for c in info["checks"]}
    if not by["torch_importable"]["ok"]:
        return JobError("PyTorch is not installed in this Python.", by["torch_importable"]["detail"] or "",
                        "Double-click setup.bat and wait for 'JOB OK'.")
    if not by["nvidia_driver"]["ok"]:
        return JobError("No NVIDIA driver found (nvidia-smi missing).", "Driver not installed or too old.",
                        "Install the latest NVIDIA driver (R570 or newer) from nvidia.com, reboot, run gpu_check.bat again.")
    if not by["torch_cuda_build"]["ok"]:
        return JobError("PyTorch is the CPU-only build.", "It was installed from PyPI instead of the CUDA 12.8 index.",
                        "Run setup.bat again; it reinstalls torch from https://download.pytorch.org/whl/cu128.")
    if not by["cuda_available"]["ok"]:
        return JobError("PyTorch cannot use the GPU (torch.cuda.is_available() is False).",
                        "Driver older than CUDA 12.8 needs (R570+), or the GPU is disabled.",
                        "Update the NVIDIA driver, reboot, run gpu_check.bat again.")
    if not by["gpu_kernel_runs"]["ok"]:
        return JobError("A test computation on the GPU failed: " + str(by["gpu_kernel_runs"]["detail"]),
                        "The installed PyTorch build does not support this GPU (RTX 50xx needs CUDA 12.8 builds).",
                        "Run setup.bat again (cu128 index).")
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "pc_info.json"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if CUDA training is not possible")
    ap.add_argument("--need-disk-gb", type=float, default=DEFAULT_NEED_DISK_GB)
    a = ap.parse_args(argv)
    log = start_job("gpu_check")
    try:
        info = collect(a.need_disk_gb)
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        g0 = (info["torch"].get("gpus") or [None])[0]
        log.info("CPU   : %s (%s logical cores)", info["cpu"]["name"], info["cpu"]["logical_cores"])
        log.info("RAM   : %s GB total, %s GB available", info["ram"]["total_gb"], info["ram"]["available_gb"])
        log.info("Disk  : %s GB free of %s GB on %s", info["disk"]["free_gb"], info["disk"]["total_gb"], info["disk"]["path"])
        log.info("Torch : %s (CUDA build %s)", info["torch"].get("version"), info["torch"].get("cuda_build"))
        if g0:
            log.info("GPU   : %s, %s GB VRAM (%s GB free), compute capability %s, kernel test %s",
                     g0["name"], g0["vram_total_gb"], g0.get("vram_free_gb"), g0["compute_capability"], g0["kernel_test"])
        else:
            log.info("GPU   : none usable by PyTorch")
        for c in info["checks"]:
            log.info("check %-17s %s  %s", c["name"], "OK  " if c["ok"] else "FAIL", c["detail"])
        if not next(c for c in info["checks"] if c["name"] == "disk_free")["ok"]:
            log.warning("Less free disk than the ~%.0f GB planning estimate; downloads may fill the drive.", a.need_disk_gb)
        log.info("wrote %s", out)
        err = explain(info)
        if err is not None:
            if a.strict:
                job_failed(err)
                return 1
            log.warning("CUDA training not possible on this machine: %s (fine for CPU-only use)", err.what)
        job_done("CUDA training ready: %s" % info["cuda_training_ready"])
        return 0
    except Exception as e:
        job_failed(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
