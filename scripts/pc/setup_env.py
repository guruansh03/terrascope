"""PC environment installer (OPS) - called by setup.bat with the .venv python.

Steps (each one logged to logs/setup_<ts>.txt):
  1. upgrade pip
  2. install torch + torchvision from the CUDA 12.8 wheel index (RTX 50xx / Blackwell needs cu128 or newer);
     if that index has no wheel for this Python, try cu130, then cu132 (same GPU support, newer driver needed)
  3. pin the installed torch/torchvision in a constraints file so requirements-pc.txt can never swap them
     for the CPU-only PyPI build
  4. pip install -r requirements-pc.txt -c <constraints>
  5. verify: torch.cuda.is_available(), GPU name, a real GPU kernel, imports of terratorch / rasterio / onnxruntime
  6. save `pip freeze` next to the log (exact versions for reproducibility)

Exit codes: 0 ok, 1 install failed, 2 installed but the GPU is not usable (CUDA check failed).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from terrascope.utils.joblog import JobError, current_log_path, diagnose_text, job_done, job_failed, start_job  # noqa: E402

TORCH_INDEXES = ["https://download.pytorch.org/whl/cu128",
                 "https://download.pytorch.org/whl/cu130",
                 "https://download.pytorch.org/whl/cu132"]
SUPPORTED_PY = ((3, 12), (3, 13))

VERIFY = r"""
import json, sys
r = {}
import torch
r['torch'] = torch.__version__; r['cuda_build'] = torch.version.cuda; r['cuda_available'] = torch.cuda.is_available()
if r['cuda_available']:
    p = torch.cuda.get_device_properties(0)
    r['gpu'] = p.name; r['vram_gb'] = round(p.total_memory / 1e9, 2); r['cc'] = f'{p.major}.{p.minor}'
    try:
        a = torch.randn(256, 256, device='cuda'); float((a @ a).sum()); torch.cuda.synchronize(); r['kernel'] = 'ok'
    except Exception as e:
        r['kernel'] = f'FAILED {type(e).__name__}: {e}'
for m in ('torchvision', 'terratorch', 'rasterio', 'onnxruntime', 'fastapi', 'numpy'):
    try:
        mod = __import__(m); r[m] = getattr(mod, '__version__', 'ok')
    except Exception as e:
        r[m] = f'IMPORT FAILED {type(e).__name__}: {e}'
print('VERIFY_JSON=' + json.dumps(r))
"""


def sh(log, args: list[str], what: str) -> tuple[int, str]:
    """Run a command, stream its output to console + log, return (code, output)."""
    log.info("$ %s", " ".join(args))
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", cwd=str(ROOT))
    lines = []
    assert p.stdout is not None
    for line in p.stdout:
        line = line.rstrip()
        lines.append(line)
        log.info("  %s", line)
    p.wait()
    out = "\n".join(lines)
    if p.returncode:
        log.error("%s failed with exit code %s", what, p.returncode)
    return p.returncode, out


def fail_from_output(what: str, out: str) -> JobError:
    hit = diagnose_text(out)
    if hit:
        return JobError(what, hit[0], hit[1])
    return JobError(what, "pip reported an error (see the lines above).",
                    "Send this log file to Claude. Re-running setup.bat is safe.")


def installed_torch() -> dict | None:
    code = ("import json,torch,torchvision;print(json.dumps({'torch':torch.__version__,"
            "'tv':torchvision.__version__,'cuda':torch.version.cuda}))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode:
        return None
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requirements", default=str(ROOT / "requirements-pc.txt"))
    ap.add_argument("--reinstall-torch", action="store_true")
    a = ap.parse_args(argv)
    log = start_job("setup", probe_torch=False)
    try:
        if sys.version_info[:2] not in SUPPORTED_PY:
            raise JobError(f"Python {sys.version.split()[0]} is not supported for training.",
                           "terratorch 1.2.13 needs torchgeo 0.9, which requires Python >= 3.12; Python 3.12/3.13 "
                           "are the versions with Windows wheels for every dependency.",
                           "Install 64-bit Python 3.12 from python.org, delete the .venv folder, run setup.bat again.")
        pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
        code, out = sh(log, pip + ["install", "--upgrade", "pip"], "pip upgrade")
        if code:
            raise fail_from_output("Upgrading pip failed.", out)

        cur = installed_torch()
        if cur and cur.get("cuda") and not a.reinstall_torch:
            log.info("torch %s (CUDA %s) already installed - keeping it", cur["torch"], cur["cuda"])
        else:
            last = ""
            for idx in TORCH_INDEXES:
                log.info("installing torch + torchvision from %s (large download: roughly 3 GB; estimate)", idx)
                extra = ["--force-reinstall"] if a.reinstall_torch else []
                code, out = sh(log, pip + ["install", "--upgrade", *extra, "torch", "torchvision",
                                           "--index-url", idx], "torch install")
                if code == 0:
                    break
                last = out
                if "no matching distribution" not in out.lower() and "could not find a version" not in out.lower():
                    raise fail_from_output(f"Installing PyTorch from {idx} failed.", out)
                log.warning("no torch wheel for this Python on %s - trying the next CUDA index", idx)
            else:
                raise fail_from_output("No CUDA PyTorch wheel was found for this Python version.", last)
            cur = installed_torch()
        if not cur:
            raise JobError("PyTorch was installed but cannot be imported.", "Broken install.",
                           "Delete the .venv folder and run setup.bat again; if it repeats, send this log to Claude.")
        cons = ROOT / ".venv" / "torch-constraints.txt"
        cons.parent.mkdir(exist_ok=True)
        cons.write_text(f"torch=={cur['torch']}\ntorchvision=={cur['tv']}\n", encoding="utf-8")
        log.info("torch pinned via %s: torch==%s torchvision==%s", cons, cur["torch"], cur["tv"])

        code, out = sh(log, pip + ["install", "-r", a.requirements, "-c", str(cons)], "requirements-pc install")
        if code:
            raise fail_from_output("Installing requirements-pc.txt failed.", out)

        code, out = sh(log, [sys.executable, "-c", VERIFY], "verification")
        line = next((l for l in out.splitlines() if l.startswith("VERIFY_JSON=")), None)
        if code or not line:
            raise fail_from_output("The post-install check crashed.", out)
        import json
        v = json.loads(line.split("=", 1)[1])
        lp = current_log_path()
        if lp:
            frz =subprocess.run(pip + ["freeze"], capture_output=True, text=True).stdout
            (lp.parent / (lp.stem + "_pip_freeze.txt")).write_text(frz, encoding="utf-8")
            log.info("pip freeze saved to %s", lp.parent / (lp.stem + "_pip_freeze.txt"))
        bad = [k for k, val in v.items() if isinstance(val, str) and val.startswith("IMPORT FAILED")]
        if bad:
            raise JobError(f"Packages installed but not importable: {', '.join(bad)}",
                           "; ".join(f"{k}: {v[k]}" for k in bad),
                           "Send this log file to Claude.")
        if not v.get("cuda_available") or v.get("kernel") != "ok":
            job_failed(JobError(
                f"Packages are installed, but PyTorch {v.get('torch')} (CUDA build {v.get('cuda_build')}) cannot use the GPU "
                f"(cuda_available={v.get('cuda_available')}, kernel={v.get('kernel')}).",
                "NVIDIA driver missing/too old (CUDA 12.8 needs R570+, CUDA 13 needs R580+) or GPU disabled.",
                "Update the NVIDIA driver from nvidia.com, reboot, then run gpu_check.bat."))
            return 2
        log.info("GPU: %s, %s GB VRAM, compute capability %s, test kernel %s", v["gpu"], v["vram_gb"], v["cc"], v["kernel"])
        job_done(f"Environment ready: torch {v['torch']} CUDA {v['cuda_build']}, terratorch {v['terratorch']}")
        return 0
    except BaseException as e:  # noqa: BLE001
        job_failed(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
