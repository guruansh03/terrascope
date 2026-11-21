import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def reqs(name):
    return [l.split("#")[0].strip() for l in (ROOT / name).read_text().splitlines() if l.split("#")[0].strip()]


def test_pc_requirements_pinned_without_torch():
    r = reqs("requirements-pc.txt")
    assert "terratorch==1.2.13" in r
    for line in r:
        assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[\w.]+", line), f"not pinned: {line}"
        assert not re.match(r"torch(vision)?==", line), "torch is installed from the cu128 index by setup_env.py"
    names = {l.split("==")[0].split("[")[0].lower() for l in r}
    for need in ("rasterio", "onnxruntime", "onnx", "fastapi", "uvicorn", "psutil", "pytest", "httpx"):
        assert need in names


def test_demo_requirements_cpu_offline_friendly():
    r = reqs("requirements-demo.txt")
    names = {re.split(r"[<>=!~\[]", l)[0].lower() for l in r}
    assert "terratorch" not in names, "laptop runs foundation models from ONNX"
    assert {"torch", "onnxruntime", "fastapi", "uvicorn", "numpy", "python-multipart"} <= names
    assert not any(l.startswith("--") for l in r), "index options belong on the pip command line (offline mode)"


def test_gitignore_policy():
    g = (ROOT / ".gitignore").read_text().splitlines()
    for need in (".venv*/", "data/", "cache/", "logs/", "dist/", "*.onnx", "artifacts/ckpt/real/", "wheels/"):
        assert need in g


def test_gitattributes_keeps_bat_crlf():
    assert "*.bat text eol=crlf" in (ROOT / ".gitattributes").read_text()
