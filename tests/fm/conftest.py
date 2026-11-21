import json, os, sys, warnings

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore", category=UserWarning)

import torch  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture
def fm_env(tmp_path, monkeypatch):
    """Isolated checkpoint / onnx / weights locations."""
    from terrascope.fm import registry, weights
    ck, ox = tmp_path / "ckpt", tmp_path / "onnx"
    monkeypatch.setenv("TERRASCOPE_REAL_CKPT", str(ck))
    monkeypatch.setenv("TERRASCOPE_ONNX_DIR", str(ox))
    monkeypatch.setenv("TERRASCOPE_RUNTIME", "torch")
    monkeypatch.setattr(weights, "LOCAL_WEIGHTS", str(tmp_path / "weights"))
    monkeypatch.setattr(weights, "HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("TERRASCOPE_DEVICE", "cpu")
    registry.reset()
    yield dict(ckpt=str(ck), onnx=str(ox), weights=str(tmp_path / "weights"), tmp=tmp_path)
    registry.reset()
