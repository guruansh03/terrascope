import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "pc"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(autouse=True)
def _logs_to_tmp(tmp_path, monkeypatch):
    """Job logs of the tests go to a temp folder, never to the repo's logs/."""
    monkeypatch.setenv("TERRASCOPE_LOG_DIR", str(tmp_path / "logs"))
    yield
