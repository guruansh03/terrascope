import os, sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("TERRASCOPE_NO_SLEEP", "1")

try:
    import torch
    torch.set_num_threads(1)
except Exception:  # pragma: no cover
    pass


@pytest.fixture
def utm_transform():
    from affine import Affine
    # 10 m pixels, UTM zone 33N, upper-left at (400000, 5300000)  (~ 47.8 N, 13.7 E)
    return Affine(10.0, 0.0, 400000.0, 0.0, -10.0, 5300000.0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)

from rs_helpers import fake_s1_db, fake_s2_l1c  # noqa: E402,F401
