"""Shared fixture builders for tests/real (unique module name: avoids conftest import clashes)."""
import numpy as np


def fake_s2_l1c(rng, H, W):
    """13-band L1C-like DN (uint16), vegetation-ish spectrum."""
    base = np.array([1200, 1000, 900, 700, 1100, 2200, 2600, 2800, 2900, 600, 20, 1800, 1000], np.float32)
    a = base[:, None, None] * (0.8 + 0.4 * rng.random((13, H, W), dtype=np.float32))
    return a.astype(np.uint16)


def fake_s1_db(rng, H, W):
    return np.stack([-10 + 2 * rng.standard_normal((H, W)), -17 + 2 * rng.standard_normal((H, W))]).astype(np.float32)
