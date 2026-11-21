"""Vegetation indices from the 6-band optical stack [B2,B3,B4,B8,B11,B12] (reflectance 0..1).

IMPORTANT: ``health_index`` is a DERIVED SPECTRAL INDEX, not ground truth and not a learned prediction.
On real data no labelled vegetation-health reference is available here, so TerraScope reports this index
(and labels it as derived) instead of pretending a model regressed something it was never supervised on.

    NDVI   = (B8 - B4) / (B8 + B4)
    NDMI   = (B8 - B11) / (B8 + B11)                  (canopy water content)
    health = clip(0.75 * clip((NDVI - 0.15) / 0.65, 0, 1) + 0.25 * clip((NDMI + 0.10) / 0.50, 0, 1), 0, 1)

NDVI is rescaled so bare soil (~0.15) -> 0 and dense healthy canopy (~0.80) -> 1; NDMI adds moisture stress.
The weights/thresholds are fixed heuristics chosen for interpretability, not fitted to data.
"""
from __future__ import annotations

import numpy as np

FORMULA = ("health = clip(0.75*clip((NDVI-0.15)/0.65,0,1) + 0.25*clip((NDMI+0.10)/0.50,0,1), 0, 1); "
           "NDVI=(B8-B4)/(B8+B4), NDMI=(B8-B11)/(B8+B11)")
KIND = "derived_index"


def _nd(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return ((a - b) / np.maximum(a + b, 1e-6)).astype(np.float32)


def ndvi(optical) -> np.ndarray:
    o = np.asarray(optical, np.float32)
    return _nd(o[3], o[2])


def ndmi(optical) -> np.ndarray:
    o = np.asarray(optical, np.float32)
    return _nd(o[3], o[4])


def ndwi(optical) -> np.ndarray:
    """McFeeters NDWI (green vs NIR); > 0 suggests open water."""
    o = np.asarray(optical, np.float32)
    return _nd(o[1], o[3])


def health_index(optical) -> np.ndarray:
    n = np.clip((ndvi(optical) - 0.15) / 0.65, 0, 1)
    m = np.clip((ndmi(optical) + 0.10) / 0.50, 0, 1)
    return np.clip(0.75 * n + 0.25 * m, 0, 1).astype(np.float32)


def describe() -> dict:
    return dict(kind=KIND, formula=FORMULA, ground_truth=False)
