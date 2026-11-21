"""Band bookkeeping: Sentinel-2 L1C (13 bands), HLS (6 bands), Sentinel-1 and TerraScope's 6-band optical.

TerraScope optical order (app contract): [B2 blue, B3 green, B4 red, B8 NIR, B11 SWIR1, B12 SWIR2],
surface/TOA reflectance scaled to 0..1.

Sentinel-2 L1C product order (13 bands, as stored in Sen1Floods11 S2Hand and SEN12MS-CR s2 patches):
B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12. Digital numbers are reflectance x 10000.
NOTE: for processing baseline >= 04.00 (2022-01-25 onward) ESA adds a +1000 radiometric offset to L1C/L2A DNs;
the datasets used here (Sen1Floods11 2016-2019, SEN12MS-CR 2018) predate that, so no offset is removed by default
(``boa_offset`` argument lets callers handle newer imagery).

HLS v1.4/v2 (hls_burn_scars ``*_merged.tif``) 6-band order: Blue, Green, Red, NIR-narrow, SWIR1, SWIR2
(S30 bands B02, B03, B04, B8A, B11, B12 / L30 bands B2, B3, B4, B5, B6, B7). The NIR band is therefore B8A-like,
not B8: a small, documented spectral mismatch versus our B8 slot.
"""
from __future__ import annotations

import numpy as np

S2_L1C_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
S2_L2A_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]  # no B10 cirrus
OUR_OPTICAL = ["B2", "B3", "B4", "B8", "B11", "B12"]
OUR_FROM_L1C = [S2_L1C_BANDS.index(b) for b in OUR_OPTICAL]  # [1, 2, 3, 7, 11, 12]
OUR_FROM_L2A = [S2_L2A_BANDS.index(b) for b in OUR_OPTICAL]  # [1, 2, 3, 7, 10, 11]
HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
SAR_BANDS = ["VV", "VH"]
DEM_BANDS = ["elevation_m", "slope_deg"]

# names used by terratorch (HLSBands / TerraMind PRETRAINED_BANDS) for OUR 6 optical slots
TERRATORCH_NAMES_L1C = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]   # B8 = NIR_BROAD
TERRATORCH_NAMES_HLS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]  # HLS NIR = B8A

REFLECTANCE_SCALE = 10000.0


def dn_to_reflectance(a, scale: float = REFLECTANCE_SCALE, boa_offset: float = 0.0):
    """Sentinel-2 / HLS integer DN -> reflectance (float32, clipped to [0, 1.5])."""
    a = np.asarray(a, np.float32)
    return np.clip((a - boa_offset) / scale, 0.0, 1.5).astype(np.float32)


def looks_like_dn(a) -> bool:
    """Heuristic: reflectance stored as DN (x10000) if the 99th percentile of finite values exceeds 2."""
    a = np.asarray(a, np.float32)
    f = a[np.isfinite(a)]
    if f.size == 0:
        return False
    return float(np.percentile(f, 99)) > 2.0


def to_reflectance(a, scale: float = REFLECTANCE_SCALE):
    """Auto-detect DN vs reflectance and return reflectance 0..1 (NaN preserved)."""
    a = np.asarray(a, np.float32)
    if looks_like_dn(a):
        return np.clip(a / scale, 0.0, 1.5).astype(np.float32)
    return np.clip(a, 0.0, 1.5).astype(np.float32)


def l1c_to_ours(s2_13):
    """(13,H,W) L1C (DN or reflectance) -> (6,H,W) reflectance in TerraScope order."""
    s2 = np.asarray(s2_13)
    if s2.shape[0] != 13:
        raise ValueError(f"expected 13 Sentinel-2 L1C bands, got {s2.shape[0]}")
    return to_reflectance(s2[OUR_FROM_L1C])


def l2a_to_ours(s2_12):
    s2 = np.asarray(s2_12)
    if s2.shape[0] != 12:
        raise ValueError(f"expected 12 Sentinel-2 L2A bands, got {s2.shape[0]}")
    return to_reflectance(s2[OUR_FROM_L2A])


def hls_to_ours(hls6):
    """(6,H,W) HLS [B,G,R,NIRn,SW1,SW2] -> our 6 bands (NIR-narrow placed in the B8 slot)."""
    h = np.asarray(hls6)
    if h.shape[0] != 6:
        raise ValueError(f"expected 6 HLS bands, got {h.shape[0]}")
    return to_reflectance(h)


def sar_to_db(sar):
    """SAR backscatter -> dB. Values that look linear (all finite values >= 0 and 99th pct < 5) are converted."""
    s = np.asarray(sar, np.float32)
    f = s[np.isfinite(s)]
    if f.size and f.min() >= 0 and np.percentile(f, 99) < 5:
        return (10.0 * np.log10(np.clip(s, 1e-6, None))).astype(np.float32)
    return s.astype(np.float32)
