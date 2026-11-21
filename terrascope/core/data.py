"""Dataset caching, in-memory tensors and (augmented) batch construction."""
from __future__ import annotations

import os
import numpy as np
import torch

from .model import normalize
from .simulator import build_parallel

DATA_DIR = os.environ.get("TERRASCOPE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# name -> (seed range, size). Seed ranges are disjoint; val64 is only used for calibration.
SPLITS = {
    "train64": (range(10_000, 13_000), 64),
    "val64": (range(800_000, 800_200), 64),
    "test64": (range(900_000, 900_300), 64),
    "test128": (range(950_000, 950_100), 128),
}

# training-time augmentation ranges (physical units)
AUG = dict(elev_offset_m=(-800.0, 1500.0), sar_offset_db=(-3.0, 3.0), optical_gain=(0.8, 1.25), p=0.85)


def ensure_data(names=("train64", "val64", "test64", "test128"), n_train=3000):
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in names:
        seeds, size = SPLITS[name]
        if name == "train64":
            seeds = range(seeds.start, seeds.start + min(n_train, len(seeds)))
        path = os.path.join(DATA_DIR, f"{name}.npz")
        if not os.path.exists(path):
            print(f"[data] generating {name} ({len(seeds)} scenes @ {size}px)…", flush=True)
            tmp = path + ".tmp.npz"
            np.savez(tmp, **build_parallel(seeds, size))
            os.replace(tmp, path)


def load_raw(name):
    """Physical-unit tensors (kept in their storage dtype, mostly float16, to save RAM)."""
    z = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    d = {k: torch.from_numpy(z[k]) for k in z.files}
    d["lc"] = d["lc"].long()
    d["flood"] = d["flood"].float()
    d["veg"] = d["veg"].float()
    d["cloud"] = d["cloud"].float()
    return d


def with_inputs(d, norm="v2", o_scale=1.0, sar_off=0.0, dem_off=0.0):
    """Add normalised model input ``x`` and reconstruction target to a raw split (optionally perturbed)."""
    o = d["optical"].float() * o_scale
    oc = d["optical_clean"].float() * o_scale
    s = d["sar"].float() + sar_off
    de = d["dem"].float().clone()
    de[:, 0] += dem_off
    out = dict(d)
    out["x"] = normalize(o, s, de, norm)
    out["target"] = normalize(oc, s, de, norm)
    return out


def load(name, norm="v2"):
    return with_inputs(load_raw(name), norm)


def make_batch(d, idx, norm="v2", aug=True, g=None, device=None):
    """Normalised (x, target) for the scenes ``idx`` with optional calibration augmentation:
    elevation offset, SAR calibration offset (dB) and optical gain, each per scene (prob AUG['p']).
    ``device``: move the raw slice there first so normalisation (HAND filter) runs on the GPU too.
    Random draws always come from the CPU generator, so results do not depend on the device's RNG."""
    o = d["optical"][idx].float().to(device)
    oc = d["optical_clean"][idx].float().to(device)
    s = d["sar"][idx].float().to(device)
    de = d["dem"][idx].float().clone().to(device)
    if aug:
        B = o.shape[0]

        def u(lo, hi):
            r = torch.rand(B, 1, 1, 1, generator=g) * (hi - lo) + lo
            on = (torch.rand(B, 1, 1, 1, generator=g) < AUG["p"]).float()
            return r.to(o.device), on.to(o.device)

        gain, on = u(*AUG["optical_gain"])
        gain = 1 + on * (gain - 1)
        o = (o * gain).clamp(0, 1)
        oc = (oc * gain).clamp(0, 1)
        off, on = u(*AUG["sar_offset_db"])
        s = s + on * off
        eo, on = u(*AUG["elev_offset_m"])
        de[:, :1] += on * eo   # no-op for v2 (scene-relative DEM) by construction; kept for other norms
    return normalize(o, s, de, norm), normalize(oc, s, de, norm)


def augment(*ts):
    """Random flips / 90deg rotations applied identically to all tensors (…,H,W)."""
    k = int(torch.randint(0, 4, (1,)))
    fl = bool(torch.randint(0, 2, (1,)))
    out = []
    for t in ts:
        t = torch.rot90(t, k, (-2, -1))
        if fl:
            t = torch.flip(t, (-1,))
        out.append(t)
    return out
