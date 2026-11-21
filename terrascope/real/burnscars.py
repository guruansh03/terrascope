"""HLS Burn Scars (``ibm-nasa-geospatial/hls_burn_scars`` on Hugging Face): burn-scar segmentation,
HLS 6 bands [Blue, Green, Red, NIR-narrow, SWIR1, SWIR2], 512 x 512 @ 30 m (license CC BY 4.0 per dataset card).

Layout relied on (from terratorch's ``FireScarsNonGeo``, installed here and read): two folders ``training`` and
``validation`` with ``*_merged.tif`` images and ``*.mask.tif`` masks (1 burn scar, 0 not burned, -1 missing).
How the HF repo packages them (loose files vs. a tar.gz) is NOT verified offline, so ``download`` lists the repo
and handles both; files are then found by suffix anywhere under the root.

Splits: the dataset has no test split. We use ``validation`` as our TEST set and carve a deterministic 10 %
VALIDATION subset (by file-name hash) out of ``training`` for early stopping.
Reflectance scale: auto-detected (``bands.to_reflectance``): values > 2 are treated as DN x 10000.
"""
from __future__ import annotations

import logging, os

import numpy as np

from .bands import hls_to_ours
from .common import data_root, fill_nan, read_tif, stable_fraction
from . import download as dl

NAME = "hls_burn_scars"
TASK = "burn"
HF_REPO = os.environ.get("BURNSCARS_HF_REPO", "ibm-nasa-geospatial/hls_burn_scars")
VAL_FRAC = 0.10
log = logging.getLogger("terrascope.real.burnscars")


def default_root() -> str:
    return os.path.join(data_root(), NAME)


def download(root: str | None = None, logger=None, api=None) -> dict:
    lg = logger or log
    root = root or default_root()
    os.makedirs(root, exist_ok=True)
    api = api or dl.hf_api()
    files = dl.hf_list_files(HF_REPO, api=api)
    lg.info("HF %s: %d files listed", HF_REPO, len(files))
    sizes = {}
    try:  # sizes for the disk check (best effort; API shape differs across huggingface_hub versions)
        for it in api.list_repo_tree(HF_REPO, repo_type="dataset", recursive=True):
            if getattr(it, "size", None):
                sizes[it.path] = int(it.size)
    except Exception as e:
        lg.info("  (no size info: %s)", type(e).__name__)
    tifs = [f for f in files if f.endswith(("_merged.tif", ".mask.tif"))]
    archives = [f for f in files if f.endswith((".tar.gz", ".tgz", ".zip", ".tar"))]
    pick = tifs if tifs else archives
    if not pick:
        raise dl.DownloadError(f"{HF_REPO} contains neither *_merged.tif files nor an archive: {files[:20]}")
    need = sum(sizes.get(f, 0) for f in pick) * (2 if pick is archives else 1)
    dl.check_disk(root, need or 4 * 10**9, lg)
    for i, f in enumerate(pick, 1):
        p = dl.hf_download(HF_REPO, f, os.path.join(root, "hf"))
        if pick is archives:
            dl.safe_extract(p, os.path.join(root, "extracted"), logger=lg)
        if i % 100 == 0:
            lg.info("  %d/%d", i, len(pick))
    return verify(root, lg)


def _pairs(root):
    imgs, masks = {}, {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith("_merged.tif"):
                imgs[(_split_of(dp), fn[: -len("_merged.tif")])] = os.path.join(dp, fn)
            elif fn.endswith(".mask.tif"):
                masks[(_split_of(dp), fn[: -len(".mask.tif")])] = os.path.join(dp, fn)
    out = []
    for k, p in sorted(imgs.items()):
        if k in masks:
            out.append(dict(id=k[1], group=k[0], image=p, mask=masks[k]))
    return out


def _split_of(dirpath: str) -> str:
    d = dirpath.replace("\\", "/").lower()
    if "valid" in d:
        return "validation"
    if "train" in d:
        return "training"
    return "unknown"


def list_split(root: str | None = None, split: str = "train") -> list[dict]:
    root = root or default_root()
    pairs = _pairs(root)
    if not pairs:
        raise FileNotFoundError(f"no *_merged.tif / *.mask.tif pairs under {root}; run get_data.bat burn")
    if split == "test":
        return [p for p in pairs if p["group"] == "validation"]
    tr = [p for p in pairs if p["group"] == "training"]
    is_val = [stable_fraction(p["id"]) < VAL_FRAC for p in tr]
    return [p for p, v in zip(tr, is_val) if v == (split == "val")]


def verify(root: str, logger=None) -> dict:
    lg = logger or log
    sizes = {s: len(list_split(root, s)) for s in ("train", "val", "test")}
    lg.info("HLS burn scars verify: split sizes %s (test = official 'validation' folder)", sizes)
    return dict(split_sizes=sizes)


def load(rec: dict) -> dict:
    img = read_tif(rec["image"])
    if img.shape[0] < 6:
        raise ValueError(f"{rec['image']}: expected 6 HLS bands, got {img.shape[0]}")
    optical = fill_nan(hls_to_ours(img[:6]), 0.0)
    m = read_tif(rec["mask"])[0]
    mask = np.where(np.isfinite(m), m, -1).astype(np.int64)
    mask[(mask != 0) & (mask != 1)] = -1
    return dict(id=rec["id"], optical=optical, sar=None, dem=None, mask=mask)
