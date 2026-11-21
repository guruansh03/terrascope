"""Pretrained backbone weights: where they live and how to fetch them.

Weights are cached in ``data/hf_cache`` with the standard Hugging Face layout (``HF_HOME=data/hf_cache`` ->
``data/hf_cache/hub/models--<org>--<name>/...``). A plain copy of the file in ``data/weights/<hf_file>`` is
also accepted (offline laptops / manual download). TerraFM's backbone is the sim-pretrained
``artifacts/ckpt/pretrained.pt`` written by ``terrascope.core.train``.
"""
from __future__ import annotations

import os

from .specs import MODELS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HF_HOME = os.environ.get("TERRASCOPE_HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
HUB_CACHE = os.path.join(HF_HOME, "hub")
LOCAL_WEIGHTS = os.path.join(ROOT, "data", "weights")
TERRAFM_PRETRAINED = [os.path.join(ROOT, "artifacts", "ckpt", "pretrained.pt"),
                      os.path.join(ROOT, "artifacts", "ckpt", "v1", "pretrained.pt")]


def weights_path(model_id: str) -> str | None:
    spec = MODELS[model_id]
    if spec["family"] == "terrafm":
        return next((p for p in TERRAFM_PRETRAINED if os.path.exists(p)), None)
    if not spec.get("hf_repo"):  # e.g. unet_r18_scratch: trained from random init, no file
        return None
    local = os.path.join(LOCAL_WEIGHTS, spec["hf_file"])
    if os.path.exists(local):
        return local
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(spec["hf_repo"], spec["hf_file"], cache_dir=HUB_CACHE,
                                   revision=spec.get("hf_revision"))
        if isinstance(p, str) and os.path.exists(p):
            return p
    except Exception:
        pass
    return None


def fetch_weights(model_id: str, logger=None) -> str | None:
    """Download (or reuse) the pretrained file into the HF cache; returns the local path (None: no weights)."""
    spec = MODELS[model_id]
    if not spec.get("hf_repo") and spec["family"] != "terrafm":
        return None
    if spec["family"] == "terrafm":
        p = weights_path(model_id)
        if not p:
            raise FileNotFoundError("TerraFM pretrained checkpoint missing; run train.bat sim first")
        return p
    from huggingface_hub import hf_hub_download
    os.makedirs(HUB_CACHE, exist_ok=True)
    if logger:
        logger.info("fetching %s/%s into %s", spec["hf_repo"], spec["hf_file"], HUB_CACHE)
    if spec.get("hf_revision"):  # smp also fetches config.json; harmless, keeps its own cache layout complete
        try:
            hf_hub_download(repo_id=spec["hf_repo"], filename="config.json", cache_dir=HUB_CACHE,
                            revision=spec["hf_revision"])
        except Exception:
            pass
    return hf_hub_download(repo_id=spec["hf_repo"], filename=spec["hf_file"], cache_dir=HUB_CACHE,
                           revision=spec.get("hf_revision"))
