"""Static facts about the models TerraScope compares. Parameter counts are NOT stored here: they are measured
from the built modules at runtime (see ``registry.measure_params``).

Per-model preprocessing (each model's documented normalisation; constants are imported from terratorch at
build time so they cannot drift from the library):
* Prithvi-EO-2.0: HLS reflectance x 10000, then (x - PRITHVI_V2_MEAN) / PRITHVI_V2_STD per band, 6 bands
  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2], patch 16, pretraining size 224. Our B8 (NIR broad) feeds the
  NIR_NARROW slot (documented spectral mismatch; Sentinel-2 L1C is TOA, not HLS surface reflectance).
* TerraMind-1.0: per-modality standardisation with terratorch's ``v1_pretraining_mean/std`` for
  ``untok_sen2l1c@224`` (L1C sources) or ``untok_sen2l2a@224`` (HLS/L2A sources), ``untok_sen1grd@224`` (dB) and
  ``untok_dem@224`` (metres). Optical uses a 6-band subset of the pretrained patch embedding (terratorch
  ``bands=`` mechanism selects the matching pretrained weights per band).
* TerraFM: its own fixed normalisation (``terrascope.core.model.normalize``).
"""
from __future__ import annotations

ALL_TASKS = ["landcover", "flood", "veg", "cloud", "burn"]
LEARNED_TASKS = ["flood", "burn", "landcover", "cloud"]
TILE = 224

MODELS: dict[str, dict] = {
    "terrafm": dict(
        name="TerraFM (ours)", family="terrafm", sensors=["optical", "sar", "dem"],
        tasks=["landcover", "flood", "veg", "cloud", "burn"], license="project code (this repository)",
        source_url=None, patch=4, out_layers=None,
    ),
    "prithvi_tiny": dict(
        name="Prithvi-EO-2.0-tiny-TL", family="prithvi", backbone="prithvi_eo_v2_tiny_tl",
        hf_repo="ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL", hf_file="Prithvi_EO_V2_tiny_TL.pt",
        sensors=["optical"], tasks=["flood", "burn", "landcover", "cloud", "veg"], license="Apache-2.0",
        source_url="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL", patch=16,
        out_layers=[2, 5, 8, 11],
    ),
    "terramind_tiny": dict(
        name="TerraMind-1.0-tiny", family="terramind", backbone="terramind_v1_tiny",
        hf_repo="ibm-esa-geospatial/TerraMind-1.0-tiny", hf_file="TerraMind_v1_tiny.pt",
        sensors=["optical", "sar", "dem"], tasks=["flood", "burn", "landcover", "cloud", "veg"],
        license="Apache-2.0", source_url="https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny",
        patch=16, out_layers=[2, 5, 8, 11],
    ),
    "prithvi_100m": dict(
        name="Prithvi-EO-2.0-100M-TL", family="prithvi", backbone="prithvi_eo_v2_100_tl",
        hf_repo="ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL", hf_file="Prithvi_EO_V2_100M_TL.pt",
        sensors=["optical"], tasks=["flood", "burn", "landcover", "cloud", "veg"], license="Apache-2.0",
        source_url="https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL", patch=16,
        out_layers=[2, 5, 8, 11],
    ),
    # supervised baselines (terrascope.fm.baselines): smp U-Net, ResNet-18 encoder, smp decoder
    "unet_r18_scratch": dict(
        name="U-Net ResNet-18 (scratch)", family="unet", sensors=["optical", "sar", "dem"],
        tasks=["flood", "burn", "landcover", "cloud", "veg"], license="MIT (segmentation_models_pytorch)",
        source_url="https://github.com/qubvel-org/segmentation_models.pytorch", patch=32, out_layers=None,
        needs_weights=False,
    ),
    "unet_r18_imagenet": dict(
        name="U-Net ResNet-18 (ImageNet init)", family="unet", sensors=["optical", "sar", "dem"],
        tasks=["flood", "burn", "landcover", "cloud", "veg"], license="MIT (segmentation_models_pytorch)",
        source_url="https://huggingface.co/smp-hub/resnet18.imagenet", patch=32, out_layers=None,
        hf_repo="smp-hub/resnet18.imagenet", hf_file="model.safetensors",
        hf_revision="3f2325ff978283d47aa6a1d6878ca20565622683",
    ),
}
MODEL_ORDER = ["terrafm", "prithvi_tiny", "terramind_tiny", "prithvi_100m", "unet_r18_scratch", "unet_r18_imagenet"]
FOUNDATION_MODELS = ["terrafm", "prithvi_tiny", "terramind_tiny", "prithvi_100m"]
BASELINES = ["unet_r18_scratch", "unet_r18_imagenet"]
FM_IDS = [m for m in MODEL_ORDER if m != "terrafm"]

# optical source flavour per task (decides TerraMind's S2 modality and band names)
TASK_OPTICAL_SOURCE = {"flood": "l1c", "landcover": "l1c", "cloud": "l1c", "burn": "hls"}
# sensors each task's dataset provides (models use the intersection with their own sensors)
TASK_SENSORS = {"flood": ["optical", "sar", "dem"], "landcover": ["optical", "sar", "dem"],
                "cloud": ["optical", "sar"], "burn": ["optical"]}
TASK_OUT = {"flood": 1, "burn": 1, "landcover": 6, "cloud": 6}

# default fine-tuning hyper-parameters (8 GB GPU, bf16/fp16 autocast, 224 px crops)
TRAIN_DEFAULTS = {
    "terrafm": dict(batch_size=16, lr_encoder=3e-4, lr_head=1e-3),
    "prithvi_tiny": dict(batch_size=16, lr_encoder=5e-5, lr_head=5e-4),
    "terramind_tiny": dict(batch_size=16, lr_encoder=5e-5, lr_head=5e-4),
    "prithvi_100m": dict(batch_size=8, lr_encoder=3e-5, lr_head=5e-4),
    # U-Nets: "head" = smp decoder + 1x1 conv
    "unet_r18_scratch": dict(batch_size=16, lr_encoder=3e-4, lr_head=1e-3),
    "unet_r18_imagenet": dict(batch_size=16, lr_encoder=1e-4, lr_head=1e-3),
}


def needs_weights(model_id: str) -> bool:
    return MODELS[model_id].get("needs_weights", True)
