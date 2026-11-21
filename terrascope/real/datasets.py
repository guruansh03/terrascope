"""Task -> dataset registry used by fine-tuning and evaluation.

    flood      -> sen1floods11    (S1 + S2 L1C + optional Copernicus DEM; official hand-labelled splits)
    burn       -> hls_burn_scars  (HLS 6-band optical only)
    cloud      -> sen12mscr       (S1 + cloudy S2 -> clear S2)
    landcover  -> worldcover      (Sen1Floods11 chips + ESA WorldCover 2021 labels mapped to 6 classes)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from . import burnscars, sen12mscr, sen1floods11


@dataclass
class DatasetSpec:
    name: str
    task: str
    list_split: Callable[..., list]
    load: Callable[[dict], dict]
    default_root: Callable[[], str]
    sensors: tuple            # sensors the dataset provides
    n_classes: int | None     # None for regression (cloud)
    notes: str = ""


def _lc_list(root=None, split="train"):
    return sen1floods11.list_split(root or sen1floods11.default_root(), split, require=("s2", "label", "wc"))


DATASETS: dict[str, DatasetSpec] = {
    "sen1floods11": DatasetSpec("sen1floods11", "flood", sen1floods11.list_split, sen1floods11.load,
                                sen1floods11.default_root, ("optical", "sar", "dem"), 2,
                                "official hand-labelled train/valid/test splits; label -1 ignored"),
    "hls_burn_scars": DatasetSpec("hls_burn_scars", "burn", burnscars.list_split, burnscars.load,
                                  burnscars.default_root, ("optical",), 2,
                                  "test = official 'validation' folder; val = 10% hash split of 'training'"),
    "sen12mscr": DatasetSpec("sen12mscr", "cloud", sen12mscr.list_split, sen12mscr.load, sen12mscr.default_root,
                             ("optical", "sar"), None, "subset; scene-level 80/10/10 hash split"),
    "worldcover": DatasetSpec("worldcover", "landcover", _lc_list, sen1floods11.load_landcover,
                              sen1floods11.default_root, ("optical", "sar", "dem"), 6,
                              "Sen1Floods11 chips (2016-19) + WorldCover 2021 labels: noisy reference"),
}
TASK_DATASET = {d.task: d.name for d in DATASETS.values()}


def for_task(task: str) -> DatasetSpec:
    if task not in TASK_DATASET:
        raise KeyError(f"no real dataset for task {task!r}; known: {sorted(TASK_DATASET)}")
    return DATASETS[TASK_DATASET[task]]


def available(task: str, root: str | None = None) -> bool:
    try:
        spec = for_task(task)
        return len(spec.list_split(root or spec.default_root(), "train")) > 0
    except Exception:
        return False
