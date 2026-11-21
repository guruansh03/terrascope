"""SEN12MS-CR (Ebel et al., IEEE TGRS 2021): co-registered Sentinel-1, cloudy Sentinel-2 and cloud-free
Sentinel-2 (L1C, 13 bands) patches of 256 x 256 px @ 10 m, 2018, four seasons. Used for cloud reconstruction.

Source: TUM mediaTUM record https://mediatum.ub.tum.de/1554803 (license CC BY 4.0 per the record - verify).
The full dataset is very large (many tens of GB per season), so we take a SUBSET, best effort:

* archives per season: ``ROIs{1158_spring|1868_summer|1970_fall|2017_winter}_{s1|s2|s2_cloudy}.tar.gz``
  (names as listed on the mediaTUM record - from memory, NOT verified offline)
* download URL template: env ``SEN12MSCR_URL_TEMPLATE`` (default below is the TUM dataserv share pattern
  ``https://dataserv.ub.tum.de/s/m1554803/download?path=/&files={file}`` - an UNVERIFIED assumption).
  If the template does not work, download the archives manually from the mediaTUM page into
  ``data/real/sen12mscr/raw/`` - local archives are used first.
* subset selection: stream the ``s2_cloudy`` archive and keep patches of the requested ROI/season, at most
  ``per_scene`` patches per scene, until ``max_patches``; then stream the ``s2`` and ``s1`` archives extracting only
  the matching patch ids (stops early once complete, or after a byte budget).
* member naming relied on: ``.../ROIs1158_spring_s2_cloudy_<scene>_p<patch>.tif`` (and ``_s1_``, ``_s2_``);
  parsed with a regex - files that do not match are logged and skipped.

Splits: no official split is assumed; we split by SCENE id with a stable hash (80/10/10) to avoid leakage between
neighbouring patches.
"""
from __future__ import annotations

import logging, os, re

import numpy as np

from .bands import l1c_to_ours, sar_to_db
from .common import data_root, fill_nan, read_tif, stable_fraction
from . import download as dl

NAME = "sen12mscr"
TASK = "cloud"
SEASONS = {"spring": "ROIs1158_spring", "summer": "ROIs1868_summer", "fall": "ROIs1970_fall",
           "winter": "ROIs2017_winter"}
URL_TEMPLATE = os.environ.get("SEN12MSCR_URL_TEMPLATE",
                              "https://dataserv.ub.tum.de/s/m1554803/download?path=/&files={file}")
PATCH_RE = re.compile(r"(ROIs\d+_[a-z]+)_(s1|s2|s2_cloudy)_(\d+)_p(\d+)\.tif$")
log = logging.getLogger("terrascope.real.sen12mscr")


def default_root() -> str:
    return os.path.join(data_root(), NAME)


def parse(name: str):
    m = PATCH_RE.search(os.path.basename(name))
    if not m:
        return None
    return dict(roi=m.group(1), kind=m.group(2), scene=int(m.group(3)), patch=int(m.group(4)))


def _key(p) -> str:
    return f"{p['roi']}_{p['scene']}_p{p['patch']}"


def _archive_source(root, roi, kind):
    fn = f"{roi}_{kind}.tar.gz"
    local = os.path.join(root, "raw", fn)
    if os.path.exists(local):
        return "file", local
    return "url", URL_TEMPLATE.format(file=fn)


def _iter_archive(kind_src, out_dir, want, stop, session, lg, max_bytes):
    kind, src = kind_src
    if kind == "file":
        import tarfile
        out = []
        with tarfile.open(src, "r|*") as tf:
            for m in tf:
                if stop():
                    break
                if not m.isfile() or not want(m.name):
                    continue
                tgt = dl._safe_target(out_dir, m.name)
                if not (os.path.exists(tgt) and os.path.getsize(tgt) == m.size):
                    os.makedirs(os.path.dirname(tgt), exist_ok=True)
                    with tf.extractfile(m) as f, open(tgt, "wb") as g:
                        g.write(f.read())
                out.append(tgt)
        return out
    return dl.stream_tar_members(src, out_dir, want, stop, session=session, logger=lg, max_bytes=max_bytes)


def download(root: str | None = None, logger=None, session=None, season: str = "spring", max_patches: int = 1500,
             per_scene: int = 40, max_gb_per_archive: float = 60.0) -> dict:
    lg = logger or log
    root = root or default_root()
    roi = SEASONS[season]
    out_dir = os.path.join(root, "patches")
    os.makedirs(out_dir, exist_ok=True)
    have = _index(root)
    complete = {k for k, v in have.items() if len(v) == 3}
    lg.info("SEN12MS-CR %s: %d complete patches already on disk (target %d)", roi, len(complete), max_patches)
    # ~1.7 MB per 13-band S2 patch x2 + ~0.5 MB S1 (uint16/float32 256x256) -> ~4 MB per triplet (estimate)
    dl.check_disk(root, max(0, max_patches - len(complete)) * 4_000_000, lg)
    budget = int(max_gb_per_archive * 1e9)
    chosen: set[str] = {k for k, v in have.items() if "s2_cloudy" in v}
    per: dict[int, int] = {}
    for k in chosen:
        sc = int(k.split("_")[-2])
        per[sc] = per.get(sc, 0) + 1

    def want_cloudy(name):
        p = parse(name)
        if not p or p["kind"] != "s2_cloudy" or p["roi"] != roi:
            return False
        k = _key(p)
        if k in chosen:
            return True
        if len(chosen) >= max_patches or per.get(p["scene"], 0) >= per_scene:
            return False
        chosen.add(k)
        per[p["scene"]] = per.get(p["scene"], 0) + 1
        return True

    if len(chosen) < max_patches:
        lg.info("streaming %s_s2_cloudy (selecting up to %d patches, <= %d per scene)", roi, max_patches, per_scene)
        _iter_archive(_archive_source(root, roi, "s2_cloudy"), out_dir, want_cloudy,
                      lambda: len(chosen) >= max_patches, session, lg, budget)
    else:
        lg.info("subset already complete")
    for kind in ("s2", "s1"):
        have = _index(root)
        missing = {k for k in chosen if kind not in have.get(k, {})}
        if not missing:
            continue
        lg.info("streaming %s_%s for %d patches", roi, kind, len(missing))

        def want(name, kind=kind, missing=missing):
            p = parse(name)
            if p and p["kind"] == kind and _key(p) in missing:
                missing.discard(_key(p))
                return True
            return False

        _iter_archive(_archive_source(root, roi, kind), out_dir, want, lambda m=missing: not m, session, lg, budget)
        if missing:
            lg.warning("  %d patches still lack %s after streaming (byte budget or archive order)", len(missing), kind)
    return verify(root, lg)


def _index(root):
    out: dict[str, dict] = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = parse(fn)
            if p:
                out.setdefault(_key(p), {})[p["kind"]] = os.path.join(dp, fn)
    return out


def list_split(root: str | None = None, split: str = "train") -> list[dict]:
    root = root or default_root()
    idx = _index(root)
    recs = []
    for k, v in sorted(idx.items()):
        if not all(t in v for t in ("s1", "s2", "s2_cloudy")):
            continue
        scene = k.rsplit("_p", 1)[0]
        f = stable_fraction(scene)
        sp = "train" if f < 0.8 else ("val" if f < 0.9 else "test")
        if sp == split:
            recs.append(dict(id=k, **v))
    if not recs and not idx:
        raise FileNotFoundError(f"no SEN12MS-CR patches under {root}; run get_data.bat cloud")
    return recs


def verify(root: str, logger=None) -> dict:
    lg = logger or log
    sizes = {s: len(list_split(root, s)) for s in ("train", "val", "test")} if _index(root) else {}
    lg.info("SEN12MS-CR verify: split sizes %s", sizes)
    return dict(split_sizes=sizes)


def load(rec: dict) -> dict:
    s2c = read_tif(rec["s2_cloudy"])
    s2 = read_tif(rec["s2"])
    s1 = read_tif(rec["s1"])
    optical = fill_nan(l1c_to_ours(s2c), 0.0)
    target = fill_nan(l1c_to_ours(s2), 0.0)
    sar = fill_nan(np.clip(sar_to_db(s1[:2]), -50, 30), None)
    return dict(id=rec["id"], optical=optical, sar=sar, dem=None, target=np.clip(target, 0, 1).astype(np.float32),
                mask=None)
