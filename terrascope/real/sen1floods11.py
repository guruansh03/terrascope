"""Sen1Floods11 v1.1, hand-labelled subset (flood segmentation; Sentinel-1 + Sentinel-2 L1C, 512 x 512 @ 10 m).

Reference: Bonafilia et al., CVPR-W 2020; data: public GCS bucket ``gs://sen1floods11`` (license CC BY 4.0 per
the project's README - verify before redistribution).

Layout relied on (also used by terratorch's ``Sen1Floods11NonGeo``, which is installed here and was read to
confirm it):
    v1.1/data/flood_events/HandLabeled/S1Hand/<chip>_S1Hand.tif      2 bands VV, VH (dB, NaN = nodata)
    v1.1/data/flood_events/HandLabeled/S2Hand/<chip>_S2Hand.tif      13 bands L1C B1..B12 incl. B8A (DN x1e4)
    v1.1/data/flood_events/HandLabeled/LabelHand/<chip>_LabelHand.tif 1 band: -1 nodata, 0 not water, 1 water
    v1.1/splits/flood_handlabeled/flood_{train,valid,test}_data.{csv|txt}   official splits
The S1 band order / dB scaling and the label coding are from the dataset README as remembered, NOT verified
offline -> ``load`` checks value ranges and converts linear SAR to dB if needed (``bands.sar_to_db``).
Split files: the GCS release ships CSVs (two filename columns); terratorch expects .txt - both are parsed, and
the files are DISCOVERED by listing (pattern ``flood_(train|valid|test)_data``) rather than hard-coded.

Download: GCS JSON API listing (sizes + md5) + per-object media download (resumable, md5-verified).
Fallback: Hugging Face mirrors discovered with ``HfApi.list_datasets(search="sen1floods11")`` (none is
hard-coded; set ``SEN1FLOODS11_HF_REPO`` to force one) - structure is again discovered by filename suffix.
"""
from __future__ import annotations

import logging, os, re

import numpy as np

from .bands import l1c_to_ours, sar_to_db
from .common import chip_id, data_root, fill_nan, index_files, read_split_ids, read_tif
from . import download as dl

NAME = "sen1floods11"
TASK = "flood"
BUCKET = "sen1floods11"
PREFIXES = [
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/S2Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/splits/flood_handlabeled/",
]
EXTRA_OBJECTS_PREFIX = "v1.1/Sen1Floods11_Metadata"  # geojson with acquisition dates (optional)
SPLIT_RE = re.compile(r"flood_(train|valid|test|bolivia)_data\.(csv|txt)$", re.I)
SPLITS = {"train": "train", "val": "valid", "test": "test"}
SUFFIX = {"s1": "_S1Hand.tif", "s2": "_S2Hand.tif", "label": "_LabelHand.tif",
          "dem": "_DEM.tif", "wc": "_WC.tif"}

log = logging.getLogger("terrascope.real.sen1floods11")


def default_root() -> str:
    return os.path.join(data_root(), NAME)


def derived_dir(root: str, kind: str) -> str:
    return os.path.join(root, "derived", kind)


# --------------------------------------------------------------------------------------------- download
def _want(name: str) -> bool:
    return any(name.startswith(p) for p in PREFIXES) or name.startswith(EXTRA_OBJECTS_PREFIX)


def download(root: str | None = None, logger=None, session=None, max_files: int | None = None) -> dict:
    """Fetch the hand-labelled subset + splits. Resumable; skips verified files. Returns a summary dict."""
    lg = logger or log
    root = root or default_root()
    os.makedirs(root, exist_ok=True)
    objs = None
    try:
        objs = []
        for p in PREFIXES + [EXTRA_OBJECTS_PREFIX]:
            got = dl.gcs_list(BUCKET, p, session=session)
            lg.info("GCS gs://%s/%s: %d objects", BUCKET, p, len(got))
            objs += got
        objs = [o for o in objs if _want(o["name"]) and not o["name"].endswith("/")]
        if not any(o["name"].endswith("_S2Hand.tif") for o in objs):
            raise dl.DownloadError("GCS listing returned no *_S2Hand.tif objects (layout changed?)")
    except Exception as e:  # only a failed LISTING switches source; download errors below propagate
        lg.warning("GCS listing failed (%s: %s) - trying Hugging Face mirrors", type(e).__name__, e)
        objs = None
    if objs is not None:
        if max_files:
            objs = objs[:max_files]
        need = sum(o["size"] for o in objs if not dl.file_ok(os.path.join(root, o["name"]), o["size"]))
        dl.check_disk(root, need, lg)
        lg.info("downloading %d objects (%.2f GB still missing) from GCS", len(objs), need / 1e9)
        for i, o in enumerate(objs, 1):
            dest = os.path.join(root, *o["name"].split("/"))
            dl.http_download(dl.gcs_media_url(BUCKET, o["name"]), dest, size=o["size"], md5=o.get("md5Hash"),
                             session=session, logger=lg)
            if i % 50 == 0 or i == len(objs):
                lg.info("  %d/%d files ok", i, len(objs))
        source = f"gs://{BUCKET}"
    else:
        source = _download_hf(root, lg)
    summ = verify(root, lg)
    summ["source"] = source
    return summ


def _download_hf(root, lg, api=None) -> str:
    api = api or dl.hf_api()
    forced = os.environ.get("SEN1FLOODS11_HF_REPO")
    if forced:
        cands = [forced]
    else:
        cands = [d.id for d in api.list_datasets(search="sen1floods11", limit=20)]
        lg.info("HF datasets matching 'sen1floods11' (UNVERIFIED community mirrors): %s", cands)
    for repo in cands:
        try:
            files = dl.hf_list_files(repo, api=api)
        except Exception as e:
            lg.info("  %s: cannot list (%s)", repo, e)
            continue
        hand = [f for f in files if f.endswith(("_S1Hand.tif", "_S2Hand.tif", "_LabelHand.tif")) or SPLIT_RE.search(f)]
        archives = [f for f in files if f.endswith((".tar.gz", ".tgz", ".zip", ".tar"))]
        if hand:
            lg.info("  using %s: %d hand-labelled files found", repo, len(hand))
            for i, f in enumerate(hand, 1):
                dl.hf_download(repo, f, os.path.join(root, "hf"))
                if i % 50 == 0:
                    lg.info("  %d/%d", i, len(hand))
            return f"hf:{repo}"
        if archives:
            lg.info("  using %s: archives %s", repo, archives)
            for f in archives:
                p = dl.hf_download(repo, f, os.path.join(root, "hf_archives"))
                dl.safe_extract(p, os.path.join(root, "hf"), logger=lg)
            if index_files(os.path.join(root, "hf"), ("_S2Hand.tif",)):
                return f"hf:{repo}"
        lg.info("  %s does not contain the hand-labelled chips; skipping", repo)
    raise dl.DownloadError("Sen1Floods11 could not be downloaded from GCS or any Hugging Face mirror. "
                           "Set SEN1FLOODS11_HF_REPO=<owner/name> to a mirror you trust, or copy the "
                           "v1.1 folder into " + root)


# --------------------------------------------------------------------------------------------- indexing
def _index(root):
    files = index_files(root, tuple(SUFFIX.values()))
    out: dict[str, dict] = {}
    for fn, p in files.items():
        for k, suf in SUFFIX.items():
            if fn.endswith(suf):
                out.setdefault(fn[: -len(suf)], {})[k] = p
    return out


def split_files(root: str) -> dict[str, str]:
    out = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            m = SPLIT_RE.search(fn)
            if m:
                key = m.group(1).lower()
                # prefer the hand-labelled split directory if several exist
                if key not in out or "handlabeled" in dp.lower():
                    out[key] = os.path.join(dp, fn)
    return out


def list_split(root: str | None = None, split: str = "train", require=("s2", "label")) -> list[dict]:
    root = root or default_root()
    sf = split_files(root)
    name = SPLITS.get(split, split)
    if name not in sf:
        raise FileNotFoundError(f"Sen1Floods11 split file flood_{name}_data.(csv|txt) not found under {root}; "
                                "run get_data.bat floods")
    ids = read_split_ids(sf[name])
    idx = _index(root)
    recs, missing = [], 0
    for c in ids:
        r = idx.get(c)
        if r is None or any(k not in r for k in require):
            missing += 1
            continue
        recs.append(dict(id=c, **r))
    if missing:
        log.warning("Sen1Floods11 %s: %d of %d listed chips have missing files", split, missing, len(ids))
    return recs


def verify(root: str, logger=None) -> dict:
    lg = logger or log
    idx = _index(root)
    complete = [c for c, r in idx.items() if all(k in r for k in ("s1", "s2", "label"))]
    sizes = {}
    for s in SPLITS:
        try:
            sizes[s] = len(list_split(root, s))
        except FileNotFoundError:
            sizes[s] = 0
    lg.info("Sen1Floods11 verify: %d chips with S1+S2+label; split sizes %s", len(complete), sizes)
    return dict(chips=len(complete), split_sizes=sizes)


# --------------------------------------------------------------------------------------------- loading
def load(rec: dict, with_dem: bool = True) -> dict:
    s2 = read_tif(rec["s2"])
    if s2.shape[0] != 13:
        raise ValueError(f"{rec['s2']}: expected 13 bands, found {s2.shape[0]}")
    optical = fill_nan(l1c_to_ours(s2), 0.0)
    sar = None
    if rec.get("s1"):
        s1 = read_tif(rec["s1"])
        sar = fill_nan(np.clip(sar_to_db(s1[:2]), -50, 30), None)
    dem = None
    if with_dem and rec.get("dem"):
        dem = fill_nan(read_tif(rec["dem"])[:2], 0.0)
    lab = read_tif(rec["label"])[0]
    mask = np.where(np.isfinite(lab), lab, -1).astype(np.int64)
    mask[(mask != 0) & (mask != 1)] = -1
    return dict(id=rec["id"], optical=optical, sar=sar, dem=dem, mask=mask)


def load_landcover(rec: dict, with_dem: bool = True) -> dict:
    """Same chip, WorldCover-derived labels (255 -> -1)."""
    s = load(dict(rec), with_dem)
    wc = read_tif(rec["wc"])[0]
    m = np.where(np.isfinite(wc), wc, 255).astype(np.int64)
    m[m > 5] = -1
    s["mask"] = m
    return s


# --------------------------------------------------------------------------------------------- derived layers
def build_dem(root: str | None = None, logger=None, session=None, tile_opener=None, limit=None) -> int:
    """Copernicus DEM (elevation, slope) on every chip grid -> derived/DEM/<chip>_DEM.tif. Resumable."""
    from .dem import dem_for_geotiff
    from .geotiff import write_geotiff
    lg = logger or log
    root = root or default_root()
    out_dir = derived_dir(root, "DEM")
    tiles = os.path.join(data_root(), "dem_tiles")
    idx = _index(root)
    todo = [(c, r) for c, r in sorted(idx.items()) if "s2" in r]
    if limit:
        todo = todo[:limit]
    n = 0
    for i, (c, r) in enumerate(todo, 1):
        dest = os.path.join(out_dir, c + SUFFIX["dem"])
        if os.path.exists(dest):
            continue
        dem, crs, T = dem_for_geotiff(r["s2"], cache_dir=None if tile_opener else tiles, session=session,
                                      logger=lg, tile_opener=tile_opener)
        write_geotiff(dest, dem, T, crs=crs, dtype="float32")
        n += 1
        if i % 25 == 0:
            lg.info("  DEM %d/%d chips", i, len(todo))
    lg.info("DEM: %d new chips written to %s", n, out_dir)
    return n


def build_worldcover(root: str | None = None, logger=None, tile_opener=None, limit=None) -> int:
    """WorldCover 2021 mapped to our 6 classes on every chip grid -> derived/WorldCover/<chip>_WC.tif."""
    import rasterio
    from .worldcover import labels_for_grid
    from .geotiff import write_geotiff
    lg = logger or log
    root = root or default_root()
    out_dir = derived_dir(root, "WorldCover")
    failed: list[str] = []
    idx = _index(root)
    todo = [(c, r) for c, r in sorted(idx.items()) if "s2" in r]
    if limit:
        todo = todo[:limit]
    n = 0
    for i, (c, r) in enumerate(todo, 1):
        dest = os.path.join(out_dir, c + SUFFIX["wc"])
        if os.path.exists(dest):
            continue
        with rasterio.open(r["s2"]) as ds:
            crs, T, H, W = ds.crs, ds.transform, ds.height, ds.width
        try:
            lab = labels_for_grid(crs, T, H, W, tile_opener=tile_opener, logger=lg)
        except Exception as e:  # not written -> retried on the next run
            failed.append(c)
            lg.warning("  WorldCover for %s failed (%s: %s); chip skipped, will retry next run", c,
                       type(e).__name__, str(e)[:150])
            continue
        write_geotiff(dest, lab, T, crs=crs, dtype="uint8", nodata=255)
        n += 1
        if i % 25 == 0:
            lg.info("  WorldCover %d/%d chips", i, len(todo))
    lg.info("WorldCover: %d new chips written to %s", n, out_dir)
    if failed:
        raise RuntimeError(f"WorldCover labels failed for {len(failed)} chips (e.g. {failed[:3]}); "
                           "check internet access and run get_data.bat landcover again")
    return n
