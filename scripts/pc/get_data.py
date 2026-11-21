"""Download / build the real datasets and pretrained weights on the PC (run by get_data.bat).

    python scripts/pc/get_data.py all            # weights, floods, dem, landcover, burn  (cloud is opt-in: get_data.bat cloud)
    python scripts/pc/get_data.py floods         # Sen1Floods11 v1.1 hand-labelled (GCS, HF-mirror fallback)
    python scripts/pc/get_data.py dem            # Copernicus GLO-30 DEM on every Sen1Floods11 chip (needs floods)
    python scripts/pc/get_data.py landcover      # ESA WorldCover 2021 labels on every Sen1Floods11 chip (needs floods)
    python scripts/pc/get_data.py burn           # HLS burn scars (Hugging Face)
    python scripts/pc/get_data.py cloud [--season spring --max-patches 1500]   # SEN12MS-CR subset (best effort)
    python scripts/pc/get_data.py weights        # Prithvi-EO-2.0 tiny/100M + TerraMind tiny into data/hf_cache

Everything is resumable: re-running skips verified files. Data goes to data/real (override with TERRASCOPE_DATA).
A manifest of what is on disk is written to data/real/manifest.json.
"""
from __future__ import annotations

import argparse, json, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_cache"))
# GDAL /vsicurl/ settings for COG windowed reads (DEM / WorldCover)
for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                 GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3", VSI_CACHE="TRUE",
                 GDAL_HTTP_TIMEOUT="60").items():
    os.environ.setdefault(k, v)
try:  # Windows rasterio wheels: give GDAL's curl a CA bundle
    import certifi
    os.environ.setdefault("CURL_CA_BUNDLE", certifi.where())
    os.environ.setdefault("GDAL_CURL_CA_BUNDLE", certifi.where())
except Exception:
    pass

STEPS_ALL = ["weights", "floods", "dem", "landcover", "burn"]  # "cloud" (SEN12MS-CR) is opt-in: it streams very large archives
MIN_FREE_GB = 5.0


def _manifest_update(key, value):
    from terrascope.real.common import data_root
    p = os.path.join(data_root(), "manifest.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    try:
        with open(p) as f:
            m = json.load(f)
    except (OSError, ValueError):
        m = {}
    m[key] = dict(value, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(p, "w") as f:
        json.dump(m, f, indent=1)


def step_weights(log, a):
    from terrascope.fm.specs import FM_IDS
    from terrascope.fm.weights import fetch_weights
    out = {}
    for mid in FM_IDS:  # 3 foundation models + ImageNet ResNet-18 for unet_r18_imagenet (scratch: nothing)
        p = fetch_weights(mid, logger=log)
        if p is None:
            log.info("  %s: no pretrained weights needed (random init)", mid)
            continue
        out[mid] = dict(path=os.path.relpath(p, ROOT), bytes=os.path.getsize(p))
        log.info("  %s -> %s (%.1f MB)", mid, out[mid]["path"], out[mid]["bytes"] / 1e6)
    return out


def step_floods(log, a):
    from terrascope.real import sen1floods11
    return sen1floods11.download(logger=log, max_files=a.max_files)


def _need_floods():
    from terrascope.real import sen1floods11
    from terrascope.utils.joblog import MissingData
    root = sen1floods11.default_root()
    if not sen1floods11._index(root):
        raise MissingData(root, bat="get_data.bat floods", what="Sen1Floods11 chips are needed first")
    return root


def step_dem(log, a):
    from terrascope.real import sen1floods11
    root = _need_floods()
    n = sen1floods11.build_dem(root, logger=log, limit=a.limit)
    return dict(new_chips=n)


def step_landcover(log, a):
    from terrascope.real import sen1floods11
    from terrascope.real.datasets import DATASETS
    root = _need_floods()
    n = sen1floods11.build_worldcover(root, logger=log, limit=a.limit)
    sizes = {}
    for s in ("train", "val", "test"):
        try:
            sizes[s] = len(DATASETS["worldcover"].list_split(root, s))
        except FileNotFoundError:
            sizes[s] = 0
    log.info("landcover (WorldCover labels) split sizes: %s", sizes)
    return dict(new_chips=n, split_sizes=sizes)


def step_burn(log, a):
    from terrascope.real import burnscars
    return burnscars.download(logger=log)


def step_cloud(log, a):
    from terrascope.real import sen12mscr
    return sen12mscr.download(logger=log, season=a.season, max_patches=a.max_patches, per_scene=a.per_scene)


STEPS = dict(weights=step_weights, floods=step_floods, dem=step_dem, landcover=step_landcover, burn=step_burn,
             cloud=step_cloud)
HINTS = dict(
    weights="Check internet access to huggingface.co (proxy / firewall), then run get_data.bat weights again.",
    floods="Check internet access to storage.googleapis.com; the download resumes where it stopped.",
    dem="Check internet access to copernicus-dem-30m.s3.amazonaws.com; already built chips are kept.",
    landcover="Check internet access to esa-worldcover.s3.eu-central-1.amazonaws.com; already built chips are kept.",
    burn="Check internet access to huggingface.co; the download resumes where it stopped.",
    cloud=("SEN12MS-CR comes from mediatum.ub.tum.de/1554803 (TUM). If the automatic link fails, download "
           "ROIs1158_spring_s1.tar.gz, ROIs1158_spring_s2.tar.gz and ROIs1158_spring_s2_cloudy.tar.gz manually "
           "into data\\real\\sen12mscr\\raw\\ and run get_data.bat cloud again."),
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["all"] + list(STEPS))
    ap.add_argument("--season", default="spring", choices=["spring", "summer", "fall", "winter"])
    ap.add_argument("--max-patches", type=int, default=1500)
    ap.add_argument("--per-scene", type=int, default=40)
    ap.add_argument("--max-files", type=int, default=None, help="debug: limit Sen1Floods11 objects")
    ap.add_argument("--limit", type=int, default=None, help="debug: limit chips for dem/landcover")
    a = ap.parse_args(argv)
    from terrascope.utils.joblog import JobError, run_job
    from terrascope.real.common import data_root
    from terrascope.real.download import free_bytes
    with run_job(f"get_data_{a.what}") as log:
        root = data_root()
        os.makedirs(root, exist_ok=True)
        free = free_bytes(root) / 1e9
        log.info("data root %s, free disk %.1f GB", root, free)
        if free < MIN_FREE_GB:
            raise JobError(f"only {free:.1f} GB free at {root}", "the disk is almost full",
                           f"Free at least {MIN_FREE_GB:.0f} GB (more for 'all': ~10 GB) and run again.")
        steps = STEPS_ALL if a.what == "all" else [a.what]
        failed = {}
        for s in steps:
            log.info("---- step %s ----", s)
            t0 = time.time()
            try:
                res = STEPS[s](log, a)
                _manifest_update(s, dict(ok=True, result=res, minutes=round((time.time() - t0) / 60, 1)))
                log.info("step %s OK (%.1f min)", s, (time.time() - t0) / 60)
            except Exception as e:  # keep going with the other datasets in 'all' mode
                log.exception("step %s FAILED: %s", s, e)
                _manifest_update(s, dict(ok=False, error=f"{type(e).__name__}: {e}"[:500]))
                failed[s] = e
                if len(steps) == 1:
                    if isinstance(e, JobError):
                        raise
                    raise JobError(f"get_data {s} failed: {type(e).__name__}: {e}", "see the traceback above",
                                   HINTS[s]) from e
        if failed:
            raise JobError(f"{len(failed)} of {len(steps)} steps failed: {', '.join(failed)}",
                           "; ".join(f"{k}: {type(v).__name__}: {v}"[:200] for k, v in failed.items()),
                           " | ".join(HINTS[k] for k in failed) + " Then run get_data.bat all again (already downloaded files are kept and skipped).")


if __name__ == "__main__":
    main()
