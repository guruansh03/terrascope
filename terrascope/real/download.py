"""Download helpers shared by the dataset modules: resumable HTTP, GCS JSON-API listing, Hugging Face,
disk-space checks and safe archive extraction.

Everything network-facing goes through :data:`SESSION_FACTORY` (``requests.Session`` by default) so tests can
inject a fake session; nothing here runs at import time.
"""
from __future__ import annotations

import base64, hashlib, logging, os, shutil, tarfile, time, urllib.parse, zipfile

log = logging.getLogger("terrascope.real.download")

CHUNK = 1 << 20
USER_AGENT = "TerraScope-get_data/1.0 (+research; resumable)"


def _default_session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


SESSION_FACTORY = _default_session


class DownloadError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- disk
def free_bytes(path: str) -> int:
    p = os.path.abspath(path)
    while not os.path.exists(p):
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free


def check_disk(path: str, need_bytes: int, logger=None, margin: float = 1.15) -> None:
    lg = logger or log
    free = free_bytes(path)
    need = int(need_bytes * margin)
    lg.info("disk check: need ~%.2f GB (incl. 15%% margin), free %.2f GB at %s", need / 1e9, free / 1e9, path)
    if free < need:
        raise DownloadError(f"not enough free disk space at {path}: need ~{need / 1e9:.1f} GB, "
                            f"free {free / 1e9:.1f} GB. Free some space or set TERRASCOPE_DATA to another drive.")


# ------------------------------------------------------------------------------------------- hashing
def md5_b64(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return base64.b64encode(h.digest()).decode()


def file_ok(path: str, size: int | None = None, md5: str | None = None) -> bool:
    if not os.path.exists(path):
        return False
    if size is not None and os.path.getsize(path) != int(size):
        return False
    if md5 and md5_b64(path) != md5:
        return False
    return True


# ------------------------------------------------------------------------------------------- http
class _Progress:
    def __init__(self, name, total, logger, every_s=10.0):
        self.name, self.total, self.lg, self.every = name, total, logger, every_s
        self.t0 = self.tl = time.time()
        self.done = 0

    def add(self, n, offset=0):
        self.done += n
        now = time.time()
        if now - self.tl >= self.every:
            self.tl = now
            rate = self.done / max(now - self.t0, 1e-6)
            tot = (self.total or 0)
            have = self.done + offset
            eta = (tot - have) / rate if tot and rate > 0 else None
            self.lg.info("  %s: %.1f/%s MB  %.1f MB/s%s", self.name, have / 1e6,
                         f"{tot / 1e6:.1f}" if tot else "?", rate / 1e6,
                         f"  ETA {eta / 60:.1f} min" if eta is not None else "")


def http_download(url: str, dest: str, size: int | None = None, md5: str | None = None, session=None,
                  retries: int = 5, logger=None, timeout: float = 60.0) -> str:
    """Resumable download (HTTP Range) into ``dest`` via ``dest + '.part'``; verifies size / md5 (base64)."""
    lg = logger or log
    if file_ok(dest, size, md5):
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    part = dest + ".part"
    sess = session or SESSION_FACTORY()
    last = None
    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if size is not None and have > size:
            os.remove(part)
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with sess.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416 and have > 0 and (size is None or have == size):
                    pass  # .part already holds the whole file (server: range not satisfiable) -> finalise below
                elif r.status_code == 404:
                    raise DownloadError(f"404 Not Found: {url}")
                elif r.status_code not in (200, 206):
                    raise DownloadError(f"HTTP {r.status_code} for {url}")
                else:
                    mode = "ab" if (have and r.status_code == 206) else "wb"
                    if mode == "wb":
                        have = 0
                    total = size
                    if total is None:
                        cl = r.headers.get("Content-Length")
                        total = (int(cl) + have) if cl and cl.isdigit() else None
                    prog = _Progress(os.path.basename(dest), total, lg)
                    with open(part, mode) as f:
                        for chunk in r.iter_content(CHUNK):
                            if chunk:
                                f.write(chunk)
                                prog.add(len(chunk), have)
            if size is not None and os.path.getsize(part) != size:
                raise DownloadError(f"size mismatch for {dest}: got {os.path.getsize(part)}, expected {size}")
            if md5 and md5_b64(part) != md5:
                os.remove(part)
                raise DownloadError(f"md5 mismatch for {dest} (file removed, will re-download)")
            os.replace(part, dest)
            return dest
        except DownloadError as e:
            last = e
            if "404" in str(e):
                raise
        except Exception as e:  # network errors
            last = e
        wait = min(60, 2 ** attempt)
        lg.warning("  download attempt %d/%d failed (%s); retrying in %ds", attempt, retries, last, wait)
        time.sleep(wait if not os.environ.get("TERRASCOPE_NO_SLEEP") else 0)
    raise DownloadError(f"giving up on {url}: {last}")


# ------------------------------------------------------------------------------------------- GCS
GCS_API = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
GCS_MEDIA = "https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{obj}?alt=media"


def gcs_list(bucket: str, prefix: str, session=None) -> list[dict]:
    """List objects of a PUBLIC bucket via the JSON API (no credentials). Returns [{name,size,md5Hash}]."""
    sess = session or SESSION_FACTORY()
    items, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000, "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            params["pageToken"] = token
        r = sess.get(GCS_API.format(bucket=bucket), params=params, timeout=60)
        if r.status_code != 200:
            raise DownloadError(f"GCS listing failed (HTTP {r.status_code}) for gs://{bucket}/{prefix}")
        js = r.json()
        for it in js.get("items", []):
            items.append(dict(name=it["name"], size=int(it.get("size", 0)), md5Hash=it.get("md5Hash")))
        token = js.get("nextPageToken")
        if not token:
            return items


def gcs_media_url(bucket: str, name: str) -> str:
    return GCS_MEDIA.format(bucket=bucket, obj=urllib.parse.quote(name, safe=""))


# ------------------------------------------------------------------------------------------- Hugging Face
def hf_api():
    from huggingface_hub import HfApi
    return HfApi()


def hf_list_files(repo_id: str, repo_type: str = "dataset", api=None) -> list[str]:
    api = api or hf_api()
    return list(api.list_repo_files(repo_id, repo_type=repo_type))


def _retryable(e: BaseException) -> bool:
    """429 / 5xx / connection-level failures are worth retrying; 401/403/404 are not."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code is not None:
        return code == 429 or code >= 500
    name = type(e).__name__
    return any(k in name for k in ("Connection", "Timeout", "ChunkedEncoding", "ProtocolError")) or isinstance(e, OSError)


def hf_download(repo_id: str, filename: str, local_dir: str, repo_type: str = "dataset", cache_dir=None,
                retries: int = 6, logger=None) -> str:
    """hf_hub_download with exponential backoff on 429 / 5xx / network errors (hub resumes partial files)."""
    from huggingface_hub import hf_hub_download
    lg = logger or log
    for attempt in range(1, retries + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=local_dir,
                                   cache_dir=cache_dir)
        except Exception as e:
            if attempt == retries or not _retryable(e):
                raise
            wait = min(120, 5 * 2 ** (attempt - 1))
            lg.warning("  HF download %s failed (%s: %s); retry %d/%d in %ds", filename, type(e).__name__,
                       str(e)[:120], attempt, retries - 1, wait)
            time.sleep(0 if os.environ.get("TERRASCOPE_NO_SLEEP") else wait)


# ------------------------------------------------------------------------------------------- archives
def _safe_target(base: str, member_name: str) -> str:
    base = os.path.abspath(base)
    tgt = os.path.abspath(os.path.join(base, member_name))
    if not (tgt == base or tgt.startswith(base + os.sep)):
        raise DownloadError(f"refusing to extract outside target dir: {member_name}")
    return tgt


def safe_extract(archive: str, out_dir: str, members_filter=None, max_total_bytes: int = 200 * 10**9,
                 logger=None) -> list[str]:
    """Extract .tar(.gz) / .zip refusing path traversal, links and absurd total size. Returns written paths."""
    lg = logger or log
    written, total = [], 0
    os.makedirs(out_dir, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir() or (members_filter and not members_filter(info.filename)):
                    continue
                total += info.file_size
                if total > max_total_bytes:
                    raise DownloadError("archive expands beyond the allowed size")
                tgt = _safe_target(out_dir, info.filename)
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                with zf.open(info) as src, open(tgt, "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                written.append(tgt)
    else:
        with tarfile.open(archive, "r:*") as tf:
            for m in tf:
                if not m.isfile() or (members_filter and not members_filter(m.name)):
                    continue
                total += m.size
                if total > max_total_bytes:
                    raise DownloadError("archive expands beyond the allowed size")
                tgt = _safe_target(out_dir, m.name)
                if os.path.exists(tgt) and os.path.getsize(tgt) == m.size:
                    written.append(tgt)
                    continue
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                src = tf.extractfile(m)
                with open(tgt + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, CHUNK)
                os.replace(tgt + ".part", tgt)
                written.append(tgt)
    lg.info("extracted %d files (%.1f MB) from %s", len(written), total / 1e6, os.path.basename(archive))
    return written


def stream_tar_members(url: str, out_dir: str, want, stop, session=None, logger=None, timeout=120.0,
                       max_bytes: int | None = None) -> list[str]:
    """Stream a remote .tar(.gz) WITHOUT storing it, extracting members for which ``want(name)`` is true,
    until ``stop()`` returns True or ``max_bytes`` of compressed data were read. Returns extracted paths.

    Used for best-effort subsets of huge archives (SEN12MS-CR). Not resumable (by nature of streaming);
    already-extracted members are skipped on re-run.
    """
    lg = logger or log
    sess = session or SESSION_FACTORY()
    out = []
    with sess.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            raise DownloadError(f"HTTP {r.status_code} for {url}")
        raw = r.raw
        raw.decode_content = True
        counted = _CountingReader(raw, max_bytes)
        mode = "r|gz" if url.split("?")[0].endswith((".gz", ".tgz")) or "gz" in url else "r|"
        try:
            with tarfile.open(fileobj=counted, mode=mode) as tf:
                for m in tf:
                    if stop():
                        break
                    if not m.isfile() or not want(m.name):
                        continue
                    tgt = _safe_target(out_dir, m.name)
                    if os.path.exists(tgt) and os.path.getsize(tgt) == m.size:
                        out.append(tgt)
                        continue
                    os.makedirs(os.path.dirname(tgt), exist_ok=True)
                    src = tf.extractfile(m)
                    with open(tgt + ".part", "wb") as dst:
                        shutil.copyfileobj(src, dst, CHUNK)
                    os.replace(tgt + ".part", tgt)
                    out.append(tgt)
                    if len(out) % 200 == 0:
                        lg.info("  streamed %d files, %.2f GB read", len(out), counted.n / 1e9)
        except _ByteBudget:
            lg.warning("  byte budget of %.1f GB reached; stopping stream", (max_bytes or 0) / 1e9)
    return out


class _ByteBudget(Exception):
    pass


class _CountingReader:
    def __init__(self, raw, max_bytes):
        self.raw, self.max, self.n = raw, max_bytes, 0

    def read(self, n=-1):
        b = self.raw.read(n)
        self.n += len(b)
        if self.max and self.n > self.max:
            raise _ByteBudget()
        return b
