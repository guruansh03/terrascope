"""Open the browser once the server answers /api/health (used by start_local.bat).

python -m terrascope.utils.open_when_ready http://127.0.0.1:8000 [--timeout 300]
Exit 0 when opened, 1 on timeout (the server window shows why).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import webbrowser


def wait_health(base: str, timeout: float = 300.0, interval: float = 1.0) -> dict | None:
    url = base.rstrip("/") + "/api/health"
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return json.loads(r.read() or b"{}")
        except Exception:
            pass
        time.sleep(interval)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    h = wait_health(a.url, a.timeout)
    if h is None:
        print(f"[open_when_ready] {a.url} did not answer /api/health within {a.timeout:.0f} s", flush=True)
        return 1
    print(f"[open_when_ready] server ready ({h}) -> opening {a.url}", flush=True)
    if not a.no_browser:
        webbrowser.open(a.url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
