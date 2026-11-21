"""Run a command, show its output in the window AND append it to a log file (used by start_local.bat).

python -m terrascope.utils.tee_run --job start_local -- <command> [args...]
-> appends to logs/start_local_<YYYYmmdd>.txt (one file per day, a header per run) and exits with the
command's exit code. Ctrl+C stops the command (it gets the same Ctrl+C from the console) and the log is kept.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def log_path(job: str) -> Path:
    d = Path(os.environ.get("TERRASCOPE_LOG_DIR") or (ROOT / "logs"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job}_{dt.datetime.now():%Y%m%d}.txt"


def run(job: str, cmd: list[str]) -> int:
    lp = log_path(job)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with open(lp, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"\n===== {dt.datetime.now():%Y-%m-%d %H:%M:%S}  $ {' '.join(cmd)}\n")
        f.flush()
        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError as e:
            msg = f"[tee_run] cannot start {cmd[0]}: {e}\n"
            sys.stdout.write(msg)
            f.write(msg)
            return 127
        assert p.stdout is not None
        try:
            for raw in iter(p.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace")
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            f.write("[tee_run] Ctrl+C - stopping\n")
        try:
            rc = p.wait(30)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            p.kill()
            rc = p.wait()
        f.write(f"===== exit code {rc}\n")
    return rc


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: python -m terrascope.utils.tee_run --job NAME -- command [args...]")
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv[:i])
    cmd = argv[i + 1:]
    if not cmd:
        return 2
    return run(a.job, cmd)


if __name__ == "__main__":
    sys.exit(main())
