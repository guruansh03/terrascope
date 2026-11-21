"""One-command test run (OPS) used by test_all.bat -> one summary in logs/test_all_<ts>.txt.

  1. pytest suites (each separately so one broken suite cannot hide the others)
  2. API smoke test (tests/e2e/smoke_api.py) against a temporary server on 127.0.0.1:<port>
  3. UI end-to-end tests (tests/e2e/*.js, Playwright) when Node.js + playwright are installed - else "skipped"

  python tests/test_all.py [--port 8104] [--no-ui] [--no-smoke] [--quick]
Exit code 0 only when nothing failed (skipped steps do not fail the run).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from terrascope.utils.joblog import JobError, current_log_path, job_done, job_failed, start_job  # noqa: E402


def suites() -> list[tuple[str, list[str]]]:
    """(label, pytest targets) for every suite present in the repo."""
    t = ROOT / "tests"
    out = []
    core = sorted(str(p) for p in t.glob("test_*.py") if p.name != "test_all.py")
    if core:
        out.append(("core+api", core))
    for d in ("ops", "real", "fm", "ml", "web"):
        p = t / d
        if p.is_dir() and (list(p.glob("test_*.py")) or list(p.glob("*_test.py"))):
            out.append((d, [str(p)]))
    return out


def parse_junit(path: Path) -> dict:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return {}
    suites_ = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    c = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites_:
        for k in c:
            c[k] += int(s.get(k, 0) or 0)
    c["passed"] = c["tests"] - c["failures"] - c["errors"] - c["skipped"]
    fails = []
    for tc in root.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            fails.append(f"{tc.get('classname')}::{tc.get('name')}")
    c["failed_tests"] = fails[:30]
    return c


def run_pytest(label: str, targets: list[str], workdir: Path, log, extra: list[str]) -> dict:
    xml = workdir / f"junit_{label.replace('+', '_')}.xml"
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *extra, *targets]
    log.info("[pytest %s] %s", label, " ".join(cmd[3:]))
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in (p.stdout + p.stderr).strip().splitlines()[-25:]:
        log.info("  %s", line)
    c = parse_junit(xml)
    # pytest exit codes: 0 ok, 1 tests failed, 5 no tests collected
    status = "passed" if p.returncode in (0, 5) else "failed"
    return {"step": f"pytest {label}", "status": status, "exit_code": p.returncode,
            "seconds": round(time.time() - t0, 1), **c}


def free_port(preferred: int) -> int:
    for port in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free TCP port near %d" % preferred)


class Server:
    def __init__(self, port: int, workdir: Path):
        self.port, self.workdir, self.proc = port, workdir, None
        self.url = f"http://127.0.0.1:{port}"

    def __enter__(self):
        self.out = open(self.workdir / "server.txt", "w", encoding="utf-8")
        env = {**os.environ, "PYTHONUTF8": "1"}
        self.proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "terrascope.server.app:app", "--host", "127.0.0.1",
                                      "--port", str(self.port)], cwd=str(ROOT), env=env, stdout=self.out,
                                     stderr=subprocess.STDOUT)
        return self

    def wait(self, timeout=300) -> dict | None:
        t_end = time.time() + timeout
        while time.time() < t_end:
            if self.proc.poll() is not None:
                return None
            try:
                with urllib.request.urlopen(self.url + "/api/health", timeout=3) as r:
                    if r.status == 200:
                        return json.loads(r.read() or b"{}")
            except Exception:
                time.sleep(1)
        return None

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.out.close()


def run_script(step: str, cmd: list[str], cwd: Path, log, timeout: float) -> dict:
    log.info("[%s] %s", step, " ".join(cmd))
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"step": step, "status": "failed", "reason": f"timed out after {timeout:.0f} s"}
    out = p.stdout + p.stderr
    lines = out.splitlines()
    n_pass = sum(l.startswith("PASS") for l in lines)
    n_fail = sum(l.startswith("FAIL") for l in lines)
    for l in [l for l in lines if l.startswith("FAIL")][:30] + lines[-8:]:
        log.info("  %s", l)
    status = "passed" if p.returncode == 0 and n_fail == 0 else "failed"
    return {"step": step, "status": status, "exit_code": p.returncode, "passed": n_pass, "failures": n_fail,
            "seconds": round(time.time() - t0, 1),
            "failed_tests": [l[:160] for l in lines if l.startswith("FAIL")][:30]}


def node_playwright() -> str | None:
    """Reason string when UI tests cannot run, else None."""
    node = shutil.which("node")
    if not node:
        return "Node.js not installed (optional; install from nodejs.org + 'npx playwright install chromium')"
    p = subprocess.run([node, "-e", "require('playwright')"], capture_output=True, text=True, cwd=str(ROOT))
    if p.returncode:
        return "playwright npm package not found (optional: 'npm i -g playwright && npx playwright install chromium')"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8104)
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--no-smoke", action="store_true")
    ap.add_argument("--quick", action="store_true", help="pytest only, deselect tests marked slow")
    ap.add_argument("--suite", action="append", help="only these pytest suites (labels, e.g. ops)")
    a = ap.parse_args(argv)
    log = start_job("test_all")
    try:
        lp = current_log_path()
        work = (lp.parent / (lp.stem + "_files")) if lp else ROOT / "logs" / "test_all_files"
        work.mkdir(parents=True, exist_ok=True)
        results = []
        extra = ["-m", "not slow"] if a.quick else []
        for label, targets in suites():
            if a.suite and label not in a.suite:
                continue
            results.append(run_pytest(label, targets, work, log, extra))

        if a.quick or a.no_smoke:
            results.append({"step": "api smoke", "status": "skipped", "reason": "--quick/--no-smoke"})
        else:
            port = free_port(a.port)
            with Server(port, work) as srv:
                h = srv.wait()
                if h is None:
                    tail = (work / "server.txt").read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
                    for l in tail:
                        log.info("  server: %s", l)
                    results.append({"step": "server start", "status": "failed",
                                    "reason": "app did not answer /api/health (see server lines above)"})
                else:
                    log.info("server up on %s: %s", srv.url, h)
                    smoke = ROOT / "tests" / "e2e" / "smoke_api.py"
                    if smoke.exists():
                        results.append(run_script("api smoke", [sys.executable, str(smoke), srv.url], ROOT, log, 900))
                    if a.no_ui:
                        results.append({"step": "ui e2e", "status": "skipped", "reason": "--no-ui"})
                    else:
                        why = node_playwright()
                        if why:
                            results.append({"step": "ui e2e", "status": "skipped", "reason": why})
                        else:
                            shots = work / "e2e"
                            (shots / "shots").mkdir(parents=True, exist_ok=True)
                            for js in sorted(glob.glob(str(ROOT / "tests" / "e2e" / "*.js"))):
                                results.append(run_script(f"ui {Path(js).name}", [shutil.which("node"), js, srv.url],
                                                          shots, log, 900))

        # ---- summary
        log.info("=" * 72)
        log.info("TEST SUMMARY")
        for r in results:
            if r["status"] == "skipped":
                log.info("  %-24s SKIPPED  %s", r["step"], r.get("reason", ""))
                continue
            counts = (f"{r.get('passed', '?')} passed, {r.get('failures', 0) + r.get('errors', 0)} failed"
                      + (f", {r['skipped']} skipped" if r.get("skipped") else ""))
            log.info("  %-24s %-8s %s  (%.0f s)", r["step"], r["status"].upper(), counts, r.get("seconds", 0))
            for t in r.get("failed_tests", [])[:10]:
                log.info("      x %s", t)
            if r.get("reason"):
                log.info("      %s", r["reason"])
        failed = [r for r in results if r["status"] == "failed"]
        summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results,
                   "ok": not failed, "log": str(lp) if lp else None}
        (work / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log.info("summary JSON: %s", work / "summary.json")
        if failed:
            log.error("%d step(s) failed: %s", len(failed), ", ".join(r["step"] for r in failed))
            job_failed(JobError(f"{len(failed)} test step(s) failed: " + ", ".join(r["step"] for r in failed),
                                "A test found a problem (details in the TEST SUMMARY above).",
                                "Send this log file and the folder next to it (same name + _files) to Claude."))
            return 1
        job_done("all test steps passed (skipped steps listed above)")
        return 0
    except Exception as e:
        job_failed(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
