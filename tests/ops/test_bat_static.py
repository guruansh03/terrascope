"""Static checks for the Windows .bat files (they cannot be executed on this Linux build machine)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BATS = sorted(ROOT.glob("*.bat"))
EXPECTED = {"setup.bat", "gpu_check.bat", "get_data.bat", "train.bat", "evaluate.bat", "test_all.bat",
            "bench_laptop.bat", "make_demo_bundle.bat", "start_local.bat",
            "paper_experiments.bat"}
# variables that hold a command or plain values, not paths (allowed unquoted)
NON_PATH_VARS = {"%PY%", "%TARGET%", "%MODE%", "%OPTS%", "%RC%", "%TERRASCOPE_PORT%", "%GITID%", "%WHEELPY%",
                 "%PYVER%", "%TERRASCOPE_RUNTIME%", "%EXTRA%", "%FORCE%", "%STEP%", "%date%", "%time%", "%%v", "%%t", "%%m", "%%p", "%~1", "%*"}
CRITICAL = re.compile(r'^(?:"%VPY%"|%PY%\s|py\s+-\d|python\s|git\s|pip\s)', re.I)


def ids(p):
    return p.name


def lines(p: Path) -> list[str]:
    return p.read_bytes().decode("ascii").split("\r\n")


def logical(p: Path) -> list[str]:
    out = []
    for l in lines(p):
        s = l.strip()
        if not s or s.upper().startswith("REM ") or s.upper() == "REM" or s.startswith("::"):
            continue
        out.append(s)
    return out


def test_all_expected_bats_exist():
    assert EXPECTED <= {p.name for p in BATS}


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_crlf_ascii(bat):
    raw = bat.read_bytes()
    raw.decode("ascii")  # cmd.exe code pages: keep the files pure ASCII
    assert raw.count(b"\n") == raw.count(b"\r\n"), "LF without CR"
    assert raw.endswith(b"\r\n")


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_header(bat):
    L = logical(bat)
    assert L[0].lower() == "@echo off"
    head = [l.lower() for l in L[:12]]
    assert any(l.startswith("setlocal") for l in head)
    assert 'cd /d "%~dp0"' in head


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_pause_before_final_exit(bat):
    L = [l.lower() for l in logical(bat)]
    i = L.index("exit /b %rc%")
    assert "pause" in L[i - 2:i], "window must stay open (pause) before the final exit"
    assert all(l.startswith("exit /b") for l in L if l.startswith("exit")), "bare 'exit' closes the window"


def errorlevel_problems(bat: Path) -> list[str]:
    L = logical(bat)
    bad = []
    for i, l in enumerate(L):
        cmd = re.sub(r"^\(\s*", "", l)
        if not CRITICAL.match(cmd) or "||" in cmd:
            continue
        nxt = L[i + 1].lower() if i + 1 < len(L) else ""
        if not re.match(r"^if (not )?errorlevel \d", nxt):
            bad.append(l)
    return bad


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_errorlevel_checked_after_critical_commands(bat):
    bad = errorlevel_problems(bat)
    assert not bad, f"no errorlevel check after: {bad}"


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_referenced_files_exist(bat):
    text = bat.read_text(encoding="ascii")
    refs = set(re.findall(r'"((?:scripts|tests)\\[^"%]+\.py)"', text))
    refs |= set(re.findall(r'"(requirements[-\w]*\.txt)"', text))
    mods = set(re.findall(r"-m (terrascope(?:\.\w+)+)", text))
    missing = [r for r in refs if not (ROOT / r.replace("\\", "/")).exists()]
    missing += [m for m in mods if not (ROOT / (m.replace(".", "/") + ".py")).exists()]
    assert not missing, f"{bat.name} references missing files: {missing}"


def quoting_problems(bat: Path) -> list:
    bad = []
    for l in logical(bat):
        low = l.lower()
        if low.startswith(("echo", "call :note", "title", ":")) or ">> \"%batlog%\" echo" in low:
            continue
        stripped = re.sub(r'"[^"]*"', '""', l)  # drop quoted segments
        for var in re.findall(r"%~dp0|%[A-Za-z_]+%|%~\d|%%\w", stripped):
            if var not in NON_PATH_VARS:
                bad.append((var, l))
        if re.search(r"[\w.-]+\\[\w.-]+", stripped):
            bad.append(("path", l))
    return bad


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_paths_are_quoted(bat):
    bad = quoting_problems(bat)
    assert not bad, f"unquoted path in {bat.name}: {bad}"


def test_checker_catches_bad_bat(tmp_path):
    """The checker itself must flag a missing errorlevel check and unquoted paths."""
    b = tmp_path / "bad.bat"
    b.write_bytes(b'@echo off\r\nsetlocal\r\ncd /d %~dp0\r\n"%VPY%" scripts\\pc\\x.py\r\necho next\r\n'
                  b'%VPY% -m pip install -r requirements-pc.txt\r\n')
    assert len(errorlevel_problems(b)) == 1
    q = quoting_problems(b)
    assert any(v == "%~dp0" for v, _ in q) and any(v == "%VPY%" for v, _ in q) and any(v == "path" for v, _ in q)


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_labels_defined(bat):
    text = bat.read_text(encoding="ascii")
    labels = {m.lower() for m in re.findall(r"^:(\w+)", text, re.M)}
    used = {m.lower() for m in re.findall(r"(?:goto|call) :(\w+)", text, re.I)} - {"eof"}
    assert used <= labels, f"undefined labels: {used - labels}"


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_choice_has_timeout_default(bat):
    for l in logical(bat):
        if l.lower().startswith("choice"):
            assert "/t " in l and "/d " in l, f"menu must default after a timeout: {l}"


@pytest.mark.parametrize("bat", [b for b in BATS if b.name != "start_local.bat"], ids=ids)
def test_job_bats_log_history(bat):
    text = bat.read_text(encoding="ascii")
    assert 'set "BATLOG=logs\\bat_history.txt"' in text
    assert '>> "%BATLOG%" echo' in text


def test_start_local_binds_localhost_and_waits_for_health():
    t = (ROOT / "start_local.bat").read_text(encoding="ascii")
    assert "--host 127.0.0.1" in t and "0.0.0.0" not in t
    assert "terrascope.utils.open_when_ready" in t
    assert "timeout /t" not in t.lower(), "browser must open on /api/health, not after a fixed sleep"
    assert t.index('.venv\\Scripts\\python.exe') < t.index('.venv-demo\\Scripts\\python.exe'), "prefer .venv"
    assert "--no-index --find-links" in t


def test_setup_prefers_python_312_then_313():
    t = (ROOT / "setup.bat").read_text(encoding="ascii")
    assert t.index("py -3.12") < t.index("py -3.13")
    assert "download.pytorch.org" not in t  # torch index lives in scripts/pc/setup_env.py (single place)
    s = (ROOT / "scripts" / "pc" / "setup_env.py").read_text()
    assert "https://download.pytorch.org/whl/cu128" in s


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_choice_255_handled_first(bat):
    L = [l.lower() for l in logical(bat)]
    for i, l in enumerate(L):
        if l.startswith("choice"):
            nxt = L[i + 1]
            assert nxt.startswith("if errorlevel 255"), "no console -> errorlevel 255 must fall back to the default"


@pytest.mark.parametrize("bat", BATS, ids=ids)
def test_note_is_safe_for_special_characters(bat):
    t = bat.read_bytes().decode("ascii").replace("\r\n", "\n")
    body = t[t.index("\n:note\n"):]
    assert 'set "NOTE=%~1"' in body and "EnableDelayedExpansion" in body and "!NOTE!" in body
    assert body.index("EnableDelayedExpansion") < body.index("echo [")
    assert "%~1" not in body.replace('set "NOTE=%~1"', ""), "%~1 must never be echoed directly"


@pytest.mark.parametrize("name, script", [("train.bat", "train_all.py"), ("get_data.bat", "get_data.py")])
def test_extra_options_forwarded(name, script):
    t = (ROOT / name).read_text(encoding="ascii")
    assert re.search(re.escape(script) + r'" %\w+% %EXTRA%', t)
    assert "set EXTRA=%EXTRA% %2" in t and "shift" in t


def test_bundle_bat_keeps_wheels_by_default():
    t = (ROOT / "make_demo_bundle.bat").read_text(encoding="ascii")
    assert 'set "OPTS=%*"' in t and "--no-wheels --no-python" in t
    assert "--with-wheels" not in t.replace("REM", "").split("choice")[1]


def test_start_local_tees_pip_and_server_output():
    t = (ROOT / "start_local.bat").read_text(encoding="ascii")
    for cmd in ("-m uvicorn", "-m pip install --no-index", '-m pip install -r "requirements-demo.txt"'):
        line = next(l for l in t.splitlines() if cmd in l)
        assert "-m terrascope.utils.tee_run --job start_local --" in line, line


def test_dockerfile_does_not_trust_any_forwarded_ip():
    if not (ROOT / "Dockerfile").exists():
        pytest.skip("no Dockerfile in this checkout")
    d = (ROOT / "Dockerfile").read_text()
    assert "forwarded-allow-ips" not in d


def test_paper_experiments_steps_in_order():
    t = (ROOT / "paper_experiments.bat").read_text(encoding="ascii")
    cmds = ['"scripts\\pc\\get_data.py" weights',
            "-m terrascope.core.train paper_sim --seeds 0,1,2 --device cuda %FORCE%",
            "-m terrascope.core.evaluate --paper --seeds 0,1,2 --device cuda",
            "-m terrascope.core.risk_sensitivity",
            '"scripts\\pc\\train_all.py" paper',
            '"scripts\\pc\\evaluate_real.py"']
    pos = [t.index(c) for c in cmds]
    assert pos == sorted(pos)
    assert 'if exist "artifacts\\ckpt\\paper_sim_started.txt" set "FORCE="' in t
    assert t.index('set "FORCE="') < t.index('> "artifacts\\ckpt\\paper_sim_started.txt" echo started') < pos[0]


def test_train_bat_offers_paper_mode():
    t = (ROOT / "train.bat").read_text(encoding="ascii")
    assert "(quick full sim paper)" in t and "choice /c QFSP" in t and 'set "MODE=paper"' in t
