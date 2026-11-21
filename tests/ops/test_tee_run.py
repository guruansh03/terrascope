import sys

from terrascope.utils import tee_run


def test_tee_shows_and_logs_output_and_keeps_exit_code(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("TERRASCOPE_LOG_DIR", str(tmp_path))
    rc = tee_run.main(["--job", "start_local", "--", sys.executable, "-c",
                       "import sys; print('hello server'); print('err line', file=sys.stderr); sys.exit(3)"])
    assert rc == 3
    out = capfd.readouterr().out
    assert "hello server" in out and "err line" in out
    logs = list(tmp_path.glob("start_local_*.txt"))
    assert len(logs) == 1
    t = logs[0].read_text(encoding="utf-8")
    assert "hello server" in t and "err line" in t and "exit code 3" in t
    tee_run.main(["--job", "start_local", "--", sys.executable, "-c", "print('second run')"])
    assert "second run" in logs[0].read_text(encoding="utf-8")  # appended, not overwritten


def test_tee_missing_command(tmp_path, monkeypatch):
    monkeypatch.setenv("TERRASCOPE_LOG_DIR", str(tmp_path))
    assert tee_run.main(["--job", "x", "--", str(tmp_path / "nope.exe")]) == 127
    assert tee_run.main(["--job", "x"]) == 2
