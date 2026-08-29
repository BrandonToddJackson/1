import sys

import pytest

from pipeline import procutil


def test_missing_binary_raises_missing_binary_error():
    with pytest.raises(procutil.MissingBinaryError, match="definitely-not-a-real-binary-xyz"):
        procutil.run_or_raise(["definitely-not-a-real-binary-xyz"], "test-step")


def test_failing_command_includes_stderr_tail():
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('boom: bad input\\n'); sys.exit(1)"]
    with pytest.raises(procutil.SubprocessFailedError, match="boom: bad input"):
        procutil.run_or_raise(cmd, "test-step")


def test_failing_command_error_is_not_the_bare_default_message():
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('specific reason\\n'); sys.exit(1)"]
    with pytest.raises(procutil.SubprocessFailedError) as exc_info:
        procutil.run_or_raise(cmd, "test-step")
    message = str(exc_info.value)
    assert "specific reason" in message
    assert message != "Command '[...]' returned non-zero exit status 1."


def test_successful_command_returns_completed_process():
    cmd = [sys.executable, "-c", "print('ok')"]
    result = procutil.run_or_raise(cmd, "test-step")
    assert result.stdout.strip() == "ok"


def test_cwd_is_respected(tmp_path):
    (tmp_path / "marker.txt").write_text("hi")
    cmd = [sys.executable, "-c", "import os; print(os.path.exists('marker.txt'))"]
    result = procutil.run_or_raise(cmd, "test-step", cwd=tmp_path)
    assert result.stdout.strip() == "True"
