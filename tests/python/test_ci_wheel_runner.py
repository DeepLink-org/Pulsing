"""CI must propagate installation/test failures instead of reporting false green."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/test-wheel.sh"


@pytest.mark.parametrize(
    ("venv_code", "install_code", "pytest_code", "expected"),
    [(13, 0, 0, 13), (0, 23, 0, 23), (0, 0, 42, 42), (0, 0, 0, 0)],
)
def test_ci_propagates_each_exit_code(
    tmp_path, venv_code, install_code, pytest_code, expected
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls"
    uv = bin_dir / "uv"
    uv.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_LOG"\n'
        'if [ "$1" = venv ]; then exit "$VENV_CODE"; fi\n'
        'exit "$INSTALL_CODE"\n'
    )
    uv.chmod(0o755)
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.write_text(
        '#!/bin/sh\nprintf "%s\\n" "pytest $*" >> "$TEST_LOG"\nexit "$PYTEST_CODE"\n'
    )
    python.chmod(0o755)
    env = os.environ | {
        "PATH": str(bin_dir),
        "TEST_LOG": str(log),
        "PULSING_CI_VENV": str(venv),
        "VENV_CODE": str(venv_code),
        "INSTALL_CODE": str(install_code),
        "PYTEST_CODE": str(pytest_code),
    }
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), "3.10"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == expected, result.stderr
    calls = log.read_text()
    assert "venv --clear --python 3.10" in calls
    if venv_code or install_code:
        assert "pytest -m" not in calls
    else:
        assert "pytest -m pytest tests/python -v" in calls
