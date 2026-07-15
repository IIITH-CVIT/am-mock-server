import shutil
import subprocess

import pytest

REPO_ROOT = "."


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff not installed (see requirements-dev.txt)")
def test_ruff_check_passes():
    """Fails the test suite on any lint violation, so `pytest` alone catches lint regressions."""
    result = subprocess.run(
        ["ruff", "check", REPO_ROOT], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
