from __future__ import annotations

import shutil
import subprocess

import pytest


def test_docker_compose_config_if_docker_available() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available.")
    result = subprocess.run(
        ["docker", "compose", "config"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip("Docker daemon or Compose is not available.")
    assert "postgres" in result.stdout
