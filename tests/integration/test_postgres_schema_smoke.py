from __future__ import annotations

import shutil
import subprocess

import pytest


def test_postgres_signal_audit_table_exists_when_docker_is_running() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available.")

    ready = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "pg_isready",
            "-U",
            "bybit",
            "-d",
            "bybit_edge",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if ready.returncode != 0:
        pytest.skip("PostgreSQL container is not running.")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "postgres",
            "psql",
            "-U",
            "bybit",
            "-d",
            "bybit_edge",
            "-tAc",
            "select to_regclass('public.signal_audit_logs');",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "signal_audit_logs"
