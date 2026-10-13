import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def cli(*args: str):
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "amlv", *map(str, args)],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=environment,
    )


@pytest.fixture(scope="session")
def cli_run():
    return cli


@pytest.fixture(scope="session")
def demo_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo") / "run"
    result = cli("demo", "--out", str(root))
    assert result.returncode == 0, result.stdout + result.stderr
    return root


@pytest.fixture(scope="session")
def fault_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("fault") / "run"
    result = cli("demo", "--out", str(root), "--inject-mismatch")
    assert result.returncode == 1, result.stdout + result.stderr
    cases = [json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()]
    failed = next(record for record in cases if record["status"] == "FAIL")
    return root, failed
