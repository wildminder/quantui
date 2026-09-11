"""Tests for scripts/precommit_ruff.sh (NTH-008, step 2.6).

The ruff count gate compares the live finding count against the frozen baseline
in the frozen baseline file, then runs mypy over the five core modules.
RUFF / MYPY can be overridden so the script is exercisable without the real
tools (and without waiting on mypy).

The regression pinned here: when ruff finds nothing it prints "All checks
passed!" instead of "Found N errors.", so the count-extraction pipeline
(`grep -oE '^Found [0-9]+ error' | grep | head`) matches nothing. Under
`set -euo pipefail` that made the assignment fail and killed the script
silently -- i.e. the gate broke at the exact moment NTH-008 reached zero
findings. `test_zero_findings_passes` is the guard.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO_ROOT, "scripts", "precommit_ruff.sh")
BASELINE_FILE = os.path.join(REPO_ROOT, "docs", "reviews", "ruff-baseline.txt")

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def _baseline() -> int:
    """Read the frozen baseline count; a missing file means baseline 0.

    This mirrors the script's clone-safety behavior: docs/ is not tracked in
    git, so on a fresh clone the baseline file is absent and the gate runs in
    zero-tolerance mode.
    """
    if not os.path.isfile(BASELINE_FILE):
        return 0
    with open(BASELINE_FILE, encoding="utf-8") as fh:
        for line in fh:
            m = re.match(r"^Found (\d+)", line)
            if m:
                return int(m.group(1))
    raise AssertionError(f"{BASELINE_FILE} has no 'Found N errors.' line")


def _stub(tmp_path, name, body):
    """Write a runnable stub script; returns its path with forward slashes."""
    if os.name == "nt":
        path = tmp_path / f"{name}.bat"
    else:
        path = tmp_path / f"{name}.sh"
        body = "#!/bin/sh\n" + body
    path.write_text(body, encoding="utf-8")
    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path.as_posix()


def _run_gate(tmp_path, ruff_out="", ruff_rc=0, mypy_rc=0):
    """Run the gate with stubbed ruff/mypy; returns (returncode, output)."""
    body = f"echo {ruff_out}\nexit /b {ruff_rc}\n" if os.name == "nt" else f"echo {ruff_out}\nexit {ruff_rc}\n"
    ruff = _stub(tmp_path, "ruff", body)
    mypy = _stub(
        tmp_path,
        "mypy",
        f"@echo off{os.linesep}exit /b {mypy_rc}{os.linesep}"
        if os.name == "nt"
        else f"exit {mypy_rc}\n",
    )
    env = dict(os.environ, RUFF=ruff, MYPY=mypy)
    proc = subprocess.run(
        [bash, GATE], cwd=REPO_ROOT, env=env, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_zero_findings_passes(tmp_path):
    """ruff prints 'All checks passed!' when clean -- the gate must still run."""
    code, out = _run_gate(tmp_path, ruff_out="All checks passed!")
    assert "ruff gate: live=0" in out, out
    assert code == 0, f"gate exited {code} with output:\n{out}"


def test_findings_at_baseline_pass(tmp_path):
    base = _baseline()
    code, out = _run_gate(tmp_path, ruff_out=f"Found {base} errors.")
    assert f"ruff gate: live={base} baseline={base}" in out, out
    assert code == 0, out


def test_findings_above_baseline_fail(tmp_path):
    base = _baseline()
    # With a missing baseline file the gate is zero-tolerance: any finding
    # fails. "Found 3 errors." exercises both the baseline>0 and the
    # baseline==0 (fresh-clone) paths identically.
    code, out = _run_gate(tmp_path, ruff_out=f"Found {base + 3} errors.")
    assert code != 0, "more findings than the baseline must fail the gate"
    assert "new ruff findings introduced" in out, out


def test_mypy_failure_blocks_gate(tmp_path):
    code, out = _run_gate(tmp_path, ruff_out="All checks passed!", mypy_rc=1)
    assert code != 0, "a mypy failure must fail the gate"
