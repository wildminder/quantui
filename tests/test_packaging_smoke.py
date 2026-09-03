"""NTH-010: tests for the opt-in packaging smoke (gate_tests.sh PACKAGING=1).

Two layers:
1. The gate script contains the opt-in switch and invokes the smoke helper
   after the pytest stage (grep-equivalent structural assertions).
2. scripts/packaging_smoke.py resolves entry points in BOTH venv layouts
   (Windows Scripts/ + POSIX bin/) and fails on a missing entry point —
   tested with injected tmp paths, no real venv required.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(REPO_ROOT, "scripts", "gate_tests.sh")
SMOKE = os.path.join(REPO_ROOT, "scripts", "packaging_smoke.py")


def _import_helper():
    # The helper lives outside the package (scripts/); load it by path.
    import importlib.util

    spec = importlib.util.spec_from_file_location("packaging_smoke", SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_gate_has_packaging_optin():
    with open(GATE, encoding="utf-8") as fh:
        src = fh.read()
    assert "PACKAGING" in src, "gate_tests.sh lost the PACKAGING=1 opt-in"
    assert "packaging_smoke.py" in src, "gate no longer invokes the smoke helper"


def test_smoke_entry_point_win_layout(tmp_path):
    helper = _import_helper()
    venv = tmp_path / "smoke-venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "quantui.exe").write_text("shim", encoding="utf-8")
    assert helper.check_entry_point(str(venv), "quantui") == 0


def test_smoke_entry_point_posix_layout(tmp_path):
    helper = _import_helper()
    venv = tmp_path / "smoke-venv"
    bindir = venv / "bin"
    bindir.mkdir(parents=True)
    (bindir / "quantui").write_text("#!/bin/sh\n", encoding="utf-8")
    assert helper.check_entry_point(str(venv), "quantui") == 0


def test_smoke_missing_entry_point_fails(tmp_path):
    helper = _import_helper()
    venv = tmp_path / "empty-venv"
    venv.mkdir(parents=True)
    # Empty venv: no entry point anywhere -> must fail loudly.
    assert helper.check_entry_point(str(venv), "quantui") == 1


def test_smoke_cli_arg_parsing(tmp_path):
    # The __main__ path takes (venv_dir, entry_name) argv and exits 0/1/2.
    import subprocess

    venv = tmp_path / "smoke-venv"
    scripts = venv / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "quantui.exe").write_text("shim", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, SMOKE, str(venv), "quantui"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc_bad = subprocess.run(
        [sys.executable, SMOKE, str(tmp_path), "quantui"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc_bad.returncode == 1
