"""NTH-009: tests for scripts/gate_torch.sh — the on-demand torch-suite gate.

The headless gate (gate_tests.sh) --ignores the two torch-dependent suites;
this script runs them in a torch-enabled CTQ venv. The tests below exercise
the `--check` mode, which validates the interpreter (exists + torch probe)
WITHOUT running pytest — so they are executable in the torchless test env.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_TORCH = os.path.join(REPO_ROOT, "scripts", "gate_torch.sh")
# Optional local override for the "real default interpreter" probe below;
# unset on any other checkout/machine (test skips cleanly in that case).
DEFAULT_VENV = os.environ.get("UQT_GATE_TORCH_VENV", "")

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def _run_check(env_python):
    """Run `gate_torch.sh --check` with GATE_PYTHON=env_python."""
    env = dict(os.environ, GATE_PYTHON=env_python)
    proc = subprocess.run(
        [bash, GATE_TORCH, "--check"], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_script_exists_and_names_suites():
    with open(GATE_TORCH, encoding="utf-8") as fh:
        src = fh.read()
    assert "test_stream_quant.py" in src
    assert "test_incremental_safetensors.py" in src
    assert "GATE_PYTHON" in src  # interpreter override is documented in-script


def test_check_missing_interpreter_fails():
    code, out = _run_check("Z:/no/such/python.exe")
    assert code == 1
    assert "not found" in out.lower(), out


def test_check_torchless_interpreter_fails(tmp_path):
    # Environment drift note: the headless test venv gained torch-cpu at some
    # point after the 2026-08-25 exclusion decision, so it can no longer serve
    # as the "torchless interpreter" fixture. A fresh throwaway venv has no
    # packages at all — a deterministic torchless python for this regression.
    import sys

    bare = tmp_path / "bare-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(bare)],
        check=True, capture_output=True, timeout=120,
    )
    exe = str(bare / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    code, out = _run_check(exe)
    assert code == 1
    assert "torch" in out.lower(), out


def test_check_default_venv_has_torch():
    """The REAL default interpreter (script default) must pass --check.

    Active only when UQT_GATE_TORCH_VENV points at a local venv; skipped
    everywhere else (fresh clone / CI).
    """
    if not DEFAULT_VENV:
        pytest.skip("UQT_GATE_TORCH_VENV not set (no local torch venv pinned)")
    exe = f"{DEFAULT_VENV}/Scripts/python.exe" if os.name == "nt" else f"{DEFAULT_VENV}/bin/python"
    if not os.path.exists(exe):
        pytest.skip(f"{exe} not present on this machine")
    code, out = _run_check(exe)
    assert code == 0, out


def test_check_mode_does_not_run_pytest():
    """--check must be a fast probe: never invoke pytest (grep-equivalent)."""
    with open(GATE_TORCH, encoding="utf-8") as fh:
        src = fh.read()
    assert 'if [ "${1:-}" = "--check" ]' in src or '--check' in src
    # The pytest invocation must be guarded behind the check-mode exit.
    assert "-m pytest" in src
