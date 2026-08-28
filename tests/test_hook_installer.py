"""Tests for scripts/install_hooks.sh and scripts/gate_tests.sh (IMP-004).

The repo has no git remote, so quality gates are enforced at commit time by a
generated ``.git/hooks/pre-commit``. These tests never touch the real
``.git/hooks``: the installer is always pointed at a throwaway temp repo, and
the two gate stages are stubbed so no test suite is actually executed.

Contract under test (plan 2026-08-28-issues-resolution-plan.md, steps 1.1-1.2):
  * installer creates an executable hook carrying the marker line
  * installer is idempotent (second run rewrites byte-identical content)
  * installer refuses a directory that is not a git checkout
  * the hook runs BOTH stages and fails on the first non-zero one
  * gate_tests.sh runs the pytest gate with the two torch ignores
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO_ROOT, "scripts")
INSTALLER = os.path.join(SCRIPTS, "install_hooks.sh")
GATE_TESTS = os.path.join(SCRIPTS, "gate_tests.sh")
PRECOMMIT_RUFF = os.path.join(SCRIPTS, "precommit_ruff.sh")
MARKER = "# unsloth-quant-tui gate hook"

bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="bash not available")


def run(cmd, cwd=None, env=None):
    """Run a command, returning (returncode, combined output)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        shell=False,
    )
    return proc.returncode, proc.stdout + proc.stderr


def make_repo(tmp_path, git_dir=True):
    """Create a throwaway 'repo' with .git/hooks (not a real git repo)."""
    root = str(tmp_path / "repo")
    if git_dir:
        os.makedirs(os.path.join(root, ".git", "hooks"))
    else:
        os.makedirs(root)
    return root


def exec_bit_supported(directory):
    """True when ``chmod +x`` is observable on this filesystem.

    pytest's tmp_path lives under ``X:\\Temp`` on this machine, where chmod is
    silently a no-op (the repo volume does honour it). Probing keeps the
    executable-bit assertion meaningful without a permanent false failure.
    """
    probe = os.path.join(directory, ".chmod-probe")
    with open(probe, "w", encoding="utf-8") as fh:
        fh.write("x")
    try:
        os.chmod(probe, 0o755)
        return bool(os.stat(probe).st_mode & 0o100)
    finally:
        if os.path.exists(probe):
            os.unlink(probe)


def stub_stage(repo, name, exit_code):
    """Write a stub stage script under <repo>/scripts/ that exits with code."""
    scripts_dir = os.path.join(repo, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    path = os.path.join(scripts_dir, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"#!/usr/bin/env bash\necho 'stub {name}'\nexit {exit_code}\n")
    return path


@pytest.fixture()
def installed(tmp_path):
    """A temp repo with the hook installed and both stages stubbed (green)."""
    repo = make_repo(tmp_path)
    stub_stage(repo, "precommit_ruff.sh", 0)
    stub_stage(repo, "gate_tests.sh", 0)
    code, out = run([bash, INSTALLER, repo])
    assert code == 0, out
    return repo


# --- step 1.1: installer --------------------------------------------------- #


def test_installer_scripts_exist():
    """Tripwire: both scripts the hook depends on are present."""
    assert os.path.isfile(INSTALLER)
    assert os.path.isfile(GATE_TESTS)
    assert os.path.isfile(PRECOMMIT_RUFF)


def test_install_creates_executable_hook(installed):
    hook = os.path.join(installed, ".git", "hooks", "pre-commit")
    assert os.path.isfile(hook), "hook file was not created"
    if exec_bit_supported(os.path.dirname(hook)):
        assert os.stat(hook).st_mode & 0o100, "hook is not executable"
    with open(hook, encoding="utf-8") as fh:
        body = fh.read()
    assert MARKER in body, "generated hook is missing its marker line"
    assert body.startswith("#!/bin/sh")


def test_install_is_idempotent(installed):
    hook = os.path.join(installed, ".git", "hooks", "pre-commit")
    with open(hook, encoding="utf-8") as fh:
        first = fh.read()
    code, out = run([bash, INSTALLER, installed])
    assert code == 0, out
    with open(hook, encoding="utf-8") as fh:
        second = fh.read()
    assert first == second, "re-running the installer changed the hook"


def test_install_refuses_without_git_dir(tmp_path):
    repo = make_repo(tmp_path, git_dir=False)
    code, out = run([bash, INSTALLER, repo])
    assert code != 0, "installer must fail outside a git checkout"
    assert not os.path.exists(os.path.join(repo, ".git", "hooks", "pre-commit"))
    assert "not a git repo" in out


# --- step 1.2: hook content and control flow -------------------------------- #


def test_hook_invokes_both_gate_stages(installed):
    with open(os.path.join(installed, ".git", "hooks", "pre-commit"), encoding="utf-8") as fh:
        body = fh.read()
    assert "scripts/precommit_ruff.sh" in body
    assert "scripts/gate_tests.sh" in body


def test_hook_passes_when_all_stages_pass(installed):
    code, out = run([bash, os.path.join(installed, ".git", "hooks", "pre-commit")], cwd=installed)
    assert code == 0, out
    assert "gate OK" in out


def test_hook_fails_when_ruff_stage_fails(installed):
    stub_stage(installed, "precommit_ruff.sh", 1)
    code, out = run([bash, os.path.join(installed, ".git", "hooks", "pre-commit")], cwd=installed)
    assert code != 0, "hook must block the commit when the ruff gate fails"
    assert "ruff/mypy gate failed" in out


def test_hook_fails_when_tests_stage_fails(installed):
    stub_stage(installed, "gate_tests.sh", 1)
    code, out = run([bash, os.path.join(installed, ".git", "hooks", "pre-commit")], cwd=installed)
    assert code != 0, "hook must block the commit when the pytest gate fails"
    assert "pytest gate failed" in out


def test_hook_propagates_stage_exit_code(installed):
    stub_stage(installed, "gate_tests.sh", 7)
    code, _ = run([bash, os.path.join(installed, ".git", "hooks", "pre-commit")], cwd=installed)
    assert code == 7, "hook must forward the failing stage's exit code"


# --- gate_tests.sh --------------------------------------------------------- #


def test_gate_tests_script_invokes_pytest_gate():
    with open(GATE_TESTS, encoding="utf-8") as fh:
        body = fh.read()
    assert "-m pytest" in body
    assert "--ignore=tests/test_incremental_safetensors.py" in body
    assert "--ignore=tests/test_stream_quant.py" in body
    assert ".venv" in body, "default interpreter is the managed headless venv"


def test_gate_tests_script_propagates_pytest_exit_code(tmp_path):
    """GATE_PYTHON overrides the interpreter so exit codes can be asserted."""
    # GATE_PYTHON replaces the interpreter, so the stub must be a single
    # directly-runnable file (the script quotes "$PY", so "bash <path>" would
    # be looked up as one literal filename and fail with 127).
    if os.name == "nt":
        stub = tmp_path / "stubpy.bat"
        stub.write_text("@echo off\nexit /b 5\n", encoding="utf-8")
    else:
        stub = tmp_path / "stubpy.sh"
        stub.write_text("#!/bin/sh\nexit 5\n", encoding="utf-8")
        os.chmod(stub, 0o755)
    # as_posix(): bash treats backslashes as escapes, so a Windows path must be
    # passed with forward slashes or the stub is not found (exit 127).
    code, _ = run([bash, GATE_TESTS], env={"GATE_PYTHON": stub.as_posix()})
    assert code == 5, "gate_tests.sh must forward pytest's exit code"
