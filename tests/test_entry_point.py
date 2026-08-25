"""NTH-002 STEP 4.1: packaging entry point contract.

Tests are written FIRST (plan order): they pin the console-script entry point
declared in pyproject.toml so the packaging cannot silently drift.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_main_is_callable():
    from quantui.__main__ import main

    assert callable(main)


def test_pyproject_declares_quantui_script():
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml must exist at repo root"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["quantui"] == "quantui.__main__:main"


def test_pyproject_declares_required_extras():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "dev" in extras, "dev extra must be declared"
    assert "torch-test" in extras, "torch-test extra must be declared"
    # Dev tooling essentials per plan.
    devJoined = " ".join(extras["dev"]).lower()
    for pkg in ("pytest", "ruff", "mypy", "pytest-cov"):
        assert pkg in devJoined, f"dev extra must include {pkg}"


def test_pyproject_declares_core_dependency():
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"]).lower()
    assert "textual" in deps, "core dependency textual must be declared"
