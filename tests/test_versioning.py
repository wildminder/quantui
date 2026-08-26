"""Tests for NTH-006 version discipline: __version__, CHANGELOG, bump script.

The bump script is exercised against a tmp-path fixture (via its ``--root`` flag)
so tests never touch the real repo.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BUMP = REPO / "scripts" / "bump_version.py"

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

INIT_TMPL = '''"""Test package init."""

__version__ = "{version}"
'''

CHANGELOG_TMPL = """# Changelog

## [Unreleased]

- some pending change

## [{version}] - 2026-08-25

### Added
- initial stuff
"""


def _write_fixture(root: Path, version: str = "0.1.0") -> None:
    (root / "quantui").mkdir(parents=True)
    (root / "quantui" / "__init__.py").write_text(INIT_TMPL.format(version=version), encoding="utf-8")
    (root / "CHANGELOG.md").write_text(CHANGELOG_TMPL.format(version=version), encoding="utf-8")


def _run_bump(root: Path, level: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BUMP), level, "--root", str(root)],
        capture_output=True, text=True,
    )


def test_version_parses_semver():
    from quantui import __version__

    assert SEMVER_RE.match(__version__), __version__


def test_changelog_has_unreleased_and_current():
    from quantui import __version__

    text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    # Exactly one section for the current released version.
    section = f"## [{__version__}]"
    assert text.count(section) == 1, f"expected exactly one {section!r} section"


def test_bump_script_patch():
    root = Path(os.environ.get("TMP_BUMP_ROOT", "")) if False else None  # placeholder guard
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root, "0.1.0")
        cp = _run_bump(root, "patch")
        assert cp.returncode == 0, cp.stderr
        init = (root / "quantui" / "__init__.py").read_text(encoding="utf-8")
        assert '__version__ = "0.1.1"' in init
        log = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        assert "## [Unreleased]" in log          # fresh empty section re-created
        assert "## [0.1.0]" in log                # old Unreleased became dated section
        assert "some pending change" in log       # content preserved under the new section
        # Keep-a-Changelog order: the fresh [Unreleased] stays ON TOP and the
        # pending content lands UNDER the NEW dated section (not under Unreleased).
        assert log.index("## [Unreleased]") < log.index("## [0.1.1]")
        assert log.index("some pending change") > log.index("## [0.1.1]")


def test_bump_script_minor_and_major():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root, "0.1.0")
        assert _run_bump(root, "minor").returncode == 0
        init = (root / "quantui" / "__init__.py").read_text(encoding="utf-8")
        assert '__version__ = "0.2.0"' in init

        root2 = Path(str(td) + "_2")
        _write_fixture(root2, "0.9.9")
        assert _run_bump(root2, "major").returncode == 0
        init2 = (root2 / "quantui" / "__init__.py").read_text(encoding="utf-8")
        assert '__version__ = "1.0.0"' in init2


def test_bump_refuses_dirty_tree(tmp_path):
    _write_fixture(tmp_path, "0.1.0")
    # A stray file outside the two managed files => dirty => refuse.
    # (tmp_path is not a git repo -> the script's non-git fallback must catch it.)
    (tmp_path / "stray.txt").write_text("x", encoding="utf-8")
    cp = _run_bump(tmp_path, "patch")
    assert cp.returncode != 0
    assert "dirty" in (cp.stdout + cp.stderr).lower()


def test_bump_allows_clean_fixture_tree(tmp_path):
    """Non-git root containing ONLY managed files + allowed dirs => proceeds."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _write_fixture(root, "0.1.0")
        (root / ".git").mkdir()  # marker dir counts as allowed
        cp = _run_bump(root, "patch")
        assert cp.returncode == 0, cp.stdout + cp.stderr
