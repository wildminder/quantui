"""P7.2: documentation + module-map link integrity.

Every local path referenced from ``README.md`` (``docs/...`` markdown links and the
project-layout module-map entries) must exist on disk, so the docs cannot silently
drift from the code.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA_DOC = REPO_ROOT / "docs" / "comfy-quant-schema.md"
INTEGRATION_PLAN = REPO_ROOT / "docs" / "plans" / "2026-08-18-quantization-toolkit-integration.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.skipif(
    not SCHEMA_DOC.is_file(),
    reason="docs/comfy-quant-schema.md not present (untracked)",
)
def test_comfy_quant_schema_doc_exists():
    assert SCHEMA_DOC.is_file()


@pytest.mark.skipif(
    not INTEGRATION_PLAN.is_file(),
    reason="docs/plans/2026-08-18-quantization-toolkit-integration.md not present (untracked)",
)
def test_integration_plan_exists():
    assert INTEGRATION_PLAN.is_file()


def test_readme_markdown_links_resolve():
    """Every relative link target in README.md must exist.

    Targets under docs/ are only checked when the docs/ tree is present —
    docs/ is not tracked in git, so on a fresh clone those links cannot
    resolve; they must not hard-fail there.
    """
    readme = _read(REPO_ROOT / "README.md")
    for tgt in re.findall(r"\]\(([^)]+)\)", readme):
        if tgt.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = tgt.split("#", 1)[0]
        if not path:
            continue
        target = REPO_ROOT / path
        if path.startswith("docs/") and not target.exists():
            pytest.skip(f"README link target under untracked docs/ absent: {tgt}")
        assert target.exists(), f"README link target missing: {tgt}"


def test_readme_module_map_resolves():
    """Every `quantui/*.py` entry in the README layout table must exist."""
    readme = _read(REPO_ROOT / "README.md")
    mods = re.findall(r"`(quantui/[A-Za-z0-9_]+\.py)`", readme)
    assert mods, "no quantui module-map entries found in README"
    for m in mods:
        assert (REPO_ROOT / m).is_file(), f"README module-map entry missing: {m}"


def test_readme_backend_requirements_mentions_both_backends():
    readme = _read(REPO_ROOT / "README.md")
    assert "Backend.CTQ" in readme
    assert "Backend.COMFY_KITCHEN" in readme
    assert "worker_ctq_kitchen.py" in readme
