#!/usr/bin/env bash
# NTH-006: bump semantic version + rotate CHANGELOG Unreleased section.
# Usage: scripts/bump_version.py {major|minor|patch} [--root PATH]
#
# - Reads __version__ from <root>/quantui/__init__.py
# - Computes next SemVer, rewrites __version__
# - Renames CHANGELOG.md "## [Unreleased]" -> "## [X.Y.Z] - <today>", opens fresh [Unreleased]
# - Refuses to run when the working tree is dirty (except the two managed files)
# - Prints suggested git add/commit/tag commands; never runs git itself
from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

INIT_RE = re.compile(r'^__version__ = "(\d+)\.(\d+)\.(\d+)"', re.M)


def _read_version(init_path: Path) -> tuple[int, int, int]:
    m = INIT_RE.search(init_path.read_text(encoding="utf-8"))
    if not m:
        sys.exit("ERROR: cannot parse __version__ in quantui/__init__.py")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _next_version(ver: tuple[int, int, int], level: str) -> str:
    major, minor, patch = ver
    if level == "major":
        return f"{major + 1}.0.0"
    if level == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def _rewrite_init(init_path: Path, new_version: str) -> None:
    text = init_path.read_text(encoding="utf-8")
    new_text, n = INIT_RE.subn(f'__version__ = "{new_version}"', text, count=1)
    if n != 1:
        sys.exit("ERROR: failed to rewrite __version__")
    init_path.write_text(new_text, encoding="utf-8")


def _rotate_changelog(changelog_path: Path, new_version: str) -> None:
    today = datetime.date.today().isoformat()
    text = changelog_path.read_text(encoding="utf-8")
    if f"## [{new_version}]" in text:
        sys.exit(f"ERROR: version {new_version} already present in CHANGELOG.md")
    if "## [Unreleased]" not in text:
        # No pending section: create both a dated (empty) and a fresh Unreleased.
        text = (
            f"# Changelog\n\n## [Unreleased]\n\n"
            f"## [{new_version}] - {today}\n\n" + text.split("# Changelog", 1)[1].lstrip("\n")
        )
    else:
        # Keep-a-Changelog semantics: the pending content UNDER [Unreleased]
        # becomes the dated release section, and a fresh EMPTY [Unreleased]
        # stays on top. So the dated header is inserted AFTER [Unreleased].
        text = text.replace(
            "## [Unreleased]",
            f"## [Unreleased]\n\n## [{new_version}] - {today}",
            1,
        )
    changelog_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bump semver + rotate CHANGELOG.")
    ap.add_argument("level", choices=["major", "minor", "patch"])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    init_path = root / "quantui" / "__init__.py"
    changelog_path = root / "CHANGELOG.md"
    for p in (init_path, changelog_path):
        if not p.is_file():
            sys.exit(f"ERROR: missing {p}")

    # Dirty-tree guard: only the two managed files may differ from HEAD.
    # Untracked cache/build dirs (unsloth_compiled_cache/, *.egg-info/) are
    # handled via .gitignore; the guard below additionally ignores the known
    # generated cache dir so a stale checkout can't block the release flow.
    _managed = ("quantui/__init__.py", "CHANGELOG.md")
    _generated = ("unsloth_compiled_cache/",)
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root,
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        proc = None
    if proc is not None and proc.returncode == 0:
        dirty = [ln for ln in proc.stdout.strip().splitlines()
                 if ln and not ln.endswith(_managed)
                 and not ln.endswith(_generated)]
    if proc is not None and proc.returncode == 0:
        dirty = [ln for ln in proc.stdout.strip().splitlines()
                 if ln and not ln.endswith(_managed)]
    elif proc is None:
        dirty = []  # no git binary at all (rare) -> skip guard
    else:
        # Not a git repository: fall back to a deterministic top-level scan.
        allowed = {"quantui", "CHANGELOG.md", ".git"}
        dirty = [f"?? {e.name}/" if e.is_dir() else f"?? {e.name}"
                 for e in sorted(root.iterdir()) if e.name not in allowed]
    if dirty:
        print("ERROR: dirty working tree (unmanaged changes):")
        print("\n".join(dirty))
        sys.exit(1)

    old = _read_version(init_path)
    new = _next_version(old, args.level)
    _rewrite_init(init_path, new)
    _rotate_changelog(changelog_path, new)
    print(f"Bumped {old[0]}.{old[1]}.{old[2]} -> {new}")
    print("Next steps:")
    print('  git add quantui/__init__.py CHANGELOG.md')
    print(f'  git commit -m "chore(release): v{new}"')
    print(f'  git tag v{new}')


if __name__ == "__main__":
    main()
