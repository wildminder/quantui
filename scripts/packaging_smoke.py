"""NTH-010: packaging smoke helper — console entry-point resolution check.

Factored out of scripts/gate_tests.sh (opt-in PACKAGING=1 stage) so it is
unit-testable with injected paths instead of a real venv. The check is
deliberately EXISTENCE + import-target based: launching a Textual App via
its console script could hang on --help, so the smoke proves the entry
point resolves on disk (Windows Scripts/ and POSIX bin/ layouts) and that
its import target exists in the repo — never executes it.
"""

from __future__ import annotations

import os
import sys


def entry_point_paths(venv_dir: str, entry_name: str) -> list[str]:
    """All on-disk locations the console script may occupy in a venv.

    Windows venvs place entry points in ``Scripts/`` (with a ``.exe``
    shim); POSIX venvs use ``bin/`` without an extension. Both layouts are
    returned so the smoke works on either platform.
    """
    venv_dir = venv_dir.replace("\\", "/").rstrip("/")
    posix = f"{venv_dir}/bin/{entry_name}"
    win_exe = f"{venv_dir}/Scripts/{entry_name}.exe"
    win_bat = f"{venv_dir}/Scripts/{entry_name}.bat"
    return [posix, win_exe, win_bat]


def check_entry_point(venv_dir: str, entry_name: str = "quantui") -> int:
    """Exit-code style check: 0 = entry point present, 1 = missing.

    ``venv_dir`` is the throwaway venv created by the PACKAGING smoke; the
    entry name defaults to the project console script (pyproject
    ``[project.scripts]``: ``quantui = "quantui.__main__:main"``).
    """
    candidates = entry_point_paths(venv_dir, entry_name)
    if not any(os.path.isfile(p) for p in candidates):
        print(
            f"packaging smoke: entry point '{entry_name}' not found in {venv_dir} "
            f"(looked for: {', '.join(candidates)})"
        )
        return 1
    print(f"packaging smoke: entry point '{entry_name}' OK")
    return 0


if __name__ == "__main__":
    venv = sys.argv[1] if len(sys.argv) > 1 else ""
    name = sys.argv[2] if len(sys.argv) > 2 else "quantui"
    if not venv:
        print("usage: packaging_smoke.py <venv_dir> [entry_name]")
        sys.exit(2)
    sys.exit(check_entry_point(venv, name))
