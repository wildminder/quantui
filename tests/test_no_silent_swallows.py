"""CRIT-002 ratchet: no NEW silent exception swallows in quantui/.

A "silent swallow" is an ``except Exception`` (or bare ``except:``) handler whose
body does nothing observable: only ``pass`` / ``return`` / ``continue``, and
carries neither a ``# boundary:`` justification comment nor a ``noqa`` marker.

The BASELINE dict freezes the count measured on 2026-08-25 during the issues-
resolution plan (STEP 2.1). It may only ever be LOWERED. Any edit that adds a
silent swallow site fails this test; eliminating sites requires tightening the
number in the same commit.
"""

from __future__ import annotations

import ast
from pathlib import Path

QUANTUI = Path(__file__).resolve().parent.parent / "quantui"

# Per-module allowances (passive-body broad excepts without boundary/noqa marks).
# Measured with this AST counter on 2026-08-25: 60 total. (Earlier regex-based
# estimate of 40 undercounted; the AST walker is authoritative.)
BASELINE: dict[str, int] = {
    "__init__.py": 0,
    "__main__.py": 0,
    "app.py": 0,
    "capabilities.py": 0,   # narrowed to json.JSONDecodeError (STEP 2.5)
    "comfy_quant_schema.py": 0,
    "dtype_cast.py": 0,     # pure bit-math module (STEP 2.1): no except blocks by design
    "handlers.py": 0,
    "ids.py": 0,
    "incremental_safetensors.py": 0,
    "live_progress.py": 0,
    "panels.py": 0,         # narrowed to NoMatches (STEP 2.5)
    # Reserved names for planned extractions (IMP-001/NTH-004); the stale-entry
    # check below ignores them until they exist.
    "palette.py": 0,          # created later by IMP-001 extraction
    "app_css.py": 0,          # created later by IMP-001 extraction
    "header_progress.py": 0,  # created later by IMP-001 extraction
    "path_checks.py": 0,      # created later by NTH-004 extraction
    "profiles_store.py": 0,   # narrowed to (OSError, JSONDecodeError) (STEP 2.5)
    "pt_convert.py": 0,
    "quant_methods.py": 0,    # boundary-marked eval guard (STEP 2.5)
    "quant_validator.py": 0,
    "run_config.py": 0,
    "run_monitor.py": 0,
    "screens.py": 0,
    "screens_wizard.py": 0,   # narrowed to NoMatches (STEP 2.5)
    "stream_parser.py": 0,
    "stream_quant.py": 0,
    "tensor_quant.py": 0,
    "ui_bridge.py": 0,
    "widgets_results.py": 0,  # narrowed to NoMatches (STEP 2.5)
    "worker.py": 0,           # boundary-marked import probe (STEP 2.5)
    "worker_ctq.py": 0,
    "worker_ctq_kitchen.py": 0,
    "worker_pt_convert.py": 0,
    "worker_runner.py": 0,    # narrowed to OSError/ChildProcessError/TimeoutExpired
                              # or boundary-marked (STEP 2.5)
}

_PASSIVE_NODES = (ast.Pass, ast.Return, ast.Continue, ast.Break)


def _has_inline_marker(line_text: str) -> bool:
    return "# boundary:" in line_text or "# noqa" in line_text


def count_silent_swallows(path: Path) -> int:
    """Count broad-except handlers with passive bodies and no justification."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        # Broad catch? (bare `except:` or `except Exception`)
        if node.type is not None:
            t = node.type
            names = [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
            if not ("Exception" in names or "BaseException" in names):
                continue
        # Passive body only?
        body_nodes = [
            n for n in node.body
            if not isinstance(n, ast.Expr) or not isinstance(n.value, ast.Constant)
        ]
        if len(body_nodes) != 1 or not isinstance(body_nodes[0], _PASSIVE_NODES):
            continue
        # Inline marker on the `except` line itself?
        except_line = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        if _has_inline_marker(except_line):
            continue
        count += 1
    return count


def test_ratchet_no_new_silent_swallows():
    """Every quantui module stays at/below its allowance; unknown modules => 0."""
    failures = []
    seen: set[str] = set()
    for path in sorted(QUANTUI.glob("*.py")):
        name = path.name
        seen.add(name)
        allow = BASELINE.get(name, 0)  # new files default to 0 tolerance
        live = count_silent_swallows(path)
        if live > allow:
            failures.append(f"{name}: {live} silent swallows > allowed {allow}")
    assert not failures, (
        "Silent exception swallowing detected (fix by narrowing the catch type, "
        "routing through QuantApp._debug_swallow, or adding '# boundary: <reason>'):\n"
        + "\n".join(failures)
    )


def test_baseline_covers_all_modules():
    """BASELINE must mention every module that exists (prevents stale entries).

    Reserved future-module names (planned extractions) are exempt from the
    stale check but still enforce 0 tolerance once the files appear.
    """
    _RESERVED = {"palette.py", "app_css.py", "header_progress.py", "path_checks.py"}
    actual = {p.name for p in QUANTUI.glob("*.py")}
    missing = actual - set(BASELINE)
    stale = (set(BASELINE) - actual) - _RESERVED
    problems = []
    if missing:
        problems.append(f"modules missing from BASELINE: {sorted(missing)}")
    if stale:
        problems.append(f"stale BASELINE entries (files gone): {sorted(stale)}")
    assert not problems, "; ".join(problems)


def test_total_never_increases():
    """Sum of allowances is monotonic downward across commits (recorded value)."""
    recorded_max_total = 60  # initial measured inventory (2026-08-25, AST-based)
    # Tightened to ZERO TOLERANCE (STEP 2.5, CRIT-002): handlers 11->0,
    # app.py 35->0, remaining modules classified as Class-W/L/boundary.
    assert sum(BASELINE.values()) <= 0
    total = sum(BASELINE.values())
    assert total <= recorded_max_total, (
        f"allowance sum {total} exceeds recorded maximum {recorded_max_total}; "
        "the ratchet may only tighten"
    )
