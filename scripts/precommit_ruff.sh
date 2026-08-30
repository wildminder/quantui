#!/usr/bin/env bash
# IMP-003: count-based ruff gate. Fails only when live finding count EXCEEDS
# the baseline recorded in docs/reviews/ruff-baseline.txt.
# Once baseline reaches 0 this becomes a strict zero-tolerance gate.
# IMP-002 S3A.5: additionally runs mypy over the five typing-clean core
# modules; any mypy error fails the gate.
# NTH-005 S4.2: opt-in coverage mode via COVERAGE=1 -- measures per-module
# floors (measured-current-minus-2, downward-only ratchet) for the five core
# modules plus a global floor. Measured 2026-08-25.
set -euo pipefail
cd "$(dirname "$0")/.."

# RUFF / MYPY can be overridden so the gate is testable without the real tools
# (see tests/test_gate_scripts.py).
RUFF="${RUFF:-ruff}"
BASELINE_FILE="docs/reviews/ruff-baseline.txt"
MYPY="${MYPY:-python}"

# ruff exits 1 when findings exist; capture output without tripping set -e.
ruff_out=$("$RUFF" check quantui/ tests/ 2>/dev/null || true)
# '|| true' matters: when ruff is clean it prints "All checks passed!" instead
# of "Found N errors", so the greps below match nothing. Without it, pipefail
# makes this assignment fail and set -e kills the script silently -- i.e. the
# gate broke precisely when the finding count reached zero (NTH-008).
live=$(printf '%s' "$ruff_out" | grep -oE "^Found [0-9]+ error" | grep -oE "[0-9]+" | head -1 || true)
[ -z "$live" ] && live=0
base=$(grep -m1 -oE "^Found [0-9]+" "$BASELINE_FILE" | grep -oE "[0-9]+$")

echo "ruff gate: live=$live baseline=$base"
if [ "$live" -gt "$base" ]; then
  echo "FAIL: new ruff findings introduced ($live > $base)."
  "$RUFF" check quantui/ tests/
  exit 1
fi

# mypy gate (IMP-002 S3A.5): the five core modules must stay typing-clean.
echo "mypy gate: checking core modules..."
"$MYPY" -m mypy \
  quantui/quant_methods.py \
  quantui/stream_parser.py \
  quantui/live_progress.py \
  quantui/run_config.py \
  quantui/profiles_store.py

# Coverage gate (NTH-005 S4.2): opt-in via COVERAGE=1 (adds ~60s).
if [ "${COVERAGE:-0}" = "1" ]; then
  echo "coverage gate: running with --cov..."
  "$MYPY" -m pytest tests/ -q \
    --ignore=tests/test_incremental_safetensors.py \
    --ignore=tests/test_stream_quant.py \
    --cov=quantui --cov-report=term --cov-report=json:coverage.json
  "$MYPY" - <<'PYEOF'
import json, sys

data = json.load(open("coverage.json", encoding="utf-8"))
files = data["files"]

def pct(name: str) -> float:
    for k, v in files.items():
        if k.endswith(name):
            return float(v["summary"]["percent_covered"])
    raise KeyError(name)

# Per-module floors for the five core modules: measured minus 2 (downward-only
# ratchet; measured values recorded in the issues tracker).
FLOORS = {
    "quant_methods.py": 87,   # measured 89.9 on 2026-08-25
    "stream_parser.py": 91,   # measured 93.8
    "live_progress.py": 91,   # measured 93.4
    "run_config.py": 83,      # measured 85.8
    "profiles_store.py": 90,  # measured 92.6
}
GLOBAL_FLOOR = 61  # total measured 63%

failures = []
for name, floor in FLOORS.items():
    p = pct(name)
    print(f"coverage: {name:<22} {p:5.1f}%  (floor {floor})")
    if p < floor:
        failures.append(f"{name}: {p:.1f}% < floor {floor}")

total = float(data["totals"]["percent_covered"])
print(f"coverage: {'TOTAL':<22} {total:5.1f}%  (floor {GLOBAL_FLOOR})")
if total < GLOBAL_FLOOR:
    failures.append(f"TOTAL: {total:.1f}% < floor {GLOBAL_FLOOR}")

if failures:
    print("FAIL: coverage below floor:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("coverage gate: OK")
PYEOF
fi

echo "OK"
