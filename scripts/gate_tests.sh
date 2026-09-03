#!/usr/bin/env bash
# IMP-004 (plan 2026-08-28): pytest gate — the headless suite that must pass
# before a commit is accepted. Kept separate from precommit_ruff.sh so the
# commit hook can report each stage independently.
#
# Usage:  bash scripts/gate_tests.sh
# Override the interpreter with GATE_PYTHON (e.g. a torch-enabled CTQ venv).
#
# Opt-in stages (both OFF by default so the commit-time gate stays fast):
#   PACKAGING=1  — packaging smoke (NTH-010): throwaway venv + editable
#                  install + console entry-point resolution check.
#   COVERAGE=1   — not here; see precommit_ruff.sh for the coverage floors.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${GATE_PYTHON:-python}"

# The two torch-dependent suites stay excluded in the headless env (user
# decision 2026-08-25); see scripts/gate_torch.sh (NTH-009) for their run.
"$PY" -m pytest tests/ -q \
  --ignore=tests/test_incremental_safetensors.py \
  --ignore=tests/test_stream_quant.py

# Packaging smoke (NTH-010) — opt-in via PACKAGING=1.
# Verifies the console entry point survives a real editable install:
# create a throwaway venv, `pip install -e . --no-deps` into it, then check
# the entry-point shim exists (scripts/packaging_smoke.py; it never LAUNCHES
# the app — a Textual App could hang on --help).
if [ "${PACKAGING:-0}" = "1" ]; then
  echo "packaging smoke: creating throwaway venv + editable install..."
  # Python manages the temp dir (mktemp/rm -rf on Git Bash mangle Windows
  # paths: mktemp returns /tmp/... which rm resolves cwd-relative, leaving
  # the venv behind). tempfile.mkdtemp gives a real Windows path; the same
  # python removes it with shutil.rmtree on exit.
  SMOKE_TMP="$("$PY" -c 'import tempfile; print(tempfile.mkdtemp(prefix="pkg-smoke-"))')"
  SMOKE_VENV="$SMOKE_TMP/venv"
  "$PY" -m venv "$SMOKE_VENV"
  # Editable install with the throwaway venv's OWN python (its pip may need
  # a first-run bootstrap; the fallback path handles old pip without --python).
  SMOKE_PY="$SMOKE_VENV/Scripts/python"
  if [ ! -f "$SMOKE_PY" ]; then SMOKE_PY="$SMOKE_VENV/bin/python"; fi
  "$SMOKE_PY" -m pip install -e . --no-deps -q
  "$PY" scripts/packaging_smoke.py "$SMOKE_VENV" quantui
  "$PY" -c "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$SMOKE_TMP"
  echo "packaging smoke: OK"
fi
