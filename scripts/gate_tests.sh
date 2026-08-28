#!/usr/bin/env bash
# IMP-004 (plan 2026-08-28): pytest gate — the headless suite that must pass
# before a commit is accepted. Kept separate from precommit_ruff.sh so the
# commit hook can report each stage independently.
#
# Usage:  bash scripts/gate_tests.sh
# Override the interpreter with GATE_PYTHON (e.g. a torch-enabled CTQ venv).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${GATE_PYTHON:-python}"

# The two torch-dependent suites stay excluded in the headless env (user
# decision 2026-08-25); see scripts/gate_torch.sh (NTH-009) for their run.
"$PY" -m pytest tests/ -q \
  --ignore=tests/test_incremental_safetensors.py \
  --ignore=tests/test_stream_quant.py
