#!/usr/bin/env bash
# NTH-009 (issues tracker Phase 2): on-demand torch-suite gate pass.
#
# The commit-time headless gate (gate_tests.sh) --ignores the two
# torch-dependent suites — the test venv .venv has NO torch:
#   * tests/test_stream_quant.py            (streaming quantization integration)
#   * tests/test_incremental_safetensors.py (resumable safetensors writer)
# Those two suites are the most safety-critical integration coverage in the
# repo, so this script runs them in a torch-enabled CTQ venv on demand:
#
#   bash scripts/gate_torch.sh           # run both suites in the default venv
#   bash scripts/gate_torch.sh --check   # fast probe: interpreter + torch only
#
# GATE_PYTHON overrides the interpreter (same convention as gate_tests.sh).
# The default venv needs pytest present:  uv pip install pytest
# (the CTQ venvs are not the test venv; install pytest on first use).
#
# Scope note (lead decision): ONLY the two torch suites run here — not the
# full headless suite. The CTQ venvs lack pytest-asyncio configuration for
# the whole app suite; the headless suite keeps running in gate_tests.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${GATE_PYTHON:-python}"

if [ ! -f "$PY" ]; then
  echo "gate_torch: interpreter not found: $PY" >&2
  echo "gate_torch: set GATE_PYTHON to a torch-enabled python (or install the default CTQ venv)." >&2
  exit 1
fi

# Torch probe: a torchless venv would fail at pytest COLLECTION with an
# opaque ImportError — probe first so the error names the actual problem.
if ! "$PY" -c "import torch" >/dev/null 2>&1; then
  echo "gate_torch: torch is not importable in: $PY" >&2
  echo "gate_torch: this script needs a torch-enabled venv (e.g. the CTQ venv)." >&2
  exit 1
fi

if [ "${1:-}" = "--check" ]; then
  echo "gate_torch: check OK — interpreter and torch present in $PY"
  exit 0
fi

"$PY" -m pytest tests/test_stream_quant.py tests/test_incremental_safetensors.py \
  -q -p no:cacheprovider
