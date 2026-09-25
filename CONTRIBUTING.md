# Contributing to QuantUI

Thank you for your interest in improving QuantUI. Contributions should be
focused, reproducible, and safe for the project's headless test environment.

## Development setup

QuantUI requires Python 3.12 or newer.

Create and activate a virtual environment, then install the package and its
development dependencies in editable mode:

```bash
python -m pip install -e ".[dev]"
```

The `dev` extra exists and installs pytest, pytest-asyncio, pytest-cov, Ruff,
and mypy. It does not install the separate environments used by Unsloth or
ComfyUI workers, and it does not install Torch.

Run the installed console command with:

```bash
quantui
```

## Quality gates

The repository's canonical checks are shell scripts. Run them from the
repository root in Git Bash or another Bash-compatible shell.

Run the Ruff and mypy gate with:

```bash
bash scripts/precommit_ruff.sh
```

This checks `quantui/` and `tests/` with Ruff, then runs mypy over the core
typing-checked modules. The Ruff gate uses the frozen baseline at
`docs/reviews/ruff-baseline.txt` when that file is available. Because `docs/`
is intentionally not distributed, a fresh clone has no baseline file and the
gate therefore enforces zero Ruff findings.

Run the headless pytest gate with:

```bash
bash scripts/gate_tests.sh
```

This runs all tests except the two Torch-dependent suites described below. To
use a specific Python interpreter, set `GATE_PYTHON`:

```bash
GATE_PYTHON=/path/to/headless-python bash scripts/gate_tests.sh
```

The script also supports two opt-in checks:

```bash
COVERAGE=1 bash scripts/precommit_ruff.sh
PACKAGING=1 bash scripts/gate_tests.sh
```

The first adds the repository's coverage floors. The second creates a temporary
environment, installs the project editable without dependencies, and checks
that the `quantui` console entry point is present.

## Torch-dependent test suites

Keep the normal headless test environment Torch-free. These two integration
suites are intentionally excluded by `scripts/gate_tests.sh` and must be run
separately in a Torch-enabled quantization environment:

- `tests/test_stream_quant.py`
- `tests/test_incremental_safetensors.py`

Install pytest in that environment and install the project's `torch-test`
extra, which provides Torch, safetensors, and NumPy:

```bash
python -m pip install pytest
python -m pip install -e ".[torch-test]"
```

The Torch suite is only the test dependency layer; a real quantization worker
may require its own Unsloth or ComfyUI environment. Then run both suites with:

```bash
bash scripts/gate_torch.sh
```

Use `GATE_PYTHON` to select the Torch-enabled interpreter:

```bash
GATE_PYTHON=/path/to/torch-python bash scripts/gate_torch.sh
```

A fast environment probe is available as:

```bash
bash scripts/gate_torch.sh --check
```

## Local commit hook

Every push and pull request runs the same four gates in CI
(`.github/workflows/ci.yml`): ruff, mypy over the five core modules, the
headless test suite, and a packaging smoke build. To catch problems before
pushing, contributors can also install the local pre-commit hook:

```bash
bash scripts/install_hooks.sh
```

The installer generates a hook that runs `scripts/precommit_ruff.sh` followed
by `scripts/gate_tests.sh`. It is safe to run repeatedly. `git commit
--no-verify` is an emergency-only escape hatch; a bypassed gate should be run
manually before sharing the change.

## Repository hygiene

Never commit model weights, GGUF or safetensors outputs, quantization results,
temporary worker files, generated coverage reports, or local environment
artifacts. Do not commit machine-specific absolute paths, personal directory
names, credentials, or configuration that identifies a contributor's computer.
Keep changes reproducible from the files in version control.

The internal `docs/` tree is deliberately ignored and is not distributed with
source archives or fresh clones. Tests that require files under `docs/`
therefore skip when those files are absent. This is expected on a fresh clone,
not by itself a test failure. When internal documentation is available, the
schema consistency test reads `docs/comfy-quant-schema.md`, and the documentation
link test checks tracked README links plus any local documentation targets that
are present.

## Submitting changes

Before requesting review:

1. Keep unrelated formatting or generated-file changes out of the patch.
2. Run `bash scripts/precommit_ruff.sh`.
3. Run `bash scripts/gate_tests.sh`.
4. Run `bash scripts/gate_torch.sh` when Torch-dependent integration behavior is
   affected.
5. Describe the change and the exact checks run. For user-visible changes, add
   an appropriate `CHANGELOG.md` entry under the unreleased section.
