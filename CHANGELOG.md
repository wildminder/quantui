# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is SemVer.

## [0.1.1] - 2026-08-25

## [0.1.2] - 2026-08-25

## [0.2.0] - 2026-08-25

### Changed
- Exception-overhaul (CRIT-002): all 60 silent `except Exception: pass` swallows
  eliminated project-wide; catches narrowed to precise types (NoMatches, OSError,
  json.JSONDecodeError, ChildProcessError, subprocess.TimeoutExpired) or
  boundary-marked with inline justifications.
- Intentionally swallowed errors are now recorded into the authoritative run log
  via QuantApp._debug_swallow instead of vanishing.
- Widget-query failures in Select writes surface InvalidSelectValueError for
  stale profile values instead of being hidden.
- AST ratchet test enforces zero-tolerance on new silent swallows.

### Added
- Historical NameError regression canary tests (red->green verified).

## [0.2.1] - 2026-08-25

### Changed
- Structure (IMP-001): app.py three-way extraction, 1296 -> 1204 lines.
  CSS -> app_css.py (MAIN_CSS alias kept), command palette -> palette.py
  (QuantCommands re-imported for the pinned import contract), header-progress
  monotonic hold -> pure HeaderProgressHold class in header_progress.py
  (_last_header_pct attribute deleted).
- Typing (IMP-002): mypy scaffolding (mypy.ini); run_config.validate gained
  explicit None-guards raising ValueError on missing family payloads;
  pre-commit gate now also runs mypy over the five core modules
  (quant_methods, stream_parser, live_progress, run_config, profiles_store).

### Added
- tests/test_header_progress_hold.py: 5 unit tests for HeaderProgressHold
  (tests-first, red->green verified).
- tests/test_app_extractions.py: identity pins for CSS and palette wiring.

## [0.3.0] - 2026-08-25

### Added
- Packaging (NTH-002): pyproject.toml with setuptools backend and a
  ``quantui`` console script ([project.scripts] -> quantui.__main__:main);
  dev extra (pytest, pytest-asyncio, pytest-cov, ruff, mypy) and torch-test
  extra (torch, safetensors, numpy). Installable via pip from checkout.
- Coverage gate (NTH-005): opt-in COVERAGE=1 mode in scripts/precommit_ruff.sh
  enforces per-module floors at measured-minus-2 (downward-only ratchet):
  quant_methods >=87%, stream_parser >=91%, live_progress >=91%,
  run_config >=83%, profiles_store >=90%; total >=61%.

## [Unreleased]

### Changed
- ruff configuration (E/F/W/I/UP/B) + autofix sweep: 114 findings -> 27 baseline
- count-based ruff gate script (scripts/precommit_ruff.sh)

## [0.1.0] - 2026-08-25

### Added
- Initial versioned release of unsloth-quant-tui: Textual TUI for GGUF (Unsloth)
  and ComfyUI (convert_to_quant) model quantization.
- ComfyUI INT8 formats (blockwise / tensorwise / ConvRot W8A8), fp8_e4m3,
  NVFP4/MXFP8 (Blackwell), W4A4/W4A8 (comfy-kitchen), on-the-fly passthrough.
- Streaming tensor-by-tensor INT8 quantization with resumable manifest checkpoints.
- .pt/.pth/.ckpt → safetensors converter with live input detection.
- Model-agnostic quantized-file structural validator.
- Capability probing with advisory badges; profiles/recents persistence;
  command palette; wizard mode; log drawer with S/M/L presets.
