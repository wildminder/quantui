# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is SemVer.

## [Unreleased]

### Fixed
- **Output field no longer reverts to an input-derived path when quant options
  change** (user report: choosing int8 then switching scaling mode overwrote the
  output folder with the input filename). New behavior:
  - a directory-shaped output (e.g. `...\out`) is treated as a DESTINATION FOLDER
    whether or not it exists on disk; only the auto-generated FILENAME inside it
    is refreshed (`out\<base>-int8-<scaling>[-convrot-gs<N>].safetensors`);
  - a hand-typed `.safetensors` filename is never rewritten (explicit-file rule);
    ownership tracking distinguishes our own suggestion from a user-typed name;
  - stale convrot/gs tags can no longer leak into filenames after leaving row
    scaling (tag emission is gated on `scaling == 'row'` AND the checkbox);
  - sharded + single mode now honors a chosen destination folder instead of
    always placing the merged file next to the input (mirrors `build_ctq_cmd`);
  - changing the format also refreshes the suggested filename.
- **comfy-kitchen formats (W4A4 / W4A8) no longer receive CTQ-only worker
  flags.** `build_ctq_cmd` gated `--simple`, `--low_memory`, `--comfy_quant`,
  `--save_quant_metadata`, `--calib_samples`, `--num_iter` and preset flags on
  the CTQ backend; the kitchen worker's argparse accepts none of them, so a
  ticked checkbox + W4A4/W4A8 previously crashed with "unrecognized arguments".

## [0.4.0] - 2026-08-25

### Changed
- **BREAKING (UI/registry): the three INT8 format entries collapsed into ONE.**
  `int8_block` / `int8_tensor` / `int8_convrot` are replaced by a single `int8`
  ("INT8 (W8A8)") entry with a first-class **Scaling mode** option
  (block / tensor / row). The UI now shows only valid combinations:
  - `Block size` appears only for scaling `block`;
  - `Apply ConvRot` + `ConvRot group size` appear only for scaling `row`
    (ConvRot mathematically requires row scales) and only after the toggle is on;
  - dead combinations such as "tensor scaling + convrot" can no longer be selected.
- The old special-cased `#ctq_scaling_mode` panel widget is gone; scaling is a
  registry-declared `OptionField` widget (`#scaling_mode`) like every other option,
  with chained `visible_when` predicates over sibling values.
- Presets: flux2 = int8 + row + ConvRot (former int8_convrot behavior); WAN /
  Hunyuan = int8 + block. `Preset` gained `recommended_options`.
- Output auto-naming tags: `int8-<scaling>[-convrot-gs<N>]` (e.g.
  `model-int8-row-convrot-gs256.safetensors`) replaces `int8_convrot-gs256`.
- Capability badge: Triton advisory now fires dynamically when the unified INT8
  format has ConvRot enabled (was keyed to a static format tag).
- Profiles persist the INT8 options (scaling mode, block size, convrot, group size).

## [0.3.1] - 2026-08-25

### Fixed
- CPU utilization during quantization diagnosed (see
  `docs/reports/2026-08-25-cpu-utilization-root-cause.md`): the ~4% CPU usage is
  inherent to the learned-rounding optimizer path (convrot without --simple) --
  a GPU-latency-bound loop with one host sync per iteration, not a torch
  threading misconfiguration.

### Added
- `worker_ctq --num_iter` / `--num-iter`: passthrough of the learned-rounding
  iteration count (ctq default 4000). Lowering it (500-1000) is the primary
  speed lever for convrot runs; suppressed when `--simple` is set.
- UI: "Learned-rounding iterations per tensor" input in the ComfyUI Advanced
  collapsible (`#ctq_num_iter`), persisted in profiles.
- Worker subprocesses inherit OMP/MKL/OPENBLAS thread-pool env vars set to the
  full logical core count unless already set by the user (benefits CPU-device
  runs; no effect on GPU-bound loops).

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
