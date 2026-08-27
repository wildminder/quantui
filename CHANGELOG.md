# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is SemVer.

## [Unreleased]

### Added
- **Model audit TUI: `AuditScreen` modal + "Audit model file" palette
  command** (plan 2026-08-27, STEP 4.1). The palette command resolves the
  target from `#ctq_input` (ComfyUI family tab, `.safetensors` file or
  single-file folder) or opens the file picker first, then pushes the modal:
  summary header line, per-module DataTable (module / tensors / params /
  bytes / linears), and the suggested `exclude_layers` regex in a
  selectable/copyable `#audit_suggestion` Input. The audit runs in a worker
  thread (`@work(thread=True)`); on `AuditError` the modal shows the error
  text instead of a table.
- **Model audit CLI `python -m quantui.model_audit`** (plan 2026-08-27,
  STEP 3.2). `-i <file>` audits a checkpoint and prints the text report
  (exit 0); `--json` prints the JSON report instead; `--out PATH` also writes
  the report to PATH (parents created) while still echoing to stdout. Missing
  or malformed input prints a message naming the path on stderr and exits 2.
  Subprocess-tested; a dedicated test proves the module never imports
  torch/safetensors/numpy (they are asserted absent from `sys.modules` after a
  full audit run).
- **Model audit report renderers** (plan 2026-08-27, STEP 3.1).
  `render_text` (fixed layout: title, totals with human-readable bytes,
  category table, per-module table bytes-desc, quantized section only when
  present, and the suggested regex on its own `exclude_layers:` line plus the
  stage-1/stage-2 boundary note) and `render_json`
  (`indent=2, sort_keys=True`, byte-stable across runs) in
  `quantui/model_audit.py`.
- **Model audit exclusion advisor `suggest_exclusions`** (plan 2026-08-27,
  STEP 2.2). Proposes a starting `exclude_layers` regex covering exactly the
  2D `.weight` keep-set (`embedding` / `head` / `linear_review`): one
  alternation of `re.escape`d tensor names, anchored `^…$`, sorted for
  determinism, safe under the `re.search` semantics of
  `QuantConfig.excluded`. Returns an `ExclusionSuggestion` with the regex, the
  covered names, and a per-category rationale. Empty keep-set -> `""`.
- **Model audit engine `audit_file`** (plan 2026-08-27, STEP 2.1).
  Header-only scan of a `.safetensors` file producing a deterministic
  `AuditReport`: per-category counts/bytes, per-module summaries, total
  bytes, `__metadata__` verbatim, and already-quantized layer detection via
  `comfy_quant_schema.read_comfy_quant_configs` with a per-format histogram.
  Malformed input (missing file, garbage bytes, truncated header, bad
  `.comfy_quant` JSON) raises `AuditError` naming the path. No tensor payload
  is ever read — a 7 GiB checkpoint scans in milliseconds.
- **Model audit module grouping + aggregation** (plan 2026-08-27, STEP 1.2).
  `TensorInfo` (frozen per-tensor record), `module_of` (top-level prefix),
  `collect_tensors` (header -> classified, name-sorted infos, `__metadata__`
  skipped) and `summarize_modules` (per-module tensor/param/byte totals +
  per-category counters, sorted bytes-desc with name tie-break) in
  `quantui/model_audit.py`.
- **Model audit classification core `quantui/model_audit.py`** (plan
  2026-08-27, STEP 1.1). Pure-stdlib tensor classifier for safetensors
  headers: `classify_tensor` assigns every tensor exactly one of nine
  categories (`linear` / `embedding` / `head` / `linear_review` / `vector` /
  `bias` / `quant_meta` / `quant_scale` / `other`) via the frozen rule table
  validated against the real Breeze-TTS-2 checkpoints (1115 tensors, zero
  false negatives vs. the official 378-tensor int8-hybrid build). Includes
  the frozen `LINEAR_SEGMENTS` set, the `DTYPE_ITEMSIZE` table, and
  `tensor_bytes`. Tripwire tests freeze the table in
  `tests/test_model_audit_classify.py`.

## [0.5.0] - 2026-08-27

### Added
- **`bf16` / `fp16` cast-only formats end-to-end** (plan 2026-08-26,
  STEP 3.1). New CTQ registry entries "BF16 (bfloat16, cast only)" and
  "FP16 (half, cast only)" whose quant tag is the format id
  (`<base>-bf16.safetensors` auto-naming). The worker gains a `--cast_dtype
  {bfloat16,float16}` short-circuit checked before any `convert_to_quant`
  import: a single-file input is cast in place, a sharded input is merge-cast
  into ONE `.safetensors` (a `.safetensors` output path is required).
- **Streaming cast writer `cast_safetensors_file` / `cast_shards_to_single`**
  in `quantui/dtype_cast.py` (plan 2026-08-26, STEP 2.2). Header-driven
  per-tensor payload streaming with cumulative offset rewrite — mirrors the
  proven `merge_safetensors_files` discipline and never loads the whole model
  (~8 MiB element chunks). Dtype policy: floating tensors (F64/F32/F16/BF16)
  are RTNE-cast to the target; integer/bool tensors pass through byte-identical
  with their original dtype recorded. `__metadata__` carried from the first
  shard; duplicate tensor names raise; per-tensor `on_progress(done, total)`
  callback starting at `(0, N)`.
- **Pure bit-exact dtype-cast core `quantui/dtype_cast.py`** (plan
  2026-08-26, STEP 2.1). RTNE conversions between F32/BF16/F16 as unsigned
  integer bit math — no numpy/torch, deterministic across platforms. Includes
  zero-copy identity fast path, F64→F32 narrowing, and a little-endian
  byte-buffer boundary (`cast_tensor_bytes`). Covered by frozen known-answer
  hex tables, an exhaustive 65 536-value fp16 widen/cast round-trip sweep,
  and a ~500k-point monotonicity sample over the f32 space.

### Changed
- **Merge-to-one formats (`combine` / `bf16` / `fp16`) are exempt from the
  "sharded output must be a directory" rule** (plan 2026-08-26, STEP 3.2).
  These formats ALWAYS write one merged `.safetensors` regardless of Output
  mode, so a `.safetensors` file output is valid even for a sharded input in
  sharded mode, and `build_ctq_cmd` appends `<stem>.safetensors` to a
  directory output in sharded mode too (new `MERGE_TO_ONE_FORMATS` set in
  `run_config`). Covered by `tests/test_cast_ux.py` (validation, builder,
  auto-naming Case C/F regression guards, headless UI flow: no dynamic option
  widgets for these formats, `--cast_dtype` in the built cmd, profile
  roundtrip + unknown-id fallback).
- **BREAKING (registry/UI): the `onthefly` format id is renamed to `combine`**
  ("Combine (merge shards, no quant)", STEP 1.2 of plan 2026-08-26). Combine
  ALWAYS merges a sharded input into ONE `.safetensors` — the Output mode
  selection is irrelevant for this format — and a single-file input is copied
  byte-identical. No `.comfy_quant` metadata is baked. Saved profiles that
  still reference the old id fall back to the default format via the existing
  safe-widget-value path (same precedent as the v0.4.0 INT8 id consolidation).
- **`--combine` replaces the old `--passthrough` worker flag** (plan
  2026-08-26, STEP 1.1). The combine path now ALWAYS merges a sharded input
  into ONE `.safetensors` (output mode is irrelevant); a single-file input is
  copied byte-identical. No `.comfy_quant` metadata is baked and
  `convert_to_quant` / torch are never imported.

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
