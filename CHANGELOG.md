# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is SemVer.

## [Unreleased]

## [0.10.5] - 2026-09-09

### Fixed
- **Q8_0 denormal blocks: wrong codes + RuntimeWarnings** (user report,
  VibeVoice-ASR-HF `native_q8_0`): blocks whose max |value| is a tiny
  denormal (~1e-40 zero-init noise) made `1/d` overflow to inf, so every
  product was NaN — the kernel emitted `-127` codes (NaN → int32 wrapped
  to INT_MIN) where the llama.cpp reference writes 0, and spammed
  overflow/invalid RuntimeWarnings. Rounding now follows gguf-py's
  `np_roundf` verbatim (its `(a - floored)` shape NaN-poisons ±inf), NaN
  maps to code 0 explicitly, and the expected-FP paths are errstate-
  silenced. Verified byte-identical AND warning-free against the live
  gguf-py oracle on denormal / sparse / logspace-sweep / zero / inf / nan
  inputs; Q4_0 unaffected (its clip absorbs the case).

## [0.10.4] - 2026-09-09

### Added
- **[Close ✕] on interrupted runs** (user request 2026-09-09): the close
  toggle moved to its own always-available row (`#footer_close_row`) on the
  results card — stopped AND failed runs can now dismiss the footer, not
  just successful ones. Copy/Open stay success-gated.

### Removed
- **Per-phase progress rail** (user request 2026-09-09): the chip rows under
  the footer bar duplicated the stats line's counts AND opened the log
  drawer when clicked — both rejected. The aggregate bar (#footer_bar) +
  boxed stats line (#footer_stats) are the single progress surface; the
  Esc keybinding and the phase-filter machinery died with the rail.

### Fixed
- **Native GGUF export label** no longer embeds the counts
  (`Exporting GGUF (native_q4_0) 621/1204`): cur/total now travel as real
  fields and render once, in the stats line (`Exporting GGUF (native_q4_0)
  [621/1204]`).

## [0.10.3] - 2026-09-09

### Fixed
- **Crash when clicking a progress-rail row during quantization** (user
  report): the footer rail's click handler read `event.target`, which no
  longer exists on Textual 8.2.8 click events (`AttributeError: 'Click'
  object has no attribute 'target'`). Reads `event.widget` instead — clicks
  on a phase chip (or anywhere on its row) again open the log drawer
  filtered to that phase.

## [0.10.2] - 2026-09-09

### Added
- **[Close ✕] button on the done-mode results card** (user request
  2026-09-09): hides the whole run footer after a finished quantization.
  The next run re-reveals it automatically.
- **BlockBar widget** (`quantui/widgets_progress.py`, plan
  2026-09-09-control-panel S1.2): a chunky multi-row progress bar that fills
  its entire widget area with █/░ blocks — the native ProgressBar renders a
  single 32-cell 1-row strip and cannot do multi-row fills. Exposes the
  ProgressBar-compatible `update(total=, progress=)` API and a 0-100
  `progress` alias so the run footer can swap it in without touching callers.

### Changed
- **Control-panel visual pass** (plan 2026-09-09-control-panel S3.1): idle /
  run / done / drawer / wizard states screenshot-audited at 120x40 — no
  near-black fills anywhere, palette-v2 colors present, chunky bar verified
  (228 block glyphs at 65%). Docs refreshed: implemented.md + roadmap.md
  carry the control-panel redesign; plan completion log filled.
- **Control-panel element polish** (plan 2026-09-09-control-panel S2.2):
  phase chips render as pill boxes (round panel border + surface fill, the
  rail row grows to fit), the footer stats line becomes a boxed readout, and
  radio buttons keep a readable theme-driven label color. The run footer's
  max-height grows 16 -> 25 rows so the boxed elements keep the status line
  on screen.
- **Section headers as instrument-group strips** (plan 2026-09-09-control-panel
  S2.1): the six numbered section labels (GGUF model path / output / method,
  ComfyUI input / output / format) carry a `section_header` class and render
  as panel-tinted bold strips — no widget-tree changes.
- **Full-width chunky footer progress bar** (plan 2026-09-09-control-panel
  S1.3): the run footer's aggregate bar is now the 3-row BlockBar spanning
  the panel width (was a 32-cell single-line strip inside a ~117-wide
  widget). `#footer_bar` keeps its id and `update(progress=)` contract, so
  stats/hold behavior is unchanged.
- **Palette v2 "control room slate"** (plan 2026-09-09-control-panel S1.1):
  the deep-space near-black stack (~4-12% luminance) is replaced by a
  professional slate/navy ramp (14-26%), the electric-cyan primary softens to
  steel cyan, and the magenta accent becomes a muted violet. A permanent
  no-near-black guard pins every surface at >= 12% luminance.

## [0.10.1] - 2026-09-09

### Changed
- Consolidated the invalid-input border into the themed sci-fi rule
  (`round $error` everywhere — the old `tall ansi_red` leftover is gone).

## [0.10.0] - 2026-09-09

### Changed
- **The app is now titled "QuantUI"** (was "QuantApp", which leaked from the
  class name into the Header and terminal window title).
- **New "quantui-cyber" sci-fi theme** — deep-space blue surfaces, electric
  cyan primary, magenta accent, neon-green success. Registered and set as the
  app default; all widget colors follow the theme variables.
- **Sci-fi input styling** — inputs render with a cyan round frame that
  "powers up" to magenta when focused; the invalid (red) state is preserved.
- **Sci-fi button styling** — bold labels; the focused/hovered button's
  border powers up to the accent hue (variant colors follow the theme).
- Theme + title are stable across repeated app boots (pinned by test).

### Removed
- **"List all methods" button.** It was redundant — the "Pick from list"
  modal already lists all 35 official methods interactively (also reachable
  from the command palette).

## [0.9.2] - 2026-09-08

### Removed
- **The top header progress strip.** It duplicated the footer's aggregate
  bar + ETA during runs (both showed the same pct at the same moment) and
  showed useless "0% / --" when idle. The run footer — wide bar + stats
  line (pct / elapsed / ETA / counts / rate) + phase chips — is now the
  single progress surface. The monotonic-hold behavior survives in the
  footer's own hold; the pure `HeaderProgressHold` class is retained.

## [0.9.1] - 2026-09-08

### Fixed
- **Footer dedup — no repeated progress info.** The per-phase rows in the
  footer are now compact clickable phase-name chips (`Loading tensors` /
  `Optimizing INT8`): the counts (`[2600/4000]`) live once in the stats line,
  the wide bar is the single progress bar, and clicking a chip still opens
  the log drawer filtered to that phase. Footer height shrinks accordingly.

### Removed
- Dead `ProgressView` widget (the pre-footer live-progress box, unmounted
  since the S1.8 rail): 66 lines of dead code gone.

## [0.9.0] - 2026-09-08

### Fixed
- **Run button no longer sticks to the "Advanced" collapsible** — a bottom
  margin on `Collapsible` restores the breathing room (both family panels).

### Changed
- **Footer progress bar + stats line.** While running, the footer shows a
  wide aggregate progress bar (same monotonic-hold math as the header strip,
  with its own hold so plain log lines never reset it) plus a stats line:
  `65% • Elapsed 02:31 • ETA 01:21 • Optimizing INT8 [2600/4000] • 66.7it/s`
  (the it/s rate comes from a new tqdm rate capture in the stream parser).
- **Footer v2 — two-state mode switching.** While a quantization runs, the
  footer shows ONLY the progress panel (wide aggregate bar + stats line +
  per-phase bars + status); when the run finishes, the progress panel is
  replaced by the results card (path + buttons) stretched to full width —
  the verdict now appears exactly once instead of on both sides.
- **Layout v2 — run footer replaces the right rail.** The main area is now a
  single full-width parameter column; the stacked progress rail, the status
  line and the results card moved into a `#run_footer` bar at the bottom that
  appears when [Run Quantization] is pressed and stays visible afterwards —
  no more permanently reserved 30% side column. (Layout contract v2:
  `#body` children are `#params` + `#run_footer`; `#rail` is gone; all
  historical widget ids inside the footer are preserved, so handlers and
  profiles keep working unchanged.) The footer shows for the whole run —
  including the validation-failure early return — and through success,
  failure and user-stop outcomes.
- **[Copy path] / [Open folder] reveal only on success.** The results-card
  button row (`#result_buttons`) is hidden while idle/running/failed/stopped
  and appears exactly when a finished run has `status == "success"`.
- Footer e2e pins: structured progress envelopes render determinate bar rows
  inside the footer; exactly one `#status` exists; the log-drawer toggle
  keeps working above the footer.

## [0.8.0] - 2026-09-08

### Added
- **`native_bf16` — 5th native method (lossless BF16).** bf16 sources can
  hold magnitudes f16 cannot (|x| > 65504 → inf under `native_f16`);
  `native_bf16` stores bf16 tensors **verbatim** (GGUF qtype 30) — bit-
  lossless at the same 16 bpw. Includes vectorized
  `dtype_cast.f32_to_bf16` (RTNE, ties-to-even, NaN quiet-bit forced;
  bit-exact vs the scalar oracle on 100k random values). Registry now 40
  ids.

### Fixed
- **Blocked-architecture errors now name the fix.** When the unsloth
  backend rejects a model (unrecognized config or TTS/Seq2Seq blocklist),
  the error points at the native backend: `--method native_q8_0 --backend
  native` (or TUI: method `native_q8_0` → Run).
- **`native_*` method ids auto-route to the native backend** even without
  `--backend` — the id is the intent; older TUI profiles keep working.
- **Q8_0 rounding now bit-matches llama.cpp/quantui-rs** (user-reported:
  `native_q8_0` voice cloning slightly worse than quantui-rs output).
  The kernel used `rint(x/d)` — ties-to-even + true division — while
  llama.cpp computes `roundf(x * (1/d))`: half-away-from-zero with an
  f32 reciprocal. On ties ~0.1–0.5% of codes flipped ±1. After the fix,
  a full-byte diff against the VibeVoice oracle shows ALL tensors
  (378 Q8_0 + 103 F16 + 723 F32) byte-identical. Rounding pin test added
  (half-away on ±0.5 ties); Q4_0 re-verified bit-exact vs gguf-py.
- Native method descriptions trimmed to one short sentence (<80 chars,
  contract-tested); redundant `[NATIVE]` hint removed — badge +
  description suffice.

## [0.7.0] - 2026-09-08

### Added
- **Native GGUF backend (plan 2026-09-07) — GGUF export without
  transformers/unsloth/torch.** Four new method ids on a new
  `Backend.NATIVE`: `native_q8_0`, `native_q4_0`, `native_f16`,
  `native_f32`. The worker's `--backend native` branch runs BEFORE the
  architecture gate and any heavy import, so any HF safetensors checkpoint
  converts — including TTS/unknown architectures (e.g. VibeVoice-1.5B,
  previously rejected twice: by transformers and by the arch blocklist).
  Architecture: `gguf_names.py` (generic HF→GGUF name mapping, verbatim
  port of quantui-rs `gguf_names.rs`; unmapped tensors pass through
  unchanged), `gguf_qkernels.py` (Q8_0/Q4_0 numpy kernels byte-exact vs
  the gguf-py/llama.cpp oracle goldens; deterministic tensor plan with
  F16 demotion for non-block-divisible rows, token_embd stays F16),
  `gguf_export.py` (streaming exporter + CLI `python -m
  quantui.gguf_export`). `run_config` routes native ids to
  `--backend native` (no imatrix; one method per run). Live parity vs the
  user's quantui-rs VibeVoice-1.5B-q8_0 oracle: tensor-census equality
  (skip-guarded; runs in the worker env).
- **Native GGUF backend groundwork (plan 2026-09-07, STEP 0.1).** Pinned
  golden Q8_0/Q4_0 block bytes generated from the gguf-py oracle
  (`tests/golden/gguf_qgolden.py`): verbatim case-B hex (204/108 B with the
  mid-stream zero block) + SHA-256 digest pins for the 64/6/131072-block
  deterministic cases. Block layout confirmed scale-first
  (`block_q8_0 { f16 d; int8 qs[32]; }`); Q4_0 zero block stores `d = -0.0`
  (sign bit set) — both pinned by tests (`tests/test_gguf_qgolden.py`).

### Changed
- **Model-audit trio (NTH-012 / NTH-011 / NTH-007, commit 6e5bceb).**
  `quant_validator` now flags orphan `.input_scale` entries with no matching
  `.comfy_quant` sibling (NTH-012). The audit report gains a per-module `q%`
  column — quantized matrix params vs the module's linear + linear_review
  params (NTH-011) — and `ExclusionSuggestion.hints`: repeated (≥ 8×, same
  shape) unknown 2D-weight last-segments surfaced as whitelist *candidates*
  in both text and JSON renderers (NTH-007; the frozen `LINEAR_SEGMENTS`
  table stays authoritative, hints only suggest, never auto-apply).

### Changed
- **IMP-006: app.py composition root extraction (1306 → 1022 lines).** Two
  coherent clusters moved verbatim into duck-typed mixins so every
  `QuantApp.method` path and test monkeypatch target is unchanged:
  `quantui/form_state.py` (profile snapshot/apply, method-info line, data-
  driven ctq visibility, presets) and `quantui/run_finalization.py` (shared
  run finalizer with the output-ownership read, duration capture, bell).
- **GGUF registry aligned to the official Unsloth quant surface (35 ids).**
  The method list now mirrors `unsloth_zoo`'s `save_pretrained_gguf` exactly:
  the 24 `ALLOWED_QUANTS` ids plus the 11 `IMATRIX_QUANTS` ids, in the official
  order, each with approximate bits-per-weight from the llama.cpp qtype tables.
  The fake `q4_nl` id (never official — the real id is the imatrix-gated
  `iq4_nl`) and the `UD-*` Dynamic ids (`q4_k_xl` / `q3_k_xl` / `q2_k_xl`) are
  gone: UD mixes are proprietary download-only GGUFs that
  `save_pretrained_gguf` cannot produce, so listing them lied to the user. The
  UI now carries a footer explaining where to download official UD GGUFs.
- **`QuantMethod` carries `needs_imatrix`** and every trailing dataclass field
  is passed by keyword — a removed field can never silently inherit another
  flag's positional slot again (that bug class caused 126 test failures once).
  `list_line` badged `[IMATRIX]` instead of the removed `[DYNAMIC 2.0]`.
- **The GGUF method field is free text now (was a 35-entry dropdown).** A
  dropdown cannot express multi-method runs; the field accepts a comma list
  (`q4_k_m, q5_k_m, q8_0`) quantized left-to-right in one run. Validation
  (below) catches typos the dropdown used to prevent. The method description
  line updates live while typing, including the imatrix hint for `iq*` ids.
- **Run-time validation gate for GGUF methods (T5).** Unknown ids and `iq*`
  methods without an imatrix are refused *before* the worker starts — unsloth
  itself only fails after a full model load, so this turns a multi-minute
  failure into an instant one. `auto` skips the existence check (fetched at
  run time).
- **Default method is `q4_k_m` everywhere** (single-homed as
  `DEFAULT_GGUF_METHOD` in `quant_methods`). The wizard previously degraded to
  `not_quantized`, silently wasting the run.
- **Saved profiles referencing removed method ids downgrade on load.**
  `profiles_store.get_profile` filters dead ids out of `method`/`custom`
  (comma lists keep their valid parts; an all-dead custom is emptied) and
  records a visible `_notes` entry. The on-disk store is not rewritten on
  read.

### Added
- **Conformance validation vs unsloth references (LFM2.5-VL-3B).** Ran the
  worker end-to-end on the user's original BF16 with q8_0, q4_0, iq2_m and
  iq4_xs (the IQ* runs with an imatrix) and compared against unsloth's own
  published GGUFs with `scripts/check_gguf_conformance.py`: Q8_0 and the
  BF16 mmproj are **bit-identical** (266/266 and 441/441 tensors); IQ4_XS /
  Q4_0 are statistically identical (median per-tensor error ratio vs the
  reference 1.001–1.04, same quant-type mix; deltas trace to different
  imatrix calibration data and unsloth's dynamic tensor upgrades, not to
  spec violations). Report: `docs/reports/2026-09-04-gguf-conformance-lfm25-vl.md`
  (local-only). Fixes surfaced by the run: the architecture gate rejected
  VLM architectures (`Lfm2VlForConditionalGeneration`) that unsloth actually
  converts — it is a blocklist of TTS/Seq2Seq/audio types now; the export
  progress thread swallowed worker exceptions and reported DONE with no file
  written — exceptions now propagate; the disk-space headroom factor is
  env-tunable (`UQT_DISK_HEADROOM`, default 2.5).
- **End-to-end imatrix support for IQ* quantizations.** The GGUF panel's
  Advanced section gains an *Imatrix* path field plus an **Auto** checkbox
  (fetch the upstream Unsloth imatrix); `--imatrix` is emitted to the worker
  only when set, and `save_pretrained_gguf` / `push_to_hub_gguf` receive
  `imatrix_file=` for every `iq*` method. The worker enforces the same gate
  **before** importing unsloth (a missing imatrix fails in milliseconds, not
  after a model load). Profiles round-trip the imatrix setting; the Auto
  checkbox wins over a stale path.
- **Multi-method runs.** `parse_methods()` splits/normalizes comma lists; the
  worker accepts `--method q4_k_m, q5_k_m` and quantizes each id in order;
  `--list-methods` prints all 35 ids with `[IMATRIX]` markers.
- **Synchronous family switching (latent bug fix).** Pressing the family radio
  only *posted* `RadioSet.Changed`, so a caller that switched family and
  immediately ran (wizard / profile apply → Run) validated the **wrong**
  family's config. Previously masked by the old dropdown rejecting cross-family
  ids; free-text entry made it fatal. All family switches now apply instantly
  via a shared `_set_family` helper.
- **Capability probe moved to its own worker group.** `action_run` is an
  `exclusive=True` worker in the default group, so with the (now-synchronous)
  family switch starting the probe first, a run cancelled the probe. The probe
  runs in its own `capabilities` group instead.
- **`[IMATRIX]` badge rendering fix.** Textual's markup parser accepts
  UPPERCASE tags, so the literal badge was consumed as a style and vanished
  from the UI. Badges are now wrapped in real bold-yellow markup with the
  literal brackets escaped.

### Ruff debt paid down to zero (NTH-008) — 27 findings → 0, and the gate is
  now zero-tolerance.** Fixed in rule-homogeneous batches: E741 ×11 (ambiguous
  `l` → `line` in the RichLog-line comprehensions), B007 ×4 (unused loop vars →
  `_`), E402 ×3 (`stream_parser` imports hoisted above the
  `CTQ_PROGRESS_PREFIX` constant), F841 ×2 (dead `input_scales` / `overall`),
  UP042 ×2 (`Family`/`Backend` are now `enum.StrEnum`), B905 ×2 (explicit
  `zip(strict=…)`), E731 ×2 (assigned lambda → `def`), I001 ×1.
  The ruff baseline is ratcheted from 27 to 0, so any new
  finding fails `scripts/precommit_ruff.sh` outright.
- **`Family` / `Backend` are `enum.StrEnum`.** On Python 3.11+ a `(str, Enum)`
  mixin formats as `"Family.GGUF"`; `StrEnum` formats as the value `"gguf"`.
  The app only compared members or read `.value`, so behaviour is unchanged —
  the formatting contract is now pinned by `tests/test_quant_methods_enums.py`
  (9 tests).
- **Audit classifier recognizes `linear1`/`linear2` FFN weights.** Added to
  `LINEAR_SEGMENTS` after the VibeVoice-7B audit surfaced 156 unambiguous FFN
  matmul weights (`*.ffn.linear1/linear2.weight` in the acoustic/semantic
  tokenizers) landing in `linear_review` and bloating the suggested
  `exclude_layers` regex from 11 to 167 names. Breeze uses neither name, so the
  live golden tests are unaffected; the VibeVoice keep-set now collapses to the
  11 genuinely review-worthy tensors (embedding, head, and the small
  diffusion-head projections/modulations).

### Added
- **Method picker modal (pick from the 35 official ids).** The GGUF tab's
  *Quantization method* field gained a **Pick from list** button (`#pick_method`)
  that opens `MethodPickerScreen` — a `SelectionList` of all 35 official
  unsloth methods (registry order, `list_line` labels with `[IMATRIX]` badges
  and bits-per-weight), with the current method(s) preselected. Space toggles,
  Confirm fills the field with a comma-joined id list in **registry order**
  (deterministic regardless of click order) and refreshes the method-info line
  plus the output auto-suggestion; Cancel/Escape leaves the field untouched.
  The free-text `#method` Input stays editable (multi-method comma lists,
  profile compatibility) — the picker removes the typo risk of typing ids by
  hand; preselect semantics are additive (picking extends the current
  selection). Covered by `tests/test_method_picker_headless.py` (10 tests)
  plus lead-authored edge cases in `tests/test_method_picker_qa.py` (6 tests:
  empty-field no-op, typo-preselect, iq\* round-trip, full id-set coverage,
  and the hand-typed-typo validation gate regression).
- **On-demand torch-suite gate (NTH-009).** New `scripts/gate_torch.sh` runs
  the two torch-dependent suites (`test_stream_quant.py` +
  `test_incremental_safetensors.py` — the streaming quantization and resumable
  writer integration coverage) in the CTQ venv
  (a torch-enabled venv; `GATE_PYTHON` overrides). The script
  probes the interpreter and torch BEFORE pytest so failures name the actual
  problem; `--check` is a fast validate-only mode. Real run: 20 passed.
  Environment drift noted: the headless venv now carries torch-cpu, so both
  suites also run there with graceful `convert_to_quant` skips
  (14 passed / 6 skipped).
- **Opt-in packaging smoke (NTH-010).** `PACKAGING=1 bash scripts/gate_tests.sh`
  adds a packaging stage after pytest: a throwaway venv + `pip install -e .
  --no-deps` + console-entry-point resolution check via
  `scripts/packaging_smoke.py` (existence-based on both Windows `Scripts/`
  and POSIX `bin/` layouts; never launches the Textual app). Default OFF —
  the commit-time gate stays fast. Python manages the smoke's temp dir
  (bash `mktemp`/`rm -rf` mangle Windows paths and leaked the venv).
- **The ruff count gate survives zero findings (NTH-008).** `ruff` prints
  `All checks passed!` (not `Found N errors.`) when clean, so the
  count-extraction pipeline in `scripts/precommit_ruff.sh` matched nothing and
  — under `set -euo pipefail` — the failed assignment killed the script
  silently, i.e. the gate broke exactly when the finding count reached zero.
  The pipeline now ends in `|| true` (empty count → 0). `RUFF` / `MYPY` are
  overridable via env so the script is testable without the real tools;
  `tests/test_gate_scripts.py` (4 tests) pins the zero-finding case, the
  at-baseline pass, the above-baseline failure, and mypy blocking the gate
  (the baseline is read from the file at test time so future ratchets do not
  break the assertions).
- **Commit-time quality gate (IMP-004).** New `scripts/install_hooks.sh`
  installs an idempotent `.git/hooks/pre-commit` that runs the two gate stages
  in order and blocks the commit on the first failure, forwarding the failing
  stage's exit code. Stage 1 is the existing `scripts/precommit_ruff.sh`
  (ruff count vs. baseline + mypy); stage 2 is the new
  `scripts/gate_tests.sh`, which runs the headless pytest suite with both
  torch-dependent suites ignored (`GATE_PYTHON` overrides the interpreter).
  The repo has no remote, so push-based CI does not apply — enforcement
  happens at commit time, and the same scripts can be wrapped unchanged in CI
  later. Emergency bypass stays available via `git commit --no-verify`.
  Covered by `tests/test_hook_installer.py` (11 tests: install, idempotency,
  refusal outside a checkout, stage-failure blocking, exit-code propagation),
  which stubs both stages against throwaway temp repos so the real
  `.git/hooks` is never touched.
- **Sharded model folder support for the audit.** `model_audit` now audits
  HuggingFace sharded models directly — no merge step needed. New
  `audit_sharded_folder(path)` reads `model.safetensors.index.json`, scans
  every referenced shard header-only, and merges them into one `AuditReport`
  (tensors deduplicated by name in sorted-shard order, `quantized_layers`
  concatenated, metadata from the index's `"metadata"` key or the first
  shard's `__metadata__`). New `audit(path)` dispatcher routes single files
  to `audit_file` and sharded folders to `audit_sharded_folder`; the CLI
  (`python -m quantui.model_audit -i`), the `AuditScreen` modal, and the
  inline Audit button (`_resolve_audit_path`) all accept sharded folders now.
  Missing index, missing shard, malformed JSON, and malformed shard headers
  raise `AuditError` naming the offending path. Covered by
  `tests/test_model_audit_sharded.py` (12 tests, pure-stdlib fixtures).
- **Inline Audit button on the ComfyUI input row.** Auditing no longer requires
  Ctrl+P → "Audit model file": a small `Audit` button (`#audit_ctq_in`) now
  sits next to Browse on the `#ctq_input` row. It starts disabled and is
  enabled live (via `HandlersMixin.update_audit_button`, refreshed at every
  `#ctq_input` change site alongside `update_pt_suggest`) exactly when the
  typed path resolves to an auditable target; pressing it pushes the
  `AuditScreen` directly. Covered by new headless tests in
  `tests/test_model_audit_tui.py`.
- **PathModal path entry + Windows drive quick-jump.** `DirectoryTree` cannot
  navigate above its root, so on Windows a picker opened with an empty/relative
  start was trapped in one drive and could never reach another drive (`D:\`).
  The modal now has a path-entry `Input` (`#path_entry`, pre-filled with the
  start path): Enter on an existing directory re-roots the tree there, in file
  mode Enter on an existing file selects it and enables `Use Selected`, and
  nonexistent paths are ignored. On Windows only, a row of drive quick-jump
  buttons (`#drive_<d>` per existing drive letter) re-roots the tree at
  `<d>:\`; selecting a directory in the tree syncs its path back into the
  entry. Covered by new headless tests (`tests/test_path_modal_drives.py`).

### Fixed
- **Command palette actions crashed with `AttributeError`** (e.g. "Audit
  model file": `'Screen' object has no attribute 'action_audit_model'`).
  Textual 8.2.8 constructs palette providers with the *calling screen*
  (`app.screen_stack[-2]`), so `QuantCommands.self.screen` was the plain
  default `Screen` under the palette, not the app — while every action
  handler lives on `QuantApp`. `QuantCommands._run` / `_palette_run` now
  dispatch via `self.app`, and `_focus` queries the app DOM
  (`self.app.query_one`) so focus jumps work even when the palette is
  opened over a modal. Covered by new headless tests, including a
  tripwire asserting every palette action exists on `QuantApp`.

## [0.6.0] - 2026-08-27

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
