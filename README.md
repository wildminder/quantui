# Unsloth GGUF Quantizer (TUI)

A full-featured **terminal user interface** for converting HuggingFace / safetensors
models to **GGUF** with **official Unsloth quantization methods**.

You only need to provide three things:

1. **Model path** — a HuggingFace-format folder (`config.json` + `*.safetensors`
   + tokenizer files) **or** a single `.safetensors` file sitting next to a
   `config.json`.
2. **Output folder** — where the `.gguf` will be written (auto-suggested).
3. **Quantization method** — pick from the list (default `q4_k_m`): press
   **Pick from list** and check one or several of the 35 official methods;
   comma-separating (`q4_k_m, q5_k_m, q8_0`) quantizes several sizes in one
   run. IQ* methods require an **imatrix** (a local `.dat`/`.gguf` path or
   *Auto*).

The heavy Unsloth/CUDA work runs in a separate python process (configurable via
the "Worker Python interpreter" field), so this TUI stays light and responsive
and streams the worker's live log.

## Requirements

- A CUDA-capable GPU + a python environment with **`unsloth`** and a CUDA build of
  **torch** installed (this is what actually does the quantization).
- This TUI only needs **`textual`**.

## Install

```bash
cd unsloth-quant-tui
python -m venv .venv
.venv/Scripts/activate        # Windows  (or: source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

In the **same** environment (or any env that has `unsloth` + CUDA torch), make
sure `unsloth` is importable. Point the TUI's *Worker Python interpreter* field
at that environment's `python`/`python.exe` if it differs from the one running
the TUI.

## Run

```bash
python -m quantui
```

## Controls

- `1` Model path — type it or press **Browse** (folder/file picker).
- `2` Output folder — auto-filled as `<model>-<METHOD>`; edit or **Browse**.
- `3` Quantization method — press **Pick from list** to check the methods
  (multi-select supported; default `q4_k_m`). Comma-separate for
  multiple outputs in one run (`q4_k_m, q5_k_m`) — or just check several in
  the picker. Methods marked **[IMATRIX]** (the `iq*` family) need an
  imatrix — set a local path or check **Auto** in the Advanced section
  (fetches the upstream Unsloth imatrix). The *Custom method* box overrides
  the field above.
- Optional: max sequence length, load-in-4bit, push-to-hub repo + token.
- **Run Quantization** (or press `r`). Watch the live log on the right.
- `q` quits.

## How it works

`worker.py` is launched as a subprocess:

1. Validates the model path and required disk space (~2.5x model size).
2. `from unsloth import FastLanguageModel`.
3. `FastLanguageModel.from_pretrained(model_name=...)` (full precision by default).
4. `model.save_pretrained_gguf(output, tokenizer, quantization_method=method)`
   — pass an `imatrix_file` too when the method is an `iq*` id. A comma list
   quantizes each method in order into the same output folder.

> **About "UD-*" / Dynamic quantizations:** Unsloth's Dynamic (UD) mixes are
> **proprietary and download-only** — they are distributed as finished GGUFs on
> HuggingFace (e.g. `unsloth/<model>-GGUF`) and cannot be reproduced by
> `save_pretrained_gguf`. This tool therefore does not list them; download the
> official UD GGUFs directly instead.

## Picking a method (size vs quality)

| Method | ~bits/weight | Notes |
| --- | --- | --- |
| `f16` | 16 | Lossless intermediate. |
| `q8_0` | 8.6 | Near-lossless. |
| `q6_k` | 6.6 | High quality; uses Q8_K for all tensors. |
| `q5_k_m` | 5.5 | Recommended quality/size. |
| `q4_k_m` | 4.85 | Recommended 4-bit. **Default.** |
| `q3_k_m` | 3.9 | Small. |
| `q2_k_l` | ~3.5 | Unsloth preset (Q2_K body + Q8_0 embeddings/head). |
| `iq4_xs` | 4.25 | Imatrix-gated; better quality per bit than `q4_k`. |
| `iq2_xs` | 2.31 | Imatrix-gated; best ~2-bit quality. |

All 35 official ids are supported (24 standard + 11 `iq*` imatrix methods);
press **List all methods** in the TUI for the full annotated list.

> Method names can shift between Unsloth releases. The method field is free
> text validated against the official list at Run time, so a typo is caught
> instantly instead of after the model loads. A genuinely new id from a newer
> Unsloth can be typed into *Custom method* and is passed straight through.

## ComfyUI quantization formats (what to choose)

The ComfyUI tab's **Format** dropdown controls how weights are quantized:

| Format | Weights×Activations | Best for |
| --- | --- | --- |
| `fp8_e4m3` | 8-bit float × 8-bit | Default; good quality/size balance, no triton needed. |
| `int8` | INT8 × INT8 | One unified entry with a **Scaling mode** selector: `block` (block size 64/128/256; needs dims divisible by block size or *Skip inefficient layers*), `tensor` (simplest, robust to odd shapes), `row` (per-row scales; enables the optional **ConvRot rotation**, best accuracy, needs `triton`). |
| `nvfp4` / `mxfp8` | 4-bit / 8-bit microscaling | Blackwell (RTX 50xx) GPUs only; python 3.12 + torch 2.10 + CUDA 13. |
| `combine` | — (no quantization) | Merges a sharded input into ONE `.safetensors` without quantizing (output mode ignored); a single-file input is copied unchanged. No `.comfy_quant` baked. |
| `bf16` / `fp16` | — (dtype cast only) | Lossless RTNE cast of all floating tensors to bfloat16 / float16 (output mode ignored; sharded input is merge-cast into ONE `.safetensors`). Integer/bool tensors pass through byte-identical. No `.comfy_quant` baked; no torch quantizer needed. |
| `w4a4_convrot`, `w4a8_asym` | 4-bit × 4/8-bit | Smallest; requires the comfy-kitchen backend (a ComfyUI-python env). |

### About the dtype-cast formats (`bf16` / `fp16`)

These are **not quantization** — every floating-point tensor (F64/F32/F16/BF16)
is converted with round-to-nearest-even to the target dtype using pure integer
bit math (bit-exact, deterministic across platforms), while integer/bool
tensors (e.g. `inv_freq` buffers) are copied unchanged with their original
dtype recorded. A sharded input is always merged into ONE output file; a
single-file input is cast in place. The conversion streams tensor payloads in
~8 MiB chunks and never loads the whole model into RAM. Use `bf16` to shrink
an F32 checkpoint by half with negligible accuracy loss, or `fp16` when the
target runtime expects half precision.

### About INT8 scaling modes (unified `int8` format)

The former separate `int8_block` / `int8_tensor` / `int8_convrot` entries are
now ONE `int8` format with a **Scaling mode** dropdown — the UI only ever shows
the options that are valid for the selected scaling (`Block size` appears for
`block`; the **ConvRot rotation** toggle and its group size appear for `row`
only). Selecting row + ConvRot emits `--int8 --scaling_mode row --convrot` and
produces a W8A8 artifact whose on-disk `.comfy_quant` tag is
`format=int8_tensorwise, convrot=true` — exactly what ComfyUI's loaders look
for. (The even older `int8_row` duplicate alias was removed earlier because it
produced identical output.)

**What actually matters when ConvRot is enabled** is the *ConvRot group
size*: the quantizer picks per layer the largest group size
(64 / 256 / 1024) that divides the layer's input features — larger groups mean
fewer scales and a smaller file, smaller groups slightly better accuracy. Leave
it at the default 256 unless you have a reason.

Quick guidance:
- No triton / unsure → `fp8_e4m3`.
- Want smallest-good INT8 with rotation → `int8`, Scaling `row`, tick **Apply ConvRot**, GS 256.
- Hitting "dimensions divisible by block_size" → lower block size to 64 or tick *Skip inefficient layers*.


## Project layout

The TUI is split into a thin Textual composition root plus pure, Textual-free
modules. Only `app.py`, `panels.py`, `screens.py`, and `handlers.py` import
`textual`; everything else is testable without a TUI runtime.

| Module | Role |
| --- | --- |
| `quantui/app.py` | **Composition root.** Wires the mixin + Textual `App`; the only module that imports `textual`. |
| `quantui/quant_methods.py` | Family/method/format/preset registry (pure, Textual-free). Source of truth for GGUF + COMFY formats. |
| `quantui/stream_parser.py` | Worker stdout → segments; progress-line detection (pure). |
| `quantui/live_progress.py` | `LiveProgressStore` — thread-safe live-progress state. |
| `quantui/worker_runner.py` | `WorkerRunner` — launches the worker subprocess and pumps its log (DI seam). |
| `quantui/ui_bridge.py` | Bridges parsed log segments to UI widgets/sinks. |
| `quantui/run_config.py` | `RunConfig`/`GgufConfig`/`CtqConfig` dataclasses + pure validate/build-cmd logic (DI seam). |
| `quantui/panels.py` | Pure widget-builder functions that compose each panel. |
| `quantui/screens.py` | Modal screens (browse, confirm, etc.). |
| `quantui/handlers.py` | `HandlersMixin` — all `on_*` event handlers; reads widgets once into `RunConfig`. |
| `quantui/capabilities.py` | Worker-environment capability probing (incl. `comfy_kitchen`). |
| `quantui/incremental_safetensors.py` | Append-only `.safetensors` writer with resume support (no torch; pure stdlib). |
| `quantui/tensor_quant.py` | `QuantConfig` + per-tensor INT8 quantization core (block/tensor/row) and calibration cache. |
| `quantui/stream_quant.py` | `stream_quantize` / `stream_quantize_sharded` — lazy-read → quantize → append orchestrator with checkpoint manifest. |
| `quantui/worker.py` | GGUF worker subprocess (Unsloth). |
| `quantui/worker_ctq.py` | ComfyUI / `convert_to_quant` worker subprocess (FP8 / W8A8 / NVFP4 / MXFP8 / combine / bf16 / fp16 cast). |
| `quantui/worker_ctq_kitchen.py` | **comfy-kitchen** worker (W4A4 `convrot_w4a4`, W4A8 `asym_w4a8_int8`). Runs in a ComfyUI-python interpreter. |
| `quantui/dtype_cast.py` | Pure bit-exact RTNE dtype-cast core + streaming cast writer (F32/F64 ↔ BF16/F16; no numpy/torch). |
| `quantui/comfy_quant_schema.py` | Pure-stdlib `.comfy_quant` schema validator + serializer (no torch/safetensors import). |
| `quantui/model_audit.py` | Pure-stdlib model audit: header-only tensor classifier + per-module aggregation + `exclude_layers` suggestion + text/JSON renderers + CLI (no torch/safetensors/numpy). |

### ComfyUI backend requirements

The COMFY family supports two **mutually exclusive** worker backends, chosen per format
(see `quantui/quant_methods.py` `Backend`):

| Backend | Module | Formats | Requires |
| --- | --- | --- | --- |
| `convert_to_quant` (`Backend.CTQ`) | `quantui/worker_ctq.py` | `fp8_e4m3`, `int8` (scaling block/tensor/row, optional ConvRot), `nvfp4`, `mxfp8`, `combine`, `bf16`, `fp16` | `convert_to_quant` + CUDA torch (+ `triton` when the INT8 ConvRot toggle is on; NOT needed for `combine` / `bf16` / `fp16` — those paths never import the quantizer) |
| `comfy_kitchen` (`Backend.COMFY_KITCHEN`) | `quantui/worker_ctq_kitchen.py` | `w4a4_convrot`, `w4a8_asym` | a ComfyUI-python interpreter with `comfy-kitchen` + `comfy.quant_ops` (set *Worker Python (ctq)* to that env) |

`convert_to_quant` **cannot** emit W4A4/W4A8 — those are produced only by the
comfy-kitchen path via ComfyUI's `comfy.quant_ops`. If a W4A4/W4A8 format is
selected but the worker interpreter lacks `comfy-kitchen`, `capabilities.py`
warns and the run is blocked rather than failing obscurely mid-quantize.

On-disk outputs follow the current ComfyUI-native `{"format": ...}` `.comfy_quant`
schema; see [`docs/comfy-quant-schema.md`](docs/comfy-quant-schema.md).

## Streaming & resumable quantization (ComfyUI / `convert_to_quant`)

For **INT8 without rotation** (`int8`, Scaling `block` / `tensor`) the worker no longer
needs to materialize a giant merged unquantized temp file for a sharded model.
Instead it **streams tensor-by-tensor**:

> **Which INT8 modes stream?** Only the *non-rotation* scaling modes —
> `block` and `tensor`. The rotation mode (`row` + **Apply ConvRot**,
> W8A8) deliberately stays on the legacy whole-file path —
> ConvRot needs a per-layer pre-rotation pass that is out of scope for v1 streaming.
> FP8 / NVFP4 / MXFP4 / combine also stay on the legacy path (combine does not
> quantize at all).

```
input (single .safetensors  OR  HF sharded folder)
  → scan headers only (no load)
  → for each tensor: lazy-read it → quantize it → append to output → checkpoint
  → rewrite the small 8-aligned header at the front once at the end
```

This is the default for INT8. It replaces the `--output-mode single` temp-merge
path (`merge every shard → load the whole thing → quantize → delete temp`).

### Why it matters

| Metric | Legacy `--output-mode single` | Streaming (INT8) |
| --- | --- | --- |
| Peak extra disk | ~2× model (merged temp + output) | ~1× (output only) |
| Peak memory | whole model in RAM | one tensor + its scales |
| Resume after kill/crash | none (restart from zero) | yes (manifest checkpoint) |

### Resume / checkpoint

A run can be interrupted (Ctrl-C, crash, reboot) and continued. A checkpoint
manifest is written alongside the output:

- Single output file: `<output>.quant-manifest.json`
- Sharded output folder: `<output_dir>/.quant-manifest.json`

The manifest records the quant-config hash plus the set of tensors already
written. **To resume, simply re-run the exact same command** — the worker re-opens
the partial output, reads back which tensors are present, and continues from the
next one. The final file is byte-identical to an uninterrupted run.

If you re-run with **different quantization flags**, the config-hash guard detects
the mismatch and refuses to resume (so you never silently corrupt a half-quantized
file) — delete the manifest + partial output and start fresh in that case.

### Which paths stream vs. stay on the legacy path

- **INT8, no rotation** → streaming (single-file input *and* sharded input, either
  `--output-mode single` or the default sharded output).
- **ConvRot (rotation)** → legacy whole-file path. Rotation needs a per-layer
  pre-rotation pass and is out of scope for v1 streaming.
- **FP8 / NVFP4 / MXFP4 / combine** → unchanged legacy path (combine never
  quantizes; it only merges/copies).
- **`--no-stream`** escape hatch: force the legacy merge path for INT8 if you ever
  need it (e.g. to compare outputs). Streaming is otherwise the default for INT8.
- **`--heur`** (skip layers with poor quantization characteristics; non-divisible
  2D weights are copied unchanged instead of being quantized) and **`--manual_seed`**
  (fixed seed for bias-correction calibration) are threaded to *both* the streaming
  and the legacy paths so behavior stays consistent.
- **INT8 parameters are now exposed in the ComfyUI tab.** When the `int8` format is
  selected, extra controls appear: **Scaling mode** (block / tensor / row), **Block
  size** (select: 64 / 128 / 256 — default 128; only shown for `block` scaling),
  **Skip inefficient layers (--heur)** (checkbox), and **Manual seed** (optional
  input). If you hit `INT8 block-wise quantization requires dimensions
  divisible by block_size` (e.g. a weight shaped `(49152, 576)` is not divisible by
  128), either lower **Block size** to 64 (576 % 64 == 0) or enable **Skip inefficient
  layers** to copy those non-divisible 2D weights unchanged. With Scaling `row`, an
  **Apply ConvRot** checkbox (needs triton) and the **ConvRot group size** control appear.

### Validating quantized files

A standalone validation feature checks a ComfyUI-native INT8 `.safetensors` for
structural integrity **without re-quantizing** — modeled on the checks in
`ComfyUI-Raon-OpenTTS/tools/validate_raon_int8_convrot.py` but model-agnostic.

- **In the TUI:** the ComfyUI tab has a *Validate* input + button. Point it at any
  quantized `.safetensors`; it reads only the safetensors header (fast even for
  multi-GB files) and reports PASS/FAIL with a per-format summary
  (matrix count, ConvRot group-size histogram, quantized vs full-precision param
  share, file size).
- **From the CLI:** `python -m quantui.quant_validator <file.safetensors>`
  (`--numeric` also loads the int8 weights + scales and checks for int8 overflow
  and non-finite / non-positive scales; needs `safetensors` + `numpy`).

What it verifies: `.comfy_quant` JSON is well-formed and uses a supported format
(`int8_tensorwise` / `int8_blockwise`); the quantized `weight` is `int8` and 2-D;
`weight_scale` is `float32` with the correct shape (`[out, 1]` for tensorwise/row/
ConvRot, `[ceil(out/g), ceil(in/g)]` for blockwise); blockwise layers carry an
`input_scale` scalar; ConvRot `convrot_groupsize` is a power of four that divides
`in_features`; biases are **not** quantized; and there are no orphan
`weight_scale` / `.comfy_quant` entries. A file with no `.comfy_quant` markers is
reported OK with a warning (it is a plain FP8/FP16 or combine/merge
checkpoint).

> `quantui/quant_validator.py` reuses the pure-stdlib header parser in
> `quantui/comfy_quant_schema.py`, so the structural pass needs neither `torch`
> nor `safetensors`.

## Model audit (tensor inventory + exclusion advisor)

The **model audit tool** automates *stage 1* of the quantization-exclusion
funnel: point it at a `.safetensors` checkpoint and it classifies every tensor,
aggregates per-module statistics, detects already-quantized layers, and proposes
a starting `exclude_layers` regex for the unified INT8 format.

It is a **header-only** scan — it reads only the safetensors header (and the
tiny `.comfy_quant` JSON descriptor blobs), never the tensor payloads — so a
7 GiB checkpoint is inventoried in milliseconds without loading it into RAM.
The core (`quantui/model_audit.py`) is pure stdlib (no torch / safetensors /
numpy), so it runs in the headless test env and any worker env.

**What it does — and deliberately does not do.** The tool produces the
*inventory* for stage 2 but does **not** attempt stage 2 itself: call-frequency
and GEMM shape cannot be inferred from a checkpoint, so hot-loop / skinny-GEMM
exclusions remain a human decision after reading the pipeline code. The tool
suggests, the human decides. It is strictly read-only (never modifies a file)
and supports safetensors only (no GGUF).

### CLI usage

```bash
python -m quantui.model_audit -i <file.safetensors>            # text report to stdout
python -m quantui.model_audit -i <file.safetensors> --json     # JSON report instead
python -m quantui.model_audit -i <file.safetensors> --out report.json
```

`--out PATH` also writes the report to PATH (parents created) while still
echoing it to stdout. Missing or malformed input prints a message naming the
path on stderr and exits 2.

### TUI usage

Open the command palette (Ctrl+P) and run **"Audit model file"**. The command
resolves the target from the `#ctq_input` field when the ComfyUI family tab is
active and the input is a `.safetensors` file or a single-file folder; otherwise
it opens the file picker first. The resulting modal shows a summary header line,
a per-module table (module / tensors / params / bytes / linears), and the
suggested `exclude_layers` regex in a selectable/copyable box. Auditing a
checkpoint that is already kitchen-quantized additionally reports the
quantized-layer count, the per-format histogram, and the unquantized 2D
`.weight` remainder (the effective exclusion set of the existing build).

### What the suggestion covers

The suggested regex covers exactly the 2D `.weight` tensors that ctq would
otherwise quantize but that should be kept: **embeddings**, **heads**, and
unknown-role 2D weights (`linear_review`). Vectors, biases, 3D+ conv weights,
and the kitchen companion tensors are never touched by ctq, so they need no
entry. The regex is one anchored alternation of `re.escape`d, sorted tensor
names (`^(a|b|c)$`), safe under the `re.search` semantics of
`QuantConfig.excluded`. An empty keep-set yields an empty suggestion.

### Worked example (Breeze-TTS-2 hybrid recipe)

Auditing `Breeze-TTS-2-bf16.safetensors` (1115 tensors) classifies to
`linear 558 · vector 424 · bias 60 · other 67 · embedding 4 · head 1 ·
linear_review 1`, with per-module linears `backbone_model 196 · text_encoder
182 · depth_decoder 84 · codec_model 96`. The tool's suggestion covers the six
2D keep-tensors (the four embeddings, `lm_head.weight`, and
`text_encoder_proj.weight`). After reading the pipeline you learn the depth
decoder runs in a hot loop and the codec model is never instantiated at
inference, so you extend the regex with `^depth_decoder\.` (and optionally
`^codec_model\.`) — quantizing exactly `backbone_model` + `text_encoder` (378
tensors), which is precisely what the official `int8-hybrid` build did.

### Determinism

### How to confirm streaming is actually active

If you want to verify a given run streamed (vs. silently used the legacy merge),
watch the **run log** (the worker echoes the exact command it launched, then its
own progress lines):

- **Streaming active:** the log shows `Streaming quantization <shards> -> <output> ...`
  and the output `.safetensors` file appears in the destination and **grows during**
  the run (it is written tensor-by-tensor). No `unsloth-ctq-merged-*.safetensors`
  temp file is created.
- **Legacy merge active:** the log shows `Merging N shard(s) into one file ...`
  and a `unsloth-ctq-merged-<timestamp>.safetensors` temp file is created in your OS
  temp dir (`tempfile.gettempdir()`), then deleted when quantization finishes.

> Note: leftover `unsloth-ctq-merged-*` files can linger in the temp dir from earlier
> *legacy* runs (e.g. before this change, or when using a rotation/non-INT8 format).
> Their presence alone does **not** prove the current run merged — check the log line
> above. If you see them but the log says `Streaming`, they are stale and can be deleted.
INT8 streaming uses a **fixed default calibration seed** (`QuantConfig.calib_seed`,
default `233983427`) so a re-run — and a resumed run — are reproducible. The
legacy `convert_to_quant` baseline re-rolls a random seed on every run; the
streaming path does not, which is intentional. Pass `--manual_seed` to override.

## Development: quality gates

Every commit is gated. The gate has two stages, both of which must pass before
`git commit` succeeds:

| Stage | Script | What it runs |
| --- | --- | --- |
| 1 | `scripts/precommit_ruff.sh` | ruff count vs. the frozen baseline in `docs/reviews/ruff-baseline.txt`, then mypy over the five typing-clean core modules (`COVERAGE=1` additionally enforces per-module coverage floors) |
| 2 | `scripts/gate_tests.sh` | the headless pytest suite: `python -m pytest tests/ -q --ignore=tests/test_incremental_safetensors.py --ignore=tests/test_stream_quant.py` |

Install the gate as a git hook (idempotent — safe to re-run):

```bash
bash scripts/install_hooks.sh       # writes .git/hooks/pre-commit
bash scripts/install_hooks.sh /path/to/other/checkout
```

The repo currently has **no remote**, so there is no push-based CI: enforcement
happens at commit time. The hook re-runs the exact same scripts, so it can also
be invoked by hand — `bash .git/hooks/pre-commit` — and, if a remote is added
later, wrapped unchanged in a CI workflow.

Run the stages individually:

```bash
bash scripts/precommit_ruff.sh      # ruff + mypy
bash scripts/gate_tests.sh          # pytest (override interpreter via GATE_PYTHON=...)
```

Emergency bypass for a single commit:

```bash
git commit --no-verify -m "..."
```

Two suites stay excluded from the headless gate because they need a real torch
install: `tests/test_stream_quant.py` and `tests/test_incremental_safetensors.py`.
Run them in one of the CTQ environments (see `scripts/gate_torch.sh`, NTH-009).

