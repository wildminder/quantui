# Unsloth Dynamic 2.0 GGUF Quantizer (TUI)

A full-featured **terminal user interface** for converting HuggingFace / safetensors
models to **GGUF** with **Unsloth Dynamic 2.0** quantization.

You only need to provide three things:

1. **Model path** — a HuggingFace-format folder (`config.json` + `*.safetensors`
   + tokenizer files) **or** a single `.safetensors` file sitting next to a
   `config.json`.
2. **Output folder** — where the `.gguf` will be written (auto-suggested).
3. **Quantization method** — picked from an easy dropdown (Dynamic 2.0 variants
   are badged), with a free-text override in case Unsloth renames a method.

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
- `3` Quantization method — dropdown. Options marked **[DYNAMIC 2.0]** use
  Unsloth's per-layer selective quantization (e.g. `UD-Q4_K_XL`). The
  *Custom method* box overrides the dropdown for any method Unsloth supports.
- Optional: max sequence length, load-in-4bit, push-to-hub repo + token.
- **Run Quantization** (or press `r`). Watch the live log on the right.
- `q` quits.

## How it works

`worker.py` is launched as a subprocess:

1. Validates the model path and required disk space (~2.5x model size).
2. `from unsloth import FastLanguageModel`.
3. `FastLanguageModel.from_pretrained(model_name=...)` (full precision by default).
4. `model.save_pretrained_gguf(output, tokenizer, quantization_method=method)`
   — this is where Dynamic 2.0 per-layer selection happens.

## Picking a method (size vs quality)

| Method | ~bits/weight | Notes |
| --- | --- | --- |
| `f16` | 16 | Lossless intermediate. |
| `q8_0` | 8.6 | Near-lossless. |
| `q5_k_m` | 5.5 | Recommended quality/size. |
| `q4_k_m` | 4.85 | Recommended 4-bit. |
| `UD-Q4_K_XL` (`q4_k_xl`) | ~4.5 | **Dynamic 2.0** — best quality at Q4 size. |
| `UD-Q3_K_XL` (`q3_k_xl`) | ~3.5 | **Dynamic 2.0**, smaller. |
| `UD-Q2_K_XL` (`q2_k_xl`) | ~2.7 | **Dynamic 2.0**, smallest. |

> Method names can shift between Unsloth releases. If a method you want isn't in
> the dropdown, type it into *Custom method* — the worker passes it straight to
> Unsloth and warns (instead of failing) if it's unknown.

## ComfyUI quantization formats (what to choose)

The ComfyUI tab's **Format** dropdown controls how weights are quantized:

| Format | Weights×Activations | Best for |
| --- | --- | --- |
| `fp8_e4m3` | 8-bit float × 8-bit | Default; good quality/size balance, no triton needed. |
| `int8` | INT8 × INT8 | One unified entry with a **Scaling mode** selector: `block` (block size 64/128/256; needs dims divisible by block size or *Skip inefficient layers*), `tensor` (simplest, robust to odd shapes), `row` (per-row scales; enables the optional **ConvRot rotation**, best accuracy, needs `triton`). |
| `nvfp4` / `mxfp8` | 4-bit / 8-bit microscaling | Blackwell (RTX 50xx) GPUs only; python 3.12 + torch 2.10 + CUDA 13. |
| `w4a4_convrot`, `w4a8_asym` | 4-bit × 4/8-bit | Smallest; requires the comfy-kitchen backend (a ComfyUI-python env). |

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
| `quantui/worker_ctq.py` | ComfyUI / `convert_to_quant` worker subprocess (FP8 / W8A8 / NVFP4 / MXFP8 / on-the-fly passthrough). |
| `quantui/worker_ctq_kitchen.py` | **comfy-kitchen** worker (W4A4 `convrot_w4a4`, W4A8 `asym_w4a8_int8`). Runs in a ComfyUI-python interpreter. |
| `quantui/comfy_quant_schema.py` | Pure-stdlib `.comfy_quant` schema validator + serializer (no torch/safetensors import). |

### ComfyUI backend requirements

The COMFY family supports two **mutually exclusive** worker backends, chosen per format
(see `quantui/quant_methods.py` `Backend`):

| Backend | Module | Formats | Requires |
| --- | --- | --- | --- |
| `convert_to_quant` (`Backend.CTQ`) | `quantui/worker_ctq.py` | `fp8_e4m3`, `int8` (scaling block/tensor/row, optional ConvRot), `nvfp4`, `mxfp8`, `onthefly` | `convert_to_quant` + CUDA torch (+ `triton` when the INT8 ConvRot toggle is on) |
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
> FP8 / NVFP4 / MXFP4 / on-the-fly passthrough also stay on the legacy path.

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
- **FP8 / NVFP4 / MXFP4 / on-the-fly passthrough** → unchanged legacy path.
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
reported OK with a warning (it is a plain FP8/FP16 or on-the-fly passthrough
checkpoint).

> `quantui/quant_validator.py` reuses the pure-stdlib header parser in
> `quantui/comfy_quant_schema.py`, so the structural pass needs neither `torch`
> nor `safetensors`.

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

Run the test suite:

```bash
.venv-test\Scripts\python.exe -m pytest -q
```

