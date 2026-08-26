"""Backend worker for the ComfyUI / ``convert_to_quant`` (ctq) quantization family.

This script is launched by the TUI as a subprocess (``pybin_ctq worker_ctq.py ...``)
so the UI stays responsive and the heavy ``convert_to_quant`` / torch / CUDA work
lives in its own interpreter (which is typically a *separate* env from the one
running Unsloth for the GGUF family).

Differences from ``worker.py`` (GGUF):
  * Input is a single ``.safetensors`` file, or a folder containing exactly one.
  * Output is a ``.safetensors`` file.
  * **No** ``check_supported_architecture`` guard -- diffusion models are not causal
    LMs and have no ``ForCausalLM`` arch, so that guard would wrongly reject them.

All progress is printed to **stdout** (stderr merged) and streamed to the TUI,
mirroring the GGUF worker's log/fail contract.

Usage:
    python worker_ctq.py -i model.safetensors -o model-Q8.safetensors --int8 \\
        --scaling_mode block --comfy_quant --save_quant_metadata
    python worker_ctq.py -i model_folder -o model-int8-row-convrot-gs256.safetensors --flux2
"""

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass

from .comfy_quant_schema import _align_header_to_8
from .quant_methods import INDEX_NAME, is_sharded_folder
from .stream_parser import CTQ_PROGRESS_PREFIX
from .stream_quant import stream_quantize, stream_quantize_sharded
from .tensor_quant import QuantConfig


def log(msg: str) -> None:
    """Print a progress line to stdout (streamed to the TUI)."""
    print(msg, flush=True)


def progress(
    phase: str,
    cur: int | None = None,
    total: int | None = None,
    label: str = "",
    pct: float | None = None,
) -> None:
    """Emit a structured ``CTQ_PROGRESS`` envelope for the TUI progress bar.

    The TUI (``stream_parser`` + ``LiveProgressStore`` + ``ProgressView``) recognizes this
    exact prefix + JSON form and drives a real determinate ProgressBar from ``cur``/``total``
    (or ``pct``). This is our OWN reliable progress signal, unlike the third-party
    tqdm / ctq ``(N/M) Processing`` headers, which are not guaranteed to be emitted.
    """
    import json

    payload: dict = {"phase": phase}
    if cur is not None:
        payload["cur"] = int(cur)
    if total is not None:
        payload["total"] = int(total)
    if pct is not None:
        payload["pct"] = float(pct)
    if label:
        payload["label"] = label
    print(f"{CTQ_PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


def fail(msg: str) -> None:
    """Print an error to stderr and exit(1)."""
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def resolve_input(path: str) -> str:
    """Resolve the ctq input path.

    Accepts a single ``.safetensors`` file, or a folder containing exactly one
    ``.safetensors`` file (returns the folder path in that case). Diffusion models
    have no architecture guard, so we only validate the file type / presence.
    """
    if not path:
        fail("Input path is required (-i/--input).")
    if not os.path.exists(path):
        fail(f"Input does not exist: {path}")

    if os.path.isfile(path):
        if not path.endswith(".safetensors"):
            fail("Input file must be a .safetensors file (or pass a folder containing one).")
        return path

    if not os.path.isdir(path):
        fail(f"Input is neither a file nor a folder: {path}")

    # A HuggingFace sharded folder (model.safetensors.index.json present) is accepted
    # as-is; the sharded branch in main() handles it without merging.
    if is_sharded_folder(path):
        return path

    sts = [f for f in os.listdir(path) if f.endswith(".safetensors")]
    if not sts:
        fail(f"No .safetensors file found inside folder: {path}")
    if len(sts) > 1:
        fail(
            f"Found {len(sts)} .safetensors files inside {path}; pass a single "
            "file or a folder with exactly one .safetensors."
        )
    # Folder form: return the folder itself; convert_to_quant reads it directly.
    return path


def build_quantize_kwargs(args: argparse.Namespace) -> dict:
    """Map the parsed CLI ``Namespace`` to the kwargs passed to ``quantize(...)``.

    ``input`` / ``output`` are positional-style keyword args of ``quantize``; the
    rest are its optional keyword args. ``None`` is passed for unset optional flags
    so the real ``convert_to_quant`` keeps its own defaults.
    """
    kwargs: dict = {
        "input": args.input,
        "output": args.output,
        # toggles / format flags
        "comfy_quant": bool(args.comfy_quant),
        "save_quant_metadata": bool(args.save_quant_metadata),
        "int8": bool(args.int8),
        "scaling_mode": args.scaling_mode,
        "convrot": bool(args.convrot),
        "convrot_group_size": int(args.convrot_group_size) if args.convrot_group_size else None,
        "nvfp4": bool(args.nvfp4),
        "mxfp8": bool(args.mxfp8),
        # presets
        "flux2": bool(args.flux2),
        "wan": bool(args.wan),
        "t5xxl": bool(args.t5xxl),
        "hunyuan": bool(args.hunyuan),
        "zimage": bool(args.zimage),
        # misc toggles
        "simple": bool(args.simple),
        "low_memory": bool(args.low_memory),
        "heur": bool(args.heur),
        "verbose": "VERBOSE" if args.verbose else "NORMAL",
        # optional passthroughs
        "calib_samples": int(args.calib_samples) if args.calib_samples else None,
        "exclude_layers": args.exclude_layers or None,
        "output_dtype": args.output_dtype or None,
        "block_size": int(args.block_size) if args.block_size else None,
        "manual_seed": int(args.manual_seed) if args.manual_seed else None,
        "num_iter": int(args.num_iter) if getattr(args, "num_iter", None) else None,
    }
    # Drop None-valued keys so convert_to_quant.quantize() keeps its OWN parser
    # defaults (calib_samples=3072, convrot_group_size=256). If we passed None,
    # it would override the library default and crash downstream
    # (e.g. torch.randn(None, ...) -> TypeError).
    return {k: v for k, v in kwargs.items() if v is not None}


# --------------------------------------------------------------------------- #
# HuggingFace sharded-folder support (ADDITIVE -- single-file path untouched)
# --------------------------------------------------------------------------- #
@dataclass
class ShardedModel:
    """Resolved HuggingFace sharded model, ready for per-shard quantization."""

    model_dir: str                       # absolute path to the HF model folder
    index_path: str                      # absolute path to model.safetensors.index.json
    weight_map: dict[str, str]           # tensor name -> shard filename (relative to model_dir)
    shard_files: list[str]               # unique, order-preserving shard filenames (relative)
    non_weight_files: list[str]          # files to copy verbatim (config.json, tokenizer*, ...)


def discover_shards(model_dir: str) -> ShardedModel:
    """Parse ``model.safetensors.index.json`` and enumerate shards + sidecar files.

    Raises ``FileNotFoundError`` if the index json is missing. Shard order follows
    first appearance in ``weight_map.values()`` (HF writes shards in order). Any file
    in ``model_dir`` that is not the index json and not a referenced shard is treated
    as a non-weight file to copy verbatim (config/tokenizer/etc.). Orphan ``.safetensors``
    not in the index are ignored.
    """
    index_path = os.path.join(model_dir, INDEX_NAME)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(index_path)
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    weight_map = index.get("weight_map", {})
    shard_files: list[str] = []
    for shard in weight_map.values():
        if shard not in shard_files:
            shard_files.append(shard)
    shard_set = set(shard_files)
    non_weight: list[str] = []
    for entry in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, entry)
        if not os.path.isfile(full):
            continue
        if entry == INDEX_NAME or entry in shard_set:
            continue
        if entry.endswith(".safetensors"):
            continue  # orphan shard not referenced by the index
        non_weight.append(entry)
    return ShardedModel(model_dir, index_path, weight_map, shard_files, non_weight)


def quantize_shards(model: ShardedModel, output_dir: str, base_kwargs: dict, quantize) -> None:
    """Quantize each shard with IDENTICAL flags, then copy index json + sidecars.

    Per-shard quantize is mathematically identical to merge-then-quant (research
    verdict): HF sharding splits by whole tensors, and ctq scaling is per-tensor.
    The ORIGINAL index json is copied unchanged so ``weight_map`` (shard filenames
    + tensor names) stays valid; non-weight files are copied verbatim.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(model.shard_files)
    if total:
        progress("shard", cur=0, total=total, label=f"Quantizing {total} shard(s)")
    for i, shard in enumerate(model.shard_files, 1):
        in_path = os.path.join(model.model_dir, shard)
        out_path = os.path.join(output_dir, shard)
        progress("shard", cur=i, total=total, label=f"Quantizing shard {shard}")
        log(f"[{i}/{total}] Quantizing shard {shard} ...")
        kw = dict(base_kwargs)            # identical quant flags for every shard
        kw["input"] = in_path
        kw["output"] = out_path
        quantize(**kw)
        log(f"[{i}/{total}] Wrote {out_path}")
    # index json copied UNCHANGED -> weight_map stays valid
    shutil.copy2(model.index_path, os.path.join(output_dir, INDEX_NAME))
    log(f"Copied {INDEX_NAME}")
    for fname in model.non_weight_files:
        src = os.path.join(model.model_dir, fname)
        shutil.copy2(src, os.path.join(output_dir, fname))
        log(f"Copied {fname}")
    log("DONE: Sharded ComfyUI quantization written.")


def merge_safetensors_files(shard_paths: list[str], out_path: str) -> None:
    """Concatenate a list of ``.safetensors`` files into ONE valid file.

    Parses each shard header (8-byte LE uint64 length + JSON), rewrites every
    tensor's ``data_offsets`` to cumulative offsets across the concatenation, and
    streams the tensor buffers (byte-range copy in ~8 MiB chunks, never a full
    in-memory load). Tensor names must be unique across shards. ``__metadata__``
    (if present in the first shard) is carried through. The result is
    math-identical to the original model split into one file, so ``quantize``
    reads it exactly like the source shards.
    """
    merged: dict[str, dict] = {}
    meta = None
    shard_infos: list[tuple[str, int, int]] = []  # (path, buffer_start, buffer_len)
    cursor = 0  # cumulative data-region length of already-merged shards
    for sp in shard_paths:
        with open(sp, "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hdr_len).decode("utf-8"))
        if "__metadata__" in header and meta is None:
            meta = header["__metadata__"]
        buffer_start = 8 + hdr_len
        buffer_len = os.path.getsize(sp) - buffer_start
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in merged:
                raise ValueError(f"duplicate tensor {name!r} across shards")
            new_spec = dict(spec)
            # Shift this shard's tensor byte range by the cumulative prior length.
            new_spec["data_offsets"] = [s + cursor for s in spec["data_offsets"]]
            merged[name] = new_spec
        shard_infos.append((sp, buffer_start, buffer_len))
        cursor += buffer_len

    out_header: dict[str, object] = {}
    if meta is not None:
        out_header["__metadata__"] = meta
    out_header.update(merged)
    out_bytes = json.dumps(out_header).encode("utf-8")
    out_bytes = _align_header_to_8(out_bytes)

    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(out_bytes)))
        out.write(out_bytes)
        for sp, buffer_start, buffer_len in shard_infos:
            with open(sp, "rb") as fh:
                fh.seek(buffer_start)
                remaining = buffer_len
                while remaining > 0:
                    chunk = fh.read(min(8 << 20, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)


def _combine_copy_file(src: str, dst: str) -> None:
    """Byte-identical copy of a single .safetensors to the output path."""
    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    shutil.copy2(src, dst)


def _run_combine(args: argparse.Namespace) -> None:
    """Combine format (plan 2026-08-26, rev. 2): NO quantization.

    A single-file input is copied byte-identical. A sharded input is ALWAYS
    merged into ONE output .safetensors (output mode is irrelevant -- merging
    is the format's whole purpose), via ``merge_safetensors_files`` which
    streams byte ranges and never loads the whole model. No ``.comfy_quant``
    metadata is baked.
    """
    if is_sharded_folder(args.input):
        out = args.output
        if not out.endswith(".safetensors"):
            fail("Combine requires a .safetensors file path as --output "
                 "(it always merges shards into ONE file).")
        model = discover_shards(args.input)
        shard_paths = [os.path.join(args.input, s) for s in model.shard_files]
        progress("merge", cur=0, total=len(shard_paths),
                 label=f"Combining {len(shard_paths)} shard(s)")
        log(f"Combining {len(shard_paths)} shard(s) into one file ...")
        merge_safetensors_files(shard_paths, out)
        progress("merge", cur=len(shard_paths), total=len(shard_paths),
                 label="Combined")
    else:
        _combine_copy_file(args.input, args.output)


# Safetensors dtype id each --cast_dtype CLI value maps to.
_CAST_DTYPE_IDS = {"bfloat16": "BF16", "float16": "F16"}


def _run_cast(args: argparse.Namespace) -> None:
    """bf16/fp16 cast-only path (plan 2026-08-26 STEP 3.1): NO quantization.

    Dispatches single vs sharded like ``_run_combine``: a single-file input is
    cast in place to one output; a sharded input with a .safetensors output is
    merge-cast into ONE file. Floating tensors are RTNE-cast; integer/bool
    tensors pass through unchanged (see dtype_cast module docs).
    """
    from .dtype_cast import cast_safetensors_file, cast_shards_to_single

    target = _CAST_DTYPE_IDS[args.cast_dtype]
    label = args.cast_dtype

    def _on_progress(done: int, total: int) -> None:
        progress("cast", cur=done, total=total,
                 label=f"Casting to {label}" if done < total else f"Cast to {label}")

    if is_sharded_folder(args.input):
        out = args.output
        if not out.endswith(".safetensors"):
            fail("Cast requires a .safetensors file path as --output "
                 "(a sharded input is merged and cast into ONE file).")
        model = discover_shards(args.input)
        shard_paths = [os.path.join(args.input, s) for s in model.shard_files]
        log(f"Merging + casting {len(shard_paths)} shard(s) to {label} ...")
        cast_shards_to_single(shard_paths, out, target, on_progress=_on_progress)
    else:
        log(f"Casting {args.input} to {label} ...")
        cast_safetensors_file(args.input, args.output, target,
                              on_progress=_on_progress)
    log(f"DONE: cast written ({label}).")


def _input_data_bytes(path: str) -> int:
    """Total tensor-DATA bytes of a ``.safetensors`` file (header excluded).

    A folder is treated as the sum of its top-level ``.safetensors`` members. Returns
    0 on any error so the caller can fall back to a conservative estimate.
    """
    try:
        if os.path.isdir(path):
            total = 0
            for nm in os.listdir(path):
                if nm.endswith(".safetensors"):
                    total += _input_data_bytes(os.path.join(path, nm))
            return total
        with open(path, "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hdr_len).decode("utf-8"))
        total = 0
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            a, b = spec["data_offsets"]
            total += (b - a)
        return total
    except Exception:  # noqa: BLE001
        try:
            return os.path.getsize(path)
        except Exception:  # noqa: BLE001
            return 0


def _expected_output_bytes(input_path: str) -> int:
    """Rough expected quantized output size for an INT8 comfy_quant model.

    Output tensor data is ~0.55x the input tensor data (int8 weight + small per-block
    scale), plus a fixed header/metadata slack. Used ONLY to scale the overall-length
    progress bar; the bar snaps to 100% on completion regardless of the estimate.
    """
    data = _input_data_bytes(input_path)
    if data <= 0:
        try:
            data = os.path.getsize(input_path)
        except Exception:  # noqa: BLE001
            data = 0
    return max(1, int(data * 0.55) + 64 * 1024)


def _run_quantize_with_progress(quantize, kwargs, output_path, input_path, poll: float = 0.4) -> None:
    """Run ``quantize(**kwargs)`` in a background thread and emit REAL overall progress.

    The live bar tracks the OUTPUT ``.safetensors`` file GROWING against the expected
    output size -- a genuine file-based ("overall length") progress signal, NOT text
    scraping. It only emits % once the output file begins to grow, so it never clobbers
    the calibration tqdm (which is tracked separately via ``parse_tqdm_progress`` on the
    TUI side, keyed into the same "quantize" slot). Falls back to a direct call on any
    error so progress instrumentation can never block the actual job.
    """
    try:
        expected = _expected_output_bytes(input_path)
        thread = threading.Thread(target=quantize, kwargs=kwargs, daemon=True)
        thread.start()
        last = -1.0
        while thread.is_alive():
            try:
                sz = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            except OSError:
                sz = 0
            if sz > 0:
                pct = max(0.0, min(100.0, 100.0 * sz / expected))
                # Throttle to ~1% steps to avoid flooding the stream with near-identical frames.
                if abs(pct - last) >= 1.0 or last < 0:
                    progress("quantize", pct=pct, label="Writing quantized model")
                    last = pct
            thread.join(timeout=poll)
        thread.join()
        progress("quantize", pct=100.0, label="Quantization done")
    except Exception:  # noqa: BLE001
        # Fallback: never let progress instrumentation block the actual job.
        quantize(**kwargs)
        progress("quantize", pct=100.0, label="Quantization done")


def _run_streaming(input_for_stream, output_path: str, config: QuantConfig) -> None:
    """Quantize via the resumable tensor-by-tensor streaming engine (INT8).

    ``input_for_stream`` is either a single ``.safetensors`` path, or a list of shard
    paths that are streamed into ONE output file (this is what replaces the temp-merge
    previously used for ``--output-mode single``). Emits a real ``cur``/``total``
    progress envelope per tensor so the TUI bar advances deterministically.
    """
    progress("prepare", cur=0, total=1, label="Initializing streaming quantizer")
    progress("prepare", cur=1, total=1, label="streaming quantizer ready")

    n_src = len(input_for_stream) if isinstance(input_for_stream, (list, tuple)) else 1
    label = f"{n_src} shard(s)" if n_src > 1 else os.path.basename(str(input_for_stream))
    log(f"Streaming quantization {label} -> {output_path} ...")
    progress("quantize", cur=0, total=1, label=f"Streaming {label}")

    def _on_progress(cur: int, total: int) -> None:
        progress("quantize", cur=cur, total=total, label=f"Streaming {label}")

    stream_quantize(input_for_stream, output_path, config, on_progress=_on_progress)
    # Snap to 100% on completion.
    progress("quantize", cur=1, total=1, label="Streaming quantization done")
    log("DONE: ComfyUI streaming quantization written.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ctq worker argument set (see plan §5.4)."""
    p = argparse.ArgumentParser(description="ComfyUI / convert_to_quant quantizer worker")
    p.add_argument("-i", "--input", required=True,
                   help="Input .safetensors file, or a folder containing exactly one.")
    p.add_argument("-o", "--output", required=True,
                   help="Output .safetensors file path (single mode) or output directory (sharded mode).")
    p.add_argument("--output-mode", choices=["single", "sharded"], default="sharded",
                   help="How to write a HuggingFace sharded input: 'single' merges the "
                        "input shards and quantizes once into one .safetensors file; "
                        "'sharded' (default) quantizes each shard into an output folder.")
    p.add_argument("--int8", action="store_true", help="Use INT8 quantization.")
    p.add_argument("--scaling_mode", choices=["row", "block", "tensor"], default=None,
                   help="INT8 scaling mode (block is ctq default).")
    p.add_argument("--convrot", action="store_true",
                   help="Apply ConvRot (row scaling only).")
    p.add_argument("--convrot_group_size", choices=["64", "256", "1024"], default=None,
                   help="ConvRot group size.")
    p.add_argument("--nvfp4", action="store_true", help="NVFP4 (Blackwell).")
    p.add_argument("--mxfp8", action="store_true", help="MXFP8 (Blackwell).")
    p.add_argument("--comfy_quant", action="store_true", default=False,
                   help="Enable comfy_quant (default on in the TUI).")
    p.add_argument("--save_quant_metadata", action="store_true", default=False,
                   help="Save quantization metadata (default on in the TUI).")
    p.add_argument("--simple", action="store_true", help="Simpler quantization path.")
    p.add_argument("--low_memory", action="store_true", help="Lower-memory path.")
    p.add_argument("--calib_samples", default=None,
                   help="Number of synthetic calibration samples (empty => ctq default).")
    p.add_argument("--flux2", action="store_true", help="FLUX.2 preset.")
    p.add_argument("--wan", action="store_true", help="WAN preset.")
    p.add_argument("--t5xxl", action="store_true", help="T5-XXL preset.")
    p.add_argument("--hunyuan", action="store_true", help="Hunyuan preset.")
    p.add_argument("--zimage", action="store_true", help="Z-Image preset.")
    p.add_argument("--exclude_layers", default=None, help="Regex of layers to exclude.")
    p.add_argument("--output_dtype", choices=["bfloat16", "float16"], default=None,
                   help="Output/compute dtype recorded for quantized layers and the "
                        "dtype unquantized 2D weights are downcast to. bfloat16 == "
                        "the upstream quantize_raon_int8_convrot.py --downcast-fp32 "
                        "behavior (also the convert_to_quant default).")
    p.add_argument("--block_size", default=None, help="Block size override.")
    p.add_argument("--heur", action="store_true",
                   help="Skip layers with poor quantization characteristics (aspect ratio, "
                        "size) -- mirrors convert_to_quant --heur. Non-divisible 2D weights "
                        "are copied unchanged instead of being quantized.")
    p.add_argument("--num_iter", "--num-iter", dest="num_iter", default=None,
                   help="Learned-rounding optimization iterations per tensor (ctq "
                        "default 4000). Only used when NOT --simple. The optimizer "
                        "loop is GPU-latency-bound (one host sync per iteration), so "
                        "it runs single-core regardless of CPU count; lowering this "
                        "(e.g. 500-1000) is the main lever for faster convrot runs.")
    p.add_argument("--manual_seed", default=None,
                   help="Fixed seed for the simulated calibration data used in bias "
                        "correction. When omitted, streaming uses a deterministic default "
                        "(reproducible across runs); convert_to_quant otherwise re-rolls a "
                        "random seed every run.")
    p.add_argument("--no-stream", action="store_true",
                   help="Disable the resumable streaming quantizer and use the legacy "
                        "whole-file path (merges shards for --output-mode single). Streaming "
                        "is the default for INT8; this is an escape hatch.")
    p.add_argument("--combine", action="store_true",
                   help="Combine shards into ONE .safetensors without quantization "
                        "(single-file input is copied unchanged).")
    p.add_argument("--cast_dtype", choices=["bfloat16", "float16"], default=None,
                   help="Cast-only output (no quantization): RTNE-convert floating "
                        "tensors to bfloat16/float16; integer/bool tensors pass "
                        "through unchanged. A sharded input is merged and cast "
                        "into ONE .safetensors.")
    p.add_argument("--verbose", action="store_true", help="Verbose progress.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Resolve + validate input (single .safetensors, a folder with one, or a
    # HuggingFace sharded folder). No arch guard for diffusion models.
    args.input = resolve_input(args.input)

    # --- Combine (plan 2026-08-26): merge shards / copy, NO quantization. -------
    if args.combine:
        log("Combine: merging shards without quantization (no .comfy_quant baked) ...")
        _run_combine(args)
        log("DONE: combine written.")
        return

    # --- bf16/fp16 cast-only (plan 2026-08-26 STEP 3.1): NO quantization. ------
    if args.cast_dtype:
        _run_cast(args)
        return

    kwargs = build_quantize_kwargs(args)

    # --- HuggingFace sharded folder branch. ------------------------------------
    if is_sharded_folder(args.input):
        model = discover_shards(args.input)
        # INT8 streaming path: no merge, no whole-file load. Stream every shard into ONE
        # output file (single mode) or one output shard per input shard. Rotation (ConvRot)
        # needs a per-layer pre-rotation pass and is out of scope for v1 streaming, so any
        # convrot format falls through to the legacy whole-file baseline below.
        config = QuantConfig.from_args(args)
        if args.int8 and not config.convrot and not args.no_stream:
            if args.output_mode == "single":
                if not args.output.endswith(".safetensors"):
                    fail("Single-file output requires a .safetensors file path (--output-mode single).")
                shard_paths = [os.path.join(args.input, s) for s in model.shard_files]
                _run_streaming(shard_paths, args.output, config)
            else:
                if args.output.endswith(".safetensors"):
                    fail("Sharded output must be a directory (omit the .safetensors filename).")
                log(f"Discovered {len(model.shard_files)} shard(s) in {args.input}")
                stream_quantize_sharded(
                    model, args.output, config,
                    on_progress=lambda c, t: progress(
                        "quantize", cur=c, total=t, label="Streaming shards"
                    ),
                )
                log("DONE: sharded ComfyUI streaming quantization written.")
            return

        # Non-INT8 (FP8 / NVFP4 / MXFP8): legacy merge / per-shard path. Streaming of
        # learned-rounding formats is out of scope for v1, so keep the temp-merge here.
        if args.output_mode == "single":
            # OPTION 1 (primary): merge the input shards into ONE file, then quantize
            # exactly once. Zero persisted converted shards; math-identical to per-shard
            # (research verdict: HF sharding splits by whole tensors; ctq scaling is
            # per-tensor). No index json / config copied (consistent with M1 single-file).
            if not args.output.endswith(".safetensors"):
                fail("Single-file output requires a .safetensors file path (--output-mode single).")
            merged = os.path.join(
                tempfile.gettempdir(),
                f"unsloth-ctq-merged-{time.strftime('%Y%m%d-%H%M%S')}.safetensors",
            )
            progress("merge", cur=0, total=1, label=f"Merging {len(model.shard_files)} shard(s)")
            log(f"Merging {len(model.shard_files)} shard(s) into one file ...")
            merge_safetensors_files(
                [os.path.join(args.input, s) for s in model.shard_files], merged
            )
            progress("merge", cur=1, total=1, label="Merged shards")
            kw = build_quantize_kwargs(args)
            kw["input"] = merged
            kw["output"] = args.output
            progress("prepare", cur=0, total=1, label="Importing convert_to_quant")
            log("Importing convert_to_quant (this can take a few seconds)...")
            try:
                from convert_to_quant import quantize
            except Exception as exc:  # noqa: BLE001
                fail(
                    f"Could not import convert_to_quant: {exc}\n"
                    "Install it into THIS interpreter's environment:\n"
                    "    pip install convert-to-quant\n"
                    "and a matching CUDA torch build. The TUI's 'Worker Python (ctq)' field "
                    "must point at that interpreter."
                )
            progress("prepare", cur=1, total=1, label="convert_to_quant ready")
            progress("quantize", cur=0, total=1, label=f"Quantizing {args.input}")
            _run_quantize_with_progress(quantize, kw, args.output, merged)
            # Best-effort cleanup of the temporary merged input. Never crash the worker
            # if the file is already gone (e.g. an external temp-file sweeper removed it).
            try:
                os.remove(merged)
            except FileNotFoundError:
                pass
            log("DONE: single-file ComfyUI quantization written.")
            return

        # EXISTING sharded-directory branch (unchanged): per-shard quantize -> folder.
        if args.output.endswith(".safetensors"):
            fail("Sharded output must be a directory (omit the .safetensors filename).")
        progress("prepare", cur=0, total=1, label="Importing convert_to_quant")
        log("Importing convert_to_quant (this can take a few seconds)...")
        try:
            from convert_to_quant import quantize
        except Exception as exc:  # noqa: BLE001
            fail(
                f"Could not import convert_to_quant: {exc}\n"
                "Install it into THIS interpreter's environment:\n"
                "    pip install convert-to-quant\n"
                "and a matching CUDA torch build. The TUI's 'Worker Python (ctq)' field "
                "must point at that interpreter."
            )
        progress("prepare", cur=1, total=1, label="convert_to_quant ready")
        log(f"Discovered {len(model.shard_files)} shard(s) in {args.input}")
        quantize_shards(model, args.output, kwargs, quantize)
        return

    # --- Single-file path. ----------------------------------------------------
    if args.int8 and not QuantConfig.from_args(args).convrot and not args.no_stream:
        # INT8 streaming: tensor-by-tensor, resumable, no whole-file load.
        config = QuantConfig.from_args(args)
        _run_streaming(args.input, args.output, config)
        return

    # Non-INT8 legacy baseline (unchanged).
    progress("prepare", cur=0, total=1, label="Importing convert_to_quant")
    log("Importing convert_to_quant (this can take a few seconds)...")
    try:
        from convert_to_quant import quantize
    except Exception as exc:  # noqa: BLE001
        fail(
            f"Could not import convert_to_quant: {exc}\n"
            "Install it into THIS interpreter's environment:\n"
            "    pip install convert-to-quant\n"
            "and a matching CUDA torch build. The TUI's 'Worker Python (ctq)' field "
            "must point at that interpreter."
        )
    progress("prepare", cur=1, total=1, label="convert_to_quant ready")

    if args.verbose:
        log(f"quantize kwargs: {kwargs}")

    progress("quantize", cur=0, total=1, label=f"Quantizing {args.input}")
    log(f"Quantizing {args.input} -> {args.output} ...")
    _run_quantize_with_progress(quantize, kwargs, args.output, args.input)
    log("DONE: ComfyUI quantization written.")


if __name__ == "__main__":
    main()
