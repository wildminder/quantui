"""Backend worker for ComfyUI **comfy-kitchen** native quantization (W4A4 / W4A8).

This worker runs in a **ComfyUI-python interpreter** (one that has ``comfy-kitchen``
installed), i.e. the same kind of env the TUI's "Worker Python (ctq)" points at for
the Triton/CUDA path. It is a *separate* worker from ``worker_ctq.py`` (which drives
``convert_to_quant``) and is selected automatically by ``run_config.build_ctq_cmd``
for formats whose ``ComfyFormat.backend == Backend.COMFY_KITCHEN``.

What it does:
  * loads the input ``.safetensors`` (or HuggingFace sharded folder),
  * quantizes every 2-D floating weight via the toolkit's native INT4 / W4A8 ops,
  * serializes each layer to the ComfyUI-native ``.comfy_quant`` schema (see
    ``quantui/comfy_quant_schema.py``),
  * writes a single output ``.safetensors``.

IMPORTANT: this module is **import-safe** (no torch/comfy at top level). The heavy
imports happen lazily inside ``main()`` so the unit tests can import and parse args
without a ComfyUI environment. The actual quantize path is ``@comfy``-gated and is
exercised inside a ComfyUI interpreter.

Usage:
    python -m quantui.worker_ctq_kitchen -i model.safetensors -o model-w4a4.safetensors \
        --w4a4 --format-id w4a4_convrot
"""

import argparse
import sys
import tempfile

from .comfy_quant_schema import (
    FORMAT_ASYM_W4A8_INT8,
    FORMAT_CONVROT_W4A4,
    default_quant_config,
    serialize_comfy_quant_layer,
    write_safetensors,
)

# NOTE: only stdlib + our own stdlib-only modules at import time.
from .quant_methods import (
    Backend,
    comfy_format,
)
from .worker_ctq import (
    is_sharded_folder,
    merge_safetensors_files,
    progress,
    resolve_input,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


# torch dtype -> safetensors dtype string (populated lazily; see _torch_dtype_map()).
def _torch_dtype_map():
    import torch

    m = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int8: "I8",
        torch.int32: "I32",
        torch.uint8: "U8",
        torch.float64: "F64",
    }
    f8 = getattr(torch, "float8_e4m3fn", None)
    if f8 is not None:
        m[f8] = "F8_E4M3"
    return m


def _tensor_to_spec(t) -> tuple[str, list[int], bytes]:
    """Convert a torch tensor to ``(dtype_str, shape, cpu_bytes)`` for the pure writer."""
    import torch

    t = t.detach().cpu().contiguous()
    dtype_map = _torch_dtype_map()
    st = dtype_map.get(t.dtype)
    if st is None:
        # Fall back to uint8 view for anything exotic (e.g. packed int4 stored as int8).
        st = "U8" if t.dtype == torch.uint8 else "I8"
    return (st, list(t.shape), t.numpy().tobytes())


# --------------------------------------------------------------------------- #
# ComfyUI QuantizedTensor decomposition (comfy-kitchen specific) -- @comfy only.
# --------------------------------------------------------------------------- #
def _decompose_w4a4(prefix, qtensor, format_name):
    """Decompose a ``TensorCoreConvRotW4A4Layout`` QuantizedTensor into layer specs."""
    # The QuantizedTensor stores the packed int4 weight in ``.qdata`` and the scale in
    # ``.scale`` (comfy.quant_ops). We read those defensively so a toolkit version
    # drift surfaces as an actionable error rather than a crash.
    qdata = getattr(qtensor, "qdata", None) or getattr(qtensor, "weight", None)
    scale = getattr(qtensor, "scale", None)
    if qdata is None or scale is None:
        # Last-resort: try the public decompose()/to() API some comfy builds expose.
        try:
            parts = qtensor.decompose()
            qdata = parts.get("qdata", parts.get("weight"))
            scale = parts.get("scale")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot decompose ConvRot W4A4 QuantizedTensor for {prefix!r}: "
                f"missing qdata/scale (comfy-kitchen layout drift). {exc}"
            ) from exc
    local = {
        "weight": _tensor_to_spec(qdata),
        "weight_scale": _tensor_to_spec(scale),
    }
    config = default_quant_config(format_name)
    return local, config


def _decompose_w4a8(prefix, qtensor, format_name):
    """Decompose an ``AsymW4A8Int8Layout`` QuantizedTensor into layer specs."""
    qdata = getattr(qtensor, "qdata", None) or getattr(qtensor, "weight", None)
    s_rel = getattr(qtensor, "s_rel", None)
    s_channel = getattr(qtensor, "s_channel", None)
    codebook = getattr(qtensor, "codebook", None)
    correction = getattr(qtensor, "correction", None)
    missing = [n for n, v in (
        ("qdata", qdata), ("s_rel", s_rel), ("s_channel", s_channel),
        ("codebook", codebook), ("correction", correction),
    ) if v is None]
    if missing:
        raise RuntimeError(
            f"Cannot decompose W4A8 QuantizedTensor for {prefix!r}: missing {missing} "
            f"(comfy-kitchen layout drift)."
        )
    local = {
        "weight": _tensor_to_spec(qdata),
        "weight_s_rel": _tensor_to_spec(s_rel),
        "weight_s_channel": _tensor_to_spec(s_channel),
        "weight_codebook": _tensor_to_spec(codebook),
        "weight_correction": _tensor_to_spec(correction),
    }
    config = default_quant_config(format_name)
    return local, config


def _quantize_weight(weight, format_id):
    """Quantize a single 2-D float weight via the toolkit's native op. @comfy only."""
    from comfyui_quantizationtoolkit.int8_quant import (  # type: ignore
        quantize_native_int4,
        quantize_native_w4a8,
    )

    cf = comfy_format(format_id)
    fmt = cf.quant_format
    if fmt == FORMAT_CONVROT_W4A4:
        qt = quantize_native_int4(weight)
        return _decompose_w4a4(weight, qt, fmt)
    if fmt == FORMAT_ASYM_W4A8_INT8:
        qt = quantize_native_w4a8(weight)
        return _decompose_w4a8(weight, qt, fmt)
    raise RuntimeError(f"Kitchen worker cannot produce format {fmt!r} (format-id {format_id!r}).")


def _quantize_state_dict(state_dict, format_id):
    """Quantize every 2-D float weight in ``state_dict``; copy the rest through.

    Returns a flat ``name -> (dtype, shape, bytes)`` tensor spec dict ready for
    ``write_safetensors``. Quantized weights get a trailing ``.comfy_quant`` config.
    """

    out_specs = {}
    total = len(state_dict)
    for idx, (key, tensor) in enumerate(state_dict.items(), 1):
        if (
            key.endswith(".weight")
            and tensor.dim() == 2
            and tensor.is_floating_point()
        ):
            progress("quantize", cur=idx, total=total, label=f"Quantizing {key}")
            local, config = _quantize_weight(tensor, format_id)
            out_specs.update(serialize_comfy_quant_layer(key, local, config))
        else:
            # Pass through (biases, norms, embeddings, 1-D weights, buffers, ...).
            out_specs[key] = _tensor_to_spec(tensor)
    return out_specs


def _load_state_dict(path):
    from safetensors.torch import load_file  # type: ignore

    return load_file(path, device="cpu")


def quantize_file(in_path: str, out_path: str, format_id: str) -> None:
    """Quantize one ``.safetensors`` file end-to-end. @comfy only."""
    progress("load", cur=0, total=1, label=f"Loading {in_path}")
    log(f"Loading {in_path} ...")
    state_dict = _load_state_dict(in_path)
    progress("load", cur=1, total=1, label=f"Loaded {len(state_dict)} tensors")
    progress("quantize", cur=0, total=len(state_dict), label=f"Quantizing with {format_id}")
    log(f"Quantizing {len(state_dict)} tensors with format-id {format_id} ...")
    out_specs = _quantize_state_dict(state_dict, format_id)
    progress("quantize", cur=len(state_dict), total=len(state_dict), label="Writing output")
    data = write_safetensors(out_specs)
    with open(out_path, "wb") as fh:
        fh.write(data)
    progress("write", cur=1, total=1, label=f"Wrote {out_path}")
    log(f"Wrote {out_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ComfyUI comfy-kitchen quantizer worker (W4A4 / W4A8)")
    p.add_argument("-i", "--input", required=True,
                   help="Input .safetensors file, or a folder containing exactly one.")
    p.add_argument("-o", "--output", required=True,
                   help="Output .safetensors file path.")
    p.add_argument("--output-mode", choices=["single", "sharded"], default="sharded",
                   help="Sharded input handling (single = merge then quantize once).")
    p.add_argument("--format-id", required=True,
                   help="Our ComfyFormat id (w4a4_convrot / w4a8_asym) -- selects the "
                        "on-disk .comfy_quant format to serialize.")
    p.add_argument("--w4a4", action="store_true", help="Produce ConvRot W4A4.")
    p.add_argument("--w4a8", action="store_true", help="Produce asymmetric W4A8.")
    p.add_argument("--convrot_group_size", default="256", help="ConvRot group size.")
    p.add_argument("--block_size", default=None, help="Block size (forward-compat).")
    p.add_argument("--verbose", action="store_true", help="Verbose progress.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.input = resolve_input(args.input)

    # Lazily confirm the comfy-kitchen environment before doing real work.
    try:
        import comfy.model_management  # noqa: F401  (toolkit import anchor)
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        fail(
            f"comfy-kitchen / ComfyUI is not importable in this interpreter: {exc}\n"
            "Point 'Worker Python (ctq)' at a ComfyUI interpreter with comfy-kitchen installed."
        )

    cf = comfy_format(args.format_id)
    if cf.backend != Backend.COMFY_KITCHEN:
        fail(f"format-id {args.format_id!r} is not a comfy-kitchen format "
             f"(backend={cf.backend.value}).")

    # --- Sharded input: merge into one file (single mode) or loop per shard. ----
    if is_sharded_folder(args.input):
        from .worker_ctq import discover_shards  # local import keeps top level clean

        model = discover_shards(args.input)
        if args.output_mode == "single":
            if not args.output.endswith(".safetensors"):
                fail("Single-file output requires a .safetensors file path (--output-mode single).")
            merged = tempfile.mktemp(suffix=".safetensors",
                                     prefix="quantui-ctq-kitchen-merged-")
            progress("merge", cur=0, total=1, label=f"Merging {len(model.shard_files)} shard(s)")
            log(f"Merging {len(model.shard_files)} shard(s) into one file ...")
            merge_safetensors_files(
                [f"{args.input}/{s}" for s in model.shard_files], merged
            )
            progress("merge", cur=1, total=1, label="Merged shards")
            quantize_file(merged, args.output, args.format_id)
            import os
            os.remove(merged)
            log("DONE: single-file ComfyUI (comfy-kitchen) quantization written.")
            return
        # sharded output mode -> quantize each shard into the output folder
        import os
        os.makedirs(args.output, exist_ok=True)
        total = len(model.shard_files)
        if total:
            progress("shard", cur=0, total=total, label=f"Quantizing {total} shard(s)")
        for i, shard in enumerate(model.shard_files, 1):
            in_path = f"{args.input}/{shard}"
            out_path = f"{args.output}/{shard}"
            progress("shard", cur=i, total=total, label=f"Quantizing shard {shard}")
            log(f"[{i}/{total}] Quantizing shard {shard} ...")
            quantize_file(in_path, out_path, args.format_id)
        log("DONE: sharded ComfyUI (comfy-kitchen) quantization written.")
        return

    # --- Single-file path. ----------------------------------------------------
    quantize_file(args.input, args.output, args.format_id)
    log("DONE: ComfyUI (comfy-kitchen) quantization written.")


if __name__ == "__main__":
    main()
