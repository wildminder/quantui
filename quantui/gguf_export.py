"""Native GGUF export engine (S3.1/S3.2): safetensors -> GGUF, no transformers.

Streaming pipeline (plan §S3.1): audit-style input discovery (single file or
sharded HF folder) -> stdlib arch config -> deterministic tensor plan over
the name-sorted inventory -> tensors streamed shard-by-shard in plan order
into a ``gguf.GGUFWriter`` -> header/KV/tensor files written in the probe-
pinned order. Only numpy + the worker-env ``gguf`` package (imported lazily).

bf16 inputs are bit-shifted to f32 (lossless) before any quant/f16 path —
numpy has no native bf16. ``progress(tensors_done, tensors_total, name)``
is invoked after each tensor for TUI/live streaming.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from quantui.gguf_names import read_arch_config
from quantui.gguf_qkernels import (
    NATIVE_METHODS,
    TensorPlanItem,
    plan_tensor,
    quantize_q4_0,
    quantize_q8_0,
)

SHARDED_INDEX_NAME = "model.safetensors.index.json"

# safetensors dtype -> (itemsize, is_float)
_DT: dict[str, tuple[int, bool]] = {
    "F64": (8, True), "F32": (4, True), "BF16": (2, True), "F16": (2, True),
    "I64": (8, False), "I32": (4, False), "I16": (2, False), "I8": (1, False),
    "U8": (1, False), "BOOL": (1, False),
}

# method id -> general.file_type id (llama.cpp FTYPE convention)
_FILE_TYPE = {"native_q8_0": 7, "native_q4_0": 2, "native_f16": 1,
              "native_bf16": 1, "native_f32": 0}
# method id -> GGML qtype id for tensor-type reporting
_QTYPE_ID = {"native_q8_0": 8, "native_q4_0": 2, "native_f16": 1,
             "native_bf16": 30, "native_f32": 0}


class GgufExportError(ValueError):
    """Export failure with a user-facing message (path/method named)."""


@dataclass
class GgufExportReport:
    """Deterministic summary of one export run."""

    path: str
    arch: str
    method: str
    tensors_total: int
    tensors_quantized: int
    demoted: list = field(default_factory=list)   # (name, reason)
    skipped: list = field(default_factory=list)   # names
    bytes_in: int = 0
    bytes_out: int = 0
    warnings: list = field(default_factory=list)
    qtype_histogram: dict = field(default_factory=dict)


def _discover_inputs(model_path: str) -> list:
    """Return the shard/file paths to convert.

    Single file -> [file]; sharded folder -> sorted shard paths from the
    index; a folder containing a single ``model.safetensors`` (no index)
    -> that file (plain HF layout). Raises :class:`GgufExportError` for
    missing/broken inputs.
    """
    if os.path.isfile(model_path):
        if not model_path.endswith(".safetensors"):
            raise GgufExportError(f"{model_path}: not a .safetensors file")
        return [model_path]
    index_path = os.path.join(model_path, SHARDED_INDEX_NAME)
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            weight_map = index["weight_map"]
            shards = sorted({str(s) for s in weight_map.values()})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise GgufExportError(f"{index_path}: malformed index: {exc}") from exc
        out = []
        for shard in shards:
            shard_path = os.path.join(model_path, shard)
            if not os.path.isfile(shard_path):
                raise GgufExportError(f"{model_path}: index references missing shard: {shard}")
            out.append(shard_path)
        return out
    single = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    raise GgufExportError(
        f"{model_path}: not a .safetensors file or a sharded folder "
        f"({SHARDED_INDEX_NAME} missing)"
    )


def _read_inventory(paths: list) -> tuple:
    """Header-scan every shard; return (name-sorted [(name, dtype, shape,
    (path, data_start, start, end))], bytes_in)."""
    inventory: dict[str, tuple] = {}
    bytes_in = 0
    for path in paths:
        from quantui.comfy_quant_schema import read_safetensors_header

        try:
            header, data_start = read_safetensors_header(path)
        except (OSError, ValueError) as exc:
            raise GgufExportError(f"{path}: cannot read safetensors header: {exc}") from exc
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in inventory:
                continue  # dedupe (first shard wins, audit precedent)
            dtype = spec.get("dtype", "")
            shape = tuple(spec.get("shape", []))
            offsets = spec.get("data_offsets", [0, 0])
            inventory[name] = (dtype, shape, (path, data_start, offsets[0], offsets[1]))
            nbytes = _DT.get(dtype, (0, False))[0]
            n = 1
            for dim in shape:
                n *= dim
            bytes_in += nbytes * n
    ordered = [(name, *inventory[name]) for name in sorted(inventory)]
    return ordered, bytes_in


def _read_tensor_bytes(entry: tuple) -> bytes:
    path, data_start, start, end = entry
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        return fh.read(end - start)


def _to_f32(dtype: str, shape: tuple, raw: bytes) -> object:  # np.ndarray
    """Convert any float-dtype safetensors payload to an f32 ndarray."""
    import numpy as np

    n = 1
    for dim in shape:
        n *= dim
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=n).reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=n).reshape(shape).astype(np.float32)
    if dtype == "BF16":
        # lossless bit-shift: bf16 is the top 16 bits of f32
        u16 = np.frombuffer(raw, dtype="<u2", count=n).astype(np.uint32)
        f32 = (u16 << 16).view(np.float32)
        return f32.reshape(shape)
    raise GgufExportError(f"unsupported float dtype {dtype}")


def _to_bf16(dtype: str, shape: tuple, raw: bytes) -> tuple:
    """Convert a float payload to a bf16 bit-pattern uint16 array.

    BF16 source: verbatim (lossless). F32 source: RTNE round-to-nearest via
    the shared bit-math core (round-to-odd increment for ties-to-even).
    """
    import numpy as np

    from quantui.dtype_cast import f32_to_bf16

    n = 1
    for dim in shape:
        n *= dim
    if dtype == "BF16":
        return np.frombuffer(raw, dtype="<u2", count=n).reshape(shape).copy()
    f32 = _to_f32(dtype, shape, raw)
    return f32_to_bf16(f32).reshape(shape).astype(np.uint16)


def _convert_int_verbatim(dtype: str, shape: tuple, raw: bytes) -> tuple:
    """Integer payloads kept verbatim; return (gguf_type_id, ndarray).

    v1 policy: GGUF weight tensors have no integer GGML type, so integers are
    stored as F32 *values* (not bits); precision loss is possible for |v| >
    2**24 — acceptable for the positional/id tensors this path exists for.
    """
    import numpy as np

    n = 1
    for dim in shape:
        n *= dim
    np_map = {"I32": "<i4", "I16": "<i2", "I8": "<i1", "U8": "<u1", "I64": "<i8"}
    arr = np.frombuffer(raw, dtype=np_map.get(dtype, "<u1"), count=n).reshape(shape)
    f32 = arr.astype(np.float32)
    return 0, f32  # F32 qtype


def export_gguf(model_path: str, out_path: str, method: str, progress=None) -> GgufExportReport:
    """Convert an HF safetensors model (file or sharded folder) to GGUF."""
    if method not in NATIVE_METHODS:
        raise GgufExportError(
            f"unsupported native method {method!r}; v1 surface: {', '.join(NATIVE_METHODS)}"
        )
    try:
        import gguf  # noqa: F401 — lazy worker-env dependency
        from gguf import GGUFWriter
    except ImportError as exc:
        raise GgufExportError(
            "the 'gguf' package is required for the native GGUF backend "
            "(pip install gguf) — it lives in the worker env"
        ) from exc
    import numpy as np

    paths = _discover_inputs(model_path)
    ordered, bytes_in = _read_inventory(paths)
    if os.path.isdir(model_path):
        arch_dir = model_path
    else:
        arch_dir = os.path.dirname(os.path.abspath(model_path)) or "."
    arch_info = read_arch_config(arch_dir)

    # full deterministic plan
    plan: list[TensorPlanItem] = []
    entries: dict[str, tuple] = {}
    for name, dtype, shape, entry in ordered:
        plan.append(plan_tensor(name, dtype, shape, method))
        entries[name] = (dtype, shape, entry)

    report = GgufExportReport(
        path=os.path.abspath(out_path),
        arch=arch_info.arch,
        method=method,
        tensors_total=len(plan),
        tensors_quantized=0,
        bytes_in=bytes_in,
    )

    writer = GGUFWriter(out_path, arch_info.arch)
    writer.add_file_type(_FILE_TYPE[method])
    for key, value in arch_info.meta_pairs():
        if key in ("general.architecture", "general.name"):
            continue  # architecture set by the constructor; name added below once
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
    writer.add_name(arch_info.name)  # single general.name (add_name)

    total = len(plan)
    for done, item in enumerate(plan):
        dtype, shape, entry = entries[item.hf_name]
        raw = _read_tensor_bytes(entry)
        if item.action == "skip":
            report.skipped.append(item.hf_name)
        elif item.action == "verbatim" and dtype in ("I32", "I16", "I8", "U8", "BOOL"):
            qtype, arr = _convert_int_verbatim(dtype, shape, raw)
            writer.add_tensor(item.gguf_name or item.hf_name, arr)
            report.qtype_histogram["F32"] = report.qtype_histogram.get("F32", 0) + 1
        elif item.action in ("quant_q8_0", "quant_q4_0"):
            arr = _to_f32(dtype, shape, raw)
            payload = quantize_q8_0(arr) if item.action == "quant_q8_0" else quantize_q4_0(arr)
            qtype = (gguf.GGMLQuantizationType.Q8_0 if item.action == "quant_q8_0"
                     else gguf.GGMLQuantizationType.Q4_0)
            # gguf-py expects the uint8 payload shaped (…, n_bytes_per_row) so
            # quant_shape_from_byte_shape can derive the block-count shape:
            # rows = prod(shape[:-1]), bytes/row = prod-payload / rows.
            rows = 1
            for dim in shape[:-1]:
                rows *= dim
            writer.add_tensor(
                item.gguf_name or item.hf_name,
                np.frombuffer(payload, dtype=np.uint8).reshape(rows, len(payload) // rows),
                raw_dtype=qtype,
            )
            report.tensors_quantized += 1
            qt = "Q8_0" if item.action == "quant_q8_0" else "Q4_0"
            report.qtype_histogram[qt] = report.qtype_histogram.get(qt, 0) + 1
        elif item.action == "pass_bf16":
            bits = _to_bf16(dtype, shape, raw)
            writer.add_tensor(
                item.gguf_name or item.hf_name, bits,
                raw_dtype=gguf.GGMLQuantizationType.BF16,
            )
            report.qtype_histogram["BF16"] = report.qtype_histogram.get("BF16", 0) + 1
        elif item.action == "pass_f16":
            arr = _to_f32(dtype, shape, raw).astype(np.float16)
            writer.add_tensor(item.gguf_name or item.hf_name, arr)
            report.qtype_histogram["F16"] = report.qtype_histogram.get("F16", 0) + 1
            if "demoted" in item.reason:
                report.demoted.append((item.hf_name, item.reason))
                report.warnings.append(f"demoted to F16: {item.hf_name} ({item.reason})")
        else:  # pass_f32
            arr = _to_f32(dtype, shape, raw)
            writer.add_tensor(item.gguf_name or item.hf_name, arr)
            report.qtype_histogram["F32"] = report.qtype_histogram.get("F32", 0) + 1
        if progress is not None:
            progress(done + 1, total, item.hf_name)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    report.bytes_out = os.path.getsize(out_path)
    return report


# --------------------------------------------------------------------------- #
# S3.2 — CLI
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    """CLI entry: exit 0 ok, 2 export error (repo convention)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m quantui.gguf_export",
        description="Native GGUF export: HF safetensors -> GGUF (no transformers).",
    )
    parser.add_argument("-i", "--input", default=None, help="safetensors file or sharded model folder")
    parser.add_argument("-o", "--output", default=None, help="output .gguf (default <base>-<method>.gguf)")
    parser.add_argument("-m", "--method", default="native_q8_0", help="native method id")
    parser.add_argument("--list-methods", action="store_true", help="list native methods and exit")
    parser.add_argument("--json", action="store_true", help="emit the JSON report")
    args = parser.parse_args(argv)

    if args.list_methods:
        for mid in NATIVE_METHODS:
            print(mid)
        return 0
    if not args.input:
        parser.error("-i/--input is required (unless --list-methods)")

    out = args.output
    if out is None:
        base = args.input.rstrip("/\\")
        base = os.path.basename(base)
        for suffix in (".safetensors",):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
        out = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"{base}-{args.method}.gguf")

    def _progress(done, total, name):
        print(f"[gguf] {done}/{total} {name}", file=sys.stderr, flush=True)

    try:
        report = export_gguf(args.input, out, args.method, progress=_progress)
    except GgufExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({
            "path": report.path, "arch": report.arch, "method": report.method,
            "tensors_total": report.tensors_total,
            "tensors_quantized": report.tensors_quantized,
            "demoted": [list(d) for d in report.demoted],
            "skipped": report.skipped,
            "bytes_in": report.bytes_in, "bytes_out": report.bytes_out,
            "warnings": report.warnings,
            "qtype_histogram": report.qtype_histogram,
        }, indent=2, sort_keys=True))
    else:
        print(f"GGUF written: {report.path} ({report.bytes_out:,} bytes)")
        print(f"arch={report.arch} method={report.method} tensors={report.tensors_total} "
              f"quantized={report.tensors_quantized} demoted={len(report.demoted)} "
              f"skipped={len(report.skipped)}")
        for warning in report.warnings:
            print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
