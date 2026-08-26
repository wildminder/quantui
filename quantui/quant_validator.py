"""Deep structural validator for ComfyUI-native quantized ``.safetensors`` files.

Why this exists
---------------
The sibling ``tools/validate_raon_int8_convrot.py`` in ComfyUI-Raon-OpenTTS checks a
*model-specific* INT8 ConvRot checkpoint. This module provides a **model-agnostic**
equivalent that validates *any* checkpoint our TUI can emit via ``convert_to_quant``:

  * ``int8_convrot``                -> ``int8_tensorwise`` (+ ``convrot``)
  * ``int8_tensor``                 -> ``int8_tensorwise`` (no convrot)
  * ``int8_block``                  -> ``int8_blockwise`` (+ ``input_scale``)

It reuses the pure-stdlib safetensors header parser from
``comfy_quant_schema`` (no ``torch`` / ``safetensors`` import required for the
**structural** pass), so it runs in the headless test sandbox and inside the TUI
worker thread alike. An *optional* numeric pass (needs ``safetensors`` + ``numpy``)
reads the int8 weights + scales and checks for quantization overflow / non-finite
scales.

On-disk contract (verified against ``convert_to_quant`` 1.3.3
``create_comfy_quant_tensor``):
  * ``<layer>.weight``      : int8  (header dtype ``"I8"``), 2-D
  * ``<layer>.weight_scale``: float32 (header dtype ``"F32"``)
        - tensorwise / row / convrot : shape ``[out_features, 1]``
        - blockwise                  : shape ``[ceil(out/g), ceil(in/g)]``
  * ``<layer>.input_scale`` : float32 scalar (blockwise only)
  * ``<layer>.comfy_quant`` : uint8 JSON marker
        - ``{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":G,
           "per_row":true,"orig_dtype":"torch.bfloat16"}``  (ConvRot)
        - ``{"format":"int8_tensorwise",...}``               (tensorwise/row)
        - ``{"format":"int8_blockwise","group_size":G,...}``  (blockwise)
"""

from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .comfy_quant_schema import (
    read_safetensors_header,
)

# Header dtype strings (safetensors / comfy_quant_schema conventions).
I8 = "I8"
F32 = "F32"
U8 = "U8"

# Formats this validator understands (superset of the schema's KNOWN_FORMATS for
# the INT8 family we emit).
SUPPORTED_FORMATS = frozenset({"int8_tensorwise", "int8_blockwise"})

# ConvRot group size must be a power of four (4/16/64/256/1024/...); this mirrors
# comfy-kitchen's ``int8.py:validate_group_size``.
_MIN_GROUP_SIZE = 4


def _is_power_of_four(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0 and ((n.bit_length() - 1) % 2 == 0)


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #
@dataclass
class ValidationReport:
    """Result of :func:`validate_comfy_quant`."""

    path: str = ""
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    formats: set[str] = field(default_factory=set)
    layers: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def _scale_is_rowwise(fmt: str, cfg: dict) -> bool:
    """A row/tensor/convrot layer stores one scale per output row -> shape [out, 1]."""
    if fmt == "int8_blockwise":
        return False
    # tensorwise or convrot: always per-row scale.
    return True


# --------------------------------------------------------------------------- #
# Core validation
# --------------------------------------------------------------------------- #
def validate_comfy_quant(path: str, *, numeric: bool = False) -> ValidationReport:
    """Validate a ComfyUI-native quantized ``.safetensors`` file.

    Structural pass (always): header integrity, ``.comfy_quant`` JSON, format
    support, weight/scale presence + dtype + shape, ConvRot group-size legality
    (power of four + divides ``in_features``), blockwise ``input_scale`` presence,
    bias-not-quantized, and orphan marker/scale detection. Also accumulates summary
    statistics (matrix counts, group-size histogram, quantized vs full-precision
    param share).

    Numeric pass (only when ``numeric=True`` and ``safetensors``+``numpy`` are
    importable): loads int8 weights + scales, checks for int8 overflow and
    non-finite / non-positive scales.

    A file with *no* ``.comfy_quant`` tensors is reported ``ok=True`` with a
    warning (plain FP8/FP16 or combine/merge checkpoints are valid too).
    """
    report = ValidationReport(path=str(path))

    p = Path(path)
    if not p.is_file():
        report.add_error(f"file not found: {path}")
        return report

    try:
        header, data_start = read_safetensors_header(str(p))
    except Exception as exc:  # noqa: BLE001
        report.add_error(f"invalid safetensors header: {exc}")
        return report

    keys = list(header.keys())
    key_set = set(keys)
    shapes = {k: tuple(header[k]["shape"]) for k in keys}
    dtypes = {k: header[k]["dtype"] for k in keys}

    markers = [k for k in keys if k.endswith(".comfy_quant")]
    scales = [k for k in keys if k.endswith(".weight_scale")]
    input_scales = [k for k in keys if k.endswith(".input_scale")]

    if not markers:
        report.add_warning(
            "no .comfy_quant markers found -- this is a plain (FP8/FP16) or "
            "combine/merge checkpoint, not a natively quantized one."
        )
        _summarize_no_markers(report, keys, shapes, p)
        return report

    # Parse every marker once.
    parsed: list[tuple[str, dict]] = []
    for marker in markers:
        base = marker[: -len(".comfy_quant")]
        try:
            cfg = json.loads(_comfy_quant_blob(header, marker, str(p), data_start))
        except Exception as exc:  # noqa: BLE001
            report.add_error(f"{base}: .comfy_quant is not valid JSON: {exc}")
            continue
        if not isinstance(cfg, dict):
            report.add_error(f"{base}: .comfy_quant payload is not a JSON object")
            continue
        parsed.append((base, cfg))

    gs_hist = collections.Counter()
    q_params = 0
    q_layers: list[str] = []
    for base, cfg in parsed:
        fmt = cfg.get("format")
        if fmt not in SUPPORTED_FORMATS:
            report.add_error(f"{base}: unsupported format {fmt!r} (known: {sorted(SUPPORTED_FORMATS)})")
            continue
        report.formats.add(fmt)

        convrot = bool(cfg.get("convrot"))
        gs = int(cfg.get("convrot_groupsize", 0)) if convrot else 0
        block_size = int(cfg.get("group_size", 0)) if fmt == "int8_blockwise" else 0

        w_key = f"{base}.weight"
        s_key = f"{base}.weight_scale"
        if w_key not in key_set:
            report.add_error(f"{base}: missing quantized weight tensor {w_key!r}")
            continue

        # Weight is present, so its dtype can be checked even if the scale is missing.
        if dtypes[w_key] != I8:
            report.add_error(f"{base}: weight dtype {dtypes[w_key]!r} != {I8!r} (expected int8)")
        if s_key not in key_set:
            report.add_error(f"{base}: missing weight_scale tensor {s_key!r}")
            continue
        if dtypes[s_key] != F32:
            report.add_error(f"{base}: weight_scale dtype {dtypes[s_key]!r} != {F32!r}")

        w_shape = shapes[w_key]
        if len(w_shape) != 2:
            report.add_error(f"{base}: weight is not 2-D: {w_shape}")
            continue
        out_f, in_f = int(w_shape[0]), int(w_shape[1])

        # --- scale shape check ------------------------------------------------
        s_shape = tuple(shapes[s_key])
        if _scale_is_rowwise(fmt, cfg):
            if s_shape != (out_f, 1):
                report.add_error(
                    f"{base}: {fmt} weight_scale shape {s_shape} != ({out_f}, 1)"
                )
        else:  # blockwise
            exp = (math.ceil(out_f / block_size) if block_size else None,
                   math.ceil(in_f / block_size) if block_size else None)
            # be lenient: allow the toolkit's exact [bm, bn] layout
            if block_size and s_shape != exp:
                report.add_error(
                    f"{base}: int8_blockwise weight_scale shape {s_shape} != expected {exp} "
                    f"(block_size={block_size})"
                )
            # blockwise MUST carry an input_scale scalar
            is_key = f"{base}.input_scale"
            if is_key not in key_set:
                report.add_error(f"{base}: int8_blockwise requires {is_key!r} (missing)")
            elif dtypes.get(is_key) != F32:
                report.add_error(f"{base}: {is_key} dtype {dtypes.get(is_key)!r} != {F32!r}")

        # --- convrot group-size legality -------------------------------------
        if convrot:
            if not gs:
                report.add_error(f"{base}: convrot=true but convrot_groupsize missing")
            elif not _is_power_of_four(gs) or gs < _MIN_GROUP_SIZE:
                report.add_error(
                    f"{base}: convrot_groupsize {gs} must be a power of four >= {_MIN_GROUP_SIZE}"
                )
            elif in_f % gs != 0:
                report.add_error(
                    f"{base}: in_features {in_f} not divisible by convrot_groupsize {gs}"
                )
            gs_hist[gs] += 1

        # --- bias must not be quantized --------------------------------------
        bias_key = f"{base}.bias"
        if bias_key in key_set and dtypes.get(bias_key) == I8:
            report.add_error(f"{base}: bias was quantized (bias must stay full precision)")

        q_params += out_f * in_f
        q_layers.append(base)

    # --- orphan detection ----------------------------------------------------
    orphan_scales = [s for s in scales if f"{s[:-len('.weight_scale')]}.comfy_quant" not in key_set]
    orphan_markers = [m for m in markers if f"{m[:-len('.comfy_quant')]}.weight_scale" not in key_set]
    if orphan_scales:
        report.add_error(f"orphan weight_scale entries (no matching .comfy_quant): {orphan_scales[:5]}")
    if orphan_markers:
        report.add_error(f"orphan .comfy_quant entries (no matching .weight_scale): {orphan_markers[:5]}")

    _summarize(report, keys, shapes, dtypes, q_params, q_layers, gs_hist, p)

    if numeric:
        _numeric_pass(report, str(p), [b for b, _ in parsed])

    return report


def _comfy_quant_blob(header: dict, marker: str, path: str, data_start: int) -> bytes:
    """Read the raw bytes of a ``.comfy_quant`` tensor (pure, no safetensors lib)."""
    start, end = header[marker]["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        return fh.read(end - start)


def _summarize_no_markers(report, keys, shapes, p) -> None:
    fp_weights = [k for k in keys if k.endswith(".weight")]
    fp_params = sum(int(math.prod(shapes[k])) for k in fp_weights)
    report.summary = {
        "file": p.name,
        "size_gb": round(p.stat().st_size / 1e9, 3),
        "quantized_matrices": 0,
        "full_precision_weights": len(fp_weights),
        "quantized_params": 0,
        "full_precision_params": fp_params,
        "quantized_share_pct": 0.0,
        "group_size_histogram": {},
        "formats_found": [],
    }


def _summarize(report, keys, shapes, dtypes, q_params, q_layers, gs_hist, p) -> None:
    fp_weights = [
        k for k in keys
        if k.endswith(".weight")
        and f"{k[:-len('.weight')]}.comfy_quant" not in set(keys)
    ]
    fp_params = sum(int(math.prod(shapes[k])) for k in fp_weights)
    total = q_params + fp_params
    report.summary = {
        "file": p.name,
        "size_gb": round(p.stat().st_size / 1e9, 3),
        "quantized_matrices": len(q_layers),
        "full_precision_weights": len(fp_weights),
        "quantized_params": q_params,
        "full_precision_params": fp_params,
        "quantized_share_pct": round(100.0 * q_params / max(total, 1), 3),
        "group_size_histogram": dict(gs_hist),
        "formats_found": sorted(report.formats),
    }
    for base in q_layers:
        w = shapes.get(f"{base}.weight")
        report.layers.append({
            "prefix": base,
            "weight_shape": list(w) if w else None,
        })


def _numeric_pass(report: ValidationReport, path: str, bases: list[str]) -> None:
    """Optional: load int8 + scales and check overflow / finiteness."""
    try:
        import numpy as np  # type: ignore
        from safetensors import numpy as stnp  # type: ignore
    except Exception:  # noqa: BLE001
        report.add_warning("numeric pass skipped: safetensors/numpy not available")
        return

    try:
        arrays = stnp.load(path)
    except Exception as exc:  # noqa: BLE001
        report.add_warning(f"numeric pass skipped: cannot load tensors: {exc}")
        return

    overflow = 0
    bad_scale = 0
    for base in bases:
        w_key = f"{base}.weight"
        s_key = f"{base}.weight_scale"
        if w_key not in arrays or s_key not in arrays:
            continue
        q = arrays[w_key]
        scale = arrays[s_key]
        # int8 range / overflow
        if q.dtype.kind in ("i", "u"):
            mx = int(np.abs(q.astype(np.int64)).max()) if q.size else 0
            if mx > 127:
                overflow += 1
        # scale finiteness / positivity
        s = scale.astype(np.float64).ravel()
        if not np.all(np.isfinite(s)):
            bad_scale += 1
        else:
            nonzero_q = (q.astype(np.float64).ravel() != 0).any()
            if nonzero_q and (s <= 0).any():
                bad_scale += 1

    if overflow:
        report.add_error(f"numeric: {overflow} layer(s) have int8 values outside [-127, 127] (overflow)")
    if bad_scale:
        report.add_error(f"numeric: {bad_scale} layer(s) have non-finite / non-positive scales")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(report: ValidationReport) -> str:
    """Render a human-readable multi-line report (mimics the Raon validate tool)."""
    s = report.summary
    lines = []
    name = report.path if not s else s.get("file", report.path)
    if s:
        lines.append(f"file: {name} ({s.get('size_gb', 0)} GB)")
        lines.append(f"quantized matrices      : {s.get('quantized_matrices', 0)}")
        gs = s.get("group_size_histogram", {})
        for g in sorted(gs):
            lines.append(f"  GS{g:<5}              : {gs[g]}")
        lines.append(f"full-precision weights  : {s.get('full_precision_weights', 0)}")
        lines.append(f"quantized params        : {s.get('quantized_params', 0) / 1e6:.1f}M")
        lines.append(f"full-precision params   : {s.get('full_precision_params', 0) / 1e6:.1f}M")
        lines.append(f"quantized share         : {s.get('quantized_share_pct', 0):.2f}%")
        if s.get("formats_found"):
            lines.append(f"formats found           : {', '.join(s['formats_found'])}")
    if report.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in report.warnings:
            lines.append(f"  - {w}")
    if report.errors:
        lines.append("")
        lines.append(f"FAIL ({len(report.errors)} problem(s)):")
        for e in report.errors[:40]:
            lines.append(f"  - {e}")
    elif report.ok:
        lines.append("")
        lines.append("PASS")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate a ComfyUI INT8 .safetensors checkpoint.")
    ap.add_argument("path", help="path to the .safetensors file")
    ap.add_argument("--numeric", action="store_true",
                    help="also read int8 weights/scales and check overflow/finiteness")
    args = ap.parse_args(argv)

    report = validate_comfy_quant(args.path, numeric=args.numeric)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
