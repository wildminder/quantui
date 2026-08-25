"""Pure (stdlib-only) schema + validator for ComfyUI-native ``.comfy_quant`` checkpoints.

This module intentionally imports **no** ``torch`` / ``safetensors`` / ``comfy`` so it
runs in any interpreter -- including the headless test sandbox and the
``convert_to_quant`` worker env. It parses the safetensors header with ``struct`` +
``json`` and reads each ``.comfy_quant`` config blob as raw bytes.

On-disk contract (sourced from ``comfyui-quantizationtoolkit``:
``int8_quant.py`` / ``int8_model_save.py`` / ``quantization_policy.py``):

* A quantized layer stores its tensors with suffixes ``<layer>.weight``,
  ``<layer>.weight_scale`` (W8A8 / ConvRot W4A4) or ``<layer>.weight_s_rel`` +
  ``<layer>.weight_s_channel`` + ``<layer>.weight_codebook`` +
  ``<layer>.weight_correction`` (W4A8), plus a ``<layer>.comfy_quant`` metadata
  tensor.
* The ``.comfy_quant`` tensor is **U8**; its bytes are a UTF-8 JSON object whose
  ``"format"`` field is one of ``KNOWN_FORMATS``. (The toolkit encodes it via
  ``torch.tensor(list(json_bytes), dtype=torch.uint8)`` -- i.e. one byte per char.)
* Group sizes (toolkit constants): ``convrot_groupsize = 256``,
  W4A4 ``quant_group_size = 64``, W4A8 ``group_size = 16``.
"""

import json
import struct

# --------------------------------------------------------------------------- #
# Toolkit format strings (from int8_quant.py: W4A8_FORMAT / _SUPPORTED_...).
# --------------------------------------------------------------------------- #
FORMAT_INT8_TENSORWISE = "int8_tensorwise"
FORMAT_CONVROT_W4A4 = "convrot_w4a4"
FORMAT_ASYM_W4A8_INT8 = "asym_w4a8_int8"
KNOWN_FORMATS = frozenset(
    {FORMAT_INT8_TENSORWISE, FORMAT_CONVROT_W4A4, FORMAT_ASYM_W4A8_INT8}
)

# --------------------------------------------------------------------------- #
# Group-size constants copied verbatim from the toolkit.
# --------------------------------------------------------------------------- #
CONVROT_GROUP_SIZE = 256
W4A4_QUANT_GROUP_SIZE = 64
W4A8_QUANT_GROUP_SIZE = 16
QUAROT_GROUP_SIZE = 128

# Map each on-disk format -> the tensor *suffixes* that must be present somewhere
# in the file (file-level presence check, robust to the toolkit's per-layer naming
# quirk where the weight tensor is ``<layer>.weight`` and its scale is the sibling
# ``<layer>.weight_scale`` rather than a child).
REQUIRED_SUFFIXES: dict[str, tuple[str, ...]] = {
    FORMAT_INT8_TENSORWISE: (".weight", ".weight_scale"),
    FORMAT_CONVROT_W4A4: (".weight", ".weight_scale"),
    FORMAT_ASYM_W4A8_INT8: (".weight", ".weight_s_rel"),
}

# Build-time defaults published for the kitchen worker serializer.
DEFAULT_QUANT_CONFIG: dict[str, dict] = {
    FORMAT_INT8_TENSORWISE: {"format": FORMAT_INT8_TENSORWISE},
    FORMAT_CONVROT_W4A4: {
        "format": FORMAT_CONVROT_W4A4,
        "convrot_groupsize": CONVROT_GROUP_SIZE,
        "quant_group_size": W4A4_QUANT_GROUP_SIZE,
    },
    FORMAT_ASYM_W4A8_INT8: {
        "format": FORMAT_ASYM_W4A8_INT8,
        "group_size": W4A8_QUANT_GROUP_SIZE,
        "convrot_groupsize": CONVROT_GROUP_SIZE,
    },
}


# --------------------------------------------------------------------------- #
# Low-level safetensors (header) IO -- pure stdlib.
# --------------------------------------------------------------------------- #
def read_safetensors_header(path: str) -> tuple[dict, int]:
    """Return ``(header_dict, data_start_offset)`` for a safetensors file.

    ``data_start_offset`` is the byte position where tensor buffers begin (i.e.
    ``8 + header_length``). Raises on malformed input.
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)
        if len(magic) < 8:
            raise ValueError("file too small to be a safetensors file")
        hdr_len = struct.unpack("<Q", magic)[0]
        raw = fh.read(hdr_len)
        if len(raw) < hdr_len:
            raise ValueError("truncated safetensors header")
        header = json.loads(raw.decode("utf-8"))
    return header, 8 + hdr_len


def _read_tensor_bytes(path: str, data_start: int, offsets) -> bytes:
    start, end = offsets
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        return fh.read(end - start)


# --------------------------------------------------------------------------- #
# Validator
# --------------------------------------------------------------------------- #
def validate_comfy_quant_file(path: str) -> dict:
    """Validate a ``.safetensors`` checkpoint against the ``.comfy_quant`` schema.

    Returns ``{"ok": bool, "errors": list[str], "formats_found": list[str]}``.
    A file with *no* ``.comfy_quant`` tensors is considered ok (e.g. plain FP8 /
    FP16 checkpoints do not carry native ``.comfy_quant`` metadata).
    """
    result: dict = {"ok": True, "errors": [], "formats_found": []}
    try:
        header, data_start = read_safetensors_header(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "errors": [f"invalid safetensors header: {exc}"], "formats_found": []}

    comfy_quant_keys = [k for k in header if k.endswith(".comfy_quant")]
    if not comfy_quant_keys:
        return result

    seen_formats: set[str] = set()
    for key in comfy_quant_keys:
        spec = header[key]
        blob = _read_tensor_bytes(path, data_start, spec["data_offsets"])
        try:
            cfg = json.loads(blob)
        except Exception as exc:  # noqa: BLE001
            result["ok"] = False
            result["errors"].append(f"{key}: .comfy_quant is not valid JSON: {exc}")
            continue
        if not isinstance(cfg, dict):
            result["ok"] = False
            result["errors"].append(f"{key}: .comfy_quant payload is not a JSON object")
            continue
        fmt = cfg.get("format")
        if fmt not in KNOWN_FORMATS:
            result["ok"] = False
            result["errors"].append(f"{key}: unknown format {fmt!r}")
            continue
        seen_formats.add(fmt)

        # File-level presence check of the required tensor suffixes.
        for suffix in REQUIRED_SUFFIXES[fmt]:
            if not any(k.endswith(suffix) for k in header):
                result["ok"] = False
                result["errors"].append(
                    f"{key}: format {fmt!r} requires at least one {suffix!r} tensor"
                )

    result["formats_found"] = sorted(seen_formats)
    return result


def read_comfy_quant_configs(path: str) -> list[tuple[str, dict]]:
    """Return ``[(layer_prefix, config_dict), ...]`` for every ``.comfy_quant`` tensor.

    ``layer_prefix`` is the tensor name with the trailing ``.comfy_quant`` removed.
    Raises if the file is not a valid safetensors file.
    """
    header, data_start = read_safetensors_header(path)
    out: list[tuple[str, dict]] = []
    for key in header:
        if not key.endswith(".comfy_quant"):
            continue
        blob = _read_tensor_bytes(path, data_start, header[key]["data_offsets"])
        cfg = json.loads(blob)
        prefix = key[: -len(".comfy_quant")]
        out.append((prefix, cfg))
    return out


# --------------------------------------------------------------------------- #
# Serializer (pure stdlib) -- used by the kitchen worker and fully testable.
# --------------------------------------------------------------------------- #
def encode_comfy_quant_config(config: dict) -> bytes:
    """Encode a ``.comfy_quant`` config dict to the U8-byte JSON blob.

    Mirrors the toolkit's ``_encode_comfy_quant_config`` (compact JSON, UTF-8).
    """
    return json.dumps(config, separators=(",", ":")).encode("utf-8")


def serialize_comfy_quant_layer(
    prefix: str, local_tensors: dict[str, tuple[str, list[int], bytes]], config: dict
) -> dict[str, tuple[str, list[int], bytes]]:
    """Append a quantized layer's tensors + its ``.comfy_quant`` config blob.

    ``local_tensors`` maps a local name (``"weight"``, ``"weight_scale"``, ...) to
    ``(dtype:str, shape:list[int], data:bytes)``. Returns the full flattened tensor
    spec dict (keys become ``"<prefix>.<name>"`` plus ``"<prefix>.comfy_quant"``).
    """
    out: dict[str, tuple[str, list[int], bytes]] = {}
    for name, spec in local_tensors.items():
        out[f"{prefix}.{name}"] = spec
    out[f"{prefix}.comfy_quant"] = (
        "U8",
        [len(encode_comfy_quant_config(config))],
        encode_comfy_quant_config(config),
    )
    return out


def _align_header_to_8(header_bytes: bytes) -> bytes:
    """Pad a safetensors JSON header with spaces so the following data section begins
    at an 8-byte-aligned offset (required by the safetensors spec and enforced by the
    reference ``safetensors`` reader).
    """
    pad = (8 - (len(header_bytes) % 8)) % 8
    return header_bytes + b" " * pad


def write_safetensors(
    tensor_specs: dict[str, tuple[str, list[int], bytes]], metadata: dict | None = None
) -> bytes:
    """Serialize ``tensor_specs`` (name -> (dtype, shape, data)) to safetensors bytes.

    Tensors are packed **contiguously** (no gaps) as required by the safetensors
    spec/reader (its verifier rejects any gap between adjacent tensor offsets).
    ``metadata`` (if given) is written as ``__metadata__``. The header is padded to
    an 8-byte boundary so the data section itself starts aligned, which keeps the
    file spec-compliant and readable by the reference ``safetensors`` library / by
    ComfyUI's ``safe_open``.
    """
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    data = bytearray()
    for name, (dtype, shape, blob) in tensor_specs.items():
        start = len(data)
        data += blob
        end = start + len(blob)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, end],
        }
    header_bytes = json.dumps(header).encode("utf-8")
    header_bytes = _align_header_to_8(header_bytes)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data)


def default_quant_config(format_name: str, **overrides) -> dict:
    """Return a copy of the build-time default config for ``format_name`` with overrides."""
    base = dict(DEFAULT_QUANT_CONFIG[format_name])
    base.update(overrides)
    return base
