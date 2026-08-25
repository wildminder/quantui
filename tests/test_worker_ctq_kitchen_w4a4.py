"""P3.2/P4.2: kitchen worker serialization produces the exact ComfyUI ``.comfy_quant`` schema.

These run headless (no torch/comfy) because they exercise the *pure* serialization seam
(``serialize_comfy_quant_layer`` + ``write_safetensors`` + the stdlib validator), which
is identical to what the ``@comfy`` quantize path writes at runtime.
"""

import struct

from quantui import worker_ctq_kitchen as w
from quantui.comfy_quant_schema import (
    FORMAT_ASYM_W4A8_INT8,
    FORMAT_CONVROT_W4A4,
    read_comfy_quant_configs,
    validate_comfy_quant_file,
)


def _f32(n):
    import struct as _s
    return b"".join(_s.pack("<f", 0.0) for _ in range(n))


def test_kitchen_serializes_w4a4_to_schema(tmp_path):
    local = {
        "weight": ("I8", [4], b"\x01" * 4),     # packed int4
        "weight_scale": ("F32", [1], _f32(1)),
    }
    specs = w.serialize_comfy_quant_layer("model.l0.weight", local, w.default_quant_config(FORMAT_CONVROT_W4A4))
    out = tmp_path / "m.safetensors"
    out.write_bytes(w.write_safetensors(specs))

    res = validate_comfy_quant_file(str(out))
    assert res["ok"], res["errors"]
    assert res["formats_found"] == [FORMAT_CONVROT_W4A4]
    cfg = read_comfy_quant_configs(str(out))[0][1]
    assert cfg["convrot_groupsize"] == 256
    assert cfg["quant_group_size"] == 64


def test_kitchen_serializes_w4a8_to_schema(tmp_path):
    local = {
        "weight": ("I8", [4], b"\x02" * 4),
        "weight_s_rel": ("F32", [1], _f32(1)),
        "weight_s_channel": ("F32", [1], _f32(1)),
        "weight_codebook": ("F32", [4], _f32(4)),
        "weight_correction": ("F32", [1], _f32(1)),
    }
    specs = w.serialize_comfy_quant_layer("model.l0.weight", local, w.default_quant_config(FORMAT_ASYM_W4A8_INT8))
    out = tmp_path / "m.safetensors"
    out.write_bytes(w.write_safetensors(specs))

    res = validate_comfy_quant_file(str(out))
    assert res["ok"], res["errors"]
    assert res["formats_found"] == [FORMAT_ASYM_W4A8_INT8]
    cfg = read_comfy_quant_configs(str(out))[0][1]
    assert cfg["group_size"] == 16
    assert cfg["convrot_groupsize"] == 256


def test_kitchen_output_is_valid_safetensors(tmp_path):
    local = {"weight": ("I8", [4], b"\x03" * 4), "weight_scale": ("F32", [1], _f32(1))}
    specs = w.serialize_comfy_quant_layer("l0.weight", local, w.default_quant_config(FORMAT_CONVROT_W4A4))
    out = tmp_path / "m.safetensors"
    out.write_bytes(w.write_safetensors(specs))
    with open(out, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        header = __import__("json").loads(fh.read(hdr_len).decode("utf-8"))
    # 8-byte aligned data region; header parses; comfy_quant U8 tensor present.
    assert "l0.weight.comfy_quant" in header
    assert header["l0.weight.comfy_quant"]["dtype"] == "U8"
