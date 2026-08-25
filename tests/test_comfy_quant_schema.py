"""P0.2: pure-stdlib validator + serializer for ``.comfy_quant`` checkpoints.

Runs with NO torch / safetensors available (headless). Builds synthetic safetensors
files using the module's own pure writer and asserts the validator accepts/rejects
correctly.
"""

import json
import struct

from quantui.comfy_quant_schema import (
    FORMAT_ASYM_W4A8_INT8,
    FORMAT_CONVROT_W4A4,
    FORMAT_INT8_TENSORWISE,
    KNOWN_FORMATS,
    default_quant_config,
    encode_comfy_quant_config,
    read_comfy_quant_configs,
    serialize_comfy_quant_layer,
    validate_comfy_quant_file,
    write_safetensors,
)


def _write_file(path, specs, metadata=None):
    data = write_safetensors(specs, metadata=metadata)
    with open(path, "wb") as fh:
        fh.write(data)


def _f32_bytes(n):
    import struct as _s
    return b"".join(_s.pack("<f", 0.0) for _ in range(n))


def test_known_formats_three():
    assert KNOWN_FORMATS == {
        FORMAT_INT8_TENSORWISE,
        FORMAT_CONVROT_W4A4,
        FORMAT_ASYM_W4A8_INT8,
    }


def test_valid_int8_tensorwise(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {
            "weight": ("U8", [4], b"\x01" * 4),
            "weight_scale": ("F32", [1], _f32_bytes(1)),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)

    res = validate_comfy_quant_file(str(out))
    assert res["ok"], res["errors"]
    assert res["formats_found"] == [FORMAT_INT8_TENSORWISE]
    configs = read_comfy_quant_configs(str(out))
    assert configs == [("model.l0.weight", {"format": FORMAT_INT8_TENSORWISE})]


def test_valid_convrot_w4a4(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {
            "weight": ("U8", [4], b"\x02" * 4),
            "weight_scale": ("F32", [1], _f32_bytes(1)),
        },
        default_quant_config(FORMAT_CONVROT_W4A4),
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)
    res = validate_comfy_quant_file(str(out))
    assert res["ok"], res["errors"]
    assert res["formats_found"] == [FORMAT_CONVROT_W4A4]
    cfg = read_comfy_quant_configs(str(out))[0][1]
    assert cfg["convrot_groupsize"] == 256
    assert cfg["quant_group_size"] == 64


def test_valid_asym_w4a8_int8(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {
            "weight": ("U8", [4], b"\x03" * 4),
            "weight_s_rel": ("F32", [1], _f32_bytes(1)),
            "weight_s_channel": ("F32", [1], _f32_bytes(1)),
            "weight_codebook": ("F32", [4], _f32_bytes(4)),
            "weight_correction": ("F32", [1], _f32_bytes(1)),
        },
        default_quant_config(FORMAT_ASYM_W4A8_INT8),
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)
    res = validate_comfy_quant_file(str(out))
    assert res["ok"], res["errors"]
    assert res["formats_found"] == [FORMAT_ASYM_W4A8_INT8]
    cfg = read_comfy_quant_configs(str(out))[0][1]
    assert cfg["group_size"] == 16
    assert cfg["convrot_groupsize"] == 256


def test_invalid_missing_weight_scale(tmp_path):
    # int8_tensorwise requires a .weight_scale suffix somewhere in the file.
    specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {"weight": ("U8", [4], b"\x01" * 4)},
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)
    res = validate_comfy_quant_file(str(out))
    assert not res["ok"]
    assert any("weight_scale" in e for e in res["errors"])


def test_invalid_unknown_format(tmp_path):
    bad = {"format": "totally_made_up"}
    specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {
            "weight": ("U8", [4], b"\x01" * 4),
            "weight_scale": ("F32", [1], _f32_bytes(1)),
        },
        bad,
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)
    res = validate_comfy_quant_file(str(out))
    assert not res["ok"]
    assert any("unknown format" in e for e in res["errors"])


def test_no_comfy_quant_is_ok(tmp_path):
    # Plain FP8/FP16 checkpoints carry no .comfy_quant -> still valid.
    specs = {"model.weight": ("F16", [2], b"\x00\x00" * 2)}
    out = tmp_path / "m.safetensors"
    _write_file(out, specs)
    res = validate_comfy_quant_file(str(out))
    assert res["ok"]
    assert res["formats_found"] == []


def test_encode_decode_roundtrip():
    cfg = default_quant_config(FORMAT_CONVROT_W4A4)
    blob = encode_comfy_quant_config(cfg)
    import json
    assert json.loads(blob) == cfg
    assert isinstance(blob, bytes)


def test_metadata_preserved_in_serialize(tmp_path):
    specs = serialize_comfy_quant_layer(
        "l0.weight",
        {"weight": ("U8", [4], b"\x01" * 4), "weight_scale": ("F32", [1], _f32_bytes(1))},
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    out = tmp_path / "m.safetensors"
    _write_file(out, specs, metadata={"prompt": "x"})
    # Header should carry __metadata__.
    with open(out, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
    assert header.get("__metadata__", {}).get("prompt") == "x"
