"""P6.2: merge hardening -- __metadata__ preservation, duplicate rejection,
and `.comfy_quant` U8 tensor preservation (lossless for quantized shards).
"""

import json
import struct

import pytest

from quantui.comfy_quant_schema import read_safetensors_header, write_safetensors
from quantui.worker_ctq import merge_safetensors_files


def _build(path, specs, metadata=None):
    path.write_bytes(write_safetensors(specs, metadata=metadata))


def _f32(vals):
    return b"".join(struct.pack("<f", v) for v in vals)


def test_merge_preserves_metadata(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build(s1, {"a": ("F32", [1], _f32([1.0]))}, metadata={"prompt": "hello"})
    _build(s2, {"b": ("F32", [1], _f32([2.0]))})
    out = tmp_path / "merged.safetensors"
    merge_safetensors_files([str(s1), str(s2)], str(out))

    header, _ = read_safetensors_header(str(out))
    assert header.get("__metadata__", {}).get("prompt") == "hello"


def test_merge_rejects_duplicate(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build(s1, {"a": ("F32", [1], _f32([1.0]))})
    _build(s2, {"a": ("F32", [1], _f32([2.0]))})  # duplicate name
    out = tmp_path / "merged.safetensors"
    with pytest.raises(ValueError):
        merge_safetensors_files([str(s1), str(s2)], str(out))


def test_merge_preserves_comfy_quant_tensor(tmp_path):
    # A quantized shard carries a `.comfy_quant` U8 config tensor; merging must keep it
    # byte-for-byte so the native format survives the merge.
    cfg = json.dumps({"format": "convrot_w4a4", "convrot_groupsize": 256}).encode("utf-8")
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build(s1, {
        "l0.weight": ("U8", [4], b"\x01" * 4),
        "l0.weight_scale": ("F32", [1], _f32([0.5])),
        "l0.comfy_quant": ("U8", [len(cfg)], cfg),
    })
    _build(s2, {"l1.weight": ("U8", [4], b"\x02" * 4)})
    out = tmp_path / "merged.safetensors"
    merge_safetensors_files([str(s1), str(s2)], str(out))

    header, data_start = read_safetensors_header(str(out))
    # The .comfy_quant tensor is present in the merged output.
    assert "l0.comfy_quant" in header
    assert header["l0.comfy_quant"]["dtype"] == "U8"
    s, e = header["l0.comfy_quant"]["data_offsets"]
    with open(out, "rb") as fh:
        fh.seek(data_start + s)
        blob = fh.read(e - s)
    assert json.loads(blob)["format"] == "convrot_w4a4"
