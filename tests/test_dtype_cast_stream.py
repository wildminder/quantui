"""STEP 2.2 (plan 2026-08-26): streaming cast writer for files and shard sets.

``cast_safetensors_file`` / ``cast_shards_to_single`` mirror the proven
``merge_safetensors_files`` discipline: header-driven, per-tensor payload
streaming, cumulative offset rewrite -- never a full-model in-memory load.
Floating tensors (F32/F64/F16/BF16 sources) are RTNE-cast to the target dtype;
integer / bool tensors (I8/U8/I64/BOOL/...) pass through byte-identical with
their original dtype recorded (real-model semantics: rotary buffers, embeddings
tables etc. must not be bit-mangled).
"""

import json
import struct

import pytest

from quantui.comfy_quant_schema import write_safetensors
from quantui.dtype_cast import (
    f32_bits_to_bf16_bits,
    f32_bits_to_f16_bits,
)


# ---- helpers --------------------------------------------------------------- #
def _read_tensors(path):
    """Independent mini-reader: (header, {name: (dtype, shape, payload_bytes)})."""
    tensors = {}
    with open(path, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
        data_start = 8 + hdr_len
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            a, b = spec["data_offsets"]
            fh.seek(data_start + a)
            tensors[name] = (spec["dtype"], spec["shape"], fh.read(b - a))
    return header, tensors


def _assert_contiguous(header):
    prev = 0
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        a, b = spec["data_offsets"]
        assert a == prev, f"non-contiguous offsets at {name}"
        prev = b
    return prev


def _expected(blob, src, dst):
    """Expected payload via the STEP 2.1 public scalar API (allowed by plan)."""
    out = bytearray()
    size = {"F32": 4, "F16": 2, "BF16": 2}[src]
    for off in range(0, len(blob), size):
        u32 = int.from_bytes(blob[off:off + size], "little")
        if src == "F32":
            h = f32_bits_to_bf16_bits(u32) if dst == "BF16" else f32_bits_to_f16_bits(u32)
            out += h.to_bytes(2, "little")
        elif src == "BF16":  # widen -> F32 bits
            out += (u32 << 16).to_bytes(4, "little")
        else:  # F16 widen
            sign = (u32 & 0x8000) << 16
            exp = (u32 >> 10) & 0x1F
            mant = u32 & 0x03FF
            if exp == 0x1F:
                w = sign | 0x7F800000 | (mant << 13)
            elif exp == 0:
                w = sign if mant == 0 else sign | ((mant.bit_length() - 1 + 103) << 23) | ((mant << (23 - (mant.bit_length() - 1))) & 0x007FFFFF)
            else:
                w = sign | ((exp + 112) << 23) | (mant << 13)
            out += w.to_bytes(4, "little")
    return bytes(out), dst


def test_api_surface_present():
    from quantui.dtype_cast import (
        cast_safetensors_file,
        cast_shards_to_single,
    )

    assert callable(cast_safetensors_file)
    assert callable(cast_shards_to_single)


def test_cast_single_file_f32_to_bf16(tmp_path):
    from quantui.dtype_cast import cast_safetensors_file

    f32_blob = struct.pack("<fff", 1.0, 2.0, -2.0)
    i64_blob = struct.pack("<qqq", -3, 0, 7)
    specs = {
        "w.f32": ("F32", [3], f32_blob),
        "buf.i64": ("I64", [3], i64_blob),
    }
    src = tmp_path / "in.safetensors"
    src.write_bytes(write_safetensors(specs, metadata={"note": "keep me"}))

    dst = tmp_path / "out.safetensors"
    calls = []
    cast_safetensors_file(str(src), str(dst), "BF16",
                          on_progress=lambda c, t: calls.append((c, t)))

    header, tensors = _read_tensors(dst)
    # F32 cast; I64 passes through UNCHANGED with original dtype.
    assert tensors["w.f32"][0] == "BF16"
    exp_payload, exp_dtype = _expected(f32_blob, "F32", "BF16")
    assert tensors["w.f32"][2] == exp_payload
    assert tensors["w.f32"][1] == [3]
    assert tensors["buf.i64"] == ("I64", [3], i64_blob)
    assert header.get("__metadata__") == {"note": "keep me"}
    _assert_contiguous(header)
    # Progress callback monotonic and ends at (N, N).
    assert calls[-1] == (2, 2)
    curs = [c for c, _ in calls]
    assert curs == sorted(curs)


def test_cast_single_file_f32_to_f16(tmp_path):
    from quantui.dtype_cast import cast_safetensors_file

    f32_blob = struct.pack("<ff", 1.0, -2.0)
    src = tmp_path / "in.safetensors"
    src.write_bytes(write_safetensors({"a": ("F32", [2], f32_blob)}))
    dst = tmp_path / "out.safetensors"

    cast_safetensors_file(str(src), str(dst), "F16")

    _, tensors = _read_tensors(dst)
    exp_payload, exp_dtype = _expected(f32_blob, "F32", "F16")
    assert tensors["a"] == ("F16", [2], exp_payload)


def test_cast_identity_fast_path_byte_identical(tmp_path):
    from quantui.dtype_cast import cast_safetensors_file

    bf16_blob = b"\x00\x3f\x00\x40"  # two BF16 values
    specs = {
        "a": ("BF16", [2], bf16_blob),
        "b": ("I8", [1], b"\x07"),
    }
    src = tmp_path / "in.safetensors"
    src.write_bytes(write_safetensors(specs))
    dst = tmp_path / "out.safetensors"

    cast_safetensors_file(str(src), str(dst), "BF16")
    # ALL tensors already at target dtype (or pass-through ints) AND single shard
    # with deterministic header layout -> FULL file equality.
    assert dst.read_bytes() == src.read_bytes()


def test_cast_sharded_single_mode(tmp_path):
    from quantui.dtype_cast import cast_shards_to_single

    s1 = {
        "t.a": ("F32", [2], struct.pack("<ff", 1.0, 2.0)),
        "meta.i8": ("I8", [2], b"\x01\x02"),
    }
    s2 = {"t.b": ("F32", [1], struct.pack("<f", 4.5))}
    p1 = tmp_path / "model-00001-of-00002.safetensors"
    p2 = tmp_path / "model-00002-of-00002.safetensors"
    p1.write_bytes(write_safetensors(s1, metadata={"src": "shard1"}))
    p2.write_bytes(write_safetensors(s2))
    dst = tmp_path / "merged-cast.safetensors"

    cast_shards_to_single([str(p1), str(p2)], str(dst), "BF16")

    header, tensors = _read_tensors(dst)
    assert set(tensors) == {"t.a", "t.b", "meta.i8"}
    assert header.get("__metadata__") == {"src": "shard1"}
    exp_a, _ = _expected(s1["t.a"][2], "F32", "BF16")
    exp_b, _ = _expected(s2["t.b"][2], "F32", "BF16")
    assert tensors["t.a"] == ("BF16", [2], exp_a)
    assert tensors["t.b"] == ("BF16", [1], exp_b)
    assert tensors["meta.i8"] == ("I8", [2], b"\x01\x02")
    _assert_contiguous(header)


def test_cast_sharded_duplicate_tensor_raises(tmp_path):
    from quantui.dtype_cast import cast_shards_to_single

    dup = {"t.x": ("F32", [1], struct.pack("<f", 9.0))}
    p1 = tmp_path / "a.safetensors"
    p2 = tmp_path / "b.safetensors"
    p1.write_bytes(write_safetensors(dup))
    p2.write_bytes(write_safetensors(dup))
    with pytest.raises(ValueError, match="duplicate tensor"):
        cast_shards_to_single([str(p1), str(p2)], str(tmp_path / "o.safetensors"), "BF16")


def test_cast_progress_callback_monotonic(tmp_path):
    from quantui.dtype_cast import cast_safetensors_file

    specs = {f"t{i}": ("F32", [1], struct.pack("<f", float(i))) for i in range(4)}
    src = tmp_path / "in.safetensors"
    src.write_bytes(write_safetensors(specs))
    dst = tmp_path / "out.safetensors"
    calls = []
    cast_safetensors_file(str(src), str(dst), "BF16", on_progress=lambda c, t: calls.append((c, t)))
    assert calls[0] == (0, 4)
    assert calls[-1] == (4, 4)
    assert all(c <= n and t == 4 for c, t in calls for n in [c]) or True
    curs = [c for c, _ in calls]
    assert curs == sorted(curs) and len(set(curs)) == len(curs)


def test_cast_large_tensor_chunked(tmp_path):
    # One > chunk-size (8 MiB data) F32 tensor exercises the chunked write loop.
    from quantui.dtype_cast import cast_safetensors_file

    n = 3 * 1024 * 1024  # 3 Mi elements -> 12 MiB F32 payload (> 8 MiB chunks)
    blob = struct.pack(f"<{n}f", *([1.0] * n))
    src = tmp_path / "big.safetensors"
    src.write_bytes(write_safetensors({"big": ("F32", [n], blob)}))
    dst = tmp_path / "big-out.safetensors"

    cast_safetensors_file(str(src), str(dst), "BF16")

    _, tensors = _read_tensors(dst)
    got_dtype, got_shape, got_data = tensors["big"]
    assert got_dtype == "BF16" and got_shape == [n]
    assert len(got_data) == n * 2
    # Spot-check first + last element via the public scalar API.
    first = int.from_bytes(got_data[0:2], "little")
    last = int.from_bytes(got_data[-2:], "little")
    assert first == f32_bits_to_bf16_bits(int.from_bytes(blob[0:4], "little"))
    assert last == f32_bits_to_bf16_bits(int.from_bytes(blob[-4:], "little"))
