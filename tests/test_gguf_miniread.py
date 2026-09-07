"""S0.2 tests for the minimal GGUF header reader (plan STEP 0.2).

Fixtures are hand-built byte strings (no gguf dependency), so these run in
the headless gate and pin the exact container semantics our exporter relies
on: magic/version, KV subset decode, tensor infos with GGUF-order ``ne``,
32-byte payload alignment, and loud failure on anything unsupported.
"""

import struct

import pytest

from tests.gguf_miniread import MiniGgufError, read_gguf_header, tensor_data

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key: str, value: str) -> bytes:
    return _s(key) + struct.pack("<I", 10) + _s(value)


def _kv_u32(key: str, value: int) -> bytes:
    return _s(key) + struct.pack("<I", 6) + struct.pack("<I", value)


def _kv_f32(key: str, value: float) -> bytes:
    return _s(key) + struct.pack("<I", 8) + struct.pack("<f", value)


def _kv_arr_str(key: str, values: list[str]) -> bytes:
    out = _s(key) + struct.pack("<I", 11) + struct.pack("<I", 10) + struct.pack("<Q", len(values))
    return out + b"".join(_s(v) for v in values)


def _tensor_info(name: str, ne: list[int], ggml_type: int, offset: int) -> bytes:
    out = _s(name) + struct.pack("<I", len(ne))
    out += b"".join(struct.pack("<Q", d) for d in ne)
    out += struct.pack("<IQ", ggml_type, offset)
    return out


def _build_full(kv_parts: list[bytes], tensor_parts: list[bytes], payload: bytes = b"\x00" * 64) -> bytes:
    n_kv = len(kv_parts)
    n_tensors = len(tensor_parts)
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", n_tensors, n_kv)
    kv_blob = b"".join(kv_parts)
    t_blob = b"".join(tensor_parts)
    header_end = len(head) + len(kv_blob) + len(t_blob)
    pad = (32 - header_end % 32) % 32
    return head + kv_blob + t_blob + b"\x00" * pad + payload


def test_miniread_roundtrip_handwritten(tmp_path):
    """A 1-tensor handcrafted GGUF decodes: name / ne / qtype / aligned offset."""
    payload = bytes(range(256))
    blob = _build_full(
        [_kv_str("general.architecture", "llama"), _kv_u32("llama.block_count", 2)],
        [_tensor_info("blk.0.attn_q.weight", [32, 4], GGML_F32, 0)],
        payload,
    )
    p = tmp_path / "m.gguf"
    p.write_bytes(blob)
    g = read_gguf_header(str(p))
    assert g.version == 3
    assert g.kv["general.architecture"] == "llama"
    assert g.kv["llama.block_count"] == 2
    t = g.tensor("blk.0.attn_q.weight")
    assert t.ne == (32, 4)
    assert t.ggml_type == GGML_F32
    assert t.offset > 0 and t.offset % 32 == 0
    assert tensor_data(str(p), t) == payload[: 32 * 4 * 4]


def test_miniread_rejects_bad_magic(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(MiniGgufError, match="magic"):
        read_gguf_header(str(p))


def test_miniread_rejects_wrong_version(tmp_path):
    blob = b"GGUF" + struct.pack("<I", 2) + struct.pack("<QQ", 0, 0)
    p = tmp_path / "v2.gguf"
    p.write_bytes(blob)
    with pytest.raises(MiniGgufError, match="version"):
        read_gguf_header(str(p))


def test_miniread_kv_types(tmp_path):
    """string / u32 / f32 / bool / array-of-string decode exactly."""
    blob = _build_full(
        [
            _kv_str("general.name", "mini"),
            _kv_u32("x.block_count", 7),
            _kv_f32("x.eps", 1.5),
            _s("x.flag") + struct.pack("<I", 9) + struct.pack("<B", 1),
            _kv_arr_str("x.roles", ["a", "b"]),
        ],
        [],
    )
    p = tmp_path / "kv.gguf"
    p.write_bytes(blob)
    g = read_gguf_header(str(p))
    assert g.kv["general.name"] == "mini"
    assert g.kv["x.block_count"] == 7
    assert g.kv["x.eps"] == pytest.approx(1.5)
    assert g.kv["x.flag"] == 1
    assert g.kv["x.roles"] == ["a", "b"]


def test_miniread_align_and_order(tmp_path):
    """Multiple tensors: file order preserved, offsets absolute + aligned."""
    payload = b"\xab" * (64 * 4)
    blob = _build_full(
        [],
        [
            _tensor_info("a.weight", [16, 2], GGML_F32, 0),
            _tensor_info("b.weight", [32], GGML_F32, 128),
        ],
        payload,
    )
    p = tmp_path / "two.gguf"
    p.write_bytes(blob)
    g = read_gguf_header(str(p))
    assert g.tensor_names() == ["a.weight", "b.weight"]
    off_a = g.tensors[0].offset
    off_b = g.tensors[1].offset
    assert off_a % 32 == 0 and off_b % 32 == 0
    assert off_b - off_a == 128
    assert tensor_data(str(p), g.tensors[0]) == payload[: 16 * 2 * 4]
    assert tensor_data(str(p), g.tensors[1]) == payload[128 : 128 + 32 * 4]


def test_miniread_tensor_missing(tmp_path):
    blob = _build_full([], [_tensor_info("a.weight", [4], GGML_F32, 0)])
    p = tmp_path / "one.gguf"
    p.write_bytes(blob)
    g = read_gguf_header(str(p))
    with pytest.raises(MiniGgufError, match="no tensor named"):
        g.tensor("b.weight")


def test_miniread_unsupported_kv_type(tmp_path):
    """Unknown/unsupported value type id -> loud error (never guessed)."""
    blob = _build_full([_s("x.weird") + struct.pack("<I", 99) + b"\x00\x00"], [])
    p = tmp_path / "weird.gguf"
    p.write_bytes(blob)
    with pytest.raises(MiniGgufError, match="unsupported KV value type 99"):
        read_gguf_header(str(p))
