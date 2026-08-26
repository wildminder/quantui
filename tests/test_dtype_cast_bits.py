"""STEP 2.1 (plan 2026-08-26): pure bit-exact dtype-cast core.

All known-answer tables are FROZEN hex literals -- the test itself performs NO
floating-point arithmetic. Structural sweeps (round-trip, monotonicity, sign
preservation) run exhaustively over the full uint16 space where applicable.
"""

import pytest

from quantui.dtype_cast import (
    UnsupportedCastError,
    bf16_bits_to_f32_bits,
    cast_tensor_bytes,
    f16_bits_to_f32_bits,
    f32_bits_to_bf16_bits,
    f32_bits_to_f16_bits,
)

# --------------------------------------------------------------------------- #
# F32 -> BF16 (RTNE on the lower 16 mantissa bits)
# --------------------------------------------------------------------------- #
F32_TO_BF16_KAT = [
    # (input_hex, expected_hex, note)
    (0x3F800000, 0x3F80, "1.0 exact"),
    (0x40000000, 0x4000, "2.0 exact"),
    (0xBF800000, 0xBF80, "-1.0 sign kept"),
    (0x3F800001, 0x3F80, "below halfway -> down"),
    (0x3F80FFFF, 0x3F81, "above halfway -> up"),
    (0x3F808000, 0x3F80, "exact tie -> round to EVEN (down)"),
    (0x3F818000, 0x3F82, "exact tie -> round to EVEN (up)"),
    (0x3F808001, 0x3F81, "just past tie -> up"),
    (0x7F800000, 0x7F80, "+Inf preserved"),
    (0xFF800000, 0xFF80, "-Inf preserved"),
    (0x7FC00000, 0x7FC0, "qNaN keeps quiet bit"),
    (0x7F800001, 0x7FC0, "sNaN payload truncated -> forced quiet"),
    (0x00000000, 0x0000, "+0"),
    (0x80000000, 0x8000, "-0"),
    (0x00000001, 0x0000, "tiny F32 subnormal -> 0"),
    (0x007FFFFF, 0x0080, "max F32 subnormal -> smallest normal"),
    (0xBF808001, 0xBF81, "negative rounding"),
]


@pytest.mark.parametrize("src,dst,_note", F32_TO_BF16_KAT, ids=[k[-1] for k in F32_TO_BF16_KAT])
def test_f32_to_bf16_kat(src, dst, _note):
    assert f32_bits_to_bf16_bits(src) == dst


# --------------------------------------------------------------------------- #
# F32 -> F16 (RTNE w/ overflow->Inf, underflow->subnormal/zero)
# --------------------------------------------------------------------------- #
F32_TO_F16_KAT = [
    (0x3F800000, 0x3C00, "1.0 exact"),
    (0x40000000, 0x4000, "2.0 exact"),
    (0xC0000000, 0xC000, "-2.0 sign kept"),
    (0x3EAAAAAB, 0x3555, "~1/3 in-range normal"),
    (0x3F801000, 0x3C00, "exact tie -> EVEN (down)"),
    (0x3F802001, 0x3C01, "just past halfway -> up"),
    (0x3F800FFF, 0x3C00, "just under halfway -> down"),
    (0x477FE000, 0x7BFF, "max finite 65504 stays finite"),
    (0x477FF000, 0x7C00, "65520 overflows -> +Inf"),
    (0xC77FF000, 0xFC00, "-65520 overflows -> -Inf"),
    (0x7F800000, 0x7C00, "+Inf"),
    (0xFF800000, 0xFC00, "-Inf"),
    (0x7FC00000, 0x7E00, "qNaN -> quiet fp16 NaN"),
    (0xFFC00000, 0xFE00, "-qNaN -> negative quiet fp16 NaN"),
    (0x00000000, 0x0000, "+0"),
    (0x80000000, 0x8000, "-0"),
    (0x38800000, 0x0400, "2^-14 == min NORMAL fp16"),
    (0x38000000, 0x0200, "2^-15 -> subnormal, exact"),
    (0x33800000, 0x0001, "2^-24 == min subnormal fp16"),
    (0x33000000, 0x0000, "2^-25 exact tie -> EVEN (zero)"),
    (0x33C00000, 0x0002, "1.5*2^-25 tie -> EVEN (rounds UP to 2)"),
    (0x30800000, 0x0000, "2^-30 flushes to zero"),
]


@pytest.mark.parametrize("src,dst,_note", F32_TO_F16_KAT, ids=[k[-1] for k in F32_TO_F16_KAT])
def test_f32_to_f16_kat(src, dst, _note):
    assert f32_bits_to_f16_bits(src) == dst


# --------------------------------------------------------------------------- #
# Widening: F16/BF16 -> F32 is EXACT for every input
# --------------------------------------------------------------------------- #
WIDEN_KAT = [
    (0x3C00, 0x3F800000, "fp16 1.0"),
    (0x4000, 0x40000000, "fp16 2.0"),
    (0xC000, 0xC0000000, "fp16 -2.0"),
    (0x0001, 0x33800000, "fp16 min subnormal == f32 2^-24"),
    (0x03FF, 0x387FC000, "fp16 max subnormal"),
    (0x0400, 0x38800000, "fp16 min normal == 2^-14"),
    (0x7BFF, 0x477FE000, "fp16 max finite"),
    (0x7C00, 0x7F800000, "fp16 +Inf"),
    (0xFC00, 0xFF800000, "fp16 -Inf"),
]

BF16_WIDEN_KAT = [
    (0x3F80, 0x3F800000),
    (0x4000, 0x40000000),
    (0xBF80, 0xBF800000),
    (0x0000, 0x00000000),
    (0x8000, 0x80000000),
    (0x7F80, 0x7F800000),
    (0x7FC0, 0x7FC00000),
]


@pytest.mark.parametrize("src,dst,_note", WIDEN_KAT, ids=[k[-1] for k in WIDEN_KAT])
def test_f16_widen_kat(src, dst, _note):
    assert f16_bits_to_f32_bits(src) == dst


@pytest.mark.parametrize("src,dst", BF16_WIDEN_KAT)
def test_bf16_widen_kat(src, dst):
    assert bf16_bits_to_f32_bits(src) == dst


def _is_fp16_nan(h: int) -> bool:
    return (h & 0x7C00) == 0x7C00 and (h & 0x03FF) != 0


def test_exhaustive_f16_widen_roundtrip():
    """For ALL 65536 fp16 patterns (minus NaN payloads) widening then casting
    back with RTNE reproduces the exact same bits."""
    for h in range(0x10000):
        wide = f16_bits_to_f32_bits(h)
        assert (wide >> 31) == ((h >> 15) & 1), hex(h)  # sign always preserved
        if _is_fp16_nan(h):
            out = f32_bits_to_f16_bits(wide)
            assert (out & 0x7C00) == 0x7C00 and (out & 0x0200), hex(h)  # quiet NaN
            continue
        assert f32_bits_to_f16_bits(wide) == h, hex(h)


def test_exhaustive_bf16_widen_is_shift():
    for h in range(0x10000):
        assert bf16_bits_to_f32_bits(h) == h << 16


def test_monotonic_f32_downcasts():
    """Over the whole non-NaN f32 space the two downcasts are monotonic
    non-decreasing (catches off-by-one rounding anywhere in the curve)."""
    prev_b = 0x0000
    prev_h = 0x0000
    lo, hi = 0x00000001, 0x7F7FFFFF  # skip +0 and the NaN/Inf region
    step = 0x00001000  # deterministic stride sample (~500k points)
    u = lo
    while u <= hi:
        b = f32_bits_to_bf16_bits(u)
        h = f32_bits_to_f16_bits(u)
        assert b >= prev_b, hex(u)
        assert h >= prev_h, hex(u)
        prev_b, prev_h = b, h
        u += step


# --------------------------------------------------------------------------- #
# Mixed casts (widen->narrow chains) known-answer rows
# --------------------------------------------------------------------------- #
MIXED_KAT = [
    # fp16 0x3555 widens to f32 0x3EAAA000; bf16 RTNE remainder 0xA000 -> up.
    ("F16", 0x3555, "BF16", 0x3EAB),
    # bf16 0x3EAA widens to f32 0x3EAA0000; f16 truncation is exact -> 0x3550.
    ("BF16", 0x3EAA, "F16", 0x3550),
]


@pytest.mark.parametrize("src_fmt,src,dst_fmt,dst", MIXED_KAT)
def test_mixed_cast_kat(src_fmt, src, dst_fmt, dst):
    wide = (
        f16_bits_to_f32_bits(src) if src_fmt == "F16" else bf16_bits_to_f32_bits(src)
    )
    narrow = (
        f32_bits_to_f16_bits(wide) if dst_fmt == "F16" else f32_bits_to_bf16_bits(wide)
    )
    assert narrow == dst


# --------------------------------------------------------------------------- #
# cast_tensor_bytes (little-endian byte-order boundary)
# --------------------------------------------------------------------------- #
def test_cast_tensor_bytes_identity_fast_path():
    blob = bytes(range(256))
    out, dt = cast_tensor_bytes(blob, "BF16", "BF16")
    assert dt == "BF16"
    assert out is blob  # SAME object: zero-copy identity


def test_cast_tensor_bytes_f32_to_bf16_little_endian():
    # Two f32 LE floats: 1.0 (00 00 80 3F) and -2.0 (00 00 00 C0).
    blob = bytes.fromhex("0000803f000000c0")
    out, dt = cast_tensor_bytes(blob, "F32", "BF16")
    assert dt == "BF16"
    assert out.hex() == "803f00c0"


def test_cast_tensor_bytes_f32_to_f16_little_endian():
    blob = bytes.fromhex("0000803f000000c0")
    out, dt = cast_tensor_bytes(blob, "F32", "F16")
    assert dt == "F16"
    assert out.hex() == "003c00c0"


def test_cast_tensor_bytes_widen_paths():
    # BF16 LE 1.0 -> F32
    out, dt = cast_tensor_bytes(bytes.fromhex("803f"), "BF16", "F32")
    assert dt == "F32" and out.hex() == "0000803f"
    # F16 LE 2.0 -> F32
    out, dt = cast_tensor_bytes(bytes.fromhex("0040"), "F16", "F32")
    assert dt == "F32" and out.hex() == "00000040"
    # F16 1.0 -> BF16
    out, dt = cast_tensor_bytes(bytes.fromhex("003c"), "F16", "BF16")
    assert dt == "BF16" and out.hex() == "803f"
    # BF16 2.0 -> F16
    out, dt = cast_tensor_bytes(bytes.fromhex("0040"), "BF16", "F16")
    assert dt == "F16" and out.hex() == "0040"


@pytest.mark.parametrize(
    "src,dst",
    [("F32", "BF16"), ("F32", "F16"), ("F64", "BF16"), ("F64", "F16"),
     ("BF16", "F32"), ("BF16", "F16"), ("F16", "F32"), ("F16", "BF16")],
)
def test_cast_tensor_bytes_output_length(src, dst):
    sizes = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2}
    blob = b"\x00" * (sizes[src] * 4)
    out, dt = cast_tensor_bytes(blob, src, dst)
    assert dt == dst
    assert len(out) == sizes[dst] * 4


@pytest.mark.parametrize("dt", ["I8", "I16", "I32", "I64", "U8", "BOOL"])
def test_cast_tensor_bytes_rejects_integers(dt):
    with pytest.raises(UnsupportedCastError):
        cast_tensor_bytes(b"\x00" * 8, dt, "BF16")
    with pytest.raises(UnsupportedCastError):
        cast_tensor_bytes(b"\x00" * 4, "F32", dt)


def test_cast_tensor_bytes_unknown_dtype_and_bad_length():
    with pytest.raises(UnsupportedCastError):
        cast_tensor_bytes(b"\x00" * 4, "F9", "BF16")
    with pytest.raises(UnsupportedCastError):
        cast_tensor_bytes(b"\x00" * 3, "F32", "BF16")  # not a multiple of 4
