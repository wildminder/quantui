"""Pure bit-exact dtype-cast core (STEP 2.1, plan 2026-08-26).

Conversions between F32 / BF16 / F16 are implemented as pure unsigned-int bit
manipulation with round-to-nearest-even (RTNE) semantics -- deterministic
across platforms and testable in a headless environment (no numpy / torch).

All ``*_bits`` helpers take and return UNSIGNED integer BIT PATTERNS (not
floats); byte-order packing happens once at the :func:`cast_tensor_bytes`
boundary, which operates on little-endian buffers (the safetensors on-disk
byte order).

Rounding rules implemented:
* F32 -> BF16: RTNE on the lower 16 mantissa bits (overflow cannot happen:
  the bf16 exponent range equals f32's).
* F32 -> F16: RTNE with overflow -> +/-Inf and underflow -> subnormal/zero.
* F64 -> BF16/F16: narrow to F32 first (double rounding, documented behavior).
"""

import struct

__all__ = [
    "UnsupportedCastError",
    "f32_bits_to_bf16_bits",
    "f32_bits_to_f16_bits",
    "f16_bits_to_f32_bits",
    "bf16_bits_to_f32_bits",
    "cast_tensor_bytes",
]


class UnsupportedCastError(ValueError):
    """Raised when a requested tensor-dtype cast is not supported."""


def f32_bits_to_bf16_bits(u32: int) -> int:
    """Round-to-nearest-even an IEEE-754 binary32 bit pattern to bfloat16."""
    u = u32 & 0x7FFFFFFF
    if u > 0x7F800000:
        # NaN: keep top 16 bits, FORCE the quiet bit (payload otherwise lost
        # by the arithmetic rounding below).
        return (u32 >> 16) | 0x0040
    # lsb of the kept half + the 16 dropped bits decide rounding.
    rounding_bias = 0x7FFF + ((u32 >> 16) & 1)
    return (u32 + rounding_bias) >> 16


def f32_bits_to_f16_bits(u32: int) -> int:
    """Round-to-nearest-even an IEEE-754 binary32 bit pattern to binary16.

    Overflow beyond fp16's max finite (~65504) becomes +/-Inf; underflow below
    fp16's min subnormal flushes to zero; in-between magnitudes become fp16
    subnormals via RTNE (a rounding carry-out promotes to the min normal).
    """
    sign = (u32 >> 16) & 0x8000
    u = u32 & 0x7FFFFFFF

    if u >= 0x7F800000:  # Inf / NaN
        if u > 0x7F800000:
            # NaN: force quiet, keep high payload bits.
            return sign | 0x7E00 | ((u >> 13) & 0x03FF)
        return sign | 0x7C00

    exp = ((u >> 23) & 0xFF) - 127 + 15  # target fp16 BIASED exponent
    mant = u & 0x007FFFFF

    if exp >= 0x1F:
        return sign | 0x7C00  # overflow -> +/-Inf
    if exp >= 1:  # lands in the normal fp16 range
        # RTNE the 23-bit mantissa down to 10 bits.
        rounded = mant + 0x0FFF + ((mant >> 13) & 1)
        m10 = (rounded >> 13) & 0x03FF
        e = exp
        if rounded >> 23:  # mantissa carried into the exponent
            e += 1
            m10 = 0
            if e >= 0x1F:
                return sign | 0x7C00  # carry overflowed fp16 -> Inf
        return sign | (e << 10) | m10

    # Subnormal / zero fp16 (target exponent <= 0).
    mant |= 0x00800000  # implicit leading one -> 24-bit significand
    shift = 14 - exp  # number of low bits dropped (>= 14 when exp <= 0)
    if shift > 24:
        return sign  # strictly below half the min subnormal -> zero
    dropped = mant & ((1 << shift) - 1)
    result = mant >> shift
    halfway = 1 << (shift - 1)
    if dropped > halfway or (dropped == halfway and (result & 1)):
        result += 1
    if result >= 0x0400:
        # Rounding carried past the max subnormal -> smallest NORMAL fp16.
        return sign | 0x0400
    return sign | result


def f16_bits_to_f32_bits(h: int) -> int:
    """Exactly widen an IEEE-754 binary16 bit pattern to binary32."""
    sign = (h & 0x8000) << 16
    exp = (h >> 10) & 0x1F
    mant = h & 0x03FF
    if exp == 0x1F:
        return sign | 0x7F800000 | (mant << 13)  # Inf / NaN
    if exp == 0:
        if mant == 0:
            return sign  # +/- zero
        n = mant.bit_length() - 1  # leading-bit index, 0..9
        # value = mant * 2^-24 == 1.f * 2^(n-24) -> biased f32 exponent = n+103
        return sign | ((n + 103) << 23) | ((mant << (23 - n)) & 0x007FFFFF)
    return sign | ((exp + 112) << 23) | (mant << 13)


def bf16_bits_to_f32_bits(b: int) -> int:
    """Exactly widen a bfloat16 bit pattern to binary32 (pure 16-bit shift)."""
    return b << 16


# --------------------------------------------------------------------------- #
# Byte-buffer boundary (little-endian, the safetensors on-disk byte order)
# --------------------------------------------------------------------------- #
_DTYPE_SIZES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2}


def _narrow_f32_bits(u32: int, dst: str) -> int:
    if dst == "BF16":
        return f32_bits_to_bf16_bits(u32)
    if dst == "F16":
        return f32_bits_to_f16_bits(u32)
    raise UnsupportedCastError(f"unsupported destination dtype {dst!r}")


def cast_tensor_bytes(blob: bytes, src_dtype: str, dst_dtype: str) -> tuple[bytes, str]:
    """Cast a whole little-endian tensor buffer; returns ``(data, new_dtype)``.

    Identity casts return the SAME buffer object (zero-copy fast path). F64
    inputs are narrowed to F32 first (documented double rounding). Integer /
    bool dtypes and unknown dtype ids raise :class:`UnsupportedCastError`.
    """
    try:
        src_size = _DTYPE_SIZES[src_dtype]
    except KeyError:
        raise UnsupportedCastError(
            f"cannot cast from non-float or unknown dtype {src_dtype!r}"
        ) from None
    try:
        dst_size = _DTYPE_SIZES[dst_dtype]
    except KeyError:
        raise UnsupportedCastError(
            f"cannot cast to non-float or unknown dtype {dst_dtype!r}"
        ) from None
    if len(blob) % src_size:
        raise UnsupportedCastError(
            f"buffer length {len(blob)} is not a multiple of {src_size} ({src_dtype})"
        )

    if src_dtype == dst_dtype:
        return blob, src_dtype  # zero-copy identity fast path

    out = bytearray()
    if src_dtype == "F64":
        for off in range(0, len(blob), 8):
            d = struct.unpack_from("<d", blob, off)[0]
            u32 = struct.unpack("<I", struct.pack("<f", d))[0]
            out += _narrow_f32_bits(u32, dst_dtype).to_bytes(dst_size, "little")
    elif src_dtype == "F32":
        for off in range(0, len(blob), 4):
            u32 = int.from_bytes(blob[off:off + 4], "little")
            out += _narrow_f32_bits(u32, dst_dtype).to_bytes(dst_size, "little")
    else:  # src is BF16 / F16
        widen = (
            bf16_bits_to_f32_bits if src_dtype == "BF16" else f16_bits_to_f32_bits
        )
        for off in range(0, len(blob), 2):
            u32 = widen(int.from_bytes(blob[off:off + 2], "little"))
            if dst_dtype == "F32":
                out += u32.to_bytes(4, "little")
            else:  # BF16 <-> F16 goes through the exact f32 representation
                out += _narrow_f32_bits(u32, dst_dtype).to_bytes(2, "little")
    return bytes(out), dst_dtype
