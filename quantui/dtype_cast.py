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

import json
import os
import struct

__all__ = [
    "UnsupportedCastError",
    "f32_bits_to_bf16_bits",
    "f32_bits_to_f16_bits",
    "f16_bits_to_f32_bits",
    "bf16_bits_to_f32_bits",
    "f32_to_bf16",
    "cast_tensor_bytes",
    "cast_safetensors_file",
    "cast_shards_to_single",
]

# Payload chunk size for the streaming writer: elements per conversion batch so
# the intermediate buffer stays ~8 MiB regardless of dtype widths.
_CHUNK_ELEMS = 2 * 1024 * 1024


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


def f32_to_bf16(arr):
    """Vectorized RTNE f32 -> bf16 (uint16 bit patterns), numpy-native.

    Semantics mirror :func:`f32_bits_to_bf16_bits`: NaN payloads keep the top
    16 bits with the quiet bit forced; everything else rounds
    ties-to-even.  Returns a ``<u2`` array of bit patterns — NOT a float
    dtype (numpy has none for bf16).
    """
    import numpy as np

    u32 = np.ascontiguousarray(arr, dtype=np.float32).view(np.uint32)
    nan_mask = (u32 & np.uint32(0x7F80_0000)) == np.uint32(0x7F80_0000)
    inf_or_finite = ~((u32 & np.uint32(0x7FFF_FFFF)) > np.uint32(0x7F80_0000))
    # rounding bias: 0x7FFF + lsb of the kept half (ties-to-even)
    bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded = (u32 + bias) >> np.uint32(16)
    # NaN: top 16 bits + forced quiet bit
    nan_result = (u32 >> np.uint32(16)) | np.uint32(0x0040)
    out = np.where(inf_or_finite | ~nan_mask, rounded, nan_result)
    return out.astype("<u2").reshape(np.asarray(arr).shape)


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



# --------------------------------------------------------------------------- #
# Streaming cast writer (STEP 2.2): header-driven per-tensor payload streaming,
# mirroring merge_safetensors_files discipline -- never a full-model load.
#
# Dtype policy (plan section 0.3 + real-model semantics): bf16/fp16 formats cast
# ONLY floating tensors (F64/F32/F16/BF16 sources). Integer / bool tensors
# (I8 / U8 / I64 / BOOL / ...) pass through byte-identical with their ORIGINAL
# dtype kept in the output header -- rotary buffers, index tensors and masks
# must not be bit-mangled by a float narrowing. Unknown dtypes (e.g. F8) also
# pass through so a cast never hard-fails on an auxiliary tensor it cannot
# represent; the header records whatever dtype was preserved.
#
# Layout note: every conversion here is width-shrinking or identity
# (F32/F64 -> 2 bytes; F16/BF16 -> 2 bytes; identity unchanged), so each
# output payload length is computable from the INPUT headers alone. That lets
# us write the aligned header FIRST and then stream payloads in one sequential
# pass over the sources -- no temp files, no second header splice, peak extra
# memory stays at one ~8 MiB chunk.
# --------------------------------------------------------------------------- #

# Element chunk for cast conversion batches (~8 MiB of F32 source elements).
_CHUNK_ELEMS = 2 * 1024 * 1024

_COPY_CHUNK = 8 << 20  # verbatim byte-range copy chunk


def _read_header(path: str) -> tuple[dict, int]:
    """Parse a safetensors file into ``(header_dict, data_start_byte)``."""
    with open(path, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(hdr_len).decode("utf-8")), 8 + hdr_len


def _out_len_and_dtype(src_dtype: str, n_bytes: int, target: str) -> tuple[int, str]:
    """Output byte length + dtype for one tensor under the dtype policy."""
    if src_dtype == target or src_dtype not in _DTYPE_SIZES:
        return n_bytes, src_dtype  # identity or pass-through (int/bool/unknown)
    src_size = _DTYPE_SIZES[src_dtype]
    dst_size = _DTYPE_SIZES[target]
    return (n_bytes // src_size) * dst_size, target


def _stream_cast(fh, blob_start: int, n_bytes: int, src: str, target: str, out) -> None:
    """Read one tensor payload in element chunks, RTNE-cast, write to ``out``."""
    elem = _DTYPE_SIZES[src]
    remaining = n_bytes
    pos = blob_start
    while remaining > 0:
        take_elems = min(_CHUNK_ELEMS, remaining // elem)
        take = take_elems * elem
        fh.seek(pos)
        blob = fh.read(take)
        data, _dt = cast_tensor_bytes(blob, src, target)
        out.write(data)
        pos += take
        remaining -= take


def _build_plan(shard_paths: list[str], target: str):
    """First pass: parse every shard header, validate uniqueness, compute output
    specs with cumulative offsets.

    Returns ``(meta, entries, total_data_bytes)`` where ``entries`` is
    ``[(src_path, data_start, name, out_spec)]`` in deterministic shard order.
    Raises ``ValueError("duplicate tensor ...")`` on cross-shard collisions.
    """
    seen: dict[str, str] = {}
    meta = None
    entries = []
    cursor = 0
    for sp in shard_paths:
        header, data_start = _read_header(sp)
        if "__metadata__" in header and meta is None:
            meta = header["__metadata__"]
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(
                    f"duplicate tensor {name!r} across shards "
                    f"({seen[name]!r} and {sp!r})"
                )
            seen[name] = sp
            a, b = spec["data_offsets"]
            length, dtype_out = _out_len_and_dtype(spec["dtype"], b - a, target)
            entries.append((sp, data_start, name, {
                "dtype": dtype_out,
                "shape": list(spec["shape"]),
                "data_offsets": [cursor, cursor + length],
                "_raw": (a, b),
                "_converts": dtype_out != spec["dtype"],
                "_src_dtype": spec["dtype"],
            }))
            cursor += length
    return meta, entries, cursor


def _write_cast_file(
    shard_paths: list[str], dst: str, target: str, on_progress, meta, entries,
) -> None:
    """Second pass: write the aligned header, then stream payloads sequentially."""
    from .comfy_quant_schema import _align_header_to_8

    out_header: dict[str, object] = {}
    if meta is not None:
        out_header["__metadata__"] = meta
    for _sp, _ds, _name, spec in entries:
        out_header[_name] = {k: v for k, v in spec.items() if not k.startswith("_")}

    parent = os.path.dirname(os.path.abspath(dst))
    os.makedirs(parent, exist_ok=True)
    hdr_bytes = _align_header_to_8(json.dumps(out_header).encode("utf-8"))

    total = len(entries)
    if on_progress:
        on_progress(0, total)
    handles: dict[str, object] = {}

    def _fh(sp):
        if sp not in handles:
            handles[sp] = open(sp, "rb")
        return handles[sp]

    try:
        with open(dst, "wb") as out:
            out.write(struct.pack("<Q", len(hdr_bytes)))
            out.write(hdr_bytes)
            for i, (sp, data_start, _name, spec) in enumerate(entries):
                a, b = spec["_raw"]
                fh = _fh(sp)
                if spec["_converts"]:
                    # Cast path: element-chunked RTNE conversion (~8 MiB chunks).
                    _stream_cast(fh, data_start + a, b - a,
                                 spec["_src_dtype"], target, out)
                else:
                    # Identity / int-bool pass-through: verbatim byte-range copy.
                    fh.seek(data_start + a)
                    remaining = b - a
                    while remaining > 0:
                        chunk = fh.read(min(_COPY_CHUNK, remaining))
                        if not chunk:
                            break
                        out.write(chunk)
                        remaining -= len(chunk)
                if on_progress:
                    on_progress(i + 1, total)
    finally:
        for h in handles.values():
            h.close()


def cast_safetensors_file(src: str, dst: str, target: str, *, on_progress=None) -> None:
    """Cast ONE ``.safetensors`` file's floating tensors to ``target``.

    ``target`` is a safetensors dtype id ("BF16" or "F16"). Floating tensors are
    RTNE-cast; integer/bool tensors pass through byte-identical. Streams
    payloads in chunks -- never loads the whole file. ``on_progress(done, total)``
    is called per tensor, starting at ``(0, N)``.
    """
    meta, entries, _cursor = _build_plan([src], target)
    _write_cast_file([src], dst, target, on_progress, meta, entries)


def cast_shards_to_single(shard_paths: list[str], dst: str, target: str, *,
                          on_progress=None) -> None:
    """Merge-cast a shard set into ONE ``.safetensors`` (streaming).

    Mirrors :func:`quantui.worker_ctq.merge_safetensors_files`: cumulative
    offset rewrite across shards, ``__metadata__`` carried from the first shard
    that has one, duplicate tensor names raise ``ValueError``.
    """
    meta, entries, _cursor = _build_plan(shard_paths, target)
    _write_cast_file(shard_paths, dst, target, on_progress, meta, entries)
