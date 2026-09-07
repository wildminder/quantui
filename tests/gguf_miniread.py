"""Test-only minimal GGUF header reader (pure stdlib ``struct``).

Deliberately NOT a full GGUF implementation: it parses exactly the subset of
the container that ``quantui.gguf_export`` produces, so gate-tier end-to-end
tests can assert on names / shapes / qtypes / KV pairs without the ``gguf``
package (which lives only in the worker env — plan STEP 0.2).

Supported subset:
* magic ``GGUF`` + version 3 (the only version gguf-py writes today);
* KV value types: u64, i64 (decoded as int), u32, i32 (int), f32 (float),
  bool (int 0/1), string (str), and arrays of (u32 | u32-as-int | string);
* tensor infos: name, n_dims, ``ne`` (u64 array, GGUF order = reversed
  numpy), type (u32), offset (u64). Payload offsets are absolute file
  offsets (header end is 32-byte aligned per spec).

Anything else raises :class:`MiniGgufError` — loud, never guessed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# gguf metadata value type ids — **gguf-py's actual on-disk ids**
# (verified 2026-09-07: STRING=8, UINT32=4, FLOAT32=6, UINT64=10, BOOL=7,
# ARRAY=9). NOTE these differ from the older draft-GGUF numbering.
_T_U64 = 10
_T_I64 = 11
_T_U32 = 4
_T_I32 = 5
_T_F32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9

_SCALAR_FMT: dict[int, struct.Struct] = {
    _T_U64: struct.Struct("<Q"),
    _T_I64: struct.Struct("<q"),
    _T_U32: struct.Struct("<I"),
    _T_I32: struct.Struct("<i"),
    _T_F32: struct.Struct("<f"),
    _T_BOOL: struct.Struct("<B"),
}


class MiniGgufError(ValueError):
    """Raised for malformed / unsupported GGUF bytes (message names the path)."""


@dataclass
class MiniTensorInfo:
    """One tensor-info record (names + GGUF-order shape + qtype id)."""

    name: str
    n_dims: int
    ne: tuple[int, ...]  # GGUF order: ne[0] = fastest-varying (in-features)
    ggml_type: int
    offset: int  # absolute file offset of the payload


@dataclass
class MiniGguf:
    """Parsed GGUF header: KV map + tensor infos (file-order)."""

    path: str
    version: int
    kv: dict[str, object] = field(default_factory=dict)
    tensors: list[MiniTensorInfo] = field(default_factory=list)

    def tensor(self, name: str) -> MiniTensorInfo:
        """Return the named tensor info; KeyError-style ValueError if absent."""
        for t in self.tensors:
            if t.name == name:
                return t
        raise MiniGgufError(f"{self.path}: no tensor named {name!r}")

    def tensor_names(self) -> list[str]:
        return [t.name for t in self.tensors]


def _read_str(buf: memoryview, off: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    raw = bytes(buf[off : off + n])
    if len(raw) != n:
        raise MiniGgufError("truncated string")
    return raw.decode("utf-8"), off + n


def _read_value(buf: memoryview, off: int, vtype: int):
    if vtype == _T_STRING:
        return _read_str(buf, off)
    fmt = _SCALAR_FMT.get(vtype)
    if fmt is not None:
        (val,) = fmt.unpack_from(buf, off)
        return val, off + fmt.size
    if vtype == _T_ARRAY:
        (etype,) = struct.unpack_from("<I", buf, off)
        off += 4
        (count,) = struct.unpack_from("<Q", buf, off)
        off += 8
        items = []
        for _ in range(count):
            val, off = _read_value(buf, off, etype)
            items.append(val)
        return items, off
    raise MiniGgufError(f"unsupported KV value type {vtype}")


def read_gguf_header(path: str) -> MiniGguf:
    """Parse the GGUF header + tensor infos of ``path`` (subset reader)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != GGUF_MAGIC:
        raise MiniGgufError(f"{path}: bad magic {data[:4]!r} (not GGUF)")
    mv = memoryview(data)
    off = 4
    (version,) = struct.unpack_from("<I", mv, off)
    off += 4
    if version != GGUF_VERSION:
        raise MiniGgufError(f"{path}: unsupported GGUF version {version}")
    (n_tensors, n_kv) = struct.unpack_from("<QQ", mv, off)
    off += 16

    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key, off = _read_str(mv, off)
        (vtype,) = struct.unpack_from("<I", mv, off)
        off += 4
        value, off = _read_value(mv, off, vtype)
        kv[key] = value

    tensors: list[MiniTensorInfo] = []
    for _ in range(n_tensors):
        name, off = _read_str(mv, off)
        (n_dims,) = struct.unpack_from("<I", mv, off)
        off += 4
        ne = []
        for _ in range(n_dims):
            (d,) = struct.unpack_from("<Q", mv, off)
            off += 8
            ne.append(d)
        (ggml_type,) = struct.unpack_from("<I", mv, off)
        off += 4
        (offset,) = struct.unpack_from("<Q", mv, off)
        off += 8
        tensors.append(
            MiniTensorInfo(
                name=name,
                n_dims=n_dims,
                ne=tuple(ne),
                ggml_type=ggml_type,
                offset=offset,
            )
        )

    # payload start = 32-byte-aligned end of the header section
    pad = (32 - off % 32) % 32
    payload_base = off + pad
    for t in tensors:
        t.offset += payload_base
    return MiniGguf(path=path, version=version, kv=kv, tensors=tensors)


def tensor_data(path: str, info: MiniTensorInfo) -> bytes:
    """Read the raw payload bytes of one tensor (size from its type/shape)."""
    sizes = _ggml_type_row_sizes()
    row_size, n_rows = sizes[info.ggml_type](info)
    with open(path, "rb") as fh:
        fh.seek(info.offset)
        return fh.read(row_size * n_rows)


def _ggml_type_row_sizes():
    """Row-size calculators for the qtypes the native backend emits (v1)."""
    import numpy as np  # local import keeps module importable without numpy

    def _q8_0(info: MiniTensorInfo):
        ne0 = info.ne[0]
        if ne0 % 32:
            raise MiniGgufError(f"{info.name}: ne0 {ne0} not divisible by 32")
        return 34, int(np.prod(info.ne)) // 32

    def _q4_0(info: MiniTensorInfo):
        ne0 = info.ne[0]
        if ne0 % 32:
            raise MiniGgufError(f"{info.name}: ne0 {ne0} not divisible by 32")
        return 18, int(np.prod(info.ne)) // 32

    def _f(info: MiniTensorInfo):
        n = int(np.prod(info.ne))
        return {0: 4, 1: 2}[info.ggml_type], n  # F32, F16

    return {8: _q8_0, 2: _q4_0, 0: _f, 1: _f}
