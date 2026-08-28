"""Tests for :mod:`quantui.incremental_safetensors` (P1 incremental writer).

The writer is pure stdlib (no torch/safetensors import) so it can be unit-tested in a
torch-free environment, but here we also verify the *output* is loadable by the real
``safetensors`` library (the authoritative format verifier) using the user-provided venv.
"""

from __future__ import annotations

import json
import struct

import pytest
import torch
from safetensors import safe_open

from quantui.incremental_safetensors import HEADER_ALIGN, IncrementalSafetensorsWriter

# torch dtype -> safetensors header dtype string
_DTYPE_MAP = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int8: "I8",
    torch.int16: "I16",
}


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().numpy().tobytes()


def _make(name, shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed + hash(name) % 1000)
    t = torch.randn(*shape, generator=g).to(dtype)
    return name, _DTYPE_MAP[dtype], list(shape), _tensor_bytes(t), t


def _read_tensor(path, name):
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


# --------------------------------------------------------------------------- #
# 1. Round-trip: real safetensors lib can read what we wrote
# --------------------------------------------------------------------------- #
def test_roundtrip_mixed_dtypes(tmp_path):
    out = tmp_path / "model.safetensors"
    tensors = [
        _make("w1", (4, 8), torch.float32, 1),
        _make("w2", (8, 16), torch.float16, 2),
        _make("idx", (32,), torch.int64, 3),
        _make("small", (2, 2), torch.int8, 4),
    ]
    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        for name, dtype, shape, data, _ in tensors:
            assert w.add_tensor(name, dtype, shape, data) is True

    assert out.exists()
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {n for n, _, _, _, _ in tensors}
        for name, _dtype, shape, data, t in tensors:
            got = f.get_tensor(name)
            assert list(got.shape) == shape
            assert got.dtype == t.dtype
            # byte-exact reconstruction
            assert got.detach().cpu().numpy().tobytes() == data


# --------------------------------------------------------------------------- #
# 2. Header slot + data section are 8-byte aligned
# --------------------------------------------------------------------------- #
def test_slot_and_data_start_aligned(tmp_path):
    out = tmp_path / "align.safetensors"
    w = IncrementalSafetensorsWriter().open(str(out), "w", initial_slot=1 << 16)
    assert w._slot % HEADER_ALIGN == 0
    assert w.data_start % HEADER_ALIGN == 0  # 8 + slot
    # data_start must equal where the first tensor is written
    name, dtype, shape, data, _ = _make("a", (2, 2))
    w.add_tensor(name, dtype, shape, data)
    assert w._header["a"]["data_offsets"] == [0, len(data)]
    w.close()


# --------------------------------------------------------------------------- #
# 3. Tensor offsets are contiguous and gap-free
# --------------------------------------------------------------------------- #
def test_offsets_contiguous(tmp_path):
    out = tmp_path / "contig.safetensors"
    tensors = [_make(f"t{i}", (3, 3), torch.float32, i) for i in range(5)]
    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        for name, dtype, shape, data, _ in tensors:
            w.add_tensor(name, dtype, shape, data)

    # Read the header directly from the slot
    with open(str(out), "rb") as fh:
        slot = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(slot)
    header = json.loads(raw.rstrip(b" ").decode("utf-8"))
    so_far = 0
    for name in sorted(header):
        if name == "__metadata__":
            continue
        start, end = header[name]["data_offsets"]
        assert start == so_far, f"{name}: gap before start"
        so_far = end
    assert so_far == w._data_len


# --------------------------------------------------------------------------- #
# 4. Resume skips already-done tensors (idempotent add)
# --------------------------------------------------------------------------- #
def test_resume_skips_done(tmp_path):
    out = tmp_path / "resume1.safetensors"
    a = _make("a", (2, 2), torch.float32, 1)
    b = _make("b", (3, 3), torch.float32, 2)
    c = _make("c", (4, 4), torch.float32, 3)

    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        w.add_tensor(*a[:4])
        w.add_tensor(*b[:4])
    # Re-open in append mode; a/b must be skipped, c appended
    with IncrementalSafetensorsWriter().open(str(out), "a") as w:
        assert w.add_tensor(*a[:4]) is False
        assert w.add_tensor(*b[:4]) is False
        assert w.add_tensor(*c[:4]) is True

    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {"a", "b", "c"}
        assert f.get_tensor("c").detach().cpu().numpy().tobytes() == c[3]


# --------------------------------------------------------------------------- #
# 5. Multi-session resume: each committed prefix is independently loadable
# --------------------------------------------------------------------------- #
def test_multi_session_resume_prefix_valid(tmp_path):
    out = tmp_path / "multi.safetensors"
    a = _make("a", (2, 2), torch.float32, 1)
    b = _make("b", (2, 2), torch.float32, 2)
    c = _make("c", (2, 2), torch.float32, 3)

    # Session 1: write a, finalize -> committed prefix {a}
    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        w.add_tensor(*a[:4])
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {"a"}

    # Session 2 (resume): append b, finalize -> committed prefix {a,b}
    with IncrementalSafetensorsWriter().open(str(out), "a") as w:
        w.add_tensor(*b[:4])
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {"a", "b"}

    # Session 3 (resume): append c -> {a,b,c}
    with IncrementalSafetensorsWriter().open(str(out), "a") as w:
        w.add_tensor(*c[:4])
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {"a", "b", "c"}
        for n, _, _, data, _ in (a, b, c):
            assert f.get_tensor(n).detach().cpu().numpy().tobytes() == data


# --------------------------------------------------------------------------- #
# 6. Metadata survives write + resume
# --------------------------------------------------------------------------- #
def test_metadata_carried(tmp_path):
    out = tmp_path / "meta.safetensors"
    meta = {"__token__": "abc", "format": "int8", "block_size": "128"}
    a = _make("a", (2, 2), torch.float32, 1)
    with IncrementalSafetensorsWriter().open(str(out), "w", metadata=meta) as w:
        w.add_tensor(*a[:4])
    # Internal metadata preserved on resume
    with IncrementalSafetensorsWriter().open(str(out), "a") as w:
        assert w._metadata == meta
    # And the real lib reads it back
    with safe_open(str(out), framework="pt") as f:
        read_meta = f.metadata() if callable(f.metadata) else f.metadata
        assert read_meta == meta


# --------------------------------------------------------------------------- #
# 7. Header slot growth when JSON outgrows the slot
# --------------------------------------------------------------------------- #
def test_slot_growth(tmp_path):
    out = tmp_path / "grow.safetensors"
    # Small slot forces growth as the header JSON grows.
    tensors = [_make(f"layer_{i}_weight", (4, 4), torch.float32, i) for i in range(40)]
    with IncrementalSafetensorsWriter().open(str(out), "w", initial_slot=256) as w:
        for name, dtype, shape, data, _ in tensors:
            w.add_tensor(name, dtype, shape, data)
    # After growth the file must still be fully loadable & byte-exact
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {n for n, _, _, _, _ in tensors}
        for name, _dtype, _shape, data, _t in tensors:
            assert f.get_tensor(name).detach().cpu().numpy().tobytes() == data


# --------------------------------------------------------------------------- #
# 8. Duplicate add within a session is a no-op (returns False)
# --------------------------------------------------------------------------- #
def test_duplicate_in_session(tmp_path):
    out = tmp_path / "dup.safetensors"
    a = _make("a", (2, 2), torch.float32, 1)
    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        assert w.add_tensor(*a[:4]) is True
        assert w.add_tensor(*a[:4]) is False
        assert len(w.done) == 1


# --------------------------------------------------------------------------- #
# 9. Truly corrupt header slot is rejected (higher layer falls back to manifest)
# --------------------------------------------------------------------------- #
def test_corrupt_header_slot_raises(tmp_path):
    out = tmp_path / "corrupt.safetensors"
    a = _make("a", (2, 2), torch.float32, 1)
    with IncrementalSafetensorsWriter().open(str(out), "w") as w:
        w.add_tensor(*a[:4])
    # Clobber the header slot length with garbage (simulating a torn write)
    with open(str(out), "r+b") as fh:
        fh.write(struct.pack("<Q", 9_999_999_999))
    with pytest.raises(ValueError):
        IncrementalSafetensorsWriter().open(str(out), "a")


# --------------------------------------------------------------------------- #
# 10. Opening a non-existent file in "a" mode starts fresh
# --------------------------------------------------------------------------- #
def test_append_missing_file_starts_fresh(tmp_path):
    out = tmp_path / "fresh.safetensors"
    a = _make("a", (2, 2), torch.float32, 1)
    with IncrementalSafetensorsWriter().open(str(out), "a") as w:
        assert w.add_tensor(*a[:4]) is True
    with safe_open(str(out), framework="pt") as f:
        assert set(f.keys()) == {"a"}
