"""Incremental, resumable ``.safetensors`` writer (pure stdlib, no torch/safetensors).

This is the append engine for streaming quantization. A safetensors file is:

    u64 header_len | JSON header (name -> {dtype, shape, data_offsets}) | concatenated buffers

The header is kept in memory and materialized into a FIXED-SIZE "slot" at the front of the
file (8-byte aligned). Tensor buffers are appended after the slot. Because the slot size is
fixed, the header can be rewritten **in place** without shifting the data region -- so a
crash mid-run always leaves a loadable prefix. The data section starts at ``8 + slot`` and
grows only at the end, so appending is O(1) and memory stays bounded (one tensor at a time).

Resume: open an existing file with ``mode="a"``; the writer re-reads the header slot to learn
which tensors already exist and where the data region ends, then continues appending. A
separate manifest (see ``stream_quant``) is the authoritative checkpoint, but the output file
itself is always self-consistent.
"""

from __future__ import annotations

import json
import os
import struct

HEADER_ALIGN = 8


class IncrementalSafetensorsWriter:
    """Append tensors to a ``.safetensors`` file, resumable across process restarts.

    Usage::

        with IncrementalSafetensorsWriter().open(out_path, "w") as w:
            for name, dtype, shape, data in produce_tensors():
                w.add_tensor(name, dtype, shape, data)
        # file is finalized on close()

    Resume::

        with IncrementalSafetensorsWriter().open(out_path, "a") as w:
            for name in remaining:
                w.add_tensor(name, ...)   # already-written names are skipped
    """

    def __init__(self) -> None:
        self._path: str | None = None
        self._fh = None
        self._slot: int = 0
        self._data_len: int = 0
        self._header: dict = {}
        self._metadata: dict | None = None
        self._done: set[str] = set()
        self._initial_slot: int = 1 << 16  # 64 KiB; grows if the header outgrows it
        self._flush_every: int = 1  # rewrite header slot after every N add_tensor calls
        self._pending: int = 0

    # -- context manager ---------------------------------------------------- #
    def __enter__(self) -> IncrementalSafetensorsWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- properties --------------------------------------------------------- #
    @property
    def done(self) -> set[str]:
        return set(self._done)

    @property
    def data_start(self) -> int:
        return 8 + self._slot

    # -- open / resume ------------------------------------------------------ #
    def open(
        self,
        path: str,
        mode: str = "w",
        metadata: dict | None = None,
        initial_slot: int = 1 << 16,
        flush_every: int = 1,
    ) -> IncrementalSafetensorsWriter:
        """Open ``path`` for incremental writing.

        ``mode="w"`` starts a new file (existing file is truncated). ``mode="a"`` resumes an
        existing file: the header slot is parsed and the data cursor is positioned at the
        current end of the data region. Tensors already present are recorded as done so
        subsequent ``add_tensor`` calls for them are no-ops.
        """
        self._initial_slot = max(HEADER_ALIGN, (initial_slot // HEADER_ALIGN) * HEADER_ALIGN)
        self._flush_every = max(1, flush_every)
        self._pending = 0
        if mode == "a" and os.path.exists(path) and os.path.getsize(path) > 0:
            self._resume(path)
            return self

        self._path = path
        self._metadata = metadata
        self._slot = self._initial_slot
        self._data_len = 0
        self._header = {}
        if metadata:
            self._header["__metadata__"] = metadata
        self._done = {k for k in self._header if k != "__metadata__"}
        self._fh = open(path, "wb")
        self._write_header_slot()
        return self

    def _resume(self, path: str) -> None:
        size = os.path.getsize(path)
        if size < 8:
            raise ValueError(f"cannot resume {path!r}: file too small")
        with open(path, "rb") as fh:
            slot = struct.unpack("<Q", fh.read(8))[0]
            # Guard against a corrupt slot value (e.g. partial header write).
            if slot < HEADER_ALIGN or slot > size - 8 or slot % HEADER_ALIGN != 0:
                raise ValueError(f"cannot resume {path!r}: invalid header slot {slot}")
            raw = fh.read(slot)
        # The header JSON is padded with spaces to fill the slot. Trailing spaces are
        # valid JSON whitespace, so rstrip then parse directly (no fragile brace-splitting).
        self._header = json.loads(raw.rstrip(b" ").decode("utf-8"))
        self._slot = slot
        self._metadata = self._header.get("__metadata__")
        size = os.path.getsize(path)
        self._data_len = size - (8 + slot)
        if self._data_len < 0:
            self._data_len = 0
        self._done = {k for k in self._header if k != "__metadata__"}
        self._path = path
        self._fh = open(path, "r+b")
        self._fh.seek(0, os.SEEK_END)

    # -- internals ---------------------------------------------------------- #
    def _serialize_header(self) -> bytes:
        return json.dumps(self._header, separators=(",", ":")).encode("utf-8")

    def _write_header_slot(self) -> None:
        # Flush any buffered tensor data FIRST: _grow_slot() reopens the file with "wb"
        # (truncating it), so the data must be on disk before that happens.
        if self._fh is not None:
            self._fh.flush()
        hdr = self._serialize_header()
        if len(hdr) > self._slot:
            self._grow_slot(len(hdr))
            hdr = self._serialize_header()
        padded = hdr + b" " * (self._slot - len(hdr))
        self._fh.seek(0)
        self._fh.write(struct.pack("<Q", self._slot))
        self._fh.write(padded)
        self._fh.seek(0, os.SEEK_END)
        self._fh.flush()

    def _grow_slot(self, needed: int) -> None:
        """Rewrite the whole file with a larger header slot (rare; header outgrew slot)."""
        new_slot = self._slot
        while new_slot < needed + HEADER_ALIGN:
            new_slot *= 2
        new_slot = (new_slot // HEADER_ALIGN) * HEADER_ALIGN
        data_start = 8 + self._slot
        with open(self._path, "rb") as fh:
            fh.seek(data_start)
            data = fh.read()
        self._slot = new_slot
        if self._fh is not None:
            self._fh.close()
        self._fh = open(self._path, "wb")
        self._write_header_slot()
        self._fh.seek(8 + self._slot)
        self._fh.write(data)
        self._fh.seek(0, os.SEEK_END)

    # -- public API --------------------------------------------------------- #
    def add_tensor(self, name: str, dtype: str, shape: list[int], data: bytes) -> bool:
        """Append ``data`` as tensor ``name``. Returns True if written, False if skipped
        (already present from a prior run). Offsets are relative to the data section start.
        """
        if name in self._done:
            return False
        start = self._data_len
        end = start + len(data)
        self._fh.seek(self.data_start + start)
        self._fh.write(data)
        self._header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, end],
        }
        self._data_len = end
        self._done.add(name)
        self._pending += 1
        if self._pending >= self._flush_every:
            self._write_header_slot()
            self._pending = 0
        return True

    def finalize(self) -> None:
        """Rewrite the header slot one last time and flush/close the file."""
        if self._fh is None:
            return
        self._write_header_slot()
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
            self._fh = None

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self.finalize()
