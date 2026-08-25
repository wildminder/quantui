"""P6.1: our byte-offset merge is functionally equivalent to the reference load-all + save_file.

The reference tool (merge-safetensors/merge_safetensors.py) loads every tensor via
``safe_open`` and writes them back with a single ``save_file``. Our
``merge_safetensors_files`` instead rewrites ``data_offsets`` and concatenates the raw
buffers. Both must yield a valid single file whose tensors decode to identical
names/shapes/values.

The reference merge here is implemented in pure stdlib (no safetensors lib needed) and
mirrors the tool's load-all-then-save semantics. If ``safetensors`` IS available in the
environment it is used as a cross-check too.
"""

import struct

import pytest

from quantui.comfy_quant_schema import (
    read_safetensors_header,
    write_safetensors,
)
from quantui.worker_ctq import merge_safetensors_files


def _safetensors_modules():
    try:
        import importlib.util as u

        return {"safetensors"} if u.find_spec("safetensors") else set()
    except Exception:  # noqa: BLE001
        return set()


def _read_header(path):
    return read_safetensors_header(path)


def _extract(path, header, data_start, name):
    s, e = header[name]["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + s)
        return fh.read(e - s)


def _build(path, tensors, metadata=None):
    specs = {n: (spec["dtype"], spec["shape"], data) for n, (spec, data) in tensors.items()}
    path.write_bytes(write_safetensors(specs, metadata=metadata))


def _make_tensors(names_values):
    """names_values: list of (name, float-list). Returns {name: (spec, bytes)}."""
    out = {}
    for name, vals in names_values:
        n = len(vals)
        data = b"".join(struct.pack("<f", v) for v in vals)
        out[name] = ({"dtype": "F32", "shape": [n], "data_offsets": [0, 4 * n]}, data)
    return out


def test_our_merge_matches_reference(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    s3 = tmp_path / "s3.safetensors"
    _build(s1, _make_tensors([("a", [1.0]), ("b", [2.0, 3.0])]))
    _build(s2, _make_tensors([("c", [4.0, 5.0, 6.0])]))
    _build(s3, _make_tensors([("d", [7.0])]))

    ours = tmp_path / "ours.safetensors"
    merge_safetensors_files([str(s1), str(s2), str(s3)], str(ours))

    # Reference: load all tensors into a dict (load-all) then re-save (save_file semantics).
    collected = {}
    order = []
    meta = None
    for sp in (s1, s2, s3):
        header, data_start = _read_header(sp)
        if meta is None and "__metadata__" in header:
            meta = header["__metadata__"]
        for name in header:
            if name == "__metadata__":
                continue
            data = _extract(str(sp), header, data_start, name)
            collected[name] = (header[name]["dtype"], header[name]["shape"], data)
            order.append(name)
    ref_specs = {n: collected[n] for n in order}
    ref_bytes = write_safetensors(ref_specs, metadata=meta)
    ref = tmp_path / "ref.safetensors"
    ref.write_bytes(ref_bytes)

    # Same set of tensor names.
    ours_header, ours_start = _read_header(str(ours))
    ref_header, ref_start = _read_header(str(ref))
    assert set(ours_header) - {"__metadata__"} == set(ref_header) - {"__metadata__"}

    # Identical values per tensor.
    for name in set(ours_header) - {"__metadata__"}:
        o = _extract(str(ours), ours_header, ours_start, name)
        r = _extract(str(ref), ref_header, ref_start, name)
        assert o == r, f"tensor {name} differs between our merge and reference"


@pytest.mark.skipif("safetensors" not in _safetensors_modules(), reason="safetensors not installed")
def test_our_merge_matches_safetensors_lib(tmp_path):
    # Cross-check against the real safetensors library when present.
    import numpy as np
    from safetensors.numpy import load_file, save_file

    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build(s1, _make_tensors([("a", [1.0]), ("b", [2.0, 3.0])]))
    _build(s2, _make_tensors([("c", [4.0, 5.0, 6.0])]))

    ours = tmp_path / "ours.safetensors"
    merge_safetensors_files([str(s1), str(s2)], str(ours))

    # Reference via safetensors lib.
    loaded = {}
    for sp in (s1, s2):
        loaded.update({k: v for k, v in load_file(str(sp)).items()})
    ref = tmp_path / "ref.safetensors"
    save_file(loaded, str(ref))

    ours_t = load_file(str(ours))
    ref_t = load_file(str(ref))
    assert set(ours_t.keys()) == set(ref_t.keys())
    for k in ours_t:
        # safetensors.numpy.load_file returns ndarrays directly (no .numpy() needed).
        assert np.allclose(ours_t[k], ref_t[k])
