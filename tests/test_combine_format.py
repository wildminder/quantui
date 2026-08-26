"""STEP 1.1 (plan 2026-08-26): --combine replaces --passthrough.

Combine ALWAYS merges sharded input into ONE .safetensors (rev. 2 user decision:
output mode is irrelevant); a single-file input is copied byte-identical.
No .comfy_quant tensor is baked; convert_to_quant is never imported.
"""

import json
import struct
import subprocess
import sys

from quantui import run_config as rc
from quantui.comfy_quant_schema import write_safetensors

WORKER_MODULE = "quantui.worker_ctq"


def _write_plain_safetensors(path, tensor_name="model.weight"):
    specs = {tensor_name: ("F32", [2], struct.pack("<ff", 1.0, 2.0))}
    path.write_bytes(write_safetensors(specs))


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, *args], capture_output=True, text=True, check=False
    )


def _make_sharded(src):
    """2 shards / 3 tensors + config.json + index json."""
    src.mkdir(parents=True)
    s1 = {"t.a": ("F32", [2], struct.pack("<ff", 1.0, 2.0)),
          "t.b": ("F32", [1], struct.pack("<f", 3.0))}
    s2 = {"t.c": ("F32", [3], struct.pack("<fff", 4.0, 5.0, 6.0))}
    (src / "model-00001-of-00002.safetensors").write_bytes(write_safetensors(s1))
    (src / "model-00002-of-00002.safetensors").write_bytes(write_safetensors(s2))
    (src / "config.json").write_text("{}")
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 36},
        "weight_map": {"t.a": "model-00001-of-00002.safetensors",
                        "t.b": "model-00001-of-00002.safetensors",
                        "t.c": "model-00002-of-00002.safetensors"},
    }))


def _read_tensors(path):
    """Return (header, {name: (dtype, shape, payload_bytes)}) from a safetensors file.

    Per the safetensors spec, ``data_offsets`` are relative to the END of the
    8-byte length prefix + header blob (i.e. the start of the data region).
    """
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


def test_combine_single_file_copies_bytes(tmp_path):
    inp = tmp_path / "in.safetensors"
    _write_plain_safetensors(inp)
    src_bytes = inp.read_bytes()
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(inp), "-o", str(out), "--combine"])
    assert cp.returncode == 0, cp.stderr
    assert out.is_file()
    assert out.read_bytes() == src_bytes
    assert "DONE" in cp.stdout


def test_combine_sharded_merges_regardless_of_output_mode(tmp_path):
    src = tmp_path / "mymodel"
    _make_sharded(src)

    # Collect the shard payloads in shard order for the expected bytes.
    order = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    payloads = {}
    for shard in order:
        _, tensors = _read_tensors(src / shard)
        payloads.update(tensors)

    for mode in ("single", "sharded"):  # output mode MUST be ignored
        out = tmp_path / f"merged-{mode}.safetensors"
        cp = _run(["-i", str(src), "-o", str(out), "--combine",
                   "--output-mode", mode])
        assert cp.returncode == 0, cp.stderr
        assert out.is_file(), f"mode={mode}: expected ONE merged file"

        header, tensors = _read_tensors(out)
        tensor_keys = sorted(tensors)
        assert tensor_keys == ["t.a", "t.b", "t.c"]
        # NO quantization metadata baked.
        assert not any(k.endswith(".comfy_quant") for k in header)
        prev_end = 0
        for name in tensor_keys:
            spec = header[name]
            a, b = spec["data_offsets"]
            assert a == prev_end, f"non-contiguous offsets at {name}"
            prev_end = b
            got_dtype, got_shape, got_data = tensors[name]
            exp_dtype, exp_shape, exp_data = payloads[name]
            assert got_dtype == exp_dtype and got_shape == exp_shape
            assert got_data == exp_data, name
        assert "DONE" in cp.stdout


def test_combine_merge_rejects_duplicate_tensors(tmp_path):
    src = tmp_path / "dup"
    src.mkdir()
    dup_spec = {"t.x": ("F32", [1], struct.pack("<f", 9.0))}
    (src / "a.safetensors").write_bytes(write_safetensors(dup_spec))
    (src / "b.safetensors").write_bytes(write_safetensors(dup_spec))
    # weight_map must reference BOTH shard files so discover_shards yields both;
    # each shard physically contains "t.x" -> merge_safetensors_files must raise.
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"t.x": "a.safetensors", "t.y": "b.safetensors"},
    }))
    out = tmp_path / "out.safetensors"
    # Both shards carry t.x -> merger must refuse (index lists only one, but the
    # worker merges by discovered shard FILES; duplicate names must be an error).
    cp = _run(["-i", str(src), "-o", str(out), "--combine"])
    assert cp.returncode != 0
    assert "duplicate tensor" in (cp.stderr + cp.stdout)


def test_combine_emits_progress_lines(tmp_path):
    src = tmp_path / "prog"
    _make_sharded(src)
    out = tmp_path / "out.safetensors"
    cp = _run(["-i", str(src), "-o", str(out), "--combine"])
    assert cp.returncode == 0, cp.stderr
    assert "PROGRESS" in cp.stdout and "merge" in cp.stdout


def test_build_ctq_cmd_combine_emits_flag_and_ctq_worker():
    c = rc.CtqConfig(
        input="C:/nonexistent/model", output="C:/nonexistent/o.safetensors",
        pybin=sys.executable, format="combine", output_mode="single",
        quant_tags=rc.ctq_quant_tags("combine"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert "--combine" in cmd
    assert cmd[2] == "quantui.worker_ctq"
