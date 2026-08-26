"""STEP 3.1 (plan 2026-08-26): bf16/fp16 cast-only formats end-to-end.

Registry entries `bf16` / `fp16` (Backend.CTQ, quant tag = format id), worker
short-circuit `--cast_dtype {bfloat16,float16}` checked before any
convert_to_quant import (same pattern as `--combine`), and builder emission of
ONLY the cast flag (the cast path is inferred from --cast_dtype alone).
"""

import json
import struct
import subprocess
import sys

from quantui import run_config as rc
from quantui.comfy_quant_schema import write_safetensors

WORKER_MODULE = "quantui.worker_ctq"


# ---- registry + tags -------------------------------------------------------- #

def test_registry_bf16_fp16_entries():
    from quantui.quant_methods import COMFY_FORMATS, Backend, comfy_format

    bf16 = comfy_format("bf16")
    assert bf16.backend == Backend.CTQ
    assert bf16.base_flags == ["--cast_dtype", "bfloat16"]
    assert bf16.quant_format is None
    assert bf16.extra_options == []
    assert bf16.needs == []

    fp16 = comfy_format("fp16")
    assert fp16.backend == Backend.CTQ
    assert fp16.base_flags == ["--cast_dtype", "float16"]
    assert fp16.quant_format is None
    assert fp16.extra_options == []
    assert fp16.needs == []

    ids = [f.id for f in COMFY_FORMATS]
    assert len(ids) == len(set(ids))
    # Plan §3.1: registered set grows to 9.
    assert set(ids) == {
        "fp8_e4m3", "int8", "nvfp4", "mxfp8",
        "w4a4_convrot", "w4a8_asym", "combine", "bf16", "fp16",
    }


def test_ctq_quant_tags_bf16_fp16():
    assert rc.ctq_quant_tags("bf16") == ["bf16"]
    assert rc.ctq_quant_tags("fp16") == ["fp16"]


def test_build_ctq_cmd_bf16_fp16_emit_cast_flag(tmp_path):
    m = tmp_path / "model.safetensors"
    m.write_text("x")

    for fmt, dtype in (("bf16", "bfloat16"), ("fp16", "float16")):
        c = rc.CtqConfig(
            input=str(m), output=str(tmp_path / f"model-{fmt}.safetensors"),
            pybin=sys.executable, format=fmt, output_mode="sharded",
            quant_tags=rc.ctq_quant_tags(fmt),
        )
        cmd = rc.build_ctq_cmd(c)
        assert cmd[2] == "quantui.worker_ctq"
        i = cmd.index("--cast_dtype")
        assert cmd[i + 1] == dtype
        # No stray quant flags: only --cast_dtype plus the mandatory
        # --output-mode envelope reach the worker.
        stray = [t for t in cmd if t in (
            "--combine", "--int8", "--nvfp4", "--mxfp8", "--comfy_quant",
            "--scaling_mode", "--convrot",
        )]
        assert not stray, f"{fmt}: stray flags {stray} in {cmd}"


def test_validate_ctq_bf16_fp16_require_safetensors_output(tmp_path):
    from quantui.run_config import validate_ctq

    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "outdir"), pybin=sys.executable,
        format="bf16", output_mode="single",
    )
    errs = validate_ctq(c)
    assert any(".safetensors" in e for e in errs), errs

    c_ok = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-bf16.safetensors"),
        pybin=sys.executable, format="bf16", output_mode="single",
    )
    assert validate_ctq(c_ok) == []


# ---- worker subprocess end-to-end ------------------------------------------- #

def _read_tensors(path):
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


def _make_sharded(src):
    src.mkdir(parents=True)
    s1 = {"t.a": ("F32", [2], struct.pack("<ff", 1.0, 2.0)),
          "t.i": ("I64", [1], struct.pack("<q", 7))}
    s2 = {"t.b": ("F32", [1], struct.pack("<f", 3.0))}
    (src / "model-00001-of-00002.safetensors").write_bytes(write_safetensors(s1))
    (src / "model-00002-of-00002.safetensors").write_bytes(write_safetensors(s2))
    (src / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"t.a": "model-00001-of-00002.safetensors",
                        "t.i": "model-00001-of-00002.safetensors",
                        "t.b": "model-00002-of-00002.safetensors"},
    }))


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, *args],
        capture_output=True, text=True, check=False,
    )


def test_cast_flag_end_to_end_bf16(tmp_path):
    src = tmp_path / "mymodel"
    _make_sharded(src)
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(src), "-o", str(out), "--cast_dtype", "bfloat16"])
    assert cp.returncode == 0, cp.stderr
    header, tensors = _read_tensors(out)

    assert tensors["t.a"][0] == "BF16"
    assert len(tensors["t.a"][2]) == 4  # two bf16 values
    assert tensors["t.b"][0] == "BF16"
    # Integer tensor passes through UNCHANGED with original dtype.
    assert tensors["t.i"] == ("I64", [1], struct.pack("<q", 7))

    prev = 0
    for name in ("t.a", "t.i", "t.b"):
        a, b = header[name]["data_offsets"]
        assert a == prev
        prev = b
    assert "DONE" in cp.stdout
    assert "PROGRESS" in cp.stdout and "cast" in cp.stdout


def test_cast_flag_end_to_end_fp16(tmp_path):
    inp = tmp_path / "in.safetensors"
    inp.write_bytes(write_safetensors(
        {"w": ("F32", [2], struct.pack("<ff", 1.0, -2.0))}))
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(inp), "-o", str(out), "--cast_dtype", "float16"])
    assert cp.returncode == 0, cp.stderr
    _, tensors = _read_tensors(out)
    assert tensors["w"] == ("F16", [2], struct.pack("<hh", 15360, -16384))


def test_cast_single_file_input_also_works(tmp_path):
    inp = tmp_path / "in.safetensors"
    inp.write_bytes(write_safetensors(
        {"w": ("F32", [1], struct.pack("<f", 1.5))}))
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(inp), "-o", str(out), "--cast_dtype", "bfloat16"])
    assert cp.returncode == 0, cp.stderr
    _, tensors = _read_tensors(out)
    assert tensors["w"][0] == "BF16"
    assert len(tensors["w"][2]) == 2


def test_cast_requires_safetensors_extension_for_sharded(tmp_path):
    src = tmp_path / "mymodel"
    _make_sharded(src)
    out = tmp_path / "not-a-st"  # no .safetensors extension

    cp = _run(["-i", str(src), "-o", str(out), "--cast_dtype", "bfloat16",
               "--output-mode", "single"])
    assert cp.returncode != 0
    combined = cp.stderr + cp.stdout
    assert ".safetensors" in combined


def test_cast_identity_case_byte_identical(tmp_path):
    # Input already BF16 (+ an int tensor) -> payload copied byte-identical at
    # file level (deterministic single-shard header layout).
    specs = {
        "a": ("BF16", [2], b"\x00\x3f\x00\x40"),
        "i": ("I8", [1], b"\x05"),
    }
    inp = tmp_path / "in.safetensors"
    inp.write_bytes(write_safetensors(specs))
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(inp), "-o", str(out), "--cast_dtype", "bfloat16"])
    assert cp.returncode == 0, cp.stderr
    assert out.read_bytes() == inp.read_bytes()
