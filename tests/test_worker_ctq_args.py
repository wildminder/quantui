"""Tests for worker_ctq.py: arg parsing, resolve_input, build_quantize_kwargs, and an
integration run with a mocked ``quantize`` (fake convert_to_quant on PYTHONPATH).

These tests must pass in a sandbox where the real ``convert_to_quant`` is NOT installed:
the integration test injects a fake module via PYTHONPATH, and unit tests either call pure
helpers or monkeypatch ``sys.modules``.
"""

import json
import os
import struct
import subprocess
import sys
import types

import pytest

from quantui import worker_ctq as wmod
from quantui.stream_parser import parse_ctq_progress
from quantui.worker_ctq import (
    ShardedModel,
    _expected_output_bytes,
    _run_quantize_with_progress,
    build_quantize_kwargs,
    discover_shards,
    main,
    merge_safetensors_files,
    parse_args,
    progress,
    quantize_shards,
    resolve_input,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER_MODULE = "quantui.worker_ctq"

FAKE_CTQ = '''
import json


def quantize(**kwargs):
    print("PROGRESS: loading model", flush=True)
    print("PROGRESS: quantizing weights", flush=True)
    print("KWARGS:" + json.dumps(kwargs), flush=True)
    print("DONE", flush=True)
'''


def _run(args, env=None):
    return subprocess.run([sys.executable, "-m", WORKER_MODULE, *args], capture_output=True, text=True, env=env)


# ---- Worker progress envelope (drives the TUI ProgressBar) ------------------ #

def test_progress_emits_parseable_envelope(capsys):
    # The worker's progress() helper must emit a CTQ_PROGRESS envelope that the TUI
    # parser classifies as progress and the store turns into a determinate state.
    progress("shard", cur=1, total=3, label="Quantizing x")
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("CTQ_PROGRESS ")]
    assert lines, out
    obj = parse_ctq_progress(lines[0])
    assert obj == {"phase": "shard", "cur": 1, "total": 3, "label": "Quantizing x"}


def test_progress_handles_partial_payload(capsys):
    # pct-only (no cur/total) is still a valid envelope.
    progress("calib", pct=50.0)
    out = capsys.readouterr().out
    obj = parse_ctq_progress([line for line in out.splitlines() if line.startswith("CTQ_PROGRESS ")][0])
    assert obj == {"phase": "calib", "pct": 50.0}


# ---- B1: arg parsing + resolve_input (no arch guard) ----------------------- #

def test_help_lists_flags():
    cp = _run(["--help"])
    assert cp.returncode == 0, cp.stderr
    for tok in ["--input", "--output", "--int8", "--nvfp4", "--convrot",
                "--scaling_mode", "--mxfp8", "--flux2", "--calib_samples"]:
        assert tok in cp.stdout


def test_resolve_input_bad_extension(tmp_path):
    cp = _run(["--input", str(tmp_path / "x.bin"), "--output", str(tmp_path / "o.safetensors")])
    assert cp.returncode == 1
    assert "ERROR:" in cp.stderr


def test_resolve_input_missing(tmp_path):
    cp = _run(["--input", str(tmp_path / "missing.safetensors"), "--output", str(tmp_path / "o.safetensors")])
    assert cp.returncode == 1
    assert "ERROR:" in cp.stderr


def test_resolve_input_ok_and_folder():
    # success cases via direct import (fail() would sys.exit in-process otherwise)
    import tempfile
    d = tempfile.mkdtemp()
    single = os.path.join(d, "ok.safetensors")
    with open(single, "w") as f:
        f.write("x")
    assert resolve_input(single) == single
    # folder with exactly one .safetensors -> returns the folder path
    assert resolve_input(d) == d


def test_resolve_input_sharded_dir(tmp_path):
    # A HuggingFace sharded folder (with model.safetensors.index.json) is accepted.
    d = tmp_path / "sh"
    d.mkdir()
    (d / "model.safetensors.index.json").write_text("{}")
    (d / "model-00001-of-00002.safetensors").write_text("x")
    assert resolve_input(str(d)) == str(d)


# ---- B2: build_quantize_kwargs + mocked quantize --------------------------- #

def _ns(**kw):
    base = dict(
        input="in.safetensors", output="out.safetensors",
        int8=False, scaling_mode=None, convrot=False, convrot_group_size=None,
        nvfp4=False, mxfp8=False, comfy_quant=False, save_quant_metadata=False,
        simple=False, low_memory=False, verbose=False, heur=False, no_stream=False,
        flux2=False, wan=False, t5xxl=False, hunyuan=False, zimage=False,
        calib_samples=None, exclude_layers=None, block_size=None, manual_seed=None,
        output_dtype=None, num_iter=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_build_kwargs():
    args = _ns(int8=True, scaling_mode="row", convrot=True, convrot_group_size="256",
               flux2=True, calib_samples="8", comfy_quant=True, save_quant_metadata=True)
    kw = build_quantize_kwargs(args)
    assert kw["int8"] is True
    assert kw["scaling_mode"] == "row"
    assert kw["convrot"] is True
    assert kw["convrot_group_size"] == 256
    assert kw["flux2"] is True
    assert kw["calib_samples"] == 8
    assert kw["comfy_quant"] is True
    assert kw["save_quant_metadata"] is True


def test_build_kwargs_verbose_mapping():
    # Regression: verbose must map to a string token accepted by setup_logging
    # ({"DEBUG","VERBOSE","NORMAL","MINIMAL"}), NOT a bool. The worker CLI
    # --verbose is store_true, so args.verbose is always a bool; passing bool
    # downstream broke EVERY run via "AttributeError: 'bool' object has no
    # attribute 'upper'".
    kw_off = build_quantize_kwargs(_ns(verbose=False))
    assert kw_off["verbose"] == "NORMAL"
    kw_on = build_quantize_kwargs(_ns(verbose=True))
    assert kw_on["verbose"] == "VERBOSE"


def test_build_kwargs_drops_none_to_honor_library_defaults():
    # Optional args left unset must be OMITTED (not passed as None), otherwise they
    # override convert_to_quant's parser defaults (calib_samples=3072,
    # convrot_group_size=256) and crash downstream (e.g. torch.randn(None, ...)).
    kw = build_quantize_kwargs(_ns())
    for key in ("calib_samples", "convrot_group_size", "block_size", "exclude_layers", "scaling_mode"):
        assert key not in kw, f"{key} should be omitted, not None"
    # explicit values are still forwarded (as the right type)
    assert build_quantize_kwargs(_ns(calib_samples="16"))["calib_samples"] == 16
    assert build_quantize_kwargs(_ns(convrot_group_size="64"))["convrot_group_size"] == 64


def test_build_kwargs_num_iter_passthrough():
    # num_iter: unset -> omitted (ctq keeps its 4000 default); set -> int forwarded.
    kw = build_quantize_kwargs(_ns())
    assert "num_iter" not in kw, kw
    assert build_quantize_kwargs(_ns(num_iter="1000"))["num_iter"] == 1000


def test_parse_args_accepts_num_iter():
    # The CLI flag exists in both spellings and defaults to None. The raw value is a
    # string; build_quantize_kwargs does the int() coercion.
    args = parse_args(["-i", "in.safetensors", "-o", "out.safetensors"])
    assert args.num_iter is None
    args2 = parse_args(["-i", "a", "-o", "b", "--num_iter", "500"])
    assert args2.num_iter == "500"
    args3 = parse_args(["-i", "a", "-o", "b", "--num-iter", "750"])
    assert args3.num_iter == "750"


def test_main_calls_quantize(monkeypatch, capsys, tmp_path):
    calls = {}

    def fake_quantize(**kwargs):
        calls["kwargs"] = kwargs
        print("DONE", flush=True)

    mod = types.ModuleType("convert_to_quant")
    mod.quantize = fake_quantize
    monkeypatch.setitem(sys.modules, "convert_to_quant", mod)

    inp = tmp_path / "m.safetensors"
    inp.write_text("x")
    out = tmp_path / "o.safetensors"
    monkeypatch.setattr(
        sys, "argv",
        ["quantui.worker_ctq", "-i", str(inp), "-o", str(out),
         "--int8", "--scaling_mode", "row", "--convrot", "--convrot_group_size", "256",
         "--flux2", "--comfy_quant", "--save_quant_metadata", "--calib_samples", "8"],
    )
    main()
    assert "kwargs" in calls
    k = calls["kwargs"]
    assert k["int8"] is True
    assert k["scaling_mode"] == "row"
    assert k["convrot"] is True
    assert k["convrot_group_size"] == 256
    assert k["flux2"] is True
    assert k["calib_samples"] == 8
    assert k["input"] == str(inp)
    assert "DONE" in capsys.readouterr().out


# ---- B3: integration run with fake convert_to_quant on PYTHONPATH ---------- #

@pytest.fixture
def fake_ctq(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "convert_to_quant.py").write_text(FAKE_CTQ)
    return str(d)


def test_worker_integration(tmp_path, fake_ctq):
    inp = tmp_path / "model.safetensors"
    inp.write_text("x")
    out = tmp_path / "model-int8-row-convrot-gs256.safetensors"
    env = dict(os.environ)
    env["PYTHONPATH"] = fake_ctq + os.pathsep + env.get("PYTHONPATH", "")
    cp = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "-i", str(inp), "-o", str(out),
         "--int8", "--scaling_mode", "row", "--convrot", "--convrot_group_size", "256",
         "--flux2", "--comfy_quant", "--save_quant_metadata"],
        capture_output=True, text=True, env=env,
    )
    assert cp.returncode == 0, cp.stderr
    assert "PROGRESS: quantizing weights" in cp.stdout
    assert "DONE" in cp.stdout

    kw = None
    for line in cp.stdout.splitlines():
        if line.startswith("KWARGS:"):
            kw = json.loads(line[len("KWARGS:"):])
    assert kw is not None
    assert kw["int8"] is True
    assert kw["scaling_mode"] == "row"
    assert kw["convrot"] is True
    assert kw["convrot_group_size"] == 256
    assert kw["flux2"] is True
    assert kw["input"] == str(inp)
    assert kw["output"] == str(out)


# --------------------------------------------------------------------------- #
# B1: discover_shards (sharded data layer, ADDITIVE)
# --------------------------------------------------------------------------- #
def _make_sharded(d, weight_map, extra=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    shards = list(dict.fromkeys(weight_map.values()))
    for s in shards:
        (d / s).write_text("x")
    if extra:
        for name, body in extra.items():
            (d / name).write_text(body)
    return shards


def test_discover_shards(tmp_path):
    d = tmp_path / "m"
    wm = {
        "t0": "model-00001-of-00002.safetensors",
        "t1": "model-00001-of-00002.safetensors",  # same shard, must dedupe
        "t2": "model-00002-of-00002.safetensors",
    }
    _make_sharded(d, wm, extra={"config.json": "{}", "extra.safetensors": "orphan"})
    model = discover_shards(str(d))
    assert isinstance(model, ShardedModel)
    # order-preserving, unique shard list
    assert model.shard_files == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    # orphan .safetensors is ignored; config.json is a non-weight sidecar
    assert model.non_weight_files == ["config.json"]
    assert model.index_path == str(d / "model.safetensors.index.json")
    assert model.weight_map == wm


def test_discover_shards_missing_index(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.safetensors").write_text("x")
    with pytest.raises(FileNotFoundError):
        discover_shards(str(d))


# --------------------------------------------------------------------------- #
# Feature B: merge_safetensors_files (pure-Python byte-level merge)
# --------------------------------------------------------------------------- #
def _build_safetensors(path, tensors):
    """Build a valid safetensors file. ``tensors``: name -> bytes (length % 4 == 0)."""
    header = {}
    offset = 0
    for name, data in tensors.items():
        n = len(data) // 4
        header[name] = {"dtype": "F32", "shape": [n], "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for data in tensors.values():
            fh.write(data)


def _parse_header(path):
    with open(path, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(hdr_len).decode("utf-8"))


def test_merge_safetensors_files(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    s3 = tmp_path / "s3.safetensors"
    _build_safetensors(s1, {"a": b"\x00\x00\x80\x3f" * 1})  # 1 float
    _build_safetensors(s2, {"b": b"\x00\x00\x00\x40" * 2})  # 2 floats
    _build_safetensors(s3, {"c": b"\x00\x00\x40\x40" * 3})  # 3 floats
    out = tmp_path / "merged.safetensors"
    merge_safetensors_files([str(s1), str(s2), str(s3)], str(out))

    # ONE valid safetensors: 8-byte length + parseable JSON header.
    assert out.is_file()
    header = _parse_header(str(out))
    assert set(header.keys()) == {"a", "b", "c"}

    # Each tensor's payload bytes equal the original shard payload (byte equality).
    ref = {
        "a": b"\x00\x00\x80\x3f" * 1,
        "b": b"\x00\x00\x00\x40" * 2,
        "c": b"\x00\x00\x40\x40" * 3,
    }
    # Re-extract payloads from the merged file by data_offsets.
    merged_bytes = out.read_bytes()
    data_start = 8 + struct.unpack("<Q", merged_bytes[:8])[0]
    for name, spec in header.items():
        s, e = spec["data_offsets"]
        assert merged_bytes[data_start + s:data_start + e] == ref[name]

    # Optional round-trip via the real safetensors loader (skipped if absent).
    pytest.importorskip("safetensors")
    import numpy as np
    from safetensors.numpy import load_file
    loaded = load_file(str(out))
    for name in ("a", "b", "c"):
        assert name in loaded
        assert loaded[name].dtype == np.float32


def test_merge_offsets_cumulative(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build_safetensors(s1, {"a": b"\x01" * 4, "b": b"\x02" * 8})
    _build_safetensors(s2, {"c": b"\x03" * 12})
    out = tmp_path / "merged.safetensors"
    merge_safetensors_files([str(s1), str(s2)], str(out))

    header = _parse_header(str(out))
    offsets = [spec["data_offsets"] for spec in header.values()]
    offsets.sort()

    # strictly non-overlapping + cumulative: each start == previous end.
    # strict=False: offsets[1:] is intentionally one shorter (the last span has
    # no successor to compare against).
    for (_, prev_end), (cur_start, _) in zip(offsets, offsets[1:], strict=False):
        assert cur_start == prev_end, (offsets,)
    total = offsets[-1][1]
    # total data length == sum of shard buffer lengths
    expected = (
        os.path.getsize(s1) - (8 + len(json.dumps(_parse_header(str(s1))).encode("utf-8")))
        + os.path.getsize(s2) - (8 + len(json.dumps(_parse_header(str(s2))).encode("utf-8")))
    )
    assert total == expected


def test_merge_rejects_duplicate(tmp_path):
    s1 = tmp_path / "s1.safetensors"
    s2 = tmp_path / "s2.safetensors"
    _build_safetensors(s1, {"a": b"\x01" * 4})
    _build_safetensors(s2, {"a": b"\x02" * 4})  # duplicate name
    out = tmp_path / "merged.safetensors"
    with pytest.raises(ValueError):
        merge_safetensors_files([str(s1), str(s2)], str(out))


# --------------------------------------------------------------------------- #
# C1: quantize_shards (loop per shard, identical flags) + sharded main()
# --------------------------------------------------------------------------- #
def test_quantize_shards(tmp_path):
    d = tmp_path / "m"
    wm = {
        "t0": "model-00001-of-00002.safetensors",
        "t1": "model-00002-of-00002.safetensors",
    }
    _make_sharded(d, wm, extra={"config.json": "{}"})
    model = discover_shards(str(d))
    out = tmp_path / "out"

    calls = []

    def fake_quantize(**kwargs):
        calls.append(dict(kwargs))
        os.makedirs(os.path.dirname(kwargs["output"]), exist_ok=True)
        with open(kwargs["output"], "w") as fh:
            fh.write("quantized")

    base_kwargs = {"int8": True, "comfy_quant": True, "input": "IGN", "output": "IGN"}
    quantize_shards(model, str(out), base_kwargs, fake_quantize)

    # exactly one quantize() per shard
    assert len(calls) == 2
    # IDENTICAL flags across shards (ignoring per-shard input/output)
    flags0 = {k: v for k, v in calls[0].items() if k not in ("input", "output")}
    flags1 = {k: v for k, v in calls[1].items() if k not in ("input", "output")}
    assert flags0 == flags1 == {"int8": True, "comfy_quant": True}
    # per-shard input/output point at the right files
    assert calls[0]["input"] == str(d / "model-00001-of-00002.safetensors")
    assert calls[0]["output"] == str(out / "model-00001-of-00002.safetensors")
    assert calls[1]["input"] == str(d / "model-00002-of-00002.safetensors")
    # index json copied UNCHANGED + non-weight sidecars copied verbatim
    assert (out / "model.safetensors.index.json").is_file()
    assert (out / "config.json").is_file()
    idx_out = json.loads((out / "model.safetensors.index.json").read_text())
    assert idx_out["weight_map"] == wm


# ---- C1: integration run with a sharded fake convert_to_quant on PYTHONPATH -- #

FAKE_CTQ_SHARDED = '''
import os
import json


def quantize(**kwargs):
    print("PROGRESS: loading shard", flush=True)
    out = kwargs.get("output")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("quantized")
    print("KWARGS:" + json.dumps(kwargs), flush=True)
    print("DONE", flush=True)
'''


@pytest.fixture
def fake_ctq_sharded(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "convert_to_quant.py").write_text(FAKE_CTQ_SHARDED)
    return str(d)


def test_worker_sharded_integration(tmp_path, fake_ctq_sharded):
    d = tmp_path / "m"
    wm = {
        "t0": "model-00001-of-00002.safetensors",
        "t1": "model-00002-of-00002.safetensors",
    }
    _make_sharded(d, wm, extra={"config.json": "{}"})
    out = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = fake_ctq_sharded + os.pathsep + env.get("PYTHONPATH", "")
    cp = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "-i", str(d), "-o", str(out),
         "--nvfp4", "--comfy_quant", "--save_quant_metadata"],
        capture_output=True, text=True, env=env,
    )
    assert cp.returncode == 0, cp.stderr
    # quantized shards + copied index json + config.json
    assert (out / "model-00001-of-00002.safetensors").is_file()
    assert (out / "model-00002-of-00002.safetensors").is_file()
    assert (out / "model.safetensors.index.json").is_file()
    assert (out / "config.json").is_file()
    # per-shard progress + DONE
    assert "[1/2]" in cp.stdout
    assert "[2/2]" in cp.stdout
    assert "DONE" in cp.stdout


def test_worker_sharded_rejects_file_output(tmp_path, fake_ctq_sharded):
    d = tmp_path / "m"
    wm = {"t0": "model-00001-of-00002.safetensors"}
    _make_sharded(d, wm)
    out_file = tmp_path / "out.safetensors"
    env = dict(os.environ)
    env["PYTHONPATH"] = fake_ctq_sharded + os.pathsep + env.get("PYTHONPATH", "")
    cp = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "-i", str(d), "-o", str(out_file), "--int8"],
        capture_output=True, text=True, env=env,
    )
    assert cp.returncode == 1
    assert "Sharded output must be a directory" in cp.stderr


# ---- C1: single-file merge integration (merge input shards -> ONE quantize) -- #

# Fake convert_to_quant for the --output-mode single path: records the kwargs it
# received (so the test can prove the merged temp file was the input + output path),
# counts how many times quantize() was called (proves the single-merge path calls it
# exactly once, not per-shard), and writes a sentinel output file.
FAKE_CTQ_SINGLE = '''
import os
import json


def quantize(**kwargs):
    print("PROGRESS: loading merged model", flush=True)
    out = kwargs.get("output")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("quantized")
    cnt = os.environ.get("CTQ_COUNT")
    if cnt:
        n = 0
        if os.path.exists(cnt):
            n = int(open(cnt).read() or "0")
        open(cnt, "w").write(str(n + 1))
    rec = os.environ.get("CTQ_RECORD")
    if rec:
        with open(rec, "w") as fh:
            json.dump(kwargs, fh)
    print("DONE", flush=True)
'''


@pytest.fixture
def fake_ctq_single(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "convert_to_quant.py").write_text(FAKE_CTQ_SINGLE)
    return str(d)


def test_worker_single_file_merge_integration(tmp_path, fake_ctq_single):
    d = tmp_path / "m"
    # Real safetensors shards (so merge_safetensors_files can parse their headers).
    d.mkdir(parents=True, exist_ok=True)
    shards = {
        "model-00001-of-00002.safetensors": {"t0": b"\x00\x00\x80\x3f" * 2},
        "model-00002-of-00002.safetensors": {"t1": b"\x00\x00\x00\x40" * 2},
    }
    weight_map = {}
    for shard, tensors in shards.items():
        _build_safetensors(d / shard, tensors)
        for name in tensors:
            weight_map[name] = shard
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    out = tmp_path / "out.safetensors"
    record = tmp_path / "record.json"
    count = tmp_path / "count.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = fake_ctq_single + os.pathsep + env.get("PYTHONPATH", "")
    env["CTQ_RECORD"] = str(record)
    env["CTQ_COUNT"] = str(count)
    cp = subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, "-i", str(d), "-o", str(out),
         "--output-mode", "single", "--int8", "--no-stream", "--comfy_quant",
         "--save_quant_metadata"],
        capture_output=True, text=True, env=env,
    )
    assert cp.returncode == 0, cp.stderr

    # Final single-file output produced.
    assert out.is_file()
    assert out.read_text() == "quantized"

    # quantize() called EXACTLY ONCE (merge-then-quantize, never per-shard).
    assert count.is_file()
    assert int(count.read_text()) == 1

    # The single call received the merged temp input (a .safetensors in tempdir),
    # and the requested output path.
    assert record.is_file()
    kw = json.loads(record.read_text())
    assert kw["output"] == str(out)
    merged_in = kw["input"]
    assert merged_in.endswith(".safetensors")
    # It was NOT one of the original shards (those still exist in the input folder),
    # and the temp merged file was cleaned up after quantize().
    assert merged_in != str(d / "model-00001-of-00002.safetensors")
    assert merged_in != str(d / "model-00002-of-00002.safetensors")
    assert not os.path.exists(merged_in), f"temp merged file not cleaned up: {merged_in}"
    assert kw["int8"] is True
    assert "DONE" in cp.stdout


# --------------------------------------------------------------------------- #
# Task #26: compute REAL overall progress from the output file growing ("overall
# length"). The worker runs quantize() in a background thread and polls the OUTPUT
# .safetensors size, emitting CTQ_PROGRESS pct. This is a genuine file-based signal,
# not text scraping.
# --------------------------------------------------------------------------- #
def test_expected_output_bytes_is_positive_even_on_bad_input():
    # An invalid (non-safetensors) input must still yield a positive, sane estimate.
    assert _expected_output_bytes("nonexistent_file_xyz.safetensors") >= 1


def test_run_quantize_with_progress_emits_real_overall_pct(tmp_path, monkeypatch):
    # quantize() runs in a thread; the helper emits real pct from the OUTPUT file
    # growing, starting near 0 and ending at exactly 100.
    inp = tmp_path / "in.safetensors"
    inp.write_bytes(b"\x00" * 4096)
    out = tmp_path / "o.safetensors"

    captured: list[tuple[str, float | None]] = []
    orig_progress = wmod.progress

    def _cap(phase, cur=None, total=None, label="", pct=None):
        captured.append((phase, pct))
        orig_progress(phase, cur=cur, total=total, label=label, pct=pct)

    monkeypatch.setattr(wmod, "progress", _cap)

    def fake_quantize(**kwargs):
        # Grow the output file in 10 chunks so the polling loop observes real growth.
        import time as _t

        with open(str(out), "wb") as fh:
            for _ in range(10):
                fh.write(b"\x00" * 200)
                fh.flush()
                _t.sleep(0.05)

    _run_quantize_with_progress(fake_quantize, {}, str(out), str(inp), poll=0.02)

    phases = [c[0] for c in captured]
    assert "quantize" in phases
    pcts = [c[1] for c in captured if c[0] == "quantize" and c[1] is not None]
    # Ends at 100%; observed >=1 intermediate (real-growth) pct in (0, 100).
    assert 100.0 in pcts
    assert any(0.0 < p < 100.0 for p in pcts)
    # quantize actually ran and produced the (2 KB) output file.
    assert out.is_file() and out.stat().st_size == 2000


def test_run_quantize_with_progress_falls_back_on_error(tmp_path, monkeypatch):
    # If the background run raises, the helper must still call quantize() directly so
    # the job is never blocked by the progress instrumentation.
    out = tmp_path / "o.safetensors"
    captured = []
    orig_progress = wmod.progress

    def _cap(phase, cur=None, total=None, label="", pct=None):
        captured.append((phase, pct))
        orig_progress(phase, cur=cur, total=total, label=label, pct=pct)

    monkeypatch.setattr(wmod, "progress", _cap)

    called = {}

    def fake_quantize(**kwargs):
        called["ok"] = True
        with open(str(out), "w") as fh:
            fh.write("quantized")

    # Force the polling thread body to error by making the output unstatable is hard;
    # instead simulate an exception during polling setup by patching expected-bytes to
    # raise. Simpler: monkeypatch os.path.getsize used inside the loop to throw on first
    # call so the try/except falls back to a direct call.
    def _boom(path):
        raise OSError("injected")

    monkeypatch.setattr(os.path, "getsize", _boom)
    _run_quantize_with_progress(fake_quantize, {}, str(out), str(out), poll=0.02)
    assert called.get("ok") is True, "fallback must still invoke quantize()"
    assert out.read_text() == "quantized"
    assert any(p == 100.0 for _, p in captured)


# --------------------------------------------------------------------------- #
# P5: streaming routing -- INT8 single-output no longer merges shards
# --------------------------------------------------------------------------- #
def test_main_single_mode_int8_sharded_calls_stream_not_merge(monkeypatch, tmp_path):
    # INT8 + sharded input + --output-mode single must stream the shards into ONE
    # output file (no temp-merge). Prove merge_safetensors_files is never called and
    # stream_quantize receives the list of shard paths (not a merged temp file).
    d = tmp_path / "m"
    wm = {
        "t0": "model-00001-of-00002.safetensors",
        "t1": "model-00002-of-00002.safetensors",
    }
    _make_sharded(d, wm, extra={"config.json": "{}"})
    out = tmp_path / "out.safetensors"

    calls = {}
    monkeypatch.setattr(wmod, "stream_quantize", lambda *a, **k: calls.setdefault("stream", (a, k)))
    merge_calls: list = []
    monkeypatch.setattr(wmod, "merge_safetensors_files", lambda *a, **k: merge_calls.append((a, k)))
    monkeypatch.setattr(wmod, "stream_quantize_sharded", lambda *a, **k: None)

    monkeypatch.setattr(sys, "argv", [
        "quantui.worker_ctq", "-i", str(d), "-o", str(out),
        "--int8", "--output-mode", "single", "--comfy_quant",
    ])
    main()

    assert "stream" in calls, calls
    args, _ = calls["stream"]
    first = args[0]
    assert isinstance(first, (list, tuple)), first
    assert set(os.path.basename(p) for p in first) == set(wm.values())
    assert merge_calls == [], "merge_safetensors_files must NOT run for INT8 streaming"


def test_main_single_file_int8_calls_stream(monkeypatch, tmp_path):
    # A single-file INT8 input is also routed through stream_quantize (passed as a path).
    inp = tmp_path / "m.safetensors"
    inp.write_text("x")
    out = tmp_path / "o.safetensors"
    calls = {}
    monkeypatch.setattr(wmod, "stream_quantize", lambda *a, **k: calls.setdefault("stream", (a, k)))
    monkeypatch.setattr(sys, "argv", [
        "quantui.worker_ctq", "-i", str(inp), "-o", str(out), "--int8", "--comfy_quant",
    ])
    main()
    assert "stream" in calls
    args, _ = calls["stream"]
    assert isinstance(args[0], str) and args[0] == str(inp)


def test_main_no_stream_forces_merge(monkeypatch, tmp_path):
    # --no-stream forces the legacy merge path even for INT8 (escape hatch).
    d = tmp_path / "m"
    wm = {"t0": "model-00001-of-00002.safetensors", "t1": "model-00002-of-00002.safetensors"}
    _make_sharded(d, wm)
    out = tmp_path / "out.safetensors"
    merge_calls: list = []
    monkeypatch.setattr(wmod, "merge_safetensors_files", lambda *a, **k: merge_calls.append((a, k)))
    mod = types.ModuleType("convert_to_quant")
    mod.quantize = lambda **kw: None
    monkeypatch.setitem(sys.modules, "convert_to_quant", mod)
    monkeypatch.setattr(sys, "argv", [
        "quantui.worker_ctq", "-i", str(d), "-o", str(out),
        "--int8", "--output-mode", "single", "--no-stream", "--comfy_quant",
    ])
    main()
    assert merge_calls, "legacy merge must run when --no-stream is set"
