"""S3.1/S3.2 tests: streaming GGUF export engine + CLI.

Two tiers (plan §0.6):
* pure-logic tests (method validation, input discovery, exit codes without
  writer) run everywhere — no gguf needed;
* real-writer e2e tests (actual GGUF output, miniread assertions) call
  ``pytest.importorskip("gguf")`` at module scope: they are active in the
  worker env and SKIP in the headless gate (which has numpy but no gguf).
This mirrors the plan's Tier-A/Tier-B split; gate coverage for the writer
comes from the golden kernel bytes + miniread parser tests.
"""

import json
import struct
import subprocess
import sys

import pytest

from quantui.comfy_quant_schema import write_safetensors
from quantui.gguf_export import GgufExportError
from tests.gguf_miniread import read_gguf_header

pytest.importorskip("gguf", reason="export e2e needs gguf (worker env)")

GGML_F32 = 0
GGML_F16 = 1
GGML_Q8_0 = 8
GGML_Q4_0 = 2
GGML_BF16 = 30
GGML_BF16 = 30


def _export():
    """Late-bound so module import succeeds without gguf (skip path)."""
    from quantui.gguf_export import export_gguf

    return export_gguf


def _model(tmp_path, specs, config=None, name="model.safetensors"):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    (d / "model.safetensors").write_bytes(write_safetensors(specs))
    if config is not None:
        (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return d


def _tiny_specs():
    """LM-ish tiny model: q_proj [64,64], norm [64], embed [100, 64]."""
    def f32(vals):
        return struct.pack(f"<{len(vals)}f", *vals)
    return {
        "model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32([0.1] * 4096)),
        "model.layers.0.input_layernorm.weight": ("F32", [64], f32([1.0] * 64)),
        "model.embed_tokens.weight": ("F32", [100, 64], f32([0.01] * 6400)),
    }


def test_export_tiny_model_e2e(tmp_path):
    """Full pipeline: names mapped, qtype Q8_0, ne = [in, out], KV present."""

    src = _model(tmp_path, _tiny_specs(), config={"model_type": "llama", "num_hidden_layers": 1})
    out = tmp_path / "out.gguf"
    report = _export()(str(src), str(out), "native_q8_0")
    g = read_gguf_header(str(out))
    assert g.kv["general.architecture"] == "llama"
    assert g.kv["llama.block_count"] == 1
    t = g.tensor("blk.0.attn_q.weight")
    assert t.ggml_type == GGML_Q8_0
    assert t.ne == (64, 64)  # HF [out, in] -> GGUF ne = [in, out]... both 64; assert order-preserving
    assert g.tensor("token_embd.weight").ggml_type == GGML_F16  # embd stays F16
    assert g.tensor("blk.0.attn_norm.weight").ggml_type == GGML_F32
    assert report.tensors_total == 3
    assert report.tensors_quantized == 1  # only q_proj; token_embd stays F16
    # payload byte size exact: 64*64/32*34 per quantized tensor
    from tests.gguf_miniread import tensor_data

    assert len(tensor_data(str(out), t)) == 64 * 64 // 32 * 34


def test_export_demoted_conv_and_summary(tmp_path):
    """[8,10] conv -> F16 in file; report.demoted + warnings populated."""
    def f32(n):
        return struct.pack(f"<{n}f", *([0.1] * n))
    specs = {
        "model.layers.0.conv.conv.weight": ("F32", [8, 10], f32(80)),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32(4096)),
    }
    src = _model(tmp_path, specs)
    out = tmp_path / "out.gguf"
    report = _export()(str(src), str(out), "native_q8_0")
    g = read_gguf_header(str(out))
    assert g.tensor("blk.0.shortconv.conv.weight").ggml_type == GGML_F16
    assert report.demoted and any("not divisible" in r for _, r in report.demoted)
    assert report.warnings


def test_export_skips_rotary(tmp_path):
    def f32(n):
        return struct.pack(f"<{n}f", *([0.1] * n))
    specs = {
        "model.layers.0.self_attn.rotary_emb.inv_freq": ("F32", [32], f32(32)),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32(4096)),
    }
    src = _model(tmp_path, specs)
    out = tmp_path / "out.gguf"
    report = _export()(str(src), str(out), "native_q8_0")
    g = read_gguf_header(str(out))
    assert "blk.0.attn_q.weight" in g.tensor_names()
    assert not any("inv_freq" in n for n in g.tensor_names())
    assert report.skipped == ["model.layers.0.self_attn.rotary_emb.inv_freq"]


def test_export_sharded_folder_deterministic(tmp_path):
    """Two shards, reversed index orderings -> byte-identical output."""
    def f32(n):
        return struct.pack(f"<{n}f", *([0.1] * n))
    shard_a = {"model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32(4096))}
    shard_b = {"model.layers.0.mlp.down_proj.weight": ("F32", [64, 64], f32(4096))}
    d = tmp_path / "sharded"
    d.mkdir()
    (d / "a.safetensors").write_bytes(write_safetensors(shard_a))
    (d / "b.safetensors").write_bytes(write_safetensors(shard_b))
    index = {
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "a.safetensors",
            "model.layers.0.mlp.down_proj.weight": "b.safetensors",
        }
    }
    out1, out2 = tmp_path / "o1.gguf", tmp_path / "o2.gguf"
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    _export()(str(d), str(out1), "native_q8_0")
    index["weight_map"] = dict(reversed(list(index["weight_map"].items())))
    (d / "model.safetensors.index.json").write_text(json.dumps(index))
    _export()(str(d), str(out2), "native_q8_0")
    assert out1.read_bytes() == out2.read_bytes()


def test_export_determinism_single_file(tmp_path):
    src = _model(tmp_path, _tiny_specs())
    out1, out2 = tmp_path / "d1.gguf", tmp_path / "d2.gguf"
    _export()(str(src), str(out1), "native_q8_0")
    _export()(str(src), str(out2), "native_q8_0")
    assert out1.read_bytes() == out2.read_bytes()


def test_export_unsupported_method(tmp_path):
    src = _model(tmp_path, _tiny_specs())
    with pytest.raises(GgufExportError, match="native_q6_k"):
        _export()(str(src), str(tmp_path / "x.gguf"), "native_q6_k")


def test_export_missing_input(tmp_path):
    with pytest.raises(GgufExportError, match="nope.safetensors|not a"):
        _export()(str(tmp_path / "nope.safetensors"), str(tmp_path / "x.gguf"), "native_q8_0")


def test_export_report_totals(tmp_path):
    src = _model(tmp_path, _tiny_specs())  # no config.json -> arch "unknown"
    report = _export()(str(src), str(tmp_path / "r.gguf"), "native_q8_0")
    assert report.bytes_in == (4096 + 64 + 6400) * 4
    assert report.bytes_out > 0
    assert report.arch == "unknown"
    assert report.qtype_histogram == {"F16": 1, "F32": 1, "Q8_0": 1}


def test_export_progress_callback(tmp_path):
    seen = []

    def cb(done, total, name):
        seen.append((done, total, name))

    src = _model(tmp_path, _tiny_specs())
    report = _export()(str(src), str(tmp_path / "p.gguf"), "native_q8_0", progress=cb)
    assert [d for d, _, _ in seen] == list(range(1, report.tensors_total + 1))
    assert seen[-1][0] == seen[-1][1]


def test_export_unknown_arch_still_writes(tmp_path):
    src = _model(tmp_path, _tiny_specs(), config=None)
    out = tmp_path / "ua.gguf"
    report = _export()(str(src), str(out), "native_q8_0")
    assert report.arch == "unknown"
    g = read_gguf_header(str(out))
    assert g.kv["general.architecture"] == "unknown"


def test_export_vibevoice_like_config_converts(tmp_path):
    """THE headline: a vibevoice-config model converts (no arch rejection)."""
    def f32(n):
        return struct.pack(f"<{n}f", *([0.1] * n))
    specs = {
        "model.language_model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32(4096)),
        "model.acoustic_tokenizer.decoder.head.conv.conv.weight": ("F32", [8, 10], f32(80)),
    }
    src = _model(tmp_path, specs, config={"model_type": "vibevoice", "num_hidden_layers": 1})
    out = tmp_path / "vv.gguf"
    report = _export()(str(src), str(out), "native_q8_0")
    g = read_gguf_header(str(out))
    assert g.kv["general.architecture"] == "vibevoice"
    assert g.tensor("blk.0.attn_q.weight").ggml_type == GGML_Q8_0
    assert g.tensor("model.acoustic_tokenizer.decoder.head.conv.conv.weight").ggml_type == GGML_F16
    assert report.tensors_quantized == 1


# ------------------------------------------------------------------ CLI ---- #
def _run_cli(args, cwd=None):
    env_python = sys.executable
    return subprocess.run(
        [env_python, "-m", "quantui.gguf_export", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_cli_happy_path(tmp_path):
    src = _model(tmp_path, _tiny_specs())
    out = tmp_path / "cli.gguf"
    proc = _run_cli(["-i", str(src), "-o", str(out), "-m", "native_q8_0"])
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert "GGUF written" in proc.stdout
    g = read_gguf_header(str(out))
    assert g.tensors


def test_cli_default_output_name(tmp_path):
    src = _model(tmp_path, _tiny_specs())
    proc = _run_cli(["-i", str(src / "model.safetensors"), "-m", "native_q8_0"])
    assert proc.returncode == 0, proc.stderr
    expected = src / "model-native_q8_0.gguf"
    assert expected.exists()


def test_cli_list_methods():
    proc = _run_cli(["--list-methods"])
    assert proc.returncode == 0
    lines = proc.stdout.strip().splitlines()
    assert "native_q8_0" in lines and "native_f32" in lines and len(lines) == 5


def test_cli_error_exit_2(tmp_path):
    proc = _run_cli(["-i", str(tmp_path / "missing.safetensors")])
    assert proc.returncode == 2
    assert "missing.safetensors" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_json_report(tmp_path):
    src = _model(tmp_path, _tiny_specs(), config={"model_type": "llama"})
    proc = _run_cli(["-i", str(src), "-o", str(tmp_path / "j.gguf"), "--json"])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["arch"] == "llama"
    assert payload["tensors_total"] == 3
    assert payload["qtype_histogram"] == {"F16": 1, "F32": 1, "Q8_0": 1}


# --------------------------------------------------------------------------- #
# S5.1: native_bf16 — lossless for bf16 sources (no f16 inf-overflow risk)
# --------------------------------------------------------------------------- #
def test_export_bf16_source_verbatim(tmp_path):
    """bf16 source + native_bf16 -> BF16 qtype in file, bits verbatim."""
    import numpy as np

    # bf16 bits: 1.0=0x3F80, 65536.0=0x4780, -0.0=0x8000, nan=0x7FC0
    bits = np.array([0x3F80, 0x4780, 0x8000, 0x7FC0] * 64, dtype="<u2")
    raw = bits.tobytes()
    specs = {"model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 64], raw)}
    src = _model(tmp_path, specs)
    out = tmp_path / "b.gguf"
    report = _export()(str(src), str(out), "native_bf16")
    g = read_gguf_header(str(out))
    t = g.tensor("blk.0.attn_q.weight")
    assert t.ggml_type == GGML_BF16
    assert t.ne == (64, 4)
    from tests.gguf_miniread import tensor_data

    stored = tensor_data(str(out), t)
    # every bf16 bit pattern must survive verbatim (lossless contract)
    assert stored == raw
    assert report.qtype_histogram == {"BF16": 1}


def test_export_bf16_beats_f16_on_overflow(tmp_path):
    """THE regression: 65536.0 survives native_bf16; under f16 it'd be inf."""
    import struct

    import numpy as np

    # f32 65536.0 = 0x47800000
    raw = struct.pack("<4f", 65536.0, 65536.0, 65536.0, 65536.0)
    specs = {"model.layers.0.self_attn.q_proj.weight": ("F32", [2, 2], raw)}
    src = _model(tmp_path, specs)
    out = tmp_path / "c.gguf"
    _export()(str(src), str(out), "native_bf16")
    g = read_gguf_header(str(out))
    t = g.tensor("blk.0.attn_q.weight")
    assert t.ggml_type == GGML_BF16
    stored = np.frombuffer(
        __import__("tests.gguf_miniread", fromlist=["tensor_data"]).tensor_data(str(out), t),
        dtype="<u2",
    )
    assert all(b == 0x4780 for b in stored)  # bf16 65536.0, NOT 0x7C00 (inf)


def test_cli_bf16_method(tmp_path):
    src = _model(tmp_path, _tiny_specs())
    out = tmp_path / "bf.gguf"
    proc = _run_cli(["-i", str(src), "-o", str(out), "-m", "native_bf16"])
    assert proc.returncode == 0, proc.stderr
    g = read_gguf_header(str(out))
    assert g.tensor("blk.0.attn_q.weight").ggml_type == GGML_BF16


# --------------------------------------------------------------------------- #
# S5.1: native_bf16 — lossless for bf16 sources (no f16 inf-overflow risk)
# --------------------------------------------------------------------------- #
