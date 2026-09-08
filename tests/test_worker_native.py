"""S4.2 tests: worker --backend native branch.

The headline regression: a vibevoice-config model converts instead of being
blocked by the (unsloth-only) arch gate. Subprocess tests use the current
interpreter; in the headless env they exercise the SKIP path (module-level
importorskip mirrors the S3.1 tiering), in the worker env they fully run.
"""

import json
import os
import struct
import subprocess
import sys

import pytest

from quantui.comfy_quant_schema import write_safetensors

pytest.importorskip("gguf", reason="native worker e2e needs gguf (worker env)")

WORKER = "quantui.worker"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fixture_model(tmp_path):
    d = tmp_path / "vibevoice-model"
    d.mkdir()

    def f32(n):
        return struct.pack(f"<{n}f", *([0.1] * n))

    specs = {
        "model.language_model.layers.0.self_attn.q_proj.weight": ("F32", [64, 64], f32(4096)),
        "model.layers.0.input_layernorm.weight": ("F32", [64], f32(64)),
        "model.acoustic_tokenizer.decoder.head.conv.conv.weight": ("F32", [8, 10], f32(80)),
    }
    (d / "model.safetensors").write_bytes(write_safetensors(specs))
    (d / "config.json").write_text(
        json.dumps({"model_type": "vibevoice", "num_hidden_layers": 1}),
        encoding="utf-8",
    )
    return d


def _run_worker(args):
    return subprocess.run(
        [sys.executable, "-m", WORKER, *args],
        capture_output=True, text=True, cwd=PROJECT_DIR,
    )


def test_worker_native_tiny_model_e2e(tmp_path):
    """--backend native converts and skips transformers/unsloth entirely."""
    src = _fixture_model(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_worker([
        "--model", str(src), "--output", str(out_dir),
        "--method", "native_q8_0", "--backend", "native",
    ])
    assert proc.returncode == 0, proc.stderr
    ggufs = list(out_dir.glob("*.gguf"))
    assert len(ggufs) == 1
    # the bypass pin: no unsloth/transformers import evidence in the log
    assert "Importing unsloth" not in proc.stdout
    assert "Loading model from" not in proc.stdout
    assert "Native export complete" in proc.stdout
    assert "DONE" in proc.stdout


def test_worker_native_vibevoice_config_not_blocked(tmp_path):
    """THE regression: model_type vibevoice converts (arch gate not consulted)."""
    from tests.gguf_miniread import read_gguf_header

    src = _fixture_model(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_worker([
        "--model", str(src), "--output", str(out_dir),
        "--method", "native_q8_0", "--backend", "native",
    ])
    assert proc.returncode == 0, proc.stderr
    gguf_path = next(out_dir.glob("*.gguf"))
    g = read_gguf_header(str(gguf_path))
    assert g.kv["general.architecture"] == "vibevoice"
    assert "blk.0.attn_q.weight" in g.tensor_names()


def test_worker_native_unknown_method_fails_friendly(tmp_path):
    src = _fixture_model(tmp_path)
    proc = _run_worker([
        "--model", str(src), "--output", str(tmp_path / "out"),
        "--method", "q4_k_m", "--backend", "native",
    ])
    assert proc.returncode == 1
    assert "not a native method" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_worker_native_rejects_imatrix(tmp_path):
    src = _fixture_model(tmp_path)
    proc = _run_worker([
        "--model", str(src), "--output", str(tmp_path / "out"),
        "--method", "native_q8_0", "--backend", "native",
        "--imatrix", "auto",
    ])
    assert proc.returncode == 1
    assert "imatrix" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_worker_native_multi_method_rejected(tmp_path):
    src = _fixture_model(tmp_path)
    proc = _run_worker([
        "--model", str(src), "--output", str(tmp_path / "out"),
        "--method", "native_q8_0,native_f16", "--backend", "native",
    ])
    assert proc.returncode == 1
    assert "exactly ONE method" in proc.stderr


def test_worker_unsloth_default_unchanged(tmp_path, monkeypatch):
    """--backend omitted -> unsloth path (dispatch snapshot)."""
    from quantui import worker as wk

    called = {}
    monkeypatch.setattr(wk, "_run_unsloth_backend", lambda a: called.setdefault("ran", a))
    monkeypatch.setattr(
        "sys.argv",
        ["worker.py", "--model", "m", "--output", "o", "--method", "q8_0"],
    )
    wk.main()
    assert "ran" in called


def test_worker_native_progress_lines(tmp_path):
    src = _fixture_model(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_worker([
        "--model", str(src), "--output", str(out_dir),
        "--method", "native_q8_0", "--backend", "native",
    ])
    assert proc.returncode == 0, proc.stderr
    # the TUI's stream parser consumes PROGRESS lines
    assert any("PROGRESS" in line or "quantize" in line for line in proc.stdout.splitlines())


def test_native_method_id_auto_routes_without_backend_flag(tmp_path):
    """S5.2: --method native_q8_0 alone implies --backend native (id IS intent)."""
    src = _fixture_model(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run_worker([
        "--model", str(src), "--output", str(out_dir),
        "--method", "native_q8_0",  # NO --backend flag
    ])
    assert proc.returncode == 0, proc.stderr
    assert "Native GGUF export" in proc.stdout
    assert len(list(out_dir.glob("*.gguf"))) == 1


def test_arch_gate_error_suggests_native_backend(tmp_path):
    """S5.2: the blocked-arch error must route the user to --method native_*."""
    src = _fixture_model(tmp_path)  # vibevoice config
    proc = _run_worker([
        "--model", str(src), "--output", str(tmp_path / "out"),
        "--method", "q8_0",  # unsloth id, default backend -> arch gate
    ])
    assert proc.returncode == 1
    err = proc.stderr
    assert "UNSLOTH backend" in err
    assert "native_q8_0" in err and "--backend native" in err
    assert "Traceback" not in err


