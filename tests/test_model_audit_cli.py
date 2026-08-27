"""STEP 3.2: CLI entry point ``python -m quantui.model_audit``.

Subprocess-tested exactly like the worker modules (``test_worker_ctq_args.py``
pattern): the test env has NO torch, so these runs are also the standing proof
that ``quantui.model_audit`` stays stdlib-only.
"""

import json
import os
import subprocess
import sys

from quantui.comfy_quant_schema import write_safetensors

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = "quantui.model_audit"


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def _write_fixture(tmp_path, name="m.safetensors"):
    specs = {
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
        "lm_head.weight": ("BF16", [10, 4], b"\x00" * 80),
    }
    path = tmp_path / name
    path.write_bytes(write_safetensors(specs, metadata={"format": "pt"}))
    return str(path)


def test_cli_text_report_exit_0(tmp_path):
    fixture = _write_fixture(tmp_path)
    result = _run(["-i", fixture])
    assert result.returncode == 0, result.stderr
    assert "Model audit: m.safetensors" in result.stdout
    assert "exclude_layers:" in result.stdout


def test_cli_json_flag(tmp_path):
    fixture = _write_fixture(tmp_path)
    result = _run(["-i", fixture, "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {
        "path",
        "metadata",
        "totals",
        "category_counts",
        "category_bytes",
        "modules",
        "quantized_layers",
        "quant_format_histogram",
        "suggestion",
    }
    assert payload["totals"]["tensors"] == 3


def test_cli_out_writes_file(tmp_path):
    fixture = _write_fixture(tmp_path)
    out_path = tmp_path / "reports" / "nested" / "report.json"
    result = _run(["-i", fixture, "--json", "--out", str(out_path)])
    assert result.returncode == 0, result.stderr
    assert out_path.exists()  # parents created
    assert out_path.read_text(encoding="utf-8") == result.stdout


def test_cli_out_text_mode(tmp_path):
    fixture = _write_fixture(tmp_path)
    out_path = tmp_path / "report.txt"
    result = _run(["-i", fixture, "--out", str(out_path)])
    assert result.returncode == 0, result.stderr
    assert out_path.read_text(encoding="utf-8") == result.stdout
    assert "Model audit:" in out_path.read_text(encoding="utf-8")


def test_cli_missing_file_exit_2(tmp_path):
    missing = tmp_path / "nope.safetensors"
    result = _run(["-i", str(missing)])
    assert result.returncode == 2
    assert str(missing) in result.stderr


def test_cli_malformed_file_exit_2(tmp_path):
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"this is definitely not safetensors")
    result = _run(["-i", str(garbage)])
    assert result.returncode == 2
    assert str(garbage) in result.stderr


def test_cli_no_torch_required(tmp_path):
    # The stdlib-only contract, proven directly: after importing
    # quantui.model_audit and exercising the full audit path, torch /
    # safetensors / numpy must NOT be in sys.modules (this venv actually HAS
    # torch installed, so the proof is the module-set assertion, not venv
    # absence).
    fixture = _write_fixture(tmp_path)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys;"
            "from quantui.model_audit import audit_file, suggest_exclusions, render_text;"
            f"r = audit_file(r'{fixture}');"
            "render_text(r, suggest_exclusions(r));"
            "banned = [m for m in ('torch', 'safetensors', 'numpy') if m in sys.modules];"
            "assert not banned, banned;"
            "print('clean')",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert probe.returncode == 0, probe.stderr
    assert "clean" in probe.stdout
