"""P2.2: the kitchen worker must import + parse args with NO comfy/torch present.

The heavy imports are lazy inside ``main()``, so the module is importable in the
headless test sandbox (which has no torch/comfy). ``main()`` must fail with a clear
message when comfy-kitchen is absent.
"""

import subprocess
import sys

from quantui import worker_ctq_kitchen as w

WORKER_MODULE = "quantui.worker_ctq_kitchen"


def _run(args, env=None):
    return subprocess.run(
        [sys.executable, "-m", WORKER_MODULE, *args],
        capture_output=True, text=True, env=env, check=False,
    )


def test_module_imports_without_comfy():
    # Top-level imports are stdlib-only.
    assert hasattr(w, "parse_args")
    assert hasattr(w, "main")
    assert hasattr(w, "quantize_file")


def test_help_lists_kitchen_flags():
    cp = _run(["--help"])
    assert cp.returncode == 0, cp.stderr
    for tok in ["--input", "--output", "--format-id", "--w4a4", "--w4a8"]:
        assert tok in cp.stdout


def test_parse_args():
    ns = w.parse_args(["-i", "in.safetensors", "-o", "out.safetensors",
                      "--w4a4", "--format-id", "w4a4_convrot"])
    assert ns.input == "in.safetensors"
    assert ns.output == "out.safetensors"
    assert ns.w4a4 is True
    assert ns.format_id == "w4a4_convrot"


def test_fails_cleanly_without_comfy(tmp_path):
    # The test venv has no torch/comfy -> main() must exit(1) with a clear message.
    inp = tmp_path / "m.safetensors"
    inp.write_text("x")
    out = tmp_path / "o.safetensors"
    cp = _run(["-i", str(inp), "-o", str(out), "--w4a4", "--format-id", "w4a4_convrot"])
    assert cp.returncode == 1, cp.stdout + cp.stderr
    assert "ERROR" in cp.stderr
    assert "comfy-kitchen" in (cp.stderr + cp.stdout).lower() or "ComfyUI" in (cp.stderr + cp.stdout)
