"""P5: on-the-fly passthrough copies the input unchanged (no .comfy_quant baked).

ComfyUI's ``on_the_fly_quantization`` loader then quantizes at load time. The passthrough
path short-circuits before importing ``convert_to_quant``.
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


def test_passthrough_single_file_identical(tmp_path):
    inp = tmp_path / "in.safetensors"
    _write_plain_safetensors(inp)
    src_bytes = inp.read_bytes()
    out = tmp_path / "out.safetensors"

    cp = _run(["-i", str(inp), "-o", str(out), "--passthrough"])
    assert cp.returncode == 0, cp.stderr
    assert out.is_file()
    # Byte-identical copy -> no .comfy_quant tensor was added.
    assert out.read_bytes() == src_bytes
    assert "DONE" in cp.stdout


def test_passthrough_sharded_folder(tmp_path):
    src = tmp_path / "mymodel"
    src.mkdir()
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))
    (src / "s1.safetensors").write_bytes(write_safetensors({"a": ("F32", [1], struct.pack("<f", 3.0))}))
    (src / "config.json").write_text("{}")

    out = tmp_path / "outmodel"
    cp = _run(["-i", str(src), "-o", str(out), "--passthrough"])
    assert cp.returncode == 0, cp.stderr
    assert (out / "model.safetensors.index.json").is_file()
    assert (out / "s1.safetensors").is_file()
    assert (out / "config.json").is_file()


def test_passthrough_has_no_comfy_quant(tmp_path):
    inp = tmp_path / "in.safetensors"
    _write_plain_safetensors(inp)
    out = tmp_path / "out.safetensors"
    cp = _run(["-i", str(inp), "-o", str(out), "--passthrough"])
    assert cp.returncode == 0, cp.stderr
    # Parse the output header and assert no .comfy_quant tensor exists.
    with open(out, "rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
    assert not any(k.endswith(".comfy_quant") for k in header)


def test_build_ctq_cmd_onthefly_emits_passthrough():
    tmp_path = __import__("pathlib").Path(__file__).parent / "dummy.safetensors"
    # Build a minimal CtqConfig for the onthefly format.
    c = rc.CtqConfig(
        input=str(tmp_path), output=str(tmp_path.parent / "o.safetensors"),
        pybin=sys.executable, format="onthefly", output_mode="sharded",
        quant_tags=rc.ctq_quant_tags("onthefly"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert "--passthrough" in cmd
    # onthefly uses the CTQ backend worker, not the kitchen worker.
    assert cmd[2] == "quantui.worker_ctq"
