"""suggest_comfy_output Case-C semantics (user-report regression).

A directory-shaped output (extension-less, existing or not) is a DESTINATION
FOLDER: the artifact ``<stem>.safetensors`` must be placed inside it. An
explicit ``.safetensors`` output is the user's own name -> leave as-is.
"""

import os

from quantui import run_config
from quantui.run_config import suggest_comfy_output


def _mk(tmp_path, name):
    p = tmp_path / name
    if name.endswith(".safetensors"):
        p.write_bytes(b"")
    else:
        p.mkdir()
    return str(p)


def test_single_file_input_with_dir_output_places_file_inside(tmp_path):
    inp = _mk(tmp_path, "tensor-1b-1.5B.safetensors")
    out_dir = _mk(tmp_path, "out")
    tags = ["int8", "tensor", "simple", "lowmem"]
    got = suggest_comfy_output(inp, out_dir, tags, "sharded")
    assert got == os.path.join(
        out_dir, "tensor-1b-1.5B-int8-tensor-simple-lowmem.safetensors"
    )


def test_nonexistent_dir_shaped_output_is_still_a_folder(tmp_path):
    # The user typed a folder that does not exist yet -- still Case C.
    inp = _mk(tmp_path, "m.safetensors")
    out_dir = str(tmp_path / "not-yet-created")
    got = suggest_comfy_output(inp, out_dir, ["fp8_e4m3"], "sharded")
    assert got == os.path.join(out_dir, "m-fp8_e4m3.safetensors")


def test_empty_output_falls_back_next_to_input(tmp_path):
    inp = _mk(tmp_path, "m.safetensors")
    got = suggest_comfy_output(inp, "", ["fp8_e4m3"], "sharded")
    parent = os.path.dirname(os.path.abspath(inp))
    assert got == os.path.join(parent, "m-fp8_e4m3.safetensors")


def test_sharded_input_dir_output_sharded_mode(tmp_path):
    shard = tmp_path / "modelA"
    shard.mkdir()
    (shard / "model.safetensors.index.json").write_bytes(b"")
    out_dir = _mk(tmp_path, "out")
    got = suggest_comfy_output(str(shard), out_dir, ["int8", "block"], "sharded")
    assert got == os.path.join(out_dir, "modelA-int8-block.safetensors")


def test_sharded_input_single_mode_honors_dir_output(tmp_path):
    # Mirrors build_ctq_cmd: merged file goes inside the chosen folder.
    shard = tmp_path / "modelA"
    shard.mkdir()
    (shard / "model.safetensors.index.json").write_bytes(b"")
    out_dir = _mk(tmp_path, "out")
    got = suggest_comfy_output(str(shard), out_dir, ["int8", "row", "convrot", "gs64"], "single")
    assert got == os.path.join(out_dir, "modelA-int8-row-convrot-gs64.safetensors")


def test_explicit_safetensors_output_is_left_alone(tmp_path):
    inp = _mk(tmp_path, "m.safetensors")
    explicit = os.path.join(str(tmp_path / "out"), "mine.safetensors")
    assert suggest_comfy_output(inp, explicit, ["fp8_e4m3"], "sharded") is None


def test_stale_convrot_checkbox_does_not_leak_into_tags():
    tags = run_config.ctq_quant_tags(
        "int8", convrot_group_size=64, scaling="tensor", convrot=True
    )
    assert tags == ["int8", "tensor"]  # no convrot/gs64 under tensor scaling
