"""Unit tests for the quantization registry / schema (quant_methods.py)."""

import os

from quantui.quant_methods import (
    COMFY_FORMATS,
    COMFY_PRESETS,
    INDEX_NAME,
    METHODS,
    METHODS_BY_ID,
    Backend,
    ComfyFormat,
    Family,
    OptionField,
    Preset,
    classify_input,
    comfy_format,
    comfy_preset,
    eval_visible_when,
    format_options,
    get_method,
    is_sharded_folder,
    methods_for_family,
    preset_options,
)


def test_enums_and_fields():
    assert Family.COMFY.value == "comfy"
    assert Family.GGUF.value == "gguf"
    assert Backend.UNSLOTH.value == "unsloth"
    assert Backend.CTQ.value == "convert_to_quant"
    # every existing GGUF entry is tagged correctly
    for m in METHODS:
        assert m.family == Family.GGUF
        assert m.backend == Backend.UNSLOTH
        assert m.options == []
    assert METHODS_BY_ID["q4_k_m"].family == Family.GGUF
    assert METHODS_BY_ID["q4_k_m"].backend == Backend.UNSLOTH
    # OptionField constructs
    of = OptionField("k", "Label", "select", default="256",
                     choices=[("a", "a")], cli_flag="--k")
    assert of.key == "k" and of.default == "256" and of.cli_flag == "--k"


def test_comfy_formats():
    assert len(COMFY_FORMATS) == 9
    cf = comfy_format("int8_convrot")
    assert cf.base_flags == ["--int8", "--scaling_mode", "row", "--convrot"]
    # int8_convrot exposes convrot_group_size + the shared INT8 options
    # (heur, manual_seed, exclude_layers, output_dtype).
    keys = [o.key for o in cf.extra_options]
    assert keys == [
        "convrot_group_size", "heur", "manual_seed", "exclude_layers", "output_dtype",
    ], keys
    crs = cf.extra_options[0]
    assert crs.key == "convrot_group_size"
    assert crs.visible_when == "format == 'int8_convrot'"
    assert cf.needs == ["triton"]
    # The former int8_row duplicate was removed: exactly one ConvRot entry remains.
    assert all(f.id != "int8_row" for f in COMFY_FORMATS)
    assert len(COMFY_PRESETS) == 5
    assert comfy_preset("flux2").flag == "--flux2"
    assert comfy_format("nvfp4").base_flags == ["--nvfp4"]
    assert comfy_format("mxfp8").base_flags == ["--mxfp8"]


def test_int8_block_is_true_blockwise():
    # P1.1: int8_block must emit explicit block scaling (not rely on the library default
    # which silently produced tensorwise). A block_size OptionField supplies the size.
    # It also exposes the shared INT8 options (heur, manual_seed, exclude_layers,
    # output_dtype) so users can avoid the "dimensions divisible by block_size" crash
    # and pin layer exclusions / passthrough dtype.
    cf = comfy_format("int8_block")
    assert cf.base_flags == ["--int8", "--scaling_mode", "block"]
    keys = [o.key for o in cf.extra_options]
    assert keys == ["block_size", "heur", "manual_seed", "exclude_layers", "output_dtype"], keys
    bs = cf.extra_options[0]
    assert bs.key == "block_size"
    assert bs.default == "128"
    assert bs.cli_flag == "--block_size"
    assert any(o.key == "heur" and o.cli_when_true == "--heur" for o in cf.extra_options)
    assert any(o.key == "manual_seed" and o.cli_flag == "--manual_seed" for o in cf.extra_options)


def test_new_formats_registered():
    # P1.2 ConvRot + P3/P4/P5 formats.
    convrot = comfy_format("int8_convrot")
    assert convrot.quant_format == "int8_tensorwise"
    assert convrot.base_flags == ["--int8", "--scaling_mode", "row", "--convrot"]

    w4a4 = comfy_format("w4a4_convrot")
    assert w4a4.backend == Backend.COMFY_KITCHEN
    assert w4a4.quant_format == "convrot_w4a4"
    assert w4a4.needs == ["comfy_kitchen"]

    w4a8 = comfy_format("w4a8_asym")
    assert w4a8.backend == Backend.COMFY_KITCHEN
    assert w4a8.quant_format == "asym_w4a8_int8"

    otf = comfy_format("onthefly")
    assert otf.backend == Backend.CTQ
    assert otf.base_flags == ["--passthrough"]
    assert otf.quant_format is None


def test_backend_enum_has_kitchen():
    assert Backend.COMFY_KITCHEN.value == "comfy_kitchen"


def test_helpers():
    gguf = methods_for_family(Family.GGUF)
    assert all(m.family == Family.GGUF for m in gguf)
    assert len(gguf) == len(METHODS)
    assert methods_for_family(Family.COMFY) == []
    assert len(format_options()) == 9
    assert len(preset_options()) == 5
    assert comfy_format("nvfp4").requires_cuda == "13.0"
    assert comfy_format("nvfp4").requires_py == "3.12"
    assert get_method(Family.GGUF, "q4_k_m").id == "q4_k_m"
    assert get_method(Family.COMFY, "x") is None
    assert comfy_preset("nope") is None
    assert preset_options()[0][1] == "flux2"
    assert isinstance(comfy_format("fp8_e4m3"), ComfyFormat)
    assert isinstance(comfy_preset("wan"), Preset)


def test_eval_visible_when():
    assert eval_visible_when("format == 'int8_convrot'", {"format": "int8_convrot"}) is True
    assert eval_visible_when("format == 'int8_convrot'", {"format": "fp8_e4m3"}) is False
    assert eval_visible_when("format in ('int8_convrot', 'int8_block')", {"format": "int8_block"}) is True
    assert eval_visible_when("format in ('int8_convrot', 'int8_block')", {"format": "fp8_e4m3"}) is False
    assert eval_visible_when("format != 'fp8_e4m3'", {"format": "int8_convrot"}) is True
    assert eval_visible_when("x not in ('a', 'b')", {"x": "c"}) is True
    assert eval_visible_when(None, {}) is True
    assert eval_visible_when("garbage((", {}) is False  # safe on syntax error


# --------------------------------------------------------------------------- #
# A1: shared ctq input classification
# --------------------------------------------------------------------------- #
def test_index_name_constant():
    assert INDEX_NAME == "model.safetensors.index.json"


def test_classify_input(tmp_path):
    # single .safetensors file
    f = tmp_path / "model.safetensors"
    f.write_text("x")
    assert classify_input(str(f)) == ("single_file", "model")

    # folder with exactly one .safetensors -> single_file, folder name
    one = tmp_path / "mymodel"
    one.mkdir()
    (one / "model.safetensors").write_text("x")
    assert classify_input(str(one)) == ("single_file", "mymodel")

    # HuggingFace sharded folder
    sh = tmp_path / "sharded"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    (sh / "model-00001-of-00002.safetensors").write_text("x")
    assert classify_input(str(sh)) == ("sharded_folder", "sharded")

    # non-.safetensors file
    bad = tmp_path / "model.bin"
    bad.write_text("x")
    assert classify_input(str(bad)) == (None, None)

    # folder with zero .safetensors and no index json
    empty = tmp_path / "empty"
    empty.mkdir()
    assert classify_input(str(empty)) == (None, None)

    # folder with two .safetensors and no index json
    two = tmp_path / "two"
    two.mkdir()
    (two / "a.safetensors").write_text("x")
    (two / "b.safetensors").write_text("x")
    assert classify_input(str(two)) == (None, None)

    # empty path
    assert classify_input("") == (None, None)

    # trailing slash on a sharded folder dir is still recognized
    assert classify_input(str(sh) + os.sep) == ("sharded_folder", "sharded")


def test_is_sharded_folder(tmp_path):
    sh = tmp_path / "sharded"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    (sh / "model-00001-of-00002.safetensors").write_text("x")
    assert is_sharded_folder(str(sh)) is True

    one = tmp_path / "one"
    one.mkdir()
    (one / "model.safetensors").write_text("x")
    assert is_sharded_folder(str(one)) is False

    assert is_sharded_folder(str(tmp_path / "nope")) is False
