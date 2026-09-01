"""Unit tests for the quantization registry / schema (quant_methods.py)."""

import os

from quantui.quant_methods import (
    ALLOWED_QUANT_IDS,
    COMFY_FORMATS,
    COMFY_PRESETS,
    IMATRIX_QUANT_IDS,
    INDEX_NAME,
    METHODS,
    METHODS_BY_ID,
    Backend,
    ComfyFormat,
    Family,
    OptionField,
    Preset,
    QuantMethod,
    classify_input,
    comfy_format,
    comfy_preset,
    eval_visible_when,
    format_options,
    get_method,
    is_sharded_folder,
    list_line,
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
    # v0.4.0: the int8_block/int8_tensor/int8_convrot trio collapsed into ONE
    # unified "int8" format with a first-class scaling option (9 -> 7 entries).
    # STEP 3.1 (plan 2026-08-26): bf16/fp16 cast-only formats added (7 -> 9).
    assert len(COMFY_FORMATS) == 9
    cf = comfy_format("int8")
    assert cf.base_flags == ["--int8"]
    keys = [o.key for o in cf.extra_options]
    assert keys == [
        "scaling_mode", "block_size", "convrot", "convrot_group_size",
        "heur", "manual_seed", "exclude_layers", "output_dtype",
    ], keys
    sm = cf.extra_options[0]
    assert sm.key == "scaling_mode"
    assert sm.default == "block"
    assert sm.cli_flag == "--scaling_mode"
    assert [v for _, v in sm.choices] == ["block", "tensor", "row"]
    # scaling_mode is unconditional (always shown); every other option is gated.
    assert sm.visible_when is None
    assert all(o.visible_when is not None
               for o in cf.extra_options if o.key != "scaling_mode")
    # The former ids are gone for good (alias tripwires).
    assert all(f.id not in ("int8_row", "int8_block", "int8_tensor", "int8_convrot")
               for f in COMFY_FORMATS)
    assert len(COMFY_PRESETS) == 5
    assert comfy_preset("flux2").flag == "--flux2"
    assert comfy_format("nvfp4").base_flags == ["--nvfp4"]
    assert comfy_format("mxfp8").base_flags == ["--mxfp8"]


def test_int8_option_visibility_predicates():
    # Only valid combinations are visible: block_size only for block scaling;
    # convrot (+ group size) only for row scaling; group size also needs the
    # convrot toggle ON.
    fmt = comfy_format("int8")
    opts = {o.key: o for o in fmt.extra_options}

    def vis(key, **ctx):
        return eval_visible_when(opts[key].visible_when,
                                 {"format": "int8", **ctx})

    base = {"scaling_mode": "block", "convrot": False}
    assert vis("block_size", **base) is True
    assert vis("block_size", **{**base, "scaling_mode": "tensor"}) is False
    assert vis("block_size", **{**base, "scaling_mode": "row"}) is False
    assert vis("convrot", **{**base, "scaling_mode": "row"}) is True
    assert vis("convrot", **base) is False
    row = {"scaling_mode": "row"}
    assert vis("convrot_group_size", convrot=True, **row) is True
    assert vis("convrot_group_size", convrot=False, **row) is False
    # Shared INT8 options are visible in every scaling mode.
    for key in ("heur", "manual_seed", "exclude_layers", "output_dtype"):
        assert vis(key, **base) is True, key
        assert vis(key, **row, convrot=False) is True, key
    bs = opts["block_size"]
    assert bs.default == "128" and bs.cli_flag == "--block_size"
    assert any(o.key == "heur" and o.cli_when_true == "--heur"
               for o in fmt.extra_options)


def test_flux2_preset_pins_row_convrot():
    p = comfy_preset("flux2")
    assert p.recommended_format == "int8"
    assert p.recommended_options.get("scaling_mode") == "row"
    assert p.recommended_options.get("convrot") is True
    # Block-scaling presets stay rotation-free.
    for pid in ("wan", "hunyuan"):
        pp = comfy_preset(pid)
        assert pp.recommended_format == "int8"
        assert pp.recommended_options.get("scaling_mode") == "block"
        assert "convrot" not in pp.recommended_options


def test_backend_enum_has_kitchen():
    assert Backend.COMFY_KITCHEN.value == "comfy_kitchen"


def test_kitchen_and_combine_formats():
    # P3/P4 kitchen formats + the combine format (STEP 1.2 rename of the former
    # on-the-fly passthrough; ConvRot W8A8 is now an option of unified int8).
    w4a4 = comfy_format("w4a4_convrot")
    assert w4a4.backend == Backend.COMFY_KITCHEN
    assert w4a4.quant_format == "convrot_w4a4"
    assert w4a4.needs == ["comfy_kitchen"]

    w4a8 = comfy_format("w4a8_asym")
    assert w4a8.backend == Backend.COMFY_KITCHEN
    assert w4a8.quant_format == "asym_w4a8_int8"

    combine = comfy_format("combine")
    assert combine.backend == Backend.CTQ
    assert combine.base_flags == ["--combine"]
    assert combine.quant_format is None
    assert combine.extra_options == []
    assert combine.needs == []

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
    assert eval_visible_when("format == 'int8'", {"format": "int8"}) is True
    assert eval_visible_when("format == 'int8'", {"format": "fp8_e4m3"}) is False
    assert eval_visible_when("format in ('int8', 'fp8_e4m3')", {"format": "int8"}) is True
    assert eval_visible_when("format != 'fp8_e4m3'", {"format": "int8"}) is True
    # Chained predicates over sibling option values (unified INT8 UI).
    ctx = {"format": "int8", "scaling_mode": "row", "convrot": True}
    assert eval_visible_when("scaling_mode == 'row' and convrot", ctx) is True
    assert eval_visible_when("format == 'int8' and scaling_mode == 'block'", ctx) is False
    assert eval_visible_when("format == 'int8' and scaling_mode == 'block'",
                             {**ctx, "scaling_mode": "block"}) is True
    assert eval_visible_when("x not in ('a', 'b')", {"x": "c"}) is True
    assert eval_visible_when(None, {}) is True
    assert eval_visible_when("garbage((", {}) is False  # safe on syntax error
    # A missing context name resolves falsy -> hidden (never crashes).
    assert eval_visible_when("convrot", {"format": "int8"}) is False


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


# --------------------------------------------------------------------------- #
# T1 (plan 2026-08-31-gguf-unsloth-parity): canonical official unsloth ids
# --------------------------------------------------------------------------- #
def test_allowed_quant_ids_canonical():
    # Exactly the 24 ALLOWED_QUANTS ids from unsloth/save.py (dict order).
    assert list(ALLOWED_QUANT_IDS) == [
        "not_quantized", "fast_quantized", "quantized", "f32", "bf16", "f16",
        "q8_0", "q4_k_m", "q5_k_m", "q2_k", "q2_k_l", "q3_k_l", "q3_k_m",
        "q3_k_s", "q4_0", "q4_1", "q4_k_s", "q4_k", "q5_k", "q5_0", "q5_1",
        "q5_k_s", "q6_k", "q3_k_xs",
    ]


def test_imatrix_quant_ids_canonical():
    # Exactly the 11 IMATRIX_QUANTS ids from unsloth/save.py (dict order).
    assert list(IMATRIX_QUANT_IDS) == [
        "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
        "iq3_xxs", "iq3_s", "iq3_m", "iq4_nl", "iq4_xs",
    ]


def test_qm_needs_imatrix_default_false():
    m = QuantMethod("x", "X", Family.GGUF, Backend.UNSLOTH)
    assert m.needs_imatrix is False


def test_needs_imatrix_is_kwarg():
    m = QuantMethod("x", "X", Family.GGUF, Backend.UNSLOTH, needs_imatrix=True)
    assert m.needs_imatrix is True


# --------------------------------------------------------------------------- #
# T2 (plan 2026-08-31-gguf-unsloth-parity): METHODS == official 35-id surface
# --------------------------------------------------------------------------- #
def test_methods_ids_match_official_surface():
    # save.py exposes 24 ALLOWED_QUANTS + 11 IMATRIX_QUANTS = 35, no more.
    assert len(METHODS) == 35
    assert {m.id for m in METHODS} == set(ALLOWED_QUANT_IDS) | set(IMATRIX_QUANT_IDS)


def test_methods_order_pinned():
    # Registry order is the UI order: 24 ALLOWED (save.py dict order), then 11
    # IMATRIX. This is the new order pin (the enums file pins Family/Backend only).
    assert [m.id for m in METHODS] == list(ALLOWED_QUANT_IDS) + list(IMATRIX_QUANT_IDS)


def test_no_fake_dynamic_ids():
    ids = {m.id for m in METHODS}
    # q4_nl never existed in unsloth (real id is the imatrix-gated iq4_nl);
    # the three UD-* XL ids claim Dynamic 2.0 output that save_pretrained_gguf
    # cannot produce (proprietary download-only mixes).
    for gone in ("q4_nl", "q4_k_xl", "q3_k_xl", "q2_k_xl"):
        assert gone not in ids
    for m in METHODS:
        assert hasattr(m, "dynamic_v2") is False


def test_needs_imatrix_flags_correct():
    flagged = {m.id for m in METHODS if m.needs_imatrix}
    assert flagged == set(IMATRIX_QUANT_IDS)


def test_q2_k_l_description_mentions_preset():
    # q2_k_l is an unsloth PRESET (q2_k body + q8_0 output/token embeddings),
    # not a distinct llama.cpp qtype -- the UI must say so.
    assert "preset" in METHODS_BY_ID["q2_k_l"].description.lower()


def test_q6_k_description_official():
    assert "Uses Q8_K for all tensors" in METHODS_BY_ID["q6_k"].description


def test_q2_k_bpw_3_35():
    assert METHODS_BY_ID["q2_k"].approx_bpw == 3.35


# --------------------------------------------------------------------------- #
# T3 (folded into T2 -- list_line is the other reader of the removed
# dynamic_v2 field, so both readers must be fixed in the SAME commit)
# --------------------------------------------------------------------------- #
def test_list_line_imatrix_marker():
    assert "[IMATRIX]" in list_line(METHODS_BY_ID["iq2_xs"])


def test_list_line_plain_has_no_marker():
    assert "[IMATRIX]" not in list_line(METHODS_BY_ID["q4_k_m"])


def test_list_line_no_dynamic_badge():
    for m in METHODS:
        assert "DYNAMIC" not in list_line(m)
