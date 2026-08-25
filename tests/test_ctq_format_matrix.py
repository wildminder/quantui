"""P7: data-driven integration matrix over ``COMFY_FORMATS``.

For every registered ComfyUI format we assert the structural invariants the rest of
the system relies on:

* Each format has a unique id and a known backend.
* CTQ-backend formats do NOT require ``comfy_kitchen``; kitchen-backend formats DO
  (the two backends are mutually exclusive by design -- convert_to_quant cannot emit
  W4A4/W4A8, comfy-kitchen is only wired for those).
* Any ``quant_format`` a format declares is a real ``.comfy_quant`` schema format.
* A kitchen format's declared ``quant_format`` is reproducible with the pure
  serializer and the resulting file passes :func:`validate_comfy_quant_file`.

The validator/serializer path is exercised headless (stdlib only) so this test runs
without torch/comfy/safetensors.
"""

import json
import struct

import pytest

from quantui.comfy_quant_schema import (
    FORMAT_ASYM_W4A8_INT8,
    FORMAT_CONVROT_W4A4,
    FORMAT_INT8_TENSORWISE,
    KNOWN_FORMATS,
    serialize_comfy_quant_layer,
    validate_comfy_quant_file,
    write_safetensors,
)
from quantui.quant_methods import (
    COMFY_FORMATS,
    Backend,
    comfy_format,
)

# On-disk schema configs the kitchen worker would emit for each native format.
EXPECTED_KITCHEN_CONFIG = {
    FORMAT_INT8_TENSORWISE: {"format": FORMAT_INT8_TENSORWISE},
    FORMAT_CONVROT_W4A4: {
        "format": FORMAT_CONVROT_W4A4,
        "convrot_groupsize": 256,
        "quant_group_size": 64,
    },
    FORMAT_ASYM_W4A8_INT8: {
        "format": FORMAT_ASYM_W4A8_INT8,
        "group_size": 16,
        "convrot_groupsize": 256,
    },
}

# Which weight-suffix tensors each on-disk format must carry.
EXPECTED_SUFFIX_TENSORS = {
    FORMAT_INT8_TENSORWISE: {"weight": (4, [8]), "weight_scale": (4, [8])},
    FORMAT_CONVROT_W4A4: {"weight": (4, [8]), "weight_scale": (4, [8])},
    FORMAT_ASYM_W4A8_INT8: {"weight": (4, [8]), "weight_s_rel": (4, [8])},
}


def _write_kitchen_artifact(tmp_path, prefix, format_name):
    """Synthesize a valid ``.comfy_quant`` safetensors for ``format_name``."""
    cfg = EXPECTED_KITCHEN_CONFIG[format_name]
    local = {}
    for name, (nbytes, shape) in EXPECTED_SUFFIX_TENSORS[format_name].items():
        local[name] = ("F32", list(shape), b"\x00" * nbytes)
    specs = serialize_comfy_quant_layer(prefix, local, cfg)
    path = tmp_path / f"{prefix}.safetensors"
    path.write_bytes(write_safetensors(specs))
    return path


def test_all_format_ids_unique():
    ids = [f.id for f in COMFY_FORMATS]
    assert len(ids) == len(set(ids)), f"duplicate format ids: {ids}"


def test_expected_formats_registered():
    registered = {f.id for f in COMFY_FORMATS}
    # 6 original + int8_block upgrade + int8_convrot + w4a4_convrot +
    # w4a8_asym + onthefly passthrough == 9 (the int8_row alias was removed).
    for fid in (
        "fp8_e4m3", "int8_block", "int8_tensor",
        "int8_convrot", "nvfp4", "mxfp8", "w4a4_convrot", "w4a8_asym", "onthefly",
    ):
        assert fid in registered, f"{fid} missing from COMFY_FORMATS"


@pytest.mark.parametrize("fmt", COMFY_FORMATS, ids=[f.id for f in COMFY_FORMATS])
def test_backend_and_needs_invariants(fmt):
    assert fmt.backend in (Backend.CTQ, Backend.COMFY_KITCHEN)

    wants_kitchen = "comfy_kitchen" in fmt.needs
    if fmt.backend == Backend.COMFY_KITCHEN:
        # Kitchen is the ONLY path for W4A4/W4A8 and must declare the need.
        assert wants_kitchen, f"{fmt.id}: kitchen backend must need 'comfy_kitchen'"
        assert fmt.quant_format is not None, f"{fmt.id}: kitchen format must set quant_format"
    else:
        # CTQ-backend (convert_to_quant) formats must NOT require comfy-kitchen.
        assert not wants_kitchen, f"{fmt.id}: CTQ backend must not need 'comfy_kitchen'"

    if fmt.quant_format is not None:
        assert fmt.quant_format in KNOWN_FORMATS, (
            f"{fmt.id}: quant_format {fmt.quant_format!r} not a known schema format"
        )


def test_kitchen_formats_are_w4a4_or_w4a8():
    kitchen = [f for f in COMFY_FORMATS if f.backend == Backend.COMFY_KITCHEN]
    assert {f.id for f in kitchen} == {"w4a4_convrot", "w4a8_asym"}
    for f in kitchen:
        assert f.quant_format in (FORMAT_CONVROT_W4A4, FORMAT_ASYM_W4A8_INT8)


@pytest.mark.parametrize(
    "format_name",
    [FORMAT_INT8_TENSORWISE, FORMAT_CONVROT_W4A4, FORMAT_ASYM_W4A8_INT8],
)
def test_synthesized_kitchen_artifact_validates(tmp_path, format_name):
    path = _write_kitchen_artifact(tmp_path, "model.diffusers.linear", format_name)
    result = validate_comfy_quant_file(str(path))
    assert result["ok"], result["errors"]
    assert format_name in result["formats_found"]
    # The file must actually be a readable safetensors blob (8-byte magic + JSON hdr).
    with open(path, "rb") as fh:
        magic = struct.unpack("<Q", fh.read(8))[0]
        json.loads(fh.read(magic).decode("utf-8"))


def test_int8_convrot_maps_to_int8_tensorwise_schema():
    fmt = comfy_format("int8_convrot")
    assert fmt.quant_format == FORMAT_INT8_TENSORWISE
    assert "--convrot" in fmt.base_flags
    # Exactly one ConvRot entry: the int8_row alias was deliberately removed.
    assert not any(f.id == "int8_row" for f in COMFY_FORMATS)


def test_onthefly_passthrough_has_no_quant_format():
    fmt = comfy_format("onthefly")
    assert fmt.backend == Backend.CTQ
    assert fmt.quant_format is None
    assert "--passthrough" in fmt.base_flags
