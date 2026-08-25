"""P2.1: documented backend-support matrix (data-driven, no execution).

``convert_to_quant`` (Backend.CTQ) is the W8A8/profp8 backend and CANNOT emit W4A4 /
W4A8. ``comfy-kitchen`` (Backend.COMFY_KITCHEN) is the ONLY backend for the native
INT4/INT4-act formats. This test pins that invariant in the registry itself.
"""

from quantui.comfy_quant_schema import (
    FORMAT_ASYM_W4A8_INT8,
    FORMAT_CONVROT_W4A4,
)
from quantui.quant_methods import (
    COMFY_FORMATS,
    Backend,
    comfy_format,
)

KITCHEN_ONLY_FORMATS = {FORMAT_CONVROT_W4A4, FORMAT_ASYM_W4A8_INT8}


def test_convert_to_quant_cannot_emit_w4a4_w4a8():
    for cf in COMFY_FORMATS:
        if cf.backend == Backend.CTQ:
            assert cf.quant_format not in KITCHEN_ONLY_FORMATS, (
                f"{cf.id} is a CTQ backend format but claims a kitchen-only quant_format "
                f"{cf.quant_format!r}"
            )


def test_kitchen_formats_are_w4a4_w4a8():
    kitchen = [cf for cf in COMFY_FORMATS if cf.backend == Backend.COMFY_KITCHEN]
    assert {cf.quant_format for cf in kitchen} == KITCHEN_ONLY_FORMATS
    for cf in kitchen:
        assert "comfy_kitchen" in cf.needs


def test_w4a4_w4a8_registered():
    assert comfy_format("w4a4_convrot").quant_format == FORMAT_CONVROT_W4A4
    assert comfy_format("w4a8_asym").quant_format == FORMAT_ASYM_W4A8_INT8
