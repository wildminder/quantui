"""P0.1: the schema doc must stay in sync with the code constants.

If someone changes a format string or group size in ``comfy_quant_schema.py`` they
must update ``docs/comfy-quant-schema.md`` (and vice versa). This test pins that.
"""

import os

import pytest

from quantui.comfy_quant_schema import (
    CONVROT_GROUP_SIZE,
    KNOWN_FORMATS,
    W4A4_QUANT_GROUP_SIZE,
    W4A8_QUANT_GROUP_SIZE,
)

DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "comfy-quant-schema.md")

DOC_MISSING = not os.path.isfile(DOC)
pytestmark = pytest.mark.skipif(
    DOC_MISSING, reason="docs/comfy-quant-schema.md not present (untracked)"
)


def _read_doc() -> str:
    with open(DOC, encoding="utf-8") as fh:
        return fh.read()


def test_doc_exists():
    assert os.path.isfile(DOC), "docs/comfy-quant-schema.md missing"


def test_doc_lists_every_format():
    text = _read_doc()
    for fmt in KNOWN_FORMATS:
        assert fmt in text, f"format {fmt!r} missing from schema doc"


def test_doc_lists_group_sizes():
    text = _read_doc()
    assert str(CONVROT_GROUP_SIZE) in text
    assert str(W4A4_QUANT_GROUP_SIZE) in text
    assert str(W4A8_QUANT_GROUP_SIZE) in text


def test_doc_lists_all_three_format_rows():
    text = _read_doc()
    # The "Three supported formats" table enumerates exactly these.
    for row in ("int8_tensorwise", "convrot_w4a4", "asym_w4a8_int8"):
        assert row in text
