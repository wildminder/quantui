"""Section-header tests (plan 2026-09-09-control-panel S2.1).

The numbered section Labels ("1. Model path", ...) become instrument-group
strips: they carry the ``section_header`` style class and MAIN_CSS tints them
with the panel color. No widget-tree changes (class approach only), so the
widget contract tests are untouched.
"""

from quantui import panels
from quantui.app import QuantApp
from quantui.app_css import MAIN_CSS


def test_section_headers_have_class():
    """Numbered section labels in panels.py carry classes="section_header"
    (>= 6: three GGUF + three Comfy minimum)."""
    src = open(panels.__file__, encoding="utf-8").read()
    count = src.count('classes="section_header"')
    assert count >= 6, f"only {count} section_header labels in panels.py"
    # Each numbered section header is paired with the class.
    for label in ("1. Model path", "2. Output folder", "3. Quantization method",
                  "1. Input", "2. Output", "3. Format"):
        idx = src.find(label)
        assert idx != -1, f"label {label!r} missing from panels.py"
        window = src[idx: idx + 200]
        assert "section_header" in window, (
            f"label {label!r} does not carry classes= within its constructor call"
        )


def test_section_header_css_pins():
    """MAIN_CSS styles .section_header: panel-tinted strip, bold, $text color,
    full width, padding."""
    block = MAIN_CSS.split(".section_header {", 1)[1].split("}", 1)[0]
    assert "background: $panel" in block
    assert "text-style: bold" in block
    assert "color: $text" in block
    assert "width: 1fr" in block


def test_section_header_class_applied_at_runtime(tmp_path, monkeypatch):
    """A booted app's GGUF panel carries .section_header labels (no tree change:
    the same Labels, restyled)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = QuantApp()
    import asyncio

    async def main():
        async with a.run_test():
            gguf = a.query_one("#gguf_panel")
            headers = list(gguf.query(".section_header"))
            assert len(headers) >= 3
            texts = [str(h.content) for h in headers]
            assert any("1. Model path" in t for t in texts)

    asyncio.run(main())
