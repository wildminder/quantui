"""STEP 3.1: deterministic text + JSON report renderers."""

import json

from quantui.comfy_quant_schema import (
    FORMAT_INT8_TENSORWISE,
    default_quant_config,
    serialize_comfy_quant_layer,
    write_safetensors,
)
from quantui.model_audit import (
    audit_file,
    render_json,
    render_text,
    suggest_exclusions,
)


def _write(tmp_path, specs, metadata=None, name="m.safetensors"):
    path = tmp_path / name
    path.write_bytes(write_safetensors(specs, metadata=metadata))
    return str(path)


def _two_module_specs():
    return {
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.norm.weight": ("BF16", [4], b"\x00" * 8),
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
        "lm_head.weight": ("BF16", [10, 4], b"\x00" * 80),
    }


def test_render_text_sections(tmp_path):
    report = audit_file(_write(tmp_path, _two_module_specs()))
    suggestion = suggest_exclusions(report)
    text = render_text(report, suggestion)

    # Title + totals + every section header.
    assert "Model audit: m.safetensors" in text
    assert "Tensors: 4" in text
    assert "Categories:" in text
    assert "Modules:" in text
    assert "Suggested exclude_layers" in text
    # Regex line verbatim, prefixed.
    assert f"exclude_layers: {suggestion.regex}" in text
    # Module rows in bytes-desc order: text_encoder (160 B) before backbone (40 B).
    assert text.index("text_encoder") < text.index("backbone_model")
    # Raw file -> no quantized section.
    assert "Quantized layers:" not in text


def test_render_text_no_quant_section_for_raw(tmp_path):
    report = audit_file(_write(tmp_path, _two_module_specs()))
    text = render_text(report, suggest_exclusions(report))
    assert "Quantized layers:" not in text


def test_render_text_quant_section_for_quantized(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.layers.0.mlp",
        {
            "weight": ("I8", [4, 4], b"\x01" * 16),
            "weight_scale": ("F32", [1], b"\x00\x00\x80\x3f"),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    report = audit_file(_write(tmp_path, specs))
    text = render_text(report, suggest_exclusions(report))
    assert "Quantized layers: 1" in text
    assert f"{FORMAT_INT8_TENSORWISE}: 1" in text


def test_render_json_roundtrip(tmp_path):
    report = audit_file(_write(tmp_path, _two_module_specs(), metadata={"format": "pt"}))
    suggestion = suggest_exclusions(report)
    payload = json.loads(render_json(report, suggestion))

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
    assert payload["metadata"] == {"format": "pt"}
    assert payload["totals"]["tensors"] == 4
    assert payload["category_counts"] == {"linear": 1, "embedding": 1, "head": 1, "vector": 1}
    assert payload["quantized_layers"] == []
    assert payload["suggestion"]["names"] == list(suggestion.names)
    assert payload["suggestion"]["regex"] == suggestion.regex
    # Modules serialized bytes-desc: lm_head 80 B, text_encoder 80 B (name
    # tie-break), backbone_model 40 B.
    assert [m["module"] for m in payload["modules"]] == [
        "lm_head",
        "text_encoder",
        "backbone_model",
    ]

    # sort_keys stability: two renders byte-identical.
    assert render_json(report, suggestion) == render_json(report, suggestion)


def test_render_text_bytes_humanized(tmp_path):
    # 7.0 GiB exactly -> "7.00 GiB" (frozen 2-decimal humanization).
    seven_gib = 7 * 2**30
    specs = {"big.q_proj.weight": ("F32", [seven_gib // 4], b"\x00" * 0)}
    # Shape-only accounting: nbytes comes from dtype*shape, payload may be empty.
    report = audit_file(_write(tmp_path, specs))
    assert report.total_bytes == seven_gib
    text = render_text(report, suggest_exclusions(report))
    assert "7.00 GiB" in text


# --------------------------------------------------------------------------- #
# NTH-011: per-module quantized-ratio column
# --------------------------------------------------------------------------- #
def test_module_table_has_quantized_ratio_column(tmp_path):
    """NTH-011: the Modules table shows a `q%` column (share of the module's
    params that are already quantized), so partially-quantized checkpoints are
    obvious at a glance."""
    report = audit_file(_write(tmp_path, _two_module_specs()))
    text = render_text(report, suggest_exclusions(report))
    header = next(ln for ln in text.splitlines() if ln.strip().startswith("module"))
    assert "q%" in header, header


def test_module_ratio_zero_for_raw_file(tmp_path):
    """A raw (unquantized) file shows 0.0% for every module."""
    report = audit_file(_write(tmp_path, _two_module_specs()))
    text = render_text(report, suggest_exclusions(report))
    backbone_row = next(ln for ln in text.splitlines() if "backbone_model" in ln)
    assert "0.0%" in backbone_row


def test_module_ratio_partial_for_hybrid_checkpoint(tmp_path):
    """A file with ONE quantized layer in a multi-linear module shows a
    partial ratio. The q% column measures the share of the module's
    MATRIX params (linear+linear_review weights) that are quantized:
    backbone_model has two 4x4 linears; quantizing one -> 50.0%."""
    # backbone_model has one 4x4 linear (q_proj) + one 4-vector norm.
    specs = _two_module_specs()
    # Add a second linear to backbone_model so its linear count is 2.
    specs["backbone_model.layers.1.mlp.down_proj.weight"] = ("BF16", [4, 4], b"\x00" * 32)
    path = _write(tmp_path, specs)
    report = audit_file(path)
    assert report.total_bytes > 0

    # Quantize exactly ONE of backbone_model's two linears.
    quantized = serialize_comfy_quant_layer(
        "backbone_model.layers.0.self_attn.q_proj",
        {
            "weight": ("I8", [4, 4], b"\x01" * 16),
            "weight_scale": ("F32", [1], b"\x00\x00\x80\x3f"),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    # Rebuild the file with the quantized layer replacing the BF16 q_proj.
    specs2 = {k: v for k, v in specs.items()
              if k != "backbone_model.layers.0.self_attn.q_proj.weight"}
    specs2.update(quantized)
    report2 = audit_file(_write(tmp_path, specs2, name="hybrid.safetensors"))
    text = render_text(report2, suggest_exclusions(report2))
    backbone_row = next(ln for ln in text.splitlines() if "backbone_model" in ln)
    assert "50.0%" in backbone_row, backbone_row
