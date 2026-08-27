"""STEP 2.1: audit_file engine — header scan + quantized-layer detection.

Fixtures are synthetic safetensors files built with the pure-stdlib
``comfy_quant_schema.write_safetensors`` / ``serialize_comfy_quant_layer``
(same pattern as ``tests/test_comfy_quant_schema.py``); no torch, no live files.
"""

import pytest

from quantui.comfy_quant_schema import (
    FORMAT_INT8_TENSORWISE,
    default_quant_config,
    serialize_comfy_quant_layer,
    write_safetensors,
)
from quantui.model_audit import AuditError, AuditReport, audit_file


def _write(tmp_path, specs, metadata=None, name="m.safetensors"):
    path = tmp_path / name
    path.write_bytes(write_safetensors(specs, metadata=metadata))
    return str(path)


def _raw_two_module_specs():
    """A Breeze-like raw bf16 file: 2 modules, one of each basic category."""
    return {
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.norm.weight": ("BF16", [4], b"\x00" * 8),
        "backbone_model.layers.0.self_attn.q_proj.bias": ("BF16", [4], b"\x00" * 8),
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
    }


def test_audit_raw_file(tmp_path):
    path = _write(tmp_path, _raw_two_module_specs())
    report = audit_file(path)

    assert isinstance(report, AuditReport)
    assert report.path == path
    assert len(report.tensors) == 4
    assert report.category_counts == {"linear": 1, "embedding": 1, "vector": 1, "bias": 1}
    assert report.category_bytes == {"linear": 32, "embedding": 80, "vector": 8, "bias": 8}
    assert report.total_bytes == 128
    assert report.quantized_layers == []
    assert report.quant_format_histogram == {}
    # Modules sorted by bytes desc: backbone_model (32+8+8=48) > text_encoder (80)?
    # No: text_encoder embedding is 80 bytes -> text_encoder first.
    assert [m.module for m in report.modules] == ["text_encoder", "backbone_model"]
    assert [m.tensors for m in report.modules] == [1, 3]


def test_audit_quantized_file(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.layers.0.mlp",
        {
            "weight": ("I8", [4, 4], b"\x01" * 16),
            "weight_scale": ("F32", [1], b"\x00\x00\x80\x3f"),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    path = _write(tmp_path, specs)
    report = audit_file(path)

    assert report.quantized_layers == [("model.layers.0.mlp", {"format": FORMAT_INT8_TENSORWISE})]
    assert report.quant_format_histogram == {FORMAT_INT8_TENSORWISE: 1}
    # The I8 weight is still a 2D .weight of a linear segment -> "linear"
    # (frozen rule table: rule 3 covers only the companion suffixes); the
    # scale/blob companions are quant_scale / quant_meta, never model linears.
    by_name = {t.name: t.category for t in report.tensors}
    assert by_name == {
        "model.layers.0.mlp.weight": "linear",
        "model.layers.0.mlp.weight_scale": "quant_scale",
        "model.layers.0.mlp.comfy_quant": "quant_meta",
    }
    assert report.category_counts["linear"] == 1


def test_audit_metadata_preserved(tmp_path):
    meta = {"format": "pt", "recipe": "breeze-hybrid", "n": "42"}
    path = _write(tmp_path, _raw_two_module_specs(), metadata=meta)
    report = audit_file(path)
    assert report.metadata == meta


def test_audit_no_metadata_is_empty_dict(tmp_path):
    path = _write(tmp_path, _raw_two_module_specs())
    assert audit_file(path).metadata == {}


def test_audit_malformed_raises(tmp_path):
    # Truncated file.
    good = _write(tmp_path, _raw_two_module_specs(), name="good.safetensors")
    truncated = tmp_path / "truncated.safetensors"
    truncated.write_bytes(open(good, "rb").read(10))
    with pytest.raises(AuditError) as exc_trunc:
        audit_file(str(truncated))
    assert str(truncated) in str(exc_trunc.value)

    # Non-safetensors garbage bytes.
    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"this is definitely not safetensors")
    with pytest.raises(AuditError) as exc_garbage:
        audit_file(str(garbage))
    assert str(garbage) in str(exc_garbage.value)


def test_audit_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.safetensors"
    with pytest.raises(AuditError) as exc:
        audit_file(str(missing))
    assert str(missing) in str(exc.value)


def test_audit_empty_file(tmp_path):
    path = _write(tmp_path, {}, metadata={"format": "pt"})
    report = audit_file(path)
    assert report.tensors == []
    assert report.modules == []
    assert report.total_bytes == 0
    assert report.category_counts == {}
    assert report.quantized_layers == []
    assert report.metadata == {"format": "pt"}


def test_audit_deterministic(tmp_path):
    specs = serialize_comfy_quant_layer(
        "b.layer",
        {"weight": ("I8", [2, 2], b"\x01" * 4), "weight_scale": ("F32", [1], b"\x00" * 4)},
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    specs.update(_raw_two_module_specs())
    path = _write(tmp_path, specs, metadata={"k": "v"})
    assert repr(audit_file(path)) == repr(audit_file(path))
