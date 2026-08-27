"""STEP 2.2: deterministic exclude_layers regex suggestion.

The suggestion covers exactly the 2D ``.weight`` keep-set (categories
``embedding`` / ``head`` / ``linear_review``) because ctq only quantizes 2D
``.weight`` tensors. The regex is a single anchored alternation of
``re.escape``d, sorted names so it reproduces the keep-set under the
``re.search`` semantics used by ``QuantConfig.excluded`` (tensor_quant.py:113).
"""

import re

from quantui.comfy_quant_schema import (
    FORMAT_INT8_TENSORWISE,
    default_quant_config,
    serialize_comfy_quant_layer,
    write_safetensors,
)
from quantui.model_audit import ExclusionSuggestion, audit_file, suggest_exclusions


def _write(tmp_path, specs, metadata=None, name="m.safetensors"):
    path = tmp_path / name
    path.write_bytes(write_safetensors(specs, metadata=metadata))
    return str(path)


def _breeze_like_specs():
    """Mimic Breeze: embed table, lm_head, a review proj, 2 linears, norm, bias."""
    return {
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
        "lm_head.weight": ("BF16", [10, 4], b"\x00" * 80),
        "text_encoder_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.layers.0.mlp.down_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.norm.weight": ("BF16", [4], b"\x00" * 8),
        "backbone_model.layers.0.self_attn.q_proj.bias": ("BF16", [4], b"\x00" * 8),
    }


def test_suggest_breeze_like_fixture(tmp_path):
    report = audit_file(_write(tmp_path, _breeze_like_specs()))
    suggestion = suggest_exclusions(report)
    assert isinstance(suggestion, ExclusionSuggestion)

    expected_names = (
        "lm_head.weight",
        "text_encoder.embed_tokens.weight",
        "text_encoder_proj.weight",
    )
    assert suggestion.names == expected_names
    assert suggestion.rationale == {"embedding": 1, "head": 1, "linear_review": 1}

    # Regex matches every keep-tensor under re.search...
    for name in expected_names:
        assert re.search(suggestion.regex, name), name
    # ...and none of the quantizable / keep-verbatim tensors.
    for name in (
        "backbone_model.layers.0.self_attn.q_proj.weight",
        "backbone_model.layers.0.mlp.down_proj.weight",
        "backbone_model.norm.weight",
        "backbone_model.layers.0.self_attn.q_proj.bias",
    ):
        assert not re.search(suggestion.regex, name), name


def test_suggest_empty_when_all_linear(tmp_path):
    specs = {
        "m.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "m.layers.0.mlp.up_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
    }
    report = audit_file(_write(tmp_path, specs))
    suggestion = suggest_exclusions(report)
    assert suggestion.regex == ""
    assert suggestion.names == ()
    assert suggestion.rationale == {}


def test_suggest_anchored_against_search(tmp_path):
    # Keep-set contains exactly x.lm_head.weight; near-misses must NOT match.
    specs = {
        "x.lm_head.weight": ("BF16", [4, 4], b"\x00" * 32),
        "x.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
    }
    report = audit_file(_write(tmp_path, specs))
    regex = suggest_exclusions(report).regex
    assert regex == "^(x\\.lm_head\\.weight)$"
    assert re.search(regex, "x.lm_head.weight")
    # Prefix over-match guard (^ anchor).
    assert not re.search(regex, "prefix_x.lm_head.weight")
    # Suffix over-match guard ($ anchor).
    assert not re.search(regex, "x.lm_head.weight_v2")


def test_suggest_deterministic_order(tmp_path):
    specs = _breeze_like_specs()
    # Rebuild the same logical file with reversed insertion order.
    reversed_specs = dict(reversed(list(specs.items())))
    a = suggest_exclusions(audit_file(_write(tmp_path, specs, name="a.safetensors")))
    b = suggest_exclusions(audit_file(_write(tmp_path, reversed_specs, name="b.safetensors")))
    assert a.regex == b.regex
    assert a.names == b.names


def test_suggest_ignores_quantized_companions(tmp_path):
    specs = serialize_comfy_quant_layer(
        "model.layers.0.mlp",
        {
            "weight": ("I8", [4, 4], b"\x01" * 16),
            "weight_scale": ("F32", [1], b"\x00\x00\x80\x3f"),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    report = audit_file(_write(tmp_path, specs))
    suggestion = suggest_exclusions(report)
    # The I8 weight is a "linear" (quantizable) and the companions are
    # quant_scale / quant_meta — none of them belong in the keep-set.
    assert suggestion.names == ()
    assert suggestion.regex == ""
    for name in report.tensors:
        assert name.name not in suggestion.names
