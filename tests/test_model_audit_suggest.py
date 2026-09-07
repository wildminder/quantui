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


# --------------------------------------------------------------------------- #
# NTH-007: whitelist-candidate heuristic hint
# --------------------------------------------------------------------------- #
def _vibevoice_like_specs(n_layers: int = 10):
    """Mimic an unknown architecture: many repeated last-segment `ffn.xw_proj`
    2D weights that the frozen whitelist does NOT know (-> linear_review).
    (Uses a still-unknown segment: `linear1`/`linear2` were whitelisted
    2026-08-27 after the VibeVoice-7B audit, so they no longer exercise this.)"""
    specs = {
        "backbone_model.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
    }
    for i in range(n_layers):
        specs[f"backbone_model.layers.{i}.ffn.xw_proj.weight"] = ("BF16", [4, 4], b"\x00" * 32)
    return specs


def test_whitelist_hint_surfaces_repeated_review_segment(tmp_path):
    """NTH-007: >=8 linear_review tensors sharing a last segment and a shape
    produce a candidate hint in the suggestion."""
    report = audit_file(_write(tmp_path, _vibevoice_like_specs(10)))
    s = suggest_exclusions(report)
    assert s.hints, "expected whitelist-candidate hints"
    seg, count, shape = s.hints[0]
    assert seg == "xw_proj"
    assert count == 10
    assert shape == "(4, 4)"


def test_whitelist_hint_below_threshold_is_silent(tmp_path):
    """Fewer than 8 repeats -> no hint (avoids noisy one-off suggestions)."""
    report = audit_file(_write(tmp_path, _vibevoice_like_specs(5)))
    s = suggest_exclusions(report)
    assert s.hints == ()


def test_whitelist_hint_requires_consistent_shape(tmp_path):
    """Same last segment but differing shapes -> no hint (not a naming convention)."""
    specs = {
        "backbone_model.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
    }
    for i in range(10):
        shape = (4, 4) if i % 2 == 0 else (8, 4)
        specs[f"backbone_model.layers.{i}.ffn.xw_proj.weight"] = ("BF16", list(shape), b"\x00" * (shape[0] * shape[1] * 2))
    report = audit_file(_write(tmp_path, specs))
    s = suggest_exclusions(report)
    assert s.hints == ()


def test_whitelist_hint_in_text_and_json_reports(tmp_path):
    from quantui.model_audit import render_json, render_text

    report = audit_file(_write(tmp_path, _vibevoice_like_specs(10)))
    s = suggest_exclusions(report)
    text = render_text(report, s)
    assert "candidate whitelist segment" in text
    assert "xw_proj" in text
    payload = __import__("json").loads(render_json(report, s))
    assert payload["suggestion"]["hints"] == [{"segment": "xw_proj", "count": 10, "shape": "(4, 4)"}]
