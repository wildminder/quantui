"""STEP 1.1: frozen-table tests for the model-audit tensor classifier.

The classification table is evidence-based: it was validated against the real
``Breeze-TTS-2-bf16.safetensors`` checkpoint on 2026-08-27 (1115 tensors ->
linear 558 / vector 424 / bias 60 / other 67 / embedding 4 / head 1 /
linear_review 1). These tests are tripwires: changing a rule or a segment is a
deliberate, reviewed act that must update this table AND the live golden tests.
"""

import pytest

from quantui.model_audit import (
    CATEGORIES,
    DTYPE_ITEMSIZE,
    LINEAR_SEGMENTS,
    classify_tensor,
    tensor_bytes,
)


def test_categories_frozen():
    assert CATEGORIES == (
        "linear",
        "embedding",
        "head",
        "linear_review",
        "vector",
        "bias",
        "quant_meta",
        "quant_scale",
        "other",
    )


def test_linear_segments_frozen():
    # Exact validated set from plan §0.2 — adding a segment is a reviewed act.
    # linear1/linear2 added 2026-08-27 after the VibeVoice-7B audit: they are
    # unambiguous FFN matmul weights (acoustic/semantic tokenizer ffn.linear1/2)
    # that the original Breeze-validated set did not cover. Breeze uses neither
    # name, so the live golden tests are unaffected.
    assert LINEAR_SEGMENTS == frozenset(
        {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "out_proj",
            "in_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "fc1",
            "fc2",
            "dense",
            "linear",
            "linear1",
            "linear2",
            "mlp",
        }
    )


@pytest.mark.parametrize("segment", sorted(LINEAR_SEGMENTS))
def test_classify_every_linear_segment(segment):
    name = f"backbone_model.layers.0.self_attn.{segment}.weight"
    assert classify_tensor(name, [2048, 2048]) == "linear"


@pytest.mark.parametrize(
    ("name", "shape", "expected"),
    [
        # Embeddings: substring match on embed|wte|wpe|tokens in the segment
        # before ".weight" (validated: Breeze has exactly 4 embeddings, incl.
        # inputs_embeds_projector via the "embed" substring).
        ("text_encoder.embed_tokens.weight", [50272, 2048], "embedding"),
        ("depth_decoder.model.embed_tokens.weight", [4096, 2048], "embedding"),
        ("model.wte.weight", [50272, 768], "embedding"),
        ("model.wpe.weight", [1024, 768], "embedding"),
        ("embed_text_tokens.weight", [4096, 2048], "embedding"),
        ("depth_decoder.model.inputs_embeds_projector.weight", [2048, 2048], "embedding"),
        # Heads: anchored ^lm_head$ or ^..._head$ segment.
        ("lm_head.weight", [50272, 2048], "head"),
        ("codebooks_head.weight", [1024, 2048], "head"),
        ("foo_head.weight", [10, 20], "head"),
        # linear_review: 2D .weight of unknown role (Breeze: text_encoder_proj).
        ("text_encoder_proj.weight", [2048, 2048], "linear_review"),
        ("model.layers.0.foo.weight", [64, 64], "linear_review"),
        # Bias / vector.
        ("model.layers.0.self_attn.q_proj.bias", [2048], "bias"),
        ("model.layers.0.input_layernorm.weight", [2048], "vector"),
        # Other: 3D conv weight, scalar.
        ("codec_model.encoder.conv1.weight", [64, 32, 3], "other"),
        ("global_scale", [], "other"),
        # Kitchen quant companions.
        ("model.layers.0.comfy_quant", [37], "quant_meta"),
        ("model.layers.0.weight_scale", [1], "quant_scale"),
        ("model.layers.0.weight_s_rel", [1], "quant_scale"),
        ("model.layers.0.weight_s_channel", [2048], "quant_scale"),
        ("model.layers.0.weight_codebook", [16, 64], "quant_scale"),
        ("model.layers.0.weight_correction", [2048], "quant_scale"),
    ],
)
def test_classify_frozen_table(name, shape, expected):
    assert classify_tensor(name, shape) == expected


@pytest.mark.parametrize(
    ("name", "shape", "expected"),
    [
        # Rule 1 (.bias) fires before any name-based rule.
        ("embed.bias", [2048], "bias"),
        ("lm_head.bias", [50272], "bias"),
        # ndim gate fires before the 2D name rules.
        ("embed_tokens.weight", [2048], "vector"),
        ("lm_head.weight", [50272], "vector"),
    ],
)
def test_classify_rule_order(name, shape, expected):
    assert classify_tensor(name, shape) == expected


def test_dtype_itemsize_frozen():
    assert DTYPE_ITEMSIZE == {
        "F64": 8,
        "F32": 4,
        "BF16": 2,
        "F16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }


def test_tensor_bytes_known_and_unknown_dtype():
    assert tensor_bytes("F32", [2, 3]) == 24
    assert tensor_bytes("BF16", [2048, 2048]) == 2048 * 2048 * 2
    assert tensor_bytes("BOOL", []) == 1  # scalar: prod([]) == 1
    # Unknown dtype -> 0 bytes (callers count such tensors separately).
    assert tensor_bytes("NOPE", [2, 3]) == 0


def test_head_regex_no_false_positives():
    # "overhead" ends in "head" but not "_head" -> must NOT be a head.
    assert classify_tensor("overhead.weight", [10, 10]) == "linear_review"
    assert classify_tensor("model.overhead.weight", [10, 10]) == "linear_review"
    # Anchored positive cases.
    assert classify_tensor("proj_head.weight", [10, 10]) == "head"
    assert classify_tensor("model.proj_head.weight", [10, 10]) == "head"
