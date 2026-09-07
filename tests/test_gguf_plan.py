"""S2.3 tests: the deterministic per-tensor conversion plan.

Pins the policy table (plan §S2.3) including the VibeVoice contracts:
unmapped 2-D weights still quantize, non-divisible ne0 rows demote to F16
with the real conv widths (4/7/8/10/16), rotary skips.
"""

import pytest

from quantui.gguf_qkernels import demote_summary, plan_tensor


def test_plan_known_linear_quantizes():
    item = plan_tensor("model.layers.3.self_attn.q_proj.weight", "F32", (2048, 2048), "native_q8_0")
    assert item.action == "quant_q8_0"
    assert item.gguf_name == "blk.3.attn_q.weight"
    assert "2048" in item.reason


def test_plan_unmapped_2d_still_quantizes():
    """VibeVoice audio linears: unknown name, still a quantizable 2-D weight."""
    item = plan_tensor(
        "model.acoustic_tokenizer.decoder.stages.0.1.ffn.linear1.weight",
        "BF16", (128, 128), "native_q8_0",
    )
    assert item.action == "quant_q8_0"
    assert item.gguf_name is None  # kept under original name


def test_plan_demotion_nonmultiple_ne0():
    """Conv row widths 4/7/8/10/16 (the real VibeVoice convs) all demote."""
    for ne0 in (4, 7, 8, 10, 16):
        item = plan_tensor("model.layers.0.conv.conv.weight", "F32", (8, ne0), "native_q8_0")
        assert item.action == "pass_f16", ne0
        assert "not divisible" in item.reason


def test_plan_divisible_ne0_quantizes_q4_0():
    item = plan_tensor("model.layers.1.mlp.down_proj.weight", "F16", (64, 96), "native_q4_0")
    assert item.action == "quant_q4_0"


def test_plan_1d_f32():
    item = plan_tensor("model.layers.0.input_layernorm.weight", "F32", (1536,), "native_q8_0")
    assert item.action == "verbatim"
    assert item.gguf_name == "blk.0.attn_norm.weight"


def test_plan_skip_rotary():
    item = plan_tensor("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", (32,), "native_q8_0")
    assert item.action == "skip"


def test_plan_int_verbatim():
    for dtype in ("I32", "U8", "BOOL"):
        item = plan_tensor("model.embed_tokens.weight", dtype, (100, 4), "native_q8_0")
        assert item.action == "verbatim", dtype


def test_plan_f16_method_no_demotion_needed():
    """f16 method with non-divisible ne0 is just a pass (no fake warning)."""
    item = plan_tensor("model.layers.0.conv.conv.weight", "F32", (8, 10), "native_f16")
    assert item.action == "pass_f16"
    assert "demoted" not in item.reason


def test_plan_f32_method():
    item = plan_tensor("model.layers.0.self_attn.q_proj.weight", "F32", (64, 64), "native_f32")
    assert item.action == "pass_f32"


def test_plan_exotic_dtype():
    item = plan_tensor("model.layers.0.self_attn.q_proj.weight", "F8_E4M3", (64, 64), "native_q8_0")
    assert item.action == "pass_f32"
    assert "exotic" in item.reason


def test_plan_token_embd_stays_f16():
    """llama.cpp convention: token_embd.weight is never method-quantized."""
    item = plan_tensor("model.language_model.embed_tokens.weight", "BF16",
                       (151936, 1536), "native_q8_0")
    assert item.action == "pass_f16"
    assert item.gguf_name == "token_embd.weight"
    assert "F16" in item.reason


def test_plan_unknown_method_raises():
    with pytest.raises(ValueError, match="native_q6_k"):
        plan_tensor("x.weight", "F32", (64, 64), "native_q6_k")


def test_plan_3d_float_demotes():
    """3-D float weights follow the same divisibility rule on the last axis."""
    item = plan_tensor("model.layers.0.attn.conv.weight", "F32", (2, 8, 64), "native_q8_0")
    assert item.action == "quant_q8_0"  # 64 % 32 == 0
    item2 = plan_tensor("model.layers.0.attn.conv.weight", "F32", (2, 8, 30), "native_q8_0")
    assert item2.action == "pass_f16"


def test_demote_summary_counts():
    items = [
        plan_tensor("model.layers.0.self_attn.q_proj.weight", "F32", (64, 64), "native_q8_0"),
        plan_tensor("model.layers.0.self_attn.k_proj.weight", "F32", (64, 64), "native_q8_0"),
        plan_tensor("model.layers.0.conv.conv.weight", "F32", (8, 10), "native_q8_0"),
        plan_tensor("model.layers.0.input_layernorm.weight", "F32", (64,), "native_q8_0"),
        plan_tensor("model.layers.0.self_attn.rotary_emb.inv_freq", "F32", (32,), "native_q8_0"),
    ]
    assert demote_summary(items) == {
        "pass_f16": 1, "quant_q8_0": 2, "skip": 1, "verbatim": 1,
    }
