"""STEP 2.3: frozen golden tests against the real Breeze-TTS-2 checkpoints.

Every number below was frozen from the 2026-08-27 investigation of the real
files; these tests reconcile the classifier with the official int8-hybrid
build (378 quantized tensors = backbone_model 196 + text_encoder 182). They
run on the machine that holds the checkpoints and skip cleanly elsewhere, so
the headless GATE stays hermetic either way.
"""

from pathlib import Path

import pytest

from quantui.model_audit import audit_file, suggest_exclusions

BF16 = Path("models/tts/Breeze-TTS-2-comfyui/Breeze-TTS-2-bf16.safetensors")
HYBRID = Path(
    "models/tts/Breeze-TTS-2-comfyui/Breeze-TTS-2-int8-hybrid.safetensors"
)

breeze_files = pytest.mark.skipif(
    not (BF16.exists() and HYBRID.exists()),
    reason="Breeze files not present",
)


@breeze_files
def test_live_bf16_categories():
    report = audit_file(str(BF16))
    assert len(report.tensors) == 1115
    assert report.category_counts == {
        "linear": 558,
        "vector": 424,
        "bias": 60,
        "other": 67,
        "embedding": 4,
        "head": 1,
        "linear_review": 1,
    }


@breeze_files
def test_live_bf16_module_linears():
    report = audit_file(str(BF16))
    linears_per_module: dict[str, int] = {}
    for info in report.tensors:
        if info.category == "linear":
            linears_per_module[info.module] = linears_per_module.get(info.module, 0) + 1
    assert linears_per_module == {
        "backbone_model": 196,
        "text_encoder": 182,
        "depth_decoder": 84,
        "codec_model": 96,
    }


@breeze_files
def test_live_bf16_suggestion():
    report = audit_file(str(BF16))
    suggestion = suggest_exclusions(report)
    assert set(suggestion.names) == {
        "depth_decoder.model.embed_tokens.weight",
        "depth_decoder.model.inputs_embeds_projector.weight",
        "embed_text_tokens.weight",
        "text_encoder.embed_tokens.weight",
        "lm_head.weight",
        "text_encoder_proj.weight",
    }
    assert suggestion.rationale == {"embedding": 4, "head": 1, "linear_review": 1}


@breeze_files
def test_live_hybrid_quantized_detection():
    report = audit_file(str(HYBRID))
    assert len(report.quantized_layers) == 378
    assert report.quant_format_histogram == {"int8_tensorwise": 378}
    # Every layer is convrot-enabled. NOTE (deviation from the plan's frozen
    # "convrot_groupsize=256 for all"): the real checkpoint carries a MIXED
    # groupsize — 248 layers at 256 and 130 at 64 (verified on-disk
    # 2026-08-27). The frozen number was stale; the real file is ground truth.
    from collections import Counter

    groupsize_hist = Counter(c.get("convrot_groupsize") for _, c in report.quantized_layers)
    for _prefix, config in report.quantized_layers:
        assert config.get("convrot") is True
    assert dict(groupsize_hist) == {256: 248, 64: 130}


@breeze_files
def test_live_hybrid_reproduces_official_exclusion():
    """The hybrid's unquantized 2D .weight linear remainder == bf16 linears - 378.

    Proves the tool's inventory reconciles with the official build: the 180
    remaining linears are exactly the depth_decoder (84) + codec_model (96)
    modules the authors excluded for pipeline reasons (stage-2 decisions).
    """
    bf16_report = audit_file(str(BF16))
    hybrid_report = audit_file(str(HYBRID))

    bf16_linears = {t.name for t in bf16_report.tensors if t.category == "linear"}
    assert len(bf16_linears) == 558

    quantized_weights = {prefix + ".weight" for prefix, _ in hybrid_report.quantized_layers}
    assert len(quantized_weights) == 378
    assert quantized_weights <= bf16_linears  # zero false negatives vs. official build

    hybrid_linears = {t.name for t in hybrid_report.tensors if t.category == "linear"}
    unquantized_remainder = hybrid_linears - quantized_weights
    assert unquantized_remainder == bf16_linears - quantized_weights
    assert len(unquantized_remainder) == 180
    assert all(
        name.startswith(("depth_decoder.", "codec_model.")) for name in unquantized_remainder
    )
