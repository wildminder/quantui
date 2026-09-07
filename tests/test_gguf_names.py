"""S1.1/S1.2 tests: generic HF→GGUF name mapping + stdlib arch config.

The mapping table is a verbatim port of quantui-rs ``gguf_names.rs``; the
fixtures below are the Rust doc examples plus the user's VibeVoice-1.5B
oracle-file names (the audio side passes through unchanged — the headline
contract). Arch tests pin the tolerant config.json reader.
"""

import json

import pytest

from quantui.gguf_names import (
    NESTED_PREFIXES,
    ArchInfo,
    hf_to_gguf_name,
    is_skippable,
    is_wrapped,
    read_arch_config,
)


# ---------------------------------------------------------------- S1.1 ---- #
def test_nested_wrapper_stripped():
    """Both wrapped forms strip to the shared mapping path (README examples)."""
    assert hf_to_gguf_name("model.language_model.layers.3.self_attn.q_proj.weight") == "blk.3.attn_q.weight"
    assert hf_to_gguf_name("model.model.layers.1.self_attn.o_proj.weight") == "blk.1.attn_output.weight"


def test_wrapped_detector():
    assert is_wrapped("model.language_model.norm.weight")
    assert is_wrapped("model.model.layers.0.mlp.gate_proj.weight")
    assert not is_wrapped("model.layers.0.mlp.gate_proj.weight")
    assert set(NESTED_PREFIXES) == {"model.language_model.", "model.model."}


def test_toplevel_map():
    """Dense and bare top-level forms both map."""
    assert hf_to_gguf_name("model.embed_tokens.weight") == "token_embd.weight"
    assert hf_to_gguf_name("embed_tokens.weight") == "token_embd.weight"
    assert hf_to_gguf_name("lm_head.weight") == "output.weight"
    assert hf_to_gguf_name("model.norm.weight") == "output_norm.weight"
    assert hf_to_gguf_name("norm.weight") == "output_norm.weight"
    assert hf_to_gguf_name("model.embedding_norm.weight") == "token_embd_norm.weight"
    assert hf_to_gguf_name("embedding_norm.weight") == "token_embd_norm.weight"


def test_layer_core_table():
    """Every core maps for .weight and .bias; layer index carried."""
    cases = {
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "self_attn.out_proj": "attn_output",
        "self_attn.q_layernorm": "attn_q_norm",
        "self_attn.k_layernorm": "attn_k_norm",
        "self_attn.q_norm": "attn_q_norm",
        "self_attn.k_norm": "attn_k_norm",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "mlp.gate": "ffn_gate_inp",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }
    for core, gguf in cases.items():
        assert hf_to_gguf_name(f"model.layers.7.{core}.weight") == f"blk.7.{gguf}.weight"
        assert hf_to_gguf_name(f"layers.7.{core}.bias") == f"blk.7.{gguf}.bias"
    assert hf_to_gguf_name("model.layers.0.input_layernorm.weight") == "blk.0.attn_norm.weight"


def test_lfm2_cores():
    """LFM2 shortconv + feed_forward w1/w2/w3 + operator_norm/ffn_norm arms."""
    assert hf_to_gguf_name("model.layers.0.conv.conv.weight") == "blk.0.shortconv.conv.weight"
    assert hf_to_gguf_name("model.layers.0.conv.in_proj.weight") == "blk.0.shortconv.in_proj.weight"
    assert hf_to_gguf_name("model.layers.0.conv.out_proj.weight") == "blk.0.shortconv.out_proj.weight"
    assert hf_to_gguf_name("model.layers.0.feed_forward.w1.weight") == "blk.0.ffn_gate.weight"
    assert hf_to_gguf_name("model.layers.0.feed_forward.w2.weight") == "blk.0.ffn_down.weight"
    assert hf_to_gguf_name("model.layers.0.feed_forward.w3.weight") == "blk.0.ffn_up.weight"
    assert hf_to_gguf_name("model.layers.0.operator_norm.weight") == "blk.0.attn_norm.weight"
    assert hf_to_gguf_name("model.layers.0.ffn_norm.weight") == "blk.0.ffn_norm.weight"


def test_unmapped_passes_through():
    """Unknown names -> None (caller keeps original) — VibeVoice audio heads."""
    assert hf_to_gguf_name("model.acoustic_tokenizer.decoder.head.conv.conv.weight") is None
    assert hf_to_gguf_name("model.acoustic_connector.fc1.weight") is None
    assert hf_to_gguf_name("model.prediction_head.head.weight") is None
    assert hf_to_gguf_name("model.language_model.layers.3.weird_thing.weight") is None
    assert hf_to_gguf_name("totally.random.name.weight") is None


def test_skip_rotary():
    """Rotary caches are skippable (distinct contract from pass-through)."""
    assert is_skippable("model.layers.2.self_attn.rotary_emb.inv_freq")
    assert is_skippable("model.language_model.layers.2.self_attn.rotary_emb.inv_freq")
    assert not is_skippable("model.layers.2.self_attn.q_proj.weight")
    # skip wins over mapping: hf_to_gguf_name returns None for skips too
    assert hf_to_gguf_name("model.layers.2.self_attn.rotary_emb.inv_freq") is None


def test_bias_suffix_carried():
    assert hf_to_gguf_name("model.layers.0.self_attn.q_proj.bias") == "blk.0.attn_q.bias"
    assert hf_to_gguf_name("model.layers.0.mlp.down_proj.bias") == "blk.0.ffn_down.bias"


def test_purity():
    """Deterministic repeat (no hidden state / I/O)."""
    name = "model.language_model.layers.9.mlp.gate_proj.weight"
    first = hf_to_gguf_name(name)
    assert first == "blk.9.ffn_gate.weight"
    assert all(hf_to_gguf_name(name) == first for _ in range(5))


# ---------------------------------------------------------------- S1.2 ---- #
def _write_config(tmp_path, config: dict) -> str:
    d = tmp_path / "model-dir"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(d)


def test_vibevoice_config(tmp_path):
    """The headline arch: model_type vibevoice + numerics -> meta_pairs emits
    vibevoice.* KVs (matches the oracle file's KV scheme)."""
    d = _write_config(
        tmp_path,
        {
            "model_type": "vibevoice",
            "architectures": ["VibeVoiceForConditionalGeneration"],
            "num_hidden_layers": 30,
            "hidden_size": 1536,
            "max_position_embeddings": 8192,
            "rms_norm_eps": 1e-06,
        },
    )
    info = read_arch_config(d)
    assert info.arch == "vibevoice"
    assert info.name == "model-dir"
    assert info.numeric["block_count"] == 30
    assert info.numeric["embedding_length"] == 1536
    pairs = dict(info.meta_pairs())
    assert pairs["general.architecture"] == "vibevoice"
    assert pairs["vibevoice.block_count"] == 30
    assert pairs["vibevoice.attention.layer_norm_rms_epsilon"] == pytest.approx(1e-06)


def test_llama_config_minimal(tmp_path):
    """Only model_type -> known arch, empty numerics, still converts."""
    d = _write_config(tmp_path, {"model_type": "llama"})
    info = read_arch_config(d)
    assert info.arch == "llama"
    assert info.numeric == {}
    assert dict(info.meta_pairs())["general.architecture"] == "llama"


def test_unknown_type_falls_back_to_classname(tmp_path):
    """model_type not a known GGUF arch -> architectures[0] class map."""
    d = _write_config(
        tmp_path,
        {"model_type": "newthing", "architectures": ["NewthingForCausalLM"]},
    )
    assert read_arch_config(d).arch == "newthing"
    d2 = _write_config(tmp_path, {"model_type": "vibevoice"})
    assert read_arch_config(d2).arch == "vibevoice"  # passthrough when no archs


def test_missing_config_file(tmp_path):
    """No config.json -> arch unknown, no raise."""
    d = tmp_path / "empty-dir"
    d.mkdir()
    info = read_arch_config(str(d))
    assert info.arch == "unknown"
    assert info.numeric == {}
    assert dict(info.meta_pairs())["general.architecture"] == "unknown"


def test_malformed_config(tmp_path):
    """Invalid JSON -> tolerant arch unknown (pinned contract)."""
    d = tmp_path / "bad-dir"
    d.mkdir()
    (d / "config.json").write_text("{not json", encoding="utf-8")
    info = read_arch_config(str(d))
    assert info.arch == "unknown"


def test_numeric_omitted_when_absent(tmp_path):
    """No phantom keys: numeric dict equals exactly the consumed subset."""
    d = _write_config(tmp_path, {"model_type": "llama", "num_hidden_layers": 4})
    info = read_arch_config(d)
    assert info.numeric == {"block_count": 4}


def test_numeric_ignores_bools_and_strings(tmp_path):
    d = _write_config(
        tmp_path,
        {"model_type": "llama", "num_hidden_layers": True, "hidden_size": "big"},
    )
    assert read_arch_config(d).numeric == {}


def test_archinfo_frozen():
    """Dataclass frozen (accidental-mutation tripwire)."""
    import dataclasses

    info = ArchInfo(arch="llama", name="m", numeric={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.arch = "gemma"  # type: ignore[misc]
