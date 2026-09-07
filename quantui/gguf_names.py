"""Generic HF -> GGUF tensor-name mapping + arch detection (S1.1 / S1.2).

Python port of ``quantui-rs crates/quant-core/src/gguf_names.rs`` (reading the
Rust repo is sanctioned cross-referencing; *integrating* it is explicitly not
wanted — the tool stays independent). The naming table is sourced from
llama.cpp and must never drift:

* tensor names: llama.cpp ``gguf-py/tensor_mapping.py`` (SHORTCONV_*, w1/w2/w3,
  operator_norm, ffn_norm, out_proj, q/k_layernorm, embedding_norm arms) +
  ``convert_hf_to_gguf.py`` dense family;
* nested-prefix stripping: generic rule for wrapped multimodal checkpoints
  (``model.language_model.*``, ``model.model.*`` — verified against real
  VibeVoice-1.5B and LFM2.5-VL-3B sources).

Contract: :func:`hf_to_gguf_name` returns ``None`` for names it does not
know — the caller keeps the original name unchanged (the VibeVoice
audio-heads behavior). Nothing is ever guessed. Pure functions, no I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

# Nested language-model wrappers stripped before any matching, so both the
# dense form and the wrapped form share one mapping path.
NESTED_PREFIXES: tuple[str, ...] = ("model.language_model.", "model.model.")

# Top-level (non-layer) tensors — dense `model.*` form and the bare form left
# after nested-prefix stripping.
_TOP_LEVEL_MAP: dict[str, str] = {
    "model.embed_tokens.weight": "token_embd.weight",
    "embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
    "norm.weight": "output_norm.weight",
    # LFM2's pre-embedding norm (tensor_mapping.py:65 -> TOKEN_EMBD_NORM).
    "model.embedding_norm.weight": "token_embd_norm.weight",
    "embedding_norm.weight": "token_embd_norm.weight",
}

# Layer-core path map (segment between `layers.{i}.` and the `.weight`/`.bias`
# suffix) -> GGUF tensor suffix. Verbatim from gguf_names.rs.
_LAYER_CORE_MAP: dict[str, str] = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    # LFM2.5's full-attention layers (tensor_mapping.py:330, :698, :715).
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
    # LFM2 layer cores (tensor_mapping.py:1464-1468, :214; arch `lfm2`,
    # llama-arch.cpp:126).
    "conv.conv": "shortconv.conv",
    "conv.in_proj": "shortconv.in_proj",
    "conv.out_proj": "shortconv.out_proj",
    "feed_forward.w1": "ffn_gate",
    "feed_forward.w2": "ffn_down",
    "feed_forward.w3": "ffn_up",
    "operator_norm": "attn_norm",
    # LFM2's internlm2-style ffn norm (tensor_mapping.py:407).
    "ffn_norm": "ffn_norm",
}

# Rotary caches and other cache tensors are skipped entirely by the exporter.
_SKIP_LAYER_CORES: frozenset[str] = frozenset({"self_attn.rotary_emb.inv_freq"})


def is_wrapped(name: str) -> bool:
    """True when the name carries a nested language-model wrapper prefix."""
    return name.startswith(NESTED_PREFIXES)


def _strip_nested(name: str) -> str:
    for prefix in NESTED_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def is_skippable(name: str) -> bool:
    """True when this tensor must be dropped from the GGUF (rotary caches)."""
    name = _strip_nested(name)
    rest = name
    for prefix in ("model.layers.", "layers."):
        if rest.startswith(prefix):
            rest = rest[len(prefix) :]
            break
    if "." in rest:
        idx, tail = rest.split(".", 1)
        if idx.isdigit():
            return tail in _SKIP_LAYER_CORES
    return False


def hf_to_gguf_name(name: str) -> str | None:
    """Map one HF tensor name to its GGUF name; ``None`` = keep original.

    Pure function: no I/O, deterministic repeat.
    """
    if is_skippable(name):
        return None
    name = _strip_nested(name)

    top = _TOP_LEVEL_MAP.get(name)
    if top is not None:
        return top

    # Layer tensors: `model.layers.{i}.<rest>` (dense) or `layers.{i}.<rest>`
    # (after nested-prefix stripping).
    rest = None
    for prefix in ("model.layers.", "layers."):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            break
    if rest is None:
        return None
    idx_str, dot, tail = rest.partition(".")
    if not dot or not idx_str.isdigit():
        return None

    core, dot2, suffix = tail.rpartition(".")
    if not dot2 or suffix not in ("weight", "bias"):
        core, suffix = tail, ""

    mapped = _LAYER_CORE_MAP.get(core)
    if mapped is None:
        return None
    return f"blk.{idx_str}.{mapped}{('.' + suffix) if suffix else ''}"


# --------------------------------------------------------------------------- #
# S1.2 — arch detection from config.json (stdlib, tolerant)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArchInfo:
    """GGUF architecture + best-effort numeric params from config.json."""

    arch: str  # config.json model_type, else "unknown"
    name: str  # folder basename (general.name)
    numeric: dict  # {gguf_key: value} — only keys actually present

    def meta_pairs(self) -> list[tuple[str, object]]:
        """KV pairs to write: general.* identity + `{arch}.{key}` numerics."""
        out: list[tuple[str, object]] = [
            ("general.architecture", self.arch),
            ("general.name", self.name),
        ]
        for key, value in sorted(self.numeric.items()):
            out.append((f"{self.arch}.{key}", value))
        return out


def _arch_from_classname(cls: str) -> str:
    """Map an HF `architectures[0]` class name to a GGUF arch string."""
    known = {
        "LlamaForCausalLM": "llama",
        "MistralForCausalLM": "llama",
        "MixtralForCausalLM": "llama",
        "Qwen2ForCausalLM": "qwen2",
        "Qwen3ForCausalLM": "qwen2",
        "GemmaForCausalLM": "gemma",
        "Gemma2ForCausalLM": "gemma",
        "PhiForCausalLM": "phi",
        "Phi3ForCausalLM": "phi",
        "GPT2LMHeadModel": "gpt2",
        "BloomForCausalLM": "bloom",
        "FalconForCausalLM": "falcon",
        "StableLmForCausalLM": "stablelm",
        # LFM2 family (LiquidAI): llama.cpp registers `lfm2` (llama-arch.cpp:126).
        "LFM2ForCausalLM": "lfm2",
        "Lfm2ForCausalLM": "lfm2",
        "LFM2VLForConditionalGeneration": "lfm2",
        "Lfm2VLForConditionalGeneration": "lfm2",
    }
    hit = known.get(cls)
    if hit:
        return hit
    stem = cls.split("For", 1)[0]
    return stem.lower() if stem else "unknown"


# config.json key -> GGUF numeric KV key (u64 unless noted)
_NUMERIC_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (config key, gguf suffix, is_float)
    ("max_position_embeddings", "context_length", False),
    ("hidden_size", "embedding_length", False),
    ("num_hidden_layers", "block_count", False),
    ("intermediate_size", "feed_forward_length", False),
    ("num_attention_heads", "attention.head_count", False),
    ("num_key_value_heads", "attention.head_count_kv", False),
    ("rms_norm_eps", "attention.layer_norm_rms_epsilon", True),
    ("vocab_size", "vocab_size", False),
)


def read_arch_config(model_dir: str) -> ArchInfo:
    """Read `<dir>/config.json` — tolerant: missing/malformed -> arch "unknown".

    Never raises for absent or unparseable config content; only a truly
    unreadable *directory* surfaces as an error from ``open`` (OS-level).
    """
    name = os.path.basename(os.path.normpath(model_dir)) or "unknown"
    path = os.path.join(model_dir, "config.json")
    config: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                config = loaded
        except (OSError, ValueError, UnicodeDecodeError):
            config = {}

    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        # Known GGUF arch strings pass through; else fall back to class map.
        known_types = {"llama", "mistral", "mixtral", "qwen2", "qwen3", "gemma",
                       "gemma2", "phi", "phi3", "gpt2", "lfm2"}
        arch = model_type if model_type in known_types else None
        if arch is None:
            archs = config.get("architectures")
            cls = archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else ""
            arch = _arch_from_classname(cls) if cls else model_type
    else:
        archs = config.get("architectures")
        cls = archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else ""
        arch = _arch_from_classname(cls) if cls else "unknown"

    numeric: dict = {}
    for cfg_key, gguf_key, is_float in _NUMERIC_KEYS:
        value = config.get(cfg_key)
        if isinstance(value, bool):
            continue  # bool is an int subclass; never a real numeric here
        if is_float and isinstance(value, (int, float)):
            numeric[gguf_key] = float(value)
        elif not is_float and isinstance(value, int):
            numeric[gguf_key] = value
    return ArchInfo(arch=arch, name=name, numeric=numeric)
