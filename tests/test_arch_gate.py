"""Regression tests for the architecture pre-flight gate (check_supported_architecture).

History: the gate used to accept ONLY ``*ForCausalLM`` architectures. That
refused LFM2.5-VL (``Lfm2VlForConditionalGeneration``) even though the very
unsloth/llama.cpp stack we ship converts it fine (language GGUF + mmproj GGUF).
Found by the 2026-09-04 LFM2.5-VL-3B conformance run; the gate is now a
BLOCKLIST on genuinely un-convertible model types (TTS / Seq2Seq / audio),
not an allowlist on the arch-name suffix.

The tests monkeypatch ``transformers.AutoConfig.from_pretrained`` so they run
in the headless venv without torch/unsloth.
"""

import json
import types
from unittest import mock

import pytest

from quantui import worker as wk


def _model_dir(tmp_path, config_dict):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config_dict), encoding="utf-8")
    return str(d)


def _cfg(config_dict):
    cfg = types.SimpleNamespace()
    cfg.architectures = config_dict.get("architectures", [])
    cfg.model_type = config_dict.get("model_type", None)
    return cfg


def _run(tmp_path, config_dict, monkeypatch):
    """Returns the friendly error string, or None when the gate passes."""
    captured = {}

    def fake_fail(msg):
        captured["msg"] = msg
        raise SystemExit(1)

    monkeypatch.setattr(wk, "fail", fake_fail)
    with mock.patch("transformers.AutoConfig.from_pretrained", lambda *a, **k: _cfg(config_dict)):
        try:
            wk.check_supported_architecture(_model_dir(tmp_path, config_dict))
        except SystemExit:
            return captured["msg"]
    return None


def test_lfm2_vl_conditional_generation_passes(tmp_path, monkeypatch):
    """THE regression: Lfm2VlForConditionalGeneration must NOT be refused."""
    msg = _run(
        tmp_path,
        {"architectures": ["Lfm2VlForConditionalGeneration"], "model_type": "lfm2_vl"},
        monkeypatch,
    )
    assert msg is None


def test_gemma3_conditional_generation_passes(tmp_path, monkeypatch):
    """Other VLM ForConditionalGeneration archs pass too."""
    msg = _run(
        tmp_path,
        {"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3"},
        monkeypatch,
    )
    assert msg is None


def test_llama_for_causal_lm_still_passes(tmp_path, monkeypatch):
    msg = _run(
        tmp_path,
        {"architectures": ["LlamaForCausalLM"], "model_type": "llama"},
        monkeypatch,
    )
    assert msg is None


def test_vibevoice_still_blocked(tmp_path, monkeypatch):
    msg = _run(
        tmp_path,
        {"architectures": ["VibeVoiceForConditionalGeneration"], "model_type": "vibevoice"},
        monkeypatch,
    )
    assert msg is not None
    assert "Unsupported architecture" in msg


def test_t5_blocked(tmp_path, monkeypatch):
    msg = _run(
        tmp_path,
        {"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"},
        monkeypatch,
    )
    assert msg is not None


def test_missing_config_is_noop(tmp_path, monkeypatch):
    """No config.json -> gate stays silent (unsloth reports the real problem)."""
    d = tmp_path / "empty"
    d.mkdir()
    calls = []
    monkeypatch.setattr(wk, "fail", lambda msg: calls.append(msg))
    wk.check_supported_architecture(str(d))
    assert calls == []


def test_unreadable_config_falls_back_to_friendly_error(tmp_path, monkeypatch):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")

    def fake_raise(*a, **k):
        raise ValueError("Unrecognized model architecture 'WhoDisForConditionalGeneration'.")

    captured = {}

    def fake_fail(msg):
        captured["msg"] = msg
        raise SystemExit(1)

    monkeypatch.setattr(wk, "fail", fake_fail)
    with mock.patch("transformers.AutoConfig.from_pretrained", fake_raise):
        with pytest.raises(SystemExit):
            wk.check_supported_architecture(str(d))
    assert "Could not read this model's configuration" in captured["msg"]
