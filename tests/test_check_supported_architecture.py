"""QA behavioral test for worker.check_supported_architecture.

This is a TEMPORARY verification script (does NOT modify worker.py).
It exercises the pre-flight architecture guard with crafted model dirs,
using unittest.mock to make the test independent of the installed
transformers version. The real worker.fail (which calls sys.exit(1)) is
monkeypatched to raise a sentinel so we can capture the message.

Run with:  python test_check_supported_architecture.py
"""

import json
import os
import sys
import tempfile
import types
from unittest import mock

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from quantui import worker  # noqa: E402  (safe: top-level imports are only stdlib)


# Sentinel raised by the fake fail() so we avoid sys.exit(1) during tests.
class FailSentinel(Exception):
    pass


captured_messages = []


def fake_fail(msg):
    captured_messages.append(msg)
    raise FailSentinel(msg)


# Replace the real fail() (which would call sys.exit(1)) with the sentinel version.
worker.fail = fake_fail


def make_model_dir(config_dict):
    d = tempfile.mkdtemp(prefix="qa_model_")
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f)
    return d


def fake_from_pretrained_factory(config_dict):
    def _fake(*args, **kwargs):
        cfg = types.SimpleNamespace()
        cfg.architectures = config_dict.get("architectures", [])
        cfg.model_type = config_dict.get("model_type", None)
        return cfg
    return _fake


def run_case(name, model_dir, from_pretrained_fake, expect_fail, expect_substr=None):
    captured_messages.clear()
    with mock.patch("transformers.AutoConfig.from_pretrained", from_pretrained_fake):
        try:
            worker.check_supported_architecture(model_dir)
            raised = False
        except FailSentinel:
            raised = True

    print(f"=== Case: {name} ===")
    if raised:
        print("RESULT: fail() WAS triggered")
        msg = captured_messages[0] if captured_messages else ""
        print("CAPTURED MESSAGE:")
        print(msg)
        if expect_fail:
            ok = (expect_substr is None) or (expect_substr in msg)
            print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
                  f"(expected substr: {expect_substr!r} -> present={ok})")
        else:
            print("VERDICT: FAIL (did NOT expect fail() to be triggered)")
    else:
        print("RESULT: fail() was NOT triggered (function returned normally)")
        print(f"VERDICT: {'PASS' if not expect_fail else 'FAIL'}")
    print()


def main():
    overall = []

    # Case 1: Non-causal / VibeVoice (Seq2Seq / multimodal TTS model)
    vibevoice_dir = make_model_dir(
        {"architectures": ["VibeVoiceForConditionalGeneration"], "model_type": "vibevoice"}
    )
    run_case(
        "Non-causal / VibeVoiceForConditionalGeneration",
        vibevoice_dir,
        from_pretrained_fake=fake_from_pretrained_factory(
            {"architectures": ["VibeVoiceForConditionalGeneration"], "model_type": "vibevoice"}
        ),
        expect_fail=True,
        expect_substr="Unsupported architecture",
    )

    # Case 2: Causal / Llama (supported)
    llama_dir = make_model_dir({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
    run_case(
        "Causal / LlamaForCausalLM",
        llama_dir,
        from_pretrained_fake=fake_from_pretrained_factory(
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        ),
        expect_fail=False,
    )

    # Case 3: Edge - unrecognized config (from_pretrained raises; old transformers branch)
    bad_dir = make_model_dir(
        {"architectures": ["VibeVoiceForConditionalGeneration"], "model_type": "vibevoice"}
    )

    def fake_raise(*args, **kwargs):
        raise ValueError(
            "Unrecognized model architecture 'VibeVoiceForConditionalGeneration'. "
            "This could be because you are using a model that is not supported."
        )

    run_case(
        "Edge - unrecognized config (from_pretrained raises)",
        bad_dir,
        from_pretrained_fake=fake_raise,
        expect_fail=True,
        expect_substr="Could not read this model's configuration",
    )

    # Case 4: Edge - missing config.json (no config to inspect; let Unsloth attempt load)
    empty_dir = tempfile.mkdtemp(prefix="qa_empty_")
    run_case(
        "Edge - missing config.json",
        empty_dir,
        from_pretrained_fake=fake_from_pretrained_factory(
            {"architectures": ["LlamaForCausalLM"], "model_type": "llama"}
        ),  # never invoked; function returns early
        expect_fail=False,
    )


if __name__ == "__main__":
    main()
