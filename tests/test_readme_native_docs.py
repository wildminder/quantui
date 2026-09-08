"""Docs sanity tests (S4.4): the README documents the native GGUF backend.

A grep-equivalent source check — cheap, and it pins the user-facing contract
that the native backend is discoverable and described honestly (no
transformers; demotion rule; four method ids).
"""

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _readme() -> str:
    with open(os.path.join(PROJECT_DIR, "README.md"), encoding="utf-8") as fh:
        return fh.read()


def test_readme_has_native_backend_section():
    text = _readme()
    assert "## Native GGUF backend" in text
    assert "no transformers" in text.lower() or "without transformers" in text.lower()


def test_readme_documents_all_five_native_methods():
    text = _readme()
    for mid in ("native_q8_0", "native_q4_0", "native_f16", "native_bf16", "native_f32"):
        assert mid in text, mid


def test_readme_documents_f16_demotion_rule():
    text = _readme().lower()
    assert "demoted to f16" in text or "demote" in text
