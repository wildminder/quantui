"""S3.3: live parity of the native GGUF backend vs the user's oracle.

Oracle: ``VibeVoice-1.5B-q8_0_new.gguf`` produced by the user's independent
quantui-rs tool and confirmed working in ComfyUI. These tests convert the
real VibeVoice-1.5B HF checkpoint with OUR native backend and compare:

* tensor census: the (gguf_name, qtype) multiset must EQUAL the oracle's —
  same names mapped / passed through, same per-tensor qtype decisions
  (Q8_0 quantizations, F16 demotions for the 4/7/8/10/16-wide convs, F32
  1-D convention);
* arch KV: general.architecture == "vibevoice";
* numeric: for sampled LM blocks, our Q8_0 and the oracle's Q8_0 both
  dequantize to within one step of each other (same convention; exact byte
  equality is NOT required because the oracle may quantize from a different
  intermediate dtype — pinned explicitly).

Everything is skip-guarded: the test is active only when gguf AND the
oracle file AND the source checkpoint are present (worker env + this
machine). In the headless gate it records as skipped, never failed.
"""

import os
from collections import Counter

import pytest

from tests.gguf_miniread import read_gguf_header

ORACLE_GGUF = os.environ.get("UQT_ORACLE_GGUF", "")
SOURCE_DIR = os.environ.get("UQT_ORACLE_SOURCE", "")

pytest.importorskip("gguf", reason="live parity runs in the worker env (gguf installed)")
if not ORACLE_GGUF or not os.path.isfile(ORACLE_GGUF):
    pytest.skip("oracle GGUF not present (set UQT_ORACLE_GGUF)", allow_module_level=True)
if not SOURCE_DIR or not os.path.isfile(os.path.join(SOURCE_DIR, "model.safetensors.index.json")):
    pytest.skip("source checkpoint not present (set UQT_ORACLE_SOURCE)", allow_module_level=True)


def _oracle_census() -> Counter:
    g = read_gguf_header(ORACLE_GGUF)
    return Counter((t.name, t.ggml_type) for t in g.tensors)


def _convert(tmp_path):
    """Convert the source once per test; return (census, out_path, report)."""
    from quantui.gguf_export import export_gguf

    out = _out_path(tmp_path)
    report = export_gguf(SOURCE_DIR, out, "native_q8_0")
    g = read_gguf_header(out)
    census = Counter((t.name, t.ggml_type) for t in g.tensors)
    return census, out, report


def _out_path(tmp_path):
    return str(tmp_path / "vibevoice-native-q8_0.gguf")


def test_parity_arch_kv(tmp_path):
    _census, out, _report = _convert(tmp_path)
    g = read_gguf_header(out)
    assert g.kv["general.architecture"] == "vibevoice"


def test_parity_tensor_census(tmp_path):
    """(name, qtype) multiset equality with the oracle file."""
    oracle = _oracle_census()
    ours, _out, report = _convert(tmp_path)
    missing = oracle - ours
    extra = ours - oracle
    assert not missing, f"tensors/qtypes missing vs oracle: {sorted(missing)[:8]} …"
    assert not extra, f"tensors/qtypes extra vs oracle: {sorted(extra)[:8]} …"
    assert report.tensors_total == sum(oracle.values())


def test_parity_lm_block_numeric(tmp_path):
    """FULL-payload comparison: EVERY Q8_0 LM tensor (196 in the oracle) must
    dequantize within one quant step of the oracle's (kernels are
    convention-identical; exact byte equality is NOT required because the
    oracle may quantize from a different intermediate dtype — pinned
    explicitly)."""
    import gguf
    import numpy as np

    oracle_reader = gguf.GGUFReader(ORACLE_GGUF)
    _census, our_out, _report = _convert(tmp_path)
    our_reader = gguf.GGUFReader(our_out)
    ours_by_name = {t.name: t for t in our_reader.tensors}
    oracle_by_name = {t.name: t for t in oracle_reader.tensors}

    q8_lm = [t.name for t in oracle_reader.tensors
             if t.name.startswith("blk.") and t.tensor_type == gguf.GGMLQuantizationType.Q8_0]
    assert len(q8_lm) >= 150, f"expected ~196 Q8_0 LM tensors, got {len(q8_lm)}"
    worst_err, worst_name = 0.0, ""
    for name in q8_lm:
        o = gguf.dequantize(oracle_by_name[name].data, oracle_by_name[name].tensor_type)
        m = gguf.dequantize(ours_by_name[name].data, ours_by_name[name].tensor_type)
        assert o.shape == m.shape, name
        # both are ~Q8_0 of the same bf16 source: within one quant step + bf16 input rounding
        scale = float(np.abs(o).max()) / 127.0 + 1e-9
        err = float(np.abs(m.astype(np.float32) - o.astype(np.float32)).max())
        if err > worst_err:
            worst_err, worst_name = err, name
        assert err <= scale, (
            f"{name}: err {err} exceeds one quant step {scale} — convention mismatch"
        )
    # record the worst offender for the test log
    print(f"\n[parity] {len(q8_lm)} Q8_0 LM tensors checked; worst err "
          f"{worst_err:.6f} at {worst_name}")


def test_parity_audio_passthrough_names(tmp_path):
    """Every oracle non-blk tensor name is present in ours, unchanged."""
    oracle_g = read_gguf_header(ORACLE_GGUF)
    ours_g = read_gguf_header(_out_path(tmp_path))
    ours_names = set(ours_g.tensor_names())
    audio = [t.name for t in oracle_g.tensors
             if not t.name.startswith("blk.")
             and t.name not in ("token_embd.weight", "output_norm.weight", "output.weight")]
    assert audio, "no audio-side tensors found in oracle"
    absent = [n for n in audio if n not in ours_names]
    assert not absent, f"audio tensors absent from our output: {absent[:8]}"
