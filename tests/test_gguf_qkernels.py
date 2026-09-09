"""S2.1/S2.2 tests: Q8_0 / Q4_0 numpy kernels vs the gguf-py oracle goldens.

The goldens (tests/golden/gguf_qgolden.py) were generated from
``gguf.quants.quantize`` — the llama.cpp reference semantics — so bit-exact
equality here pins the kernels to the llama.cpp convention (plan STEP 2.1
"golden bit-exact" is the core parity pin).
"""

import hashlib

import numpy as np
import pytest

from quantui.gguf_qkernels import (
    Q4_0_BLOCK_BYTES,
    Q8_0_BLOCK_BYTES,
    dequantize_q4_0,
    dequantize_q8_0,
    quantize_q4_0,
    quantize_q8_0,
)
from tests.golden import gguf_qgolden as G


# --------------------------------------------------------------- Q8_0 ----- #
def test_q8_0_golden_bit_exact():
    """Case B (with the mid-stream zero block) reproduces the oracle bytes."""
    assert quantize_q8_0(G.golden_case_b()) == bytes.fromhex(G.B_Q8_0_HEX)


def test_q8_0_golden_digests_all_cases():
    """Cases A (64 blocks) and C (4096 blocks) match the pinned digests."""
    for name, maker in (("A", G.golden_case_a), ("C", G.golden_case_c)):
        raw = quantize_q8_0(maker())
        assert hashlib.sha256(raw).hexdigest() == G.Q8_0_BLOCKS_SHA256[name], name


def test_q8_0_error_bounds():
    """Dequantized mirror vs original: error within one f16-scale step.

    Bound uses the *stored* f16 d (which can round up slightly vs the f32
    d): |err| <= max|stored d| + eps covers round-to-nearest quantization.
    """
    for maker in (G.golden_case_a, G.golden_case_b, G.golden_case_c):
        arr = maker()
        raw = quantize_q8_0(arr)
        n_blocks = arr.size // 32
        blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, 34)
        max_d = float(np.frombuffer(blocks[:, 0:2].copy().tobytes(), dtype="<f2").max())
        back = dequantize_q8_0(raw, arr.size).reshape(arr.shape)
        assert float(np.max(np.abs(back - arr))) <= max_d + 1e-6


def test_q8_0_zero_block():
    """All-zero block -> scale +0.0, codes 0; dequant returns exact zeros."""
    arr = np.zeros((2, 32), dtype=np.float32)
    raw = quantize_q8_0(arr)
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(2, Q8_0_BLOCK_BYTES)
    assert np.frombuffer(blocks[0, 0:2].tobytes(), dtype="<f2")[0] == 0.0
    assert not blocks[0, 2:].any()
    back = dequantize_q8_0(raw, 64)
    assert not back.any() and np.isfinite(back).all()


def test_q8_0_shape_contract():
    """(64, 32) ok; (64, 30) raises ValueError naming the shape."""
    assert len(quantize_q8_0(np.zeros((64, 32), dtype=np.float32))) == 64 * Q8_0_BLOCK_BYTES
    with pytest.raises(ValueError, match="divisible"):
        quantize_q8_0(np.zeros((64, 30), dtype=np.float32))


def test_q8_0_determinism():
    """Same input twice -> identical bytes."""
    arr = G.golden_case_b()
    assert quantize_q8_0(arr) == quantize_q8_0(arr)


def test_q8_0_rounding_is_half_away_from_zero_on_x_inv():
    """llama.cpp roundf(x * (1/d)) semantics — rint(x/d) flips boundary codes.

    Exact-tie case: x = d/2 * (2k+1) style values. Construct x where x/d is
    exactly representable with a .5 fraction and x*inv is too: the two modes
    disagree (rint -> even, half-away -> odd-signed away from zero).
    """
    import numpy as np

    # d = 4.0 -> amax = 508.0; x = 2.0 -> x/d = 0.5 (tie: rint->0, away->1)
    # x = 6.0 -> 1.5 (tie: rint->2, away->2 even? no: 1.5 -> rint 2, away 2)
    # Use x = -2.0 -> -0.5: rint -> -0 (0), half-away -> -1. That's the pin.
    arr = np.zeros((1, 32), dtype=np.float32)
    arr[0, 0] = 508.0  # amax -> d = 4.0
    arr[0, 1] = 2.0    # +0.5 tie
    arr[0, 2] = -2.0   # -0.5 tie
    raw = quantize_q8_0(arr)
    codes = np.frombuffer(raw, np.uint8).reshape(34)[2:].view(np.int8)
    assert codes[0] == 127 and codes[1] == 1 and codes[2] == -1, codes[:4]


def test_q8_0_negative_extremes_roundtrip():
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((4, 64)).astype(np.float32)
    arr[0, 0] = 1e4  # large but f16-safe scale
    arr[1, 0] = -1e4
    raw = quantize_q8_0(arr)
    codes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, Q8_0_BLOCK_BYTES)[:, 2:34].view(np.int8)
    assert codes.min() >= -127 and codes.max() <= 127
    back = dequantize_q8_0(raw, arr.size).reshape(arr.shape)
    bound = float(np.max(np.abs(arr))) / 127.0 / 2 + 1e-2
    assert float(np.max(np.abs(back - arr))) <= bound


def test_q8_0_dequantize_size_mismatch():
    with pytest.raises(ValueError, match="expected"):
        dequantize_q8_0(b"\x00" * 10, 64)


def test_q8_0_denormal_block_silent_and_oracle_exact():
    """USER REPORT 2026-09-09 (VibeVoice-ASR-HF native_q8_0): blocks whose
    maxabs is a tiny DENORMAL (zero-init noise ~1e-40) made 1/d overflow to
    inf -> NaN products -> RuntimeWarnings (overflow in divide / invalid in
    multiply+cast) AND wrong codes (NaN->int32 wrapped to INT_MIN, clipped
    to -127 instead of the oracle's 0). The kernel must be WARNING-FREE and
    BYTE-IDENTICAL to the gguf-py oracle on all such inputs.

    Live-oracle comparison: skipped when gguf-py is absent (the gate venv
    has no torch/gguf by mandate); runs in the worker env."""
    import warnings

    pytest.importorskip("gguf")
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize as oracle_quantize

    rng = np.random.default_rng(42)
    cases = (
        # denormal maxabs blocks mixed with a normal block (user scenario)
        np.array([[1e-40, 0.0, -5e-41] + [0.0] * 29,
                  [1e-45, 2e-45] + [0.0] * 30,
                  np.linspace(-1, 1, 32, dtype=np.float32)], dtype=np.float32),
        # f32-tiny noise across the whole tensor
        rng.standard_normal((64, 256)).astype(np.float32) * 1e-38,
        # magnitudes spanning denormal..f16-scale-overflow
        rng.standard_normal((64, 256)).astype(np.float32)
        * np.logspace(-45, 7, 64, dtype=np.float32)[:, None],
        np.zeros((4, 32), dtype=np.float32),  # all-zero (already pinned, cheap guard)
    )
    for arr in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # ANY RuntimeWarning fails the test
            raw = quantize_q8_0(arr)
        oracle = np.asarray(oracle_quantize(arr, GGMLQuantizationType.Q8_0),
                            dtype=np.uint8).tobytes()
        assert raw == oracle

    # Explicit semantic pin: denormal-block codes are 0 and the f16 scale is
    # +0.0 (NOT -127 codes) — the exact bytes the user's run got wrong.
    den = np.zeros((1, 32), dtype=np.float32)
    den[0, 0] = 1e-40
    raw = quantize_q8_0(den)
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(Q8_0_BLOCK_BYTES)
    assert blk[0:2].view("<f2")[0] == 0.0
    assert (blk[2:].view(np.int8) == 0).all()


def test_q8_0_inf_nan_inputs_oracle_exact():
    """inf/NaN block values follow the same NaN->0 path as the oracle
    (np_roundf NaN-poisons them; codes 0). Warning-free too. Skipped when
    gguf-py is absent (gate venv)."""
    import warnings

    pytest.importorskip("gguf")
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize as oracle_quantize

    cases = (
        np.array([[np.inf] * 32, [1.0] * 32], dtype=np.float32),
        np.array([[np.nan] * 32, [1.0] * 32], dtype=np.float32),
        np.array([[np.inf, np.nan, -np.inf] + [1.0] * 29], dtype=np.float32),
    )
    for arr in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            raw = quantize_q8_0(arr)
        oracle = np.asarray(oracle_quantize(arr, GGMLQuantizationType.Q8_0),
                            dtype=np.uint8).tobytes()
        assert raw == oracle


def test_q8_0_f16_scale_overflow_matches_oracle():
    """maxabs > ~8.3e6 -> d overflows the f16 STORE to +inf with codes at
    the clip boundary — exactly what the oracle produces (probed 2026-09-09:
    codes 127, d inf). Warning-free (the f16 store overflow is silenced).
    Skipped when gguf-py is absent (gate venv)."""
    import warnings

    pytest.importorskip("gguf")
    from gguf import GGMLQuantizationType
    from gguf.quants import quantize as oracle_quantize

    arr = np.full((2, 32), 1e7, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw = quantize_q8_0(arr)
    oracle = np.asarray(oracle_quantize(arr, GGMLQuantizationType.Q8_0),
                        dtype=np.uint8).tobytes()
    assert raw == oracle
    blk = np.frombuffer(raw, dtype=np.uint8).reshape(2, Q8_0_BLOCK_BYTES)
    assert blk[0, 0:2].view("<f2")[0] == np.inf
    assert (blk[0, 2:].view(np.int8) == 127).all()


# --------------------------------------------------------------- Q4_0 ----- #
def test_q4_0_golden_bit_exact():
    """Case B reproduces the oracle bytes (zero block stores d = -0.0)."""
    assert quantize_q4_0(G.golden_case_b()) == bytes.fromhex(G.B_Q4_0_HEX)


def test_q4_0_golden_digests_all_cases():
    for name, maker in (("A", G.golden_case_a), ("C", G.golden_case_c)):
        raw = quantize_q4_0(maker())
        assert hashlib.sha256(raw).hexdigest() == G.Q4_0_BLOCKS_SHA256[name], name


def test_q4_0_zero_block_negative_zero_scale():
    """llama.cpp convention: d = max|x|/-8 -> -0.0 bits for a zero block."""
    import numpy as np

    arr = np.zeros((1, 32), dtype=np.float32)
    raw = quantize_q4_0(arr)
    d = np.frombuffer(raw[:2], dtype="<f2")[0]
    assert d == 0.0 and np.signbit(d)
    # oracle case B's zero block (flat idx 4): bytes 00 80 … = -0.0 f16
    oracle = bytes.fromhex(G.B_Q4_0_HEX)[4 * Q4_0_BLOCK_BYTES : 4 * Q4_0_BLOCK_BYTES + 2]
    assert raw[:2] == oracle


def test_q4_0_nibble_order_pinned():
    """gguf-py packing: byte i = (elem_i | elem_{i+16} << 4); value = nibble-8.

    Fixture cross-verified against gguf-py directly: with max=+7 at index 0,
    d = -0.875; x=7 -> x/d=-8 -> code 0 (lo nibble); x=-7 -> x/d=8 ->
    trunc(16.5)=16 -> clip 15 (hi nibble). Byte 2 = 0xf0.
    """
    arr = np.zeros((1, 32), dtype=np.float32)
    arr[0, 0] = 7.0
    arr[0, 16] = -7.0
    raw = quantize_q4_0(arr)
    assert raw[2] == 0xF0  # byte 0: lo = elem0 = 0, hi = elem16 = 15
    back = dequantize_q4_0(raw, 32)
    assert back[0] == pytest.approx(7.0, abs=1.0)
    assert back[16] == pytest.approx(-6.125, abs=1.0)  # clip cost: 15-8=7 * -0.875


def test_q4_0_error_bounds():
    for maker in (G.golden_case_a, G.golden_case_b, G.golden_case_c):
        arr = maker()
        raw = quantize_q4_0(arr)
        back = dequantize_q4_0(raw, arr.size).reshape(arr.shape)
        bound = float(np.max(np.abs(arr))) / 8.0 + 1e-6  # biased trunc: worst case = one step
        assert float(np.max(np.abs(back - arr))) <= bound


def test_q4_0_shape_contract():
    assert len(quantize_q4_0(np.zeros((8, 64), dtype=np.float32))) == 16 * Q4_0_BLOCK_BYTES
    with pytest.raises(ValueError, match="divisible"):
        quantize_q4_0(np.zeros((8, 7), dtype=np.float32))


def test_q4_0_determinism():
    arr = G.golden_case_a()
    assert quantize_q4_0(arr) == quantize_q4_0(arr)
