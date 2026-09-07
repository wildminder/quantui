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


def test_q8_0_negative_extremes_roundtrip():
    """Signed extremes map to ±127-capped codes; roundtrip stays in bound.

    (Extremes stay within f16-scale range — 1e9 would overflow f16, which is
    an upstream llama.cpp property, not a kernel bug.)
    """
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
