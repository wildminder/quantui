

# --------------------------------------------------------------------------- #
# S5.1: vectorized f32_to_bf16 (RTNE, bit-exact vs the scalar oracle)
# --------------------------------------------------------------------------- #
def test_f32_to_bf16_matches_scalar_oracle():
    """100k random f32 values: vectorized == scalar bit oracle, zero mismatches."""
    import numpy as np

    from quantui.dtype_cast import f32_bits_to_bf16_bits, f32_to_bf16

    rng = np.random.default_rng(0)
    big = (rng.standard_normal(100_000) * 100).astype(np.float32)
    vec = f32_to_bf16(big)
    scalar = np.array([f32_bits_to_bf16_bits(x) for x in big.view(np.uint32)], dtype=np.uint16)
    assert (vec == scalar).all()


def test_f32_to_bf16_special_values():
    """1.0 / -2.0 / ±0 / 65536.0 (f16 inf!) / ±inf / NaN / 1e-40."""
    import numpy as np

    from quantui.dtype_cast import f32_to_bf16

    vals = np.array([1.0, -2.0, 0.0, -0.0, 65536.0, np.inf, -np.inf, np.nan, 1e-40],
                    dtype=np.float32)
    out = f32_to_bf16(vals)
    bits = [int(x) for x in out]
    assert bits[0] == 0x3F80 and bits[1] == 0xC000      # 1.0, -2.0
    assert bits[2] == 0x0000 and bits[3] == 0x8000      # +0, -0 preserved
    assert bits[4] == 0x4780                            # 65536.0 (0x4780) — f16 would be inf
    assert bits[5] == 0x7F80 and bits[6] == 0xFF80      # ±inf
    assert bits[7] & 0x7FC0 == 0x7FC0                   # NaN quiet bit forced
    assert bits[8] != 0                                 # 1e-40 is representable as bf16 denormal


def test_f32_to_bf16_shape_preserved():
    import numpy as np

    from quantui.dtype_cast import f32_to_bf16

    arr = np.zeros((3, 5), dtype=np.float32)
    out = f32_to_bf16(arr)
    assert out.shape == (3, 5)
    assert out.dtype == np.dtype("<u2")
