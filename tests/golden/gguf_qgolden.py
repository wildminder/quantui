"""Pinned golden bytes for the native GGUF Q8_0 / Q4_0 kernels.

Generated 2026-09-07 from the **gguf-py oracle** (`gguf.quants.quantize`,
version installed in the worker env `.venv-gguf`,
gguf 0.19-era with the numpy quants module) — see plan
`native-backend plan` STEP 0.1 and the §9
completion log. gguf-py's block encoders are the llama.cpp reference
implementation semantics, so byte-equality against these constants pins our
numpy kernels to the llama.cpp convention.

Deterministic inputs (`np.random.default_rng(42)`, standard_normal * 0.13,
float32, C-order):

* Case A — ``(64, 32)``: 64 blocks, plain random.
* Case B — ``(2, 96)``: 6 blocks, block index 4 (flat) is all-zero
  (``b[1, 32:64] = 0``). Pins the zero-block semantics: Q8_0 → scale f16
  ``+0.0`` and 32 zero codes; Q4_0 → scale f16 ``-0.0`` (sign bit set) and
  zero delta nibbles (gguf-py stores ``d = max|x|/-8``; ``-0.0`` is what the
  reference produces for max == 0 — bit-pinned verbatim).
* Case C — ``(512, 256)``: 4096 blocks, bulk-size sanity (digest-pinned,
  hex not embedded — too large; SHA-256 of the raw little-endian block
  bytes is the pin).

Layout constants (llama.cpp `ggml` block structs):

* ``block_q8_0``: 32 × int8 code + f16 ``d`` = 34 B/block.
* ``block_q4_0``: f16 ``d`` (2 B) + 16 B of packed 4-bit codes
  (low nibble = element 2i, high nibble = element 2i+1) = 18 B/block.

The kernels in ``quantui/gguf_qkernels.py`` must reproduce these bytes
exactly (test_golden_bit_exact in tests/test_gguf_qkernels.py).
"""

from __future__ import annotations

# --- deterministic input regeneration ------------------------------------- #
GOLDEN_SEED = 42
GOLDEN_SCALE = 0.13


def golden_case_a() -> object:  # np.ndarray (64, 32) f32
    import numpy as np

    rng = np.random.default_rng(GOLDEN_SEED)
    return (rng.standard_normal((64, 32)) * GOLDEN_SCALE).astype(np.float32)


def golden_case_b() -> object:  # np.ndarray (2, 96) f32, zero block idx 4
    import numpy as np

    rng = np.random.default_rng(GOLDEN_SEED)
    arr = (rng.standard_normal((2, 96)) * GOLDEN_SCALE).astype(np.float32)
    arr[1, 32:64] = 0.0
    return arr


def golden_case_c() -> object:  # np.ndarray (512, 256) f32
    import numpy as np

    rng = np.random.default_rng(GOLDEN_SEED)
    return (rng.standard_normal((512, 256)) * GOLDEN_SCALE).astype(np.float32)


# --- layout constants ------------------------------------------------------ #
Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34  # 32 int8 codes + f16 scale
Q4_0_BLOCK_ELEMS = 32
Q4_0_BLOCK_BYTES = 18  # f16 scale + 16 B packed nibbles

# --- Case B verbatim hex (small enough to embed; 204 B / 108 B) ------------ #
# Zero block (flat index 4) lives mid-stream: Q8_0 bytes [4*34 : 5*34] are
# 34 zero bytes; Q4_0 bytes [4*18 : 5*18] are ``00`` + f16 -0.0 + 14 zero
# code bytes (delta nibble byte 0b10001000 = -8/+8 pattern from gguf-py).
B_Q8_0_HEX = (
    "7d1812c22d388cb308edffcd342e04431ccd16c734fdf5d848f7e7eb2016181a"
    "7fe80e17d9c32e55f7c1c2313829ce12091142113305163092e8ddd0eb71bf49"
    "81e70c2c1317363ce6dd41f2a0abbb250b34e00c2fe922cee5e3a625dd012422"
    "32f9e0fa81933e179fb71dbde460e636bbf1bae73e812011d49605d9110276ee"
    "b50d10643d1a6ca9000000000000000000000000000000000000000000000000"
    "000000000000000000007618c6d57fcf32ca3817f7fed91be5b7b40a5e0af911"
    "4e0de7421a5c0bb7ae6367f5"
)
B_Q4_0_HEX = (
    "74a877cc55849fbd3899a89b6575686406ab0027b6849bbd1764645b7cfb45d9"
    "09699cb90527ab5c66662ca76283a4aab98b66890b17302752138a64968ef67b"
    "349794e6cca0fa290080888888888888888888888888888888886da82c7b807b"
    "357ba44769287ad6da2d2d97"
)

# --- digest pins (all cases; SHA-256 of raw little-endian block bytes) ----- #
# NOTE: each case reseeds default_rng(42) independently (see makers above) —
# so case-B bytes differ from the original sequential-rng session digests;
# C was likewise regenerated under the reseeded convention (T0.1, §9 log).
Q8_0_BLOCKS_SHA256 = {
    "A": "81a5f8e60c863fd31e1135228800dbd136e70b488e37713681b8449a8fe58a3d",
    "B": "1ca653b90aaee6be0e12d1e2307dde7e30df4edcdb9730c5b7dcb05b203b92af",
    "C": "0c1ec7203558297811084ad0a3a559d9751da053bdcaa1e041e5b96dae391b06",
}
Q4_0_BLOCKS_SHA256 = {
    "A": "877ae5f548176eef3c4f4f683ebe4aa3bff285d2a1b35eb32a39624216d11d90",
    "B": "2fe75ce45c749b839597dc125e87f9cabf05df0a6d24490c165c839362ace616",
    "C": "7141c8cb627e97e7219f69743d82743f4f0b37b845e9e5e4aa110d4cc2f58ad4",
}

# --- input digests (regeneration tripwire; reseeded-per-case convention) --- #
Q8_0_INPUT_SHA256 = {
    "A": "2fca0e0bb82e0c08",  # first 16 hex chars
    "B": "ca7310ca17fa775a",
    "C": "fd2e3f77a9235a18",
}
