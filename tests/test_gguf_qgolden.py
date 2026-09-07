"""S0.1 gate-tier tests for the pinned GGUF kernel goldens (plan STEP 0.1).

The golden file is data + docstring only; these tests pin its internal
consistency so a corrupted edit (bad hex, wrong lengths, drifted layout
constants) fails loudly before any kernel is built against it.
"""

import hashlib

from tests.golden import gguf_qgolden as G


def test_golden_constants_parse():
    """B-case hex decodes and has exactly the llama.cpp block lengths."""
    q8 = bytes.fromhex(G.B_Q8_0_HEX)
    q4 = bytes.fromhex(G.B_Q4_0_HEX)
    # case B = (2, 96) = 192 elems = 6 blocks
    assert len(q8) == 6 * G.Q8_0_BLOCK_BYTES == 204
    assert len(q4) == 6 * G.Q4_0_BLOCK_BYTES == 108


def test_golden_scales_are_f16():
    """Scale lanes decode as finite f16 — block layout is SCALE FIRST
    (``block_q8_0 { f16 d; int8 qs[32]; }``, ``block_q4_0 { f16 d; … }``),
    verified against the gguf-py oracle bytes."""
    import numpy as np

    q8 = np.frombuffer(bytes.fromhex(G.B_Q8_0_HEX), dtype=np.uint8).reshape(6, 34)
    scales8 = q8[:, 0:2].copy().view("<f2").ravel()
    assert np.all(np.isfinite(scales8))
    # zero block (flat index 4): scale +0.0
    assert scales8[4] == 0.0
    assert scales8[:4].min() > 0 and scales8[5] > 0

    q4 = np.frombuffer(bytes.fromhex(G.B_Q4_0_HEX), dtype=np.uint8).reshape(6, 18)
    scales4 = q4[:, 0:2].copy().view("<f2").ravel()
    assert np.all(np.isfinite(scales4))
    # zero block: gguf-py stores d = max|x|/-8 -> -0.0 (sign bit set)
    assert scales4[4] == 0.0
    assert np.signbit(scales4[4])  # the -0.0 pin


def test_golden_q8_0_codes_in_range():
    """All Q8_0 int8 codes within [-127, 127] (never -128); codes are bytes 2:34."""
    import numpy as np

    q8 = np.frombuffer(bytes.fromhex(G.B_Q8_0_HEX), dtype=np.uint8).reshape(6, 34)
    codes = q8[:, 2:34].copy().view(np.int8)
    assert codes.min() >= -127
    assert codes.max() <= 127
    # zero block codes are all zero
    assert not codes[4].any()


def test_golden_b_case_digest_pins():
    """B-case hex is the exact bytes the digests were generated from."""
    q8 = hashlib.sha256(bytes.fromhex(G.B_Q8_0_HEX)).hexdigest()
    q4 = hashlib.sha256(bytes.fromhex(G.B_Q4_0_HEX)).hexdigest()
    assert q8 == G.Q8_0_BLOCKS_SHA256["B"]
    assert q4 == G.Q4_0_BLOCKS_SHA256["B"]


def test_golden_input_regeneration_tripwire():
    """Regenerating the deterministic inputs reproduces the pinned input digests."""
    import numpy as np

    for name, maker in (("A", G.golden_case_a), ("B", G.golden_case_b), ("C", G.golden_case_c)):
        arr = maker()
        digest = hashlib.sha256(arr.astype(np.float32).tobytes()).hexdigest()[:16]
        assert digest == G.Q8_0_INPUT_SHA256[name], name


def test_golden_block_count_math():
    """Layout constants satisfy bytes = numel / 32 * block_bytes for all cases."""
    assert 64 * 32 // G.Q8_0_BLOCK_ELEMS * G.Q8_0_BLOCK_BYTES == 2176
    assert 512 * 256 // G.Q8_0_BLOCK_ELEMS * G.Q8_0_BLOCK_BYTES == 139264
    # Q4_0 is denser: same numel, 18 B/block — the Q8_0 math must NOT hold.
    assert 64 * 32 // G.Q4_0_BLOCK_ELEMS * G.Q4_0_BLOCK_BYTES == 1152
    assert 64 * 32 // G.Q4_0_BLOCK_ELEMS * G.Q4_0_BLOCK_BYTES != 2176
