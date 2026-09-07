"""Native GGUF quantization kernels: pure numpy, llama.cpp-exact (S2.x).

Every kernel reproduces the llama.cpp ``ggml`` block format **byte-for-byte**
— verified against the gguf-py oracle goldens in ``tests/golden/gguf_qgolden.py``
(generated from ``gguf.quants.quantize``, the llama.cpp reference semantics).

Block layouts (little-endian, scale FIRST — confirmed against the oracle):

* ``block_q8_0`` (34 B): ``f16 d; int8 qs[32]`` —
  ``d = max|x| / 127`` (f16), ``q_i = round(x_i / d)`` clamped to [-127, 127].
* ``block_q4_0`` (18 B): ``f16 d; uint8 qs[16]`` —
  ``d = signed_max / -8`` (f16; an all-zero block stores ``d = -0.0``
  exactly like the reference), biased codes
  ``q = trunc(x / d + 8.5)`` clipped to [0, 15], packed low-nibble-first
  (element 2i in the low nibble, 2i+1 in the high nibble); decode is
  ``value = nibble - 8``.

Caller contract: the tensor's last axis (``ne[0]``) must be divisible by 32 —
the tensor plan (STEP 2.3) demotes non-conforming tensors to F16; the kernels
raise :class:`ValueError` on violation (loud, never silent).

Pure numpy — no torch / transformers / gguf imports.
"""

from __future__ import annotations

import numpy as np

Q8_0_BLOCK_ELEMS = 32
Q8_0_BLOCK_BYTES = 34  # f16 scale (2) + 32 int8 codes
Q4_0_BLOCK_ELEMS = 32
Q4_0_BLOCK_BYTES = 18  # f16 scale (2) + 16 B packed nibbles


def _check_ne0(arr: np.ndarray, block: int, kernel: str) -> None:
    """Loud caller-contract check: last axis divisible by the block size."""
    if arr.shape[-1] % block != 0:
        raise ValueError(
            f"{kernel}: last-axis length {arr.shape[-1]} not divisible by "
            f"{block} (demote to F16 in the tensor plan before calling)"
        )


def _blocks_of(arr: np.ndarray, block: int) -> np.ndarray:
    """Flatten C-order and cut into (-1, block) f32 blocks."""
    flat = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    return flat.reshape(-1, block)


def _f16_le(value: np.ndarray) -> bytes:
    """Little-endian f16 bytes of a scalar/0-d value."""
    return np.asarray(value, dtype="<f2").tobytes()


# --------------------------------------------------------------------------- #
# Q8_0
# --------------------------------------------------------------------------- #
def quantize_q8_0(arr: np.ndarray) -> bytes:
    """Quantize f32 -> Q8_0 block bytes (llama.cpp-exact, golden-pinned)."""
    _check_ne0(arr, Q8_0_BLOCK_ELEMS, "quantize_q8_0")
    blocks = _blocks_of(arr, Q8_0_BLOCK_ELEMS)
    maxabs = np.max(np.abs(blocks), axis=1)  # (n_blocks,) f32
    d = (maxabs / 127.0).astype(np.float32)
    # gguf-py/llama.cpp store d=0 blocks as scale +0.0 with zero codes.
    safe_d = np.where(maxabs == 0, np.float32(1.0), d)
    q = np.clip(np.rint(blocks / safe_d[:, None]), -127, 127).astype(np.int8)
    q[maxabs == 0] = 0
    out = bytearray()
    for i in range(q.shape[0]):
        out += _f16_le(np.float16(d[i]))
        out += q[i].tobytes()
    return bytes(out)


def dequantize_q8_0(raw: bytes, numel: int) -> np.ndarray:
    """Mirror of :func:`quantize_q8_0` — block bytes -> f32 array (len numel)."""
    n_blocks = numel // Q8_0_BLOCK_ELEMS
    if len(raw) != n_blocks * Q8_0_BLOCK_BYTES:
        raise ValueError(f"dequantize_q8_0: expected {n_blocks * Q8_0_BLOCK_BYTES} bytes, got {len(raw)}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q8_0_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    q = blocks[:, 2:34].copy().view(np.int8)
    return (q.astype(np.float32) * d[:, None]).reshape(-1)


# --------------------------------------------------------------------------- #
# Q4_0
# --------------------------------------------------------------------------- #
def quantize_q4_0(arr: np.ndarray) -> bytes:
    """Quantize f32 -> Q4_0 block bytes (llama.cpp-exact, golden-pinned).

    gguf-py/llama.cpp semantics (``Q4_0.quantize_blocks``): ``d = max / -8``
    (signed max, not abs!), ``id = 1/d`` (0 when d == 0), biased codes
    ``q = trunc(x * id + 8.5)`` clipped to [0, 15], packed low-nibble-first
    (element 2i low, element 2i+1 high). The zero block stores
    ``d = -0.0`` (sign bit) with zero nibbles.
    """
    _check_ne0(arr, Q4_0_BLOCK_ELEMS, "quantize_q4_0")
    blocks = _blocks_of(arr, Q4_0_BLOCK_ELEMS)
    imax = np.abs(blocks).argmax(axis=1)
    amax_val = np.take_along_axis(blocks, imax[:, None], axis=1)[:, 0]
    d = (amax_val / -8.0).astype(np.float32)  # signed: negative block -> positive d
    with np.errstate(divide="ignore"):
        inv_d = np.where(d == 0, np.float32(0.0), 1.0 / d)
    q = (np.trunc(blocks * inv_d[:, None] + np.float32(8.5)).astype(np.int32)).clip(0, 15).astype(np.uint8)
    out = bytearray()
    for i in range(blocks.shape[0]):
        out += _f16_le(np.float16(d[i]))
        # gguf-py packing: reshape(n_blocks, 2, 16) -> plane0 = low nibbles of
        # bytes 0..15 (elements 0..15), plane1 = high nibbles (elements 16..31).
        lo = q[i, 0:16]
        hi = q[i, 16:32]
        out += (lo | (hi << 4)).tobytes()
    return bytes(out)


def _unpack_q4_nibbles(raw16: np.ndarray) -> np.ndarray:
    """gguf-py decode: byte i low nibble = element i, high = element i+16;
    value = nibble - 8."""
    lo = (raw16 & 0x0F).astype(np.int8)
    hi = ((raw16 >> 4) & 0x0F).astype(np.int8)
    out = np.empty(32, dtype=np.int8)
    out[0:16] = lo - 8
    out[16:32] = hi - 8
    return out


def dequantize_q4_0(raw: bytes, numel: int) -> np.ndarray:
    """Mirror of :func:`quantize_q4_0` — block bytes -> f32 array (len numel)."""
    n_blocks = numel // Q4_0_BLOCK_ELEMS
    if len(raw) != n_blocks * Q4_0_BLOCK_BYTES:
        raise ValueError(f"dequantize_q4_0: expected {n_blocks * Q4_0_BLOCK_BYTES} bytes, got {len(raw)}")
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, Q4_0_BLOCK_BYTES)
    d = blocks[:, 0:2].copy().view("<f2").astype(np.float32).ravel()
    q = np.stack([_unpack_q4_nibbles(blocks[i, 2:18]) for i in range(n_blocks)])
    return (q.astype(np.float32) * d[:, None]).reshape(-1)


# --------------------------------------------------------------------------- #
# S2.3 — tensor plan: the deterministic per-tensor policy
# --------------------------------------------------------------------------- #
from dataclasses import dataclass  # noqa: E402  (grouped with the plan section)

# safetensors dtypes the plan understands (subset of DTYPE_ITEMSIZE)
_FLOAT_DTYPES = frozenset({"F32", "F16", "BF16"})
_INT_DTYPES = frozenset({"I32", "U8", "BOOL"})

NATIVE_METHODS: tuple[str, ...] = ("native_q8_0", "native_q4_0", "native_f16", "native_f32")


@dataclass(frozen=True)
class TensorPlanItem:
    """One tensor's deterministic conversion decision."""

    hf_name: str
    gguf_name: str | None
    hf_shape: tuple[int, ...]
    action: str  # quant_q8_0 | quant_q4_0 | pass_f16 | pass_f32 | verbatim | skip
    reason: str


def plan_tensor(name: str, dtype: str, shape: tuple[int, ...], method: str) -> TensorPlanItem:
    """Decide one tensor's action — first match wins, never guesses.

    Policy (plan §S2.3): rotary skip -> exotic dtype -> 1-D verbatim ->
    2-D float quant (ne0 divisible) -> 2-D float F16 demotion -> pass-quant
    for f16/f32 methods -> integer verbatim. Unmapped GGUF names do NOT
    change the action (VibeVoice audio 2-D linears quantize).
    """
    from quantui.gguf_names import hf_to_gguf_name, is_skippable

    if method not in NATIVE_METHODS:
        raise ValueError(f"plan_tensor: unknown native method {method!r} (v1 surface: {NATIVE_METHODS})")

    if is_skippable(name):
        return TensorPlanItem(name, None, tuple(shape), "skip", "rotary/cache tensor (not a weight)")

    gguf_name = hf_to_gguf_name(name)

    if dtype not in _FLOAT_DTYPES | _INT_DTYPES:
        return TensorPlanItem(name, gguf_name, tuple(shape), "pass_f32", f"exotic dtype {dtype} -> F32")

    ndim = len(shape)
    if ndim == 1:
        return TensorPlanItem(name, gguf_name, tuple(shape), "verbatim", "1-D vector -> F32 (shared convention)")

    if ndim >= 2 and dtype in _FLOAT_DTYPES:
        ne0 = shape[-1]
        if method == "native_q8_0":
            if ne0 % 32 == 0:
                return TensorPlanItem(name, gguf_name, tuple(shape), "quant_q8_0", f"q8_0 (ne0={ne0})")
            return TensorPlanItem(name, gguf_name, tuple(shape), "pass_f16", f"demoted: ne0={ne0} not divisible by 32")
        if method == "native_q4_0":
            if ne0 % 32 == 0:
                return TensorPlanItem(name, gguf_name, tuple(shape), "quant_q4_0", f"q4_0 (ne0={ne0})")
            return TensorPlanItem(name, gguf_name, tuple(shape), "pass_f16", f"demoted: ne0={ne0} not divisible by 32")
        if method == "native_f16":
            return TensorPlanItem(name, gguf_name, tuple(shape), "pass_f16", "f16 method")
        return TensorPlanItem(name, gguf_name, tuple(shape), "pass_f32", "f32 method")

    # ndim >= 2 integers (or odd-shape ints): keep bytes verbatim
    return TensorPlanItem(name, gguf_name, tuple(shape), "verbatim", f"{dtype} tensor kept verbatim")


def demote_summary(items: list[TensorPlanItem]) -> dict[str, int]:
    """Action -> count histogram over a plan (deterministic key order via sorted)."""
    out: dict[str, int] = {}
    for item in items:
        out[item.action] = out.get(item.action, 0) + 1
    return dict(sorted(out.items()))
