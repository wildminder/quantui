"""Per-tensor INT8 quantization core (P2).

This module wraps ``convert_to_quant``'s *true* per-tensor INT8 primitive so that
streaming quantization can quantize one weight tensor at a time (instead of having
``convert_to_quant.quantize`` materialize the entire model and write it whole-file).

For each 2D ``.weight`` tensor it produces the exact same set of output tensors the
whole-file path writes, so the streaming result is byte-for-byte equivalent to
``convert_to_quant.quantize`` on the same input:

    <name>                  -> int8 quantized weight      (keeps original tensor name)
    <base>.weight_scale    -> float32 scale(s)
    <base>.comfy_quant     -> uint8 JSON config blob (built by the library's own
                              ``create_comfy_quant_tensor`` so the bytes match exactly)
    <base>.input_scale     -> float32 scalar 1.0  (block-wise only)

A pure-numpy fallback (``quantize_weight_numpy``) is provided for torch-free
environments; it is functionally equivalent (symmetric INT8) but NOT byte-identical
to the torch path, so parity tests exercise the torch path.

The heavy imports are guarded so this module imports cleanly even when torch /
``convert_to_quant`` are unavailable (the numpy fallback then becomes the only path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .comfy_quant_schema import encode_comfy_quant_config

# --- guarded heavy imports ------------------------------------------------ #
try:  # pragma: no cover - exercised only when torch + convert_to_quant present
    import torch
    from convert_to_quant.constants import SCALE_DTYPE, TARGET_INT8_DTYPE
    from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
    from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor

    HAS_CTQ = True
except Exception:  # ImportError or any torch/ctq runtime issue
    torch = None  # type: ignore
    LearnedRoundingConverter = None  # type: ignore
    SCALE_DTYPE = None  # type: ignore
    TARGET_INT8_DTYPE = None  # type: ignore
    create_comfy_quant_tensor = None  # type: ignore
    HAS_CTQ = False


# torch dtype -> safetensors header dtype string
_TORCH_DTYPE_TO_STR = {
    "float32": "F32", "float16": "F16", "bfloat16": "BF16",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
    "uint8": "U8", "bool": "BOOL",
}


def torch_dtype_to_str(dt) -> str:
    """Map a torch dtype (or its ``.name``) to the safetensors header dtype string."""
    name = getattr(dt, "name", None) or str(dt).split(".")[-1].lower()
    return _TORCH_DTYPE_TO_STR.get(name, str(name).upper())


def resolve_orig_dtype_str(orig_dtype: str) -> str:
    """Mirror ctq's ``str(resolve_output_dtype(orig_dtype))`` without importing torch.

    ctq only supports ``'bfloat16'`` / ``'float16'`` for the output dtype and
    serializes it as the torch dtype *string* (e.g. ``'torch.bfloat16'``), which is
    exactly what must land in the ``.comfy_quant`` blob for byte-exact parity with the
    whole-file ``convert_to_quant.quantize`` path. Keeping this pure (no torch import)
    also lets the numpy fallback stay torch-free while still emitting the right bytes.
    """
    return {"bfloat16": "torch.bfloat16", "float16": "torch.float16"}.get(orig_dtype, orig_dtype)


@dataclass
class QuantConfig:
    """Streaming-quantization configuration (mirrors the relevant ctq flags)."""

    target_format: str = "int8"
    int8: bool = True
    scaling_mode: str = "block"          # "tensor" | "row" | "block"
    block_size: int = 128
    no_learned_rounding: bool = True     # --simple
    convrot: bool = False
    convrot_group_size: int = 256
    device: str = "cpu"
    orig_dtype: str = "bfloat16"         # default resolved_output_dtype in ctq
    skip_inefficient: bool = True        # mirror ctq --heur: copy (don't quantize)
                                          # layers whose dims aren't divisible by block_size
    # Seed for the "simulated calibration data" used only for bias correction. MUST be
    # pinned to the same value the whole-file baseline uses (ctq derives its seed from
    # --manual_seed, which is random by default), otherwise parity tests fail.
    # In production we keep this FIXED (deterministic) rather than ctq's random default:
    # the legacy whole-file path re-rolled a random seed every run, so it was itself
    # non-reproducible. Streaming is therefore reproducible across runs (and across the
    # merge-then-quant baseline only when that baseline is given the same --manual_seed).
    calib_seed: int = 233983427
    # Regex of tensor-name keys kept at original precision (mirrors ctq
    # ``--exclude_layers``). None disables matching. Example: "attn_norm|text_embed".
    exclude_layers: str | None = None
    # added when wiring (P5); ignored by the math here
    comfy_quant: bool = True
    save_quant_metadata: bool = False

    def excluded(self, name: str) -> bool:
        """True when ``name`` matches the :attr:`exclude_layers` regex."""
        if not self.exclude_layers:
            return False
        import re

        try:
            return re.search(self.exclude_layers, name) is not None
        except re.error:
            return False  # invalid regex -> never block quantization

    @classmethod
    def from_args(cls, args) -> QuantConfig:
        """Build a streaming :class:`QuantConfig` that mirrors ``convert_to_quant``'s
        effective flags, so ``stream_quantize`` output is byte-identical to the
        whole-file ``convert_to_quant.quantize`` baseline on the same input.

        Mirrors exactly what ``build_quantize_kwargs`` passes through to ctq:
        - ``scaling_mode`` (None -> ctq default ``block``)
        - ``block_size`` (None -> ctq default 128)
        - ``simple`` -> ``no_learned_rounding``
        - ``heur``    -> ``skip_inefficient`` (ctq: ``skip_inefficient_layers = args.heur``)
        - ``manual_seed`` -> ``calib_seed`` (None -> fixed default, fully reproducible)
        - device resolves like ctq (``None -> cuda if available``) — required for the
          float32 INT8 rounding to match the baseline byte-for-byte.
        """
        block_size = int(args.block_size) if getattr(args, "block_size", None) else 128
        scaling_mode = getattr(args, "scaling_mode", None) or "block"
        manual_seed = getattr(args, "manual_seed", None)
        calib_seed = int(manual_seed) if manual_seed is not None else cls.calib_seed
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        return cls(
            target_format="int8",
            int8=bool(getattr(args, "int8", True)),
            scaling_mode=scaling_mode,
            block_size=block_size,
            no_learned_rounding=bool(getattr(args, "simple", True)),
            convrot=bool(getattr(args, "convrot", False)),
            convrot_group_size=int(getattr(args, "convrot_group_size", 256) or 256),
            device=device,
            orig_dtype=str(getattr(args, "output_dtype", None) or "bfloat16"),
            skip_inefficient=bool(getattr(args, "heur", False)),
            calib_seed=calib_seed,
            exclude_layers=getattr(args, "exclude_layers", None) or None,
        )

    def as_converter_kwargs(self) -> dict:
        """Build the kwargs dict passed to ``LearnedRoundingConverter`` (matches ctq)."""
        return {
            "target_format": self.target_format,
            "int8": self.int8,
            "scaling_mode": self.scaling_mode,
            "block_size": self.block_size,
            "no_learned_rounding": self.no_learned_rounding,
            "device": self.device,
            "convrot": self.convrot,
            "convrot_group_size": self.convrot_group_size,
            "dynamic_convrot": False,
            "extract_lora": False,
            "lora_rank": 32,
            "lora_target": None,
            "lora_depth": 1,
            "lora_ar_threshold": 0.0,
        }

    def config_hash(self) -> str:
        import hashlib
        import json

        payload = {
            "target_format": self.target_format,
            "int8": self.int8,
            "scaling_mode": self.scaling_mode,
            "block_size": self.block_size,
            "no_learned_rounding": self.no_learned_rounding,
            "convrot": self.convrot,
            "convrot_group_size": self.convrot_group_size,
            "skip_inefficient": self.skip_inefficient,
            "calib_seed": self.calib_seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def as_dict(self) -> dict:
        return {
            "target_format": self.target_format,
            "int8": self.int8,
            "scaling_mode": self.scaling_mode,
            "block_size": self.block_size,
            "no_learned_rounding": self.no_learned_rounding,
            "convrot": self.convrot,
            "convrot_group_size": self.convrot_group_size,
            "orig_dtype": self.orig_dtype,
        }


def make_converter(config: QuantConfig):
    """Construct the ``LearnedRoundingConverter`` used for per-tensor INT8.

    Raises ``RuntimeError`` if ``convert_to_quant`` / torch are unavailable.
    """
    if not HAS_CTQ:
        raise RuntimeError(
            "convert_to_quant / torch not available; use quantize_weight_numpy fallback"
        )
    return LearnedRoundingConverter(**config.as_converter_kwargs())


# --------------------------------------------------------------------------- #
# Torch path (byte-exact parity with the whole-file ctq path)
# --------------------------------------------------------------------------- #
def quantize_weight(name: str, tensor, converter, config: QuantConfig, has_bias: bool, calibration_data=None):
    """Quantize one 2D ``.weight`` tensor and return the output tensor specs.

    Returns ``(specs, dequant_w)`` where ``specs`` is a list of
    ``(out_name, dtype_str, shape, data_bytes)`` ready for
    :class:`~quantui.incremental_safetensors.IncrementalSafetensorsWriter`, and
    ``dequant_w`` is the de-quantized weight (float) used for bias correction so the
    streaming pipeline matches the whole-file baseline byte-for-byte.

    ``calibration_data`` is the simulated-activation tensor (shape ``(CALIB_SAMPLES,
    in_features)``) the reference ``convert_to_quant`` passes into ``converter.convert``
    -- it materially affects the INT8 rounding, so it must be supplied for parity.
    """
    if not HAS_CTQ:
        raise RuntimeError("torch / convert_to_quant unavailable")

    q, dequant_s, dequant_w, _extra = converter.convert(
        tensor, key=name, calibration_data=calibration_data, has_bias=has_bias
    )
    base = name[: name.rfind(".weight")]

    out: list[tuple[str, str, list, bytes]] = []
    q = q.detach().cpu()
    out.append((name, torch_dtype_to_str(q.dtype), list(q.shape), q.numpy().tobytes()))

    scale = dequant_s.to(device="cpu", dtype=SCALE_DTYPE)
    # Mirror ctq's ``normalize_tensorwise_scales``: a 1-element scale (e.g. a single
    # (1,1) block for a square weight) is squeezed to a scalar so the safetensors
    # header shape is ``[]`` and matches the whole-file baseline byte-for-byte.
    if scale.numel() == 1 and scale.ndim > 0:
        scale = scale.squeeze()
    out.append(
        (
            f"{base}.weight_scale",
            "F32",
            list(scale.shape),
            scale.detach().cpu().numpy().tobytes(),
        )
    )

    scaling = config.scaling_mode
    if scaling in ("tensor", "row"):
        fmt = "int8_tensorwise"
        bs: int | None = None
        per_row = scaling == "row"
    else:
        fmt = "int8_blockwise"
        bs = config.block_size
        per_row = False

    blob = create_comfy_quant_tensor(
        fmt,
        block_size=bs,
        full_precision_matrix_mult=None,
        convrot=config.convrot,
        convrot_groupsize=config.convrot_group_size if config.convrot else None,
        per_row=per_row if scaling == "row" else None,
        orig_dtype=resolve_orig_dtype_str(config.orig_dtype),
    )
    out.append(
        (f"{base}.comfy_quant", "U8", [int(blob.numel())], blob.detach().cpu().numpy().tobytes())
    )

    if fmt == "int8_blockwise":
        iscale = torch.tensor(1.0, dtype=torch.float32)
        out.append(
            (f"{base}.input_scale", "F32", [], iscale.detach().cpu().numpy().tobytes())
        )
    return out, dequant_w


# --------------------------------------------------------------------------- #
# Pure-numpy fallback (torch-free; functionally equivalent, NOT byte-identical)
# --------------------------------------------------------------------------- #
def _sym_int8_block(arr: np.ndarray, maxabs: np.ndarray) -> np.ndarray:
    scale = np.where(maxabs == 0, 1.0, maxabs) / 127.0
    return np.clip(np.round(arr / scale), -127, 127).astype(np.int8)


def quantize_weight_numpy(name: str, arr: np.ndarray, config: QuantConfig, has_bias: bool):
    """Numpy INT8 quantization fallback (no torch / convert_to_quant needed).

    Mirrors the torch path's tensor layout so downstream writers are identical; the
    numeric result is a standard symmetric INT8 quantization (row/tensor/block).
    Returns ``(specs, dequant_w)`` like :func:`quantize_weight`.
    """
    arr = np.asarray(arr, dtype=np.float32)
    base = name[: name.rfind(".weight")]
    out: list[tuple[str, str, list, bytes]] = []

    if config.scaling_mode == "row":
        maxabs = np.max(np.abs(arr), axis=1, keepdims=True).astype(np.float32)
        q = _sym_int8_block(arr, maxabs)
        scale = (np.where(maxabs == 0, 1.0, maxabs) / 127.0).astype(np.float32)
        dequant_w = (q.astype(np.float32) * scale).astype(np.float32)
        fmt, bs = "int8_tensorwise", None
    elif config.scaling_mode == "tensor":
        maxabs = np.max(np.abs(arr))
        scale_val = np.float32(maxabs / 127.0) if maxabs > 0 else np.float32(1.0)
        q = np.clip(np.round(arr / scale_val), -127, 127).astype(np.int8)
        scale = np.array([scale_val], dtype=np.float32)
        dequant_w = (q.astype(np.float32) * scale_val).astype(np.float32)
        fmt, bs = "int8_tensorwise", None
    else:  # block
        bs = config.block_size
        M, N = arr.shape
        bm, bn = M // bs, N // bs
        blocked = arr.reshape(bm, bs, bn, bs)
        maxabs = np.max(np.abs(blocked), axis=(1, 3)).astype(np.float32)  # (bm, bn)
        q = np.zeros_like(arr, dtype=np.int8)
        scale = np.zeros((bm, bn), dtype=np.float32)
        for i in range(bm):
            for j in range(bn):
                block = arr[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs]
                m = maxabs[i, j]
                s = m / 127.0 if m > 0 else 1.0
                scale[i, j] = s
                q[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs] = np.clip(
                    np.round(block / s), -127, 127
                ).astype(np.int8)
        scale = (np.where(maxabs == 0, 1.0, maxabs) / 127.0).astype(np.float32)
        dequant_w = (
            q.reshape(bm, bs, bn, bs).astype(np.float32) * scale[:, None, :, None]
        ).reshape(M, N).astype(np.float32)
        fmt = "int8_blockwise"

    out.append((name, "I8", list(q.shape), q.tobytes()))
    # Mirror ctq's normalize_tensorwise_scales: a 1-element scale is squeezed to scalar.
    if scale.size == 1 and scale.ndim > 0:
        scale = scale.reshape(())
    out.append((f"{base}.weight_scale", "F32", list(scale.shape), scale.tobytes()))

    cfg: dict = {"format": fmt, "orig_dtype": resolve_orig_dtype_str(config.orig_dtype)}
    if bs is not None:
        cfg["group_size"] = bs
    blob = encode_comfy_quant_config(cfg)
    out.append((f"{base}.comfy_quant", "U8", [len(blob)], blob))

    if fmt == "int8_blockwise":
        out.append((f"{base}.input_scale", "F32", [], np.array(1.0, dtype=np.float32).tobytes()))
    return out, dequant_w
