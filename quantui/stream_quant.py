"""Streaming quantization orchestrator with resumable checkpoints (P3).

Replaces the heavy "merge shards -> quantize whole file -> copy" path in
``worker_ctq.py`` with a tensor-by-tensor pipeline:

    for each tensor in the input (in file order):
        if it is a 2D ``.weight``  -> quantize it (QuantConfig) and append the
                                       int8 weight + scale + .comfy_quant (+ input_scale)
        else                       -> copy it unchanged
        flush a manifest checkpoint after every tensor

Because the output grows **only at the end** (via
:class:`~quantui.incremental_safetensors.IncrementalSafetensorsWriter`) and the
checkpoint is rewritten after each tensor, a killed process can be resumed by simply
re-running: already-written tensors are skipped (idempotent), so the partial output
file plus the manifest fully determine where to continue.

A ``<output>.quant-manifest.json`` is the authoritative checkpoint:
    {"version": 1, "config_hash": "<sha256[:16]>", "order": [...], "done": [...]}
The ``config_hash`` guards against resuming with a *different* quantization config
(which would corrupt the file); a mismatch forces a clean restart.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from .comfy_quant_schema import read_safetensors_header
from .incremental_safetensors import IncrementalSafetensorsWriter
from .tensor_quant import (
    HAS_CTQ,
    QuantConfig,
    make_converter,
    quantize_weight,
    quantize_weight_numpy,
)

MANIFEST_VERSION = 1

# Mirrors convert_to_quant's "simulated calibration data" used for bias correction
# (see formats/fp8_conversion.py). These are the library's CLI defaults.
CALIB_SAMPLES = 3072


def _should_skip_layer_for_performance(tensor, block_size: int) -> bool:
    """Faithful mirror of ``convert_to_quant``'s ``should_skip_layer_for_performance``.

    The whole-file baseline invokes this when ``skip_inefficient_layers`` (heur) is
    True and copies the layer unchanged if it returns True. Streaming must make the
    identical decision so its output stays byte-for-byte equal to the baseline.

    Works on both torch tensors and numpy arrays (both expose ``.shape``/``.ndim``).
    """
    ndim = getattr(tensor, "ndim", len(tensor.shape))
    if ndim != 2:
        return True
    rows, cols = tensor.shape[-2], tensor.shape[-1]
    if rows < block_size or cols < block_size:
        return True
    if rows % block_size != 0 or cols % block_size != 0:
        return True
    return False


def _should_skip_shape(shape, block_size: int) -> bool:
    """Shape-only variant of :func:`_should_skip_layer_for_performance` (no tensor load)."""
    if len(shape) != 2:
        return True
    rows, cols = shape[-2], shape[-1]
    if rows < block_size or cols < block_size:
        return True
    if rows % block_size != 0 or cols % block_size != 0:
        return True
    return False


def _build_torch_calibration_cache(header: dict, names: list[str], seed: int) -> dict:
    """Replicate ctq's deterministic simulated calibration data (torch).

    For each unique ``in_features`` (weight shape[1]) among 2D ``.weight`` tensors, in
    file order, draw ``torch.randn(CALIB_SAMPLES, in_features, generator=gen)`` from a
    single shared generator seeded once with ``seed``. This exactly matches
    ``convert_to_quant``'s ``calibration_data_cache`` build (formats/fp8_conversion.py)
    so the resulting bias correction is identical -- provided ``seed`` equals the value
    the baseline used (ctq derives it from ``--manual_seed``).
    """
    import torch

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    cache: dict[int, object] = {}
    for name in names:
        info = header.get(name)
        if not info or not name.endswith(".weight"):
            continue
        shape = info.get("shape", [])
        if len(shape) != 2:
            continue
        in_features = int(shape[1])
        if in_features not in cache:
            cache[in_features] = torch.randn(
                CALIB_SAMPLES, in_features, dtype=torch.float32, generator=gen, device="cpu"
            )
    return cache


def _build_numpy_calibration_cache(header: dict, names: list[str], seed: int) -> dict:
    """Numpy equivalent of :func:`_build_torch_calibration_cache` (self-consistent only)."""
    import numpy as _np

    rng = _np.random.default_rng(seed)
    cache: dict[int, object] = {}
    for name in names:
        info = header.get(name)
        if not info or not name.endswith(".weight"):
            continue
        shape = info.get("shape", [])
        if len(shape) != 2:
            continue
        in_features = int(shape[1])
        if in_features not in cache:
            cache[in_features] = rng.standard_normal((CALIB_SAMPLES, in_features)).astype(_np.float32)
    return cache


def _correct_bias_torch(original_weight, dequant_w, bias, in_features: int, calib_cache: dict, device: str):
    """Mirror convert_to_quant's bias correction (formats/fp8_conversion.py ~L711-721).

    The matmul runs on ``device`` (matching the baseline, which moves calibration data
    to the active device) and the result is stored on CPU, identical to the baseline's
    ``new_tensors[bias_key] = b_new.to(device="cpu", dtype=original_bias.dtype)``.
    """
    import torch

    dev = torch.device(device)
    X = calib_cache[in_features].to(dev)  # (CALIB_SAMPLES, in_features) float32
    W_orig = original_weight.to(dev, torch.float32)
    W_dq = dequant_w.to(dev, torch.float32)
    b_orig = bias.to(dev, torch.float32)
    err = W_orig - W_dq
    out_err = X @ err.T
    corr = out_err.mean(dim=0)
    return (b_orig - corr).to(device="cpu", dtype=bias.dtype)


def _correct_bias_numpy(original_weight, dequant_w, bias, in_features: int, calib_cache: dict, device: str = "cpu"):
    import numpy as _np

    X = calib_cache[in_features]  # (CALIB_SAMPLES, in_features) float32
    W_orig = original_weight.astype(_np.float32)
    W_dq = dequant_w.astype(_np.float32)
    b_orig = bias.astype(_np.float32)
    err = W_orig - W_dq
    out_err = X @ err.T
    corr = out_err.mean(axis=0)
    return (b_orig - corr).astype(bias.dtype)


def enumerate_tensor_names(input_path: str | list[str]) -> list[str]:
    """Return all tensor names across one or more ``.safetensors`` files.

    A single path returns that file's tensor names (file order). A list of shard
    paths returns the *union* in first-appearance order (used when streaming several
    shards into one output). ``__metadata__`` is excluded.
    """
    shard_paths = input_path if isinstance(input_path, (list, tuple)) else [input_path]
    names: list[str] = []
    for sp in shard_paths:
        header, _ = read_safetensors_header(sp)
        for k in header:
            if k != "__metadata__" and k not in names:
                names.append(k)
    return names


def _resolve_union_header(shard_paths: list[str]):
    """Merge several shard headers into one union header for streaming.

    Returns ``(union_header, names, name_to_shard, metadata)`` where ``names`` is the
    ordered union of tensor names, ``name_to_shard`` maps each name to the shard path it
    lives in (first occurrence wins, mirroring HF weight_map precedence), and ``metadata``
    is the first ``__metadata__`` found (carried into the single output, like
    :func:`merge_safetensors_files`).
    """
    union_header: dict = {}
    name_to_shard: dict[str, str] = {}
    names: list[str] = []
    metadata = None
    for sp in shard_paths:
        header, _ = read_safetensors_header(sp)
        if metadata is None and "__metadata__" in header:
            metadata = header["__metadata__"]
        for k, v in header.items():
            if k == "__metadata__":
                continue
            if k not in name_to_shard:
                name_to_shard[k] = sp
                union_header[k] = v
                names.append(k)
    return union_header, names, name_to_shard, metadata


def _read_input_metadata(input_path: str) -> dict | None:
    try:
        header, _ = read_safetensors_header(input_path)
        return header.get("__metadata__")
    except Exception:  # noqa: BLE001
        return None


class _StreamState:
    """Incremental progress + manifest bookkeeping for one stream_quantize run."""

    def __init__(self, output_path: str, config: QuantConfig):
        self.output_path = output_path
        self.manifest_path = output_path + ".quant-manifest.json"
        self.config = config
        self.config_hash = config.config_hash()
        self.order: list[str] = []
        self.done: set[str] = set()

    # -- manifest persistence ---------------------------------------------- #
    def load_manifest(self) -> bool:
        """Load an existing manifest if it matches this run's config. Returns True if resumed."""
        if not (os.path.exists(self.manifest_path) and os.path.exists(self.output_path)):
            return False
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:  # noqa: BLE001
            return False
        if data.get("config_hash") != self.config_hash:
            # Different config -> do not trust the partial file; restart clean.
            return False
        self.order = list(data.get("order", []))
        self.done = set(data.get("done", []))
        return True

    def save_manifest(self) -> None:
        payload = {
            "version": MANIFEST_VERSION,
            "config_hash": self.config_hash,
            "order": self.order,
            "done": sorted(self.done),
        }
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        os.replace(tmp, self.manifest_path)


def stream_quantize(
    input_path: str,
    output_path: str,
    config: QuantConfig,
    on_progress: Callable[[int, int], None] | None = None,
    use_numpy_fallback: bool = False,
) -> dict:
    """Quantize ``input_path`` to ``output_path`` tensor-by-tensor, resumable.

    Args:
        input_path: source ``.safetensors`` (single file; sharded P4a variant TBD).
        output_path: destination ``.safetensors`` (incremental / resumable).
        config: quantization configuration (scaling mode, block size, ...).
        on_progress: optional ``callable(cur, total)`` invoked after each tensor.
        use_numpy_fallback: force the pure-numpy quantizer (no torch dependency).

    Returns the manifest dict (``version / config_hash / order / done``).
    """
    if not os.path.exists(input_path if isinstance(input_path, str) else input_path[0]):
        raise FileNotFoundError(input_path)
    if use_numpy_fallback or not HAS_CTQ:
        if not HAS_CTQ and not use_numpy_fallback:
            # torch path requested but unavailable: transparently fall back so the
            # pipeline still completes (parity tests will not use this branch).
            use_numpy_fallback = True

    # Accept a single file path OR a list of shard paths that should be streamed into a
    # single output (this is what replaces the temp-merge for `--output-mode single`).
    shard_paths = input_path if isinstance(input_path, (list, tuple)) else [input_path]
    header, names, name_to_shard, metadata = _resolve_union_header(shard_paths)
    total = len(names)

    def is_quantizable_2d(n: str) -> bool:
        info = header.get(n)
        if not info or not n.endswith(".weight"):
            return False
        if config.excluded(n):  # ctq --exclude_layers parity (e.g. attn_norm|text_embed)
            return False
        shape = info.get("shape", [])
        if len(shape) != 2 or 0 in shape:
            return False
        if config.skip_inefficient and _should_skip_shape(shape, config.block_size):
            return False
        return True

    state = _StreamState(output_path, config)
    resumed = state.load_manifest()

    # Build (or resume) the converter. For the numpy fallback there is no converter.
    converter = None
    if not use_numpy_fallback:
        converter = make_converter(config)

    # Open the incremental writer (resumes if the file already exists & is valid).
    mode = "a" if resumed else "w"
    metadata = header.get("__metadata__")
    writer = IncrementalSafetensorsWriter().open(
        output_path, mode, metadata=metadata
    )

    # Ensure the manifest order includes any names from a prior run (so the final
    # `order` is complete even when resuming mid-way).
    if resumed:
        for n in names:
            if n not in state.order:
                state.order.append(n)

    remaining = [n for n in names if n not in state.done]

    # Deterministic calibration data for bias correction (mirrors convert_to_quant).
    calib_cache = (
        _build_numpy_calibration_cache(header, names, config.calib_seed)
        if use_numpy_fallback
        else _build_torch_calibration_cache(header, names, config.calib_seed)
    )
    correct_bias = _correct_bias_numpy if use_numpy_fallback else _correct_bias_torch

    # Output dtype the baseline casts *skipped* (unquantized) 2D `.weight` tensors to.
    target_dtype = None
    if not use_numpy_fallback:
        from convert_to_quant.utils.output_dtype import resolve_output_dtype

        target_dtype = resolve_output_dtype(config.orig_dtype)

    # 2D `.weight` tensors that are NOT quantized (skipped via skip_inefficient) must
    # be cast to the output dtype, exactly like ctq's ``cast_unquantized_weights``.
    skip_cast_names = {
        n
        for n in names
        if n.endswith(".weight")
        and len(header[n].get("shape", [])) == 2
        and not is_quantizable_2d(n)
    }

    # Caches so a weight and its bias are written at their own file positions while
    # being quantized exactly once (handles weight-before-bias AND bias-before-weight).
    quantized_specs_cache: dict[str, list] = {}
    corrected_bias: dict[str, tuple] = {}

    fh_in = None
    handles: dict = {}
    try:
        if use_numpy_fallback:
            import numpy as np
            from safetensors.numpy import load as safetensors_load_np

            input_tensors: dict = {}
            for sp in shard_paths:
                with open(sp, "rb") as _fh:
                    input_tensors.update(safetensors_load_np(_fh.read()))
            get_tensor = lambda n: input_tensors[n]  # noqa: E731
        else:
            from safetensors import safe_open

            for sp in shard_paths:
                handles[sp] = safe_open(sp, framework="pt", device="cpu")
            get_tensor = lambda n: handles[name_to_shard[n]].get_tensor(n)  # noqa: E731

        def process_weight(wname: str) -> list:
            """Quantize one weight, cache its specs, and compute its (optional) bias
            correction. Returns the weight's output specs (does NOT write them)."""
            tensor = get_tensor(wname)
            bias_name = f"{wname[: wname.rfind('.weight')]}.bias"
            has_bias = bias_name in names
            in_features = int(tensor.shape[1])
            calibration_data = calib_cache.get(in_features)
            if use_numpy_fallback:
                arr = np.asarray(tensor, dtype=np.float32)
                specs, dequant_w = quantize_weight_numpy(wname, arr, config, has_bias=has_bias)
            else:
                specs, dequant_w = quantize_weight(
                    wname, tensor, converter, config, has_bias=has_bias, calibration_data=calibration_data
                )
            quantized_specs_cache[wname] = specs
            if has_bias:
                bias = get_tensor(bias_name)
                corrected = correct_bias(tensor, dequant_w, bias, in_features, calib_cache, config.device)
                dtype_str, shape, data = _serialize_tensor(corrected, use_numpy_fallback)
                corrected_bias[bias_name] = (dtype_str, list(corrected.shape), data)
            return specs

        def copy_tensor(name: str) -> None:
            tensor = get_tensor(name)
            # Mirror ctq's cast_unquantized_weights: a skipped 2D `.weight` is recast
            # to the output dtype (bfloat16) so the bytes match the whole-file baseline.
            if (
                name in skip_cast_names
                and not use_numpy_fallback
                and target_dtype is not None
                and str(tensor.dtype) in _CASTABLE_FLOAT_DTYPE_NAMES
                and tensor.dtype != target_dtype
            ):
                tensor = tensor.to(dtype=target_dtype)
            dtype_str, shape, data = _serialize_tensor(tensor, use_numpy_fallback)
            writer.add_tensor(name, dtype_str, shape, data)

        for name in remaining:
            if name in corrected_bias:
                # Bias whose weight was already processed -> write corrected value.
                dtype_str, shape, data = corrected_bias[name]
                writer.add_tensor(name, dtype_str, shape, data)
            elif name in quantized_specs_cache:
                # Weight whose bias was processed earlier -> write cached specs.
                for out_name, dtype, shape, data in quantized_specs_cache[name]:
                    writer.add_tensor(out_name, dtype, shape, data)
            elif is_quantizable_2d(name):
                for out_name, dtype, shape, data in process_weight(name):
                    writer.add_tensor(out_name, dtype, shape, data)
            elif name.endswith(".bias"):
                # Bias reached before its weight: process the weight now (caches its
                # specs + this bias correction) but only write the bias here.
                wname = name[: -len(".bias")] + ".weight"
                if wname in names and is_quantizable_2d(wname) and not config.excluded(wname):
                    process_weight(wname)
                    dtype_str, shape, data = corrected_bias[name]
                    writer.add_tensor(name, dtype_str, shape, data)
                else:
                    copy_tensor(name)
            else:
                copy_tensor(name)

            state.done.add(name)
            if name not in state.order:
                state.order.append(name)
            state.save_manifest()
            if on_progress is not None:
                on_progress(len(state.done), total)
    finally:
        writer.close()
        for _h in handles.values():
            try:
                _h.close()
            except Exception:  # noqa: BLE001
                pass
        if fh_in is not None:
            try:
                fh_in.close()
            except Exception:  # noqa: BLE001
                pass

    state.save_manifest()
    return {
        "version": MANIFEST_VERSION,
        "config_hash": state.config_hash,
        "order": state.order,
        "done": sorted(state.done),
    }


# --------------------------------------------------------------------------- #
# Sharded-output variant (P4a): one incremental output shard per input shard,
# plus a global manifest + the original index json / sidecars copied verbatim.
# --------------------------------------------------------------------------- #
SHARDED_MANIFEST_NAME = ".quant-manifest.json"


def stream_quantize_sharded(
    model,  # ShardedModel from worker_ctq
    output_dir: str,
    config: QuantConfig,
    on_progress: Callable[[int, int], None] | None = None,
    use_numpy_fallback: bool = False,
) -> dict:
    """Stream-quantize a HuggingFace sharded model into a sharded OUTPUT directory.

    Mirrors ``quantize_shards``: each input shard is quantized into its own output shard
    (never merged), then ``model.safetensors.index.json`` and the non-weight sidecars are
    copied verbatim. Unlike ``quantize_shards`` (which shells out to the whole-file
    ``convert_to_quant.quantize`` per shard), this streams tensor-by-tensor and checkpoint
    each shard independently, so every shard is individually resumable.

    Args:
        model: a :class:`~quantui.worker_ctq.ShardedModel` (or duck-typed equivalent with
            ``model_dir``, ``shard_files``, ``index_path``, ``non_weight_files``).
        output_dir: destination directory (created if missing).
        config: quantization configuration.
        on_progress: optional ``callable(shards_done, total_shards)``.
        use_numpy_fallback: force the pure-numpy quantizer.

    Returns a global manifest dict with per-shard done-sets.
    """
    import shutil

    from .quant_methods import INDEX_NAME

    os.makedirs(output_dir, exist_ok=True)
    total = len(model.shard_files)
    per_shard: dict[str, list[str]] = {}
    for i, shard in enumerate(model.shard_files, 1):
        in_path = os.path.join(model.model_dir, shard)
        out_path = os.path.join(output_dir, shard)
        manifest = stream_quantize(
            in_path, out_path, config, on_progress=None, use_numpy_fallback=use_numpy_fallback
        )
        per_shard[shard] = manifest["done"]
        if on_progress is not None:
            on_progress(i, total)

    # Copy index json + sidecars unchanged (shard filenames / tensor names stay valid).
    shutil.copy2(model.index_path, os.path.join(output_dir, INDEX_NAME))
    for fname in getattr(model, "non_weight_files", []):
        src = os.path.join(model.model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    global_manifest = {
        "version": MANIFEST_VERSION,
        "config_hash": config.config_hash(),
        "output_dir": output_dir,
        "shards": per_shard,
    }
    gmp = os.path.join(output_dir, SHARDED_MANIFEST_NAME)
    with open(gmp, "w", encoding="utf-8") as fh:
        json.dump(global_manifest, fh, separators=(",", ":"))
    return global_manifest


# --------------------------------------------------------------------------- #
# dtype-string helpers for the copy path
# --------------------------------------------------------------------------- #
def _torch_dtype_to_str(dt) -> str:
    from .tensor_quant import torch_dtype_to_str

    return torch_dtype_to_str(dt)


_NUMPY_DTYPE_TO_STR = {
    "float32": "F32", "float16": "F16", "float64": "F64",
    "int64": "I64", "int32": "I32", "int16": "I16", "int8": "I8",
    "uint8": "U8", "bool": "BOOL",
}


def _numpy_dtype_to_str(dt) -> str:
    import numpy as _np

    return _NUMPY_DTYPE_TO_STR.get(_np.dtype(dt).name, "F32")


# Mirrors convert_to_quant's ``cast_unquantized_weights``: dtypes that may be recast
# to the output dtype when a 2D ``.weight`` is skipped (not quantized).
_CASTABLE_FLOAT_DTYPE_NAMES = {
    "torch.float32", "torch.float16", "torch.float64", "torch.bfloat16",
}


def _serialize_tensor(tensor, use_numpy_fallback: bool):
    """Return ``(dtype_str, shape, raw_bytes)`` for a tensor, for any dtype.

    Uses a universal ``view(torch.uint8)`` extraction on the torch path so even
    non-numpy-supported dtypes (e.g. ``bfloat16``) serialize to their exact raw
    bytes, matching what ``safetensors`` writes -- needed because the baseline casts
    skipped 2D weights to the output dtype (bfloat16).
    """
    if use_numpy_fallback:
        return _numpy_dtype_to_str(tensor.dtype), list(tensor.shape), tensor.tobytes()

    # ``untyped_storage`` extracts raw bytes for ANY dtype (incl. bfloat16) AND any
    # shape (incl. scalars), unlike ``.view(uint8)`` which rejects 0-dim tensors.
    return (
        _torch_dtype_to_str(tensor.dtype),
        list(tensor.shape),
        bytes(tensor.detach().cpu().contiguous().untyped_storage()),
    )
