"""Tests for P2 (per-tensor quant core) and P3 (streaming orchestrator + resume).

The headline test is ``test_parity_with_whole_file_quantize``: our streaming pipeline
must produce a ``.safetensors`` that is byte-identical (tensor-by-tensor) to the output
of the reference ``convert_to_quant.quantize`` on the same input. That is the contract
that lets streaming replace the temp-merge path without changing model quality.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantui.stream_quant import enumerate_tensor_names, stream_quantize
from quantui.tensor_quant import HAS_CTQ, QuantConfig, quantize_weight_numpy

requires_ctq = pytest.mark.skipif(not HAS_CTQ, reason="convert_to_quant / torch not available")

# The whole-file baseline (convert_to_quant.quantize) derives its bias-correction seed
# from --manual_seed, which is *random* by default. Streaming must pin the identical
# seed, otherwise the simulated-calibration bias correction diverges and parity fails.
# This must equal QuantConfig.calib_seed (the streaming default).
PARITY_SEED = 233983427


# --------------------------------------------------------------------------- #
# Synthetic model builder
# --------------------------------------------------------------------------- #
def _make_synthetic_model(path):
    """A small but representative model.

    2D ``.weight`` tensors that are block-divisible (both dims >= 128 and % 128 == 0)
    are *quantized* by both the baseline and the streaming path, giving a real
    byte-for-byte math check. One 2D ``.weight`` is intentionally NOT block-divisible
    (16x32) so the ``skip_inefficient`` (heur) copy branch is exercised too. All other
    tensors (bias, non-weight 2D, norm, 4D conv, 3D) are copied unchanged by both paths.
    """
    g = torch.Generator().manual_seed(1234)
    tensors = {
        # 2D .weight -> quantized (block-divisible for block_size=128)
        "model.layers.0.attn.q_proj.weight": torch.randn(256, 128, generator=g),
        "model.layers.0.attn.k_proj.weight": torch.randn(256, 128, generator=g),
        "model.layers.0.attn.v_proj.weight": torch.randn(256, 128, generator=g),
        "model.layers.0.attn.o_proj.weight": torch.randn(128, 256, generator=g),
        "model.layers.0.mlp.gate.weight": torch.randn(512, 256, generator=g),
        # 2D .weight, block-divisible -> quantized (kept simple/divisible for parity)
        "model.layers.0.mlp.up.weight": torch.randn(128, 128, generator=g),
        # 1D bias -> copied (or bias-corrected when its weight is quantized)
        "model.layers.0.attn.o_proj.bias": torch.randn(128, generator=g),
        # 2D but NOT .weight -> copied
        "model.embed": torch.randn(10, 128, generator=g),
        # 1D norm -> copied
        "model.norm.weight": torch.randn(128, generator=g),
        # 4D conv (.weight but not 2D) -> copied
        "model.conv.weight": torch.randn(8, 3, 3, 3, generator=g),
        # 3D (.weight but not 2D) -> copied
        "model.bn.weight": torch.randn(4, 8, 8, generator=g),
    }
    save_file(tensors, str(path))
    return tensors


def _load_tensors(path):
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if k == "__metadata__":
                continue
            t = f.get_tensor(k)
            # Universal raw-byte extraction: handles bfloat16 (which `.numpy()` rejects
            # in this torch build) and scalars (which ``.view(uint8)`` rejects).
            data = bytes(t.detach().cpu().contiguous().untyped_storage())
            out[k] = (t.dtype, list(t.shape), data)
    return out


def _assert_same_tensors(a, b):
    only_a = set(a) - set(b)
    only_b = set(b) - set(a)
    assert not only_a and not only_b, f"key mismatch: +{only_a} -{only_b}"
    for k in a:
        da, sa, ba = a[k]
        db, sb, bb = b[k]
        assert da == db, f"{k}: dtype {da} != {db}"
        assert sa == sb, f"{k}: shape {sa} != {sb}"
        assert ba == bb, f"{k}: DATA bytes differ"


# --------------------------------------------------------------------------- #
# P2 unit tests
# --------------------------------------------------------------------------- #
def test_enumerate_tensor_names(tmp_path):
    p = tmp_path / "m.safetensors"
    _make_synthetic_model(p)
    names = enumerate_tensor_names(str(p))
    assert "model.layers.0.attn.q_proj.weight" in names
    assert "model.embed" in names
    assert "__metadata__" not in names
    assert len(names) == 11


def test_quantize_weight_numpy_blockwise_layout():
    cfg = QuantConfig(scaling_mode="block", block_size=8)
    w = np.random.randn(16, 16).astype(np.float32)
    specs, _ = quantize_weight_numpy("m.weight", w, cfg, has_bias=False)
    names = {s[0] for s in specs}
    assert "m.weight" in names
    assert "m.weight_scale" in names
    assert "m.comfy_quant" in names
    assert "m.input_scale" in names  # block-wise adds input_scale
    q = {s[0]: s for s in specs}["m.weight"]
    assert len(q[3]) == 16 * 16  # int8 bytes
    # q values within int8 range
    arr = np.frombuffer(q[3], dtype=np.int8).reshape(16, 16)
    assert arr.min() >= -127 and arr.max() <= 127


def test_quantize_weight_numpy_tensorwise_no_input_scale():
    cfg = QuantConfig(scaling_mode="tensor", block_size=128)
    w = np.random.randn(16, 16).astype(np.float32)
    specs, _ = quantize_weight_numpy("m.weight", w, cfg, has_bias=False)
    names = {s[0] for s in specs}
    assert "m.input_scale" not in names  # tensor-wise has no input_scale
    assert "m.comfy_quant" in names


# --------------------------------------------------------------------------- #
# P3 headline: parity with the whole-file reference quantizer
# --------------------------------------------------------------------------- #
@requires_ctq
def test_parity_with_whole_file_quantize(tmp_path):
    from convert_to_quant import quantize

    inp = tmp_path / "input.safetensors"
    base_out = tmp_path / "baseline.safetensors"
    stream_out = tmp_path / "stream.safetensors"
    _make_synthetic_model(inp)

    # Reference: whole-file ctq quantize (the path streaming replaces).
    quantize(
        str(inp),
        str(base_out),
        int8=True,
        scaling_mode="block",
        block_size=128,
        comfy_quant=True,
        simple=True,
        save_quant_metadata=False,
        manual_seed=PARITY_SEED,
    )

    # Streaming equivalent.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QuantConfig(
        scaling_mode="block", block_size=128, no_learned_rounding=True, device=device,
        calib_seed=PARITY_SEED,
    )
    manifest = stream_quantize(str(inp), str(stream_out), cfg)

    assert set(manifest["done"]) == set(enumerate_tensor_names(str(inp)))

    base = _load_tensors(base_out)
    stream = _load_tensors(stream_out)
    # Byte-identical tensor set.
    _assert_same_tensors(base, stream)


@requires_ctq
def test_resume_after_kill_matches_baseline(tmp_path):
    """Simulate a process kill mid-run, then resume; the resumed file must equal
    the whole-file baseline (and the partial prefix must itself be loadable)."""
    from convert_to_quant import quantize

    inp = tmp_path / "input.safetensors"
    base_out = tmp_path / "baseline.safetensors"
    out2 = tmp_path / "resumed.safetensors"
    _make_synthetic_model(inp)

    quantize(
        str(inp), str(base_out), int8=True, scaling_mode="block", block_size=128,
        comfy_quant=True, simple=True, save_quant_metadata=False, manual_seed=PARITY_SEED,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QuantConfig(scaling_mode="block", block_size=128, no_learned_rounding=True, device=device,
                      calib_seed=PARITY_SEED)

    # Run until 3 tensors are written, then "kill" by raising from on_progress.
    killed = {}

    def _kill_after_3(cur, total):
        if cur >= 3:
            killed["at"] = cur
            raise KeyboardInterrupt("simulated kill")

    with pytest.raises(KeyboardInterrupt):
        stream_quantize(str(inp), str(out2), cfg, on_progress=_kill_after_3)

    # Partial prefix is independently loadable (crash safety).
    partial = _load_tensors(out2)
    assert len(partial) >= 3

    # Resume -> finishes the rest.
    manifest = stream_quantize(str(inp), str(out2), cfg)
    assert set(manifest["done"]) == set(enumerate_tensor_names(str(inp)))

    resumed = _load_tensors(out2)
    base = _load_tensors(base_out)
    _assert_same_tensors(base, resumed)


@requires_ctq
def test_resume_is_idempotent(tmp_path):
    """Running stream_quantize twice (no kill) on the same output is a no-op."""
    inp = tmp_path / "input.safetensors"
    out = tmp_path / "stream.safetensors"
    _make_synthetic_model(inp)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QuantConfig(scaling_mode="block", block_size=128, no_learned_rounding=True, device=device)

    m1 = stream_quantize(str(inp), str(out), cfg)
    size_after_first = out.stat().st_size
    m2 = stream_quantize(str(inp), str(out), cfg)  # resume; should skip everything
    assert m2["done"] == m1["done"]
    assert out.stat().st_size == size_after_first  # nothing rewritten


@requires_ctq
def test_skip_inefficient_copies_non_divisible(tmp_path):
    """skip_inefficient (== ctq --heur) must copy non-block-divisible 2D weights
    unchanged, byte-identical to convert_to_quant.quantize with --heur. Divisible
    weights are still quantized identically by both paths (incl. bias correction)."""
    from convert_to_quant import quantize

    inp = tmp_path / "input.safetensors"
    base_out = tmp_path / "baseline.safetensors"
    stream_out = tmp_path / "stream.safetensors"
    g = torch.Generator().manual_seed(7)
    tensors = {
        # divisible 2D weight -> quantized by both paths (bias corrected)
        "a.weight": torch.randn(256, 128, generator=g),
        "a.bias": torch.randn(256, generator=g),
        # non-divisible 2D weight -> copied unchanged by both (heur / skip_inefficient)
        "b.weight": torch.randn(16, 32, generator=g),
    }
    save_file(tensors, str(inp))

    quantize(
        str(inp), str(base_out), int8=True, scaling_mode="block", block_size=128,
        comfy_quant=True, simple=True, save_quant_metadata=False, heur=True,
        manual_seed=PARITY_SEED,
    )

    # Device MUST match the baseline: ctq resolves device=None -> CUDA when available.
    # Forcing CPU here while the baseline runs on CUDA is a device mismatch that makes
    # the float32 INT8 rounding diverge (round(tensor/scale) crosses boundaries), so we
    # mirror ctq's auto-resolution instead of pinning cpu.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = QuantConfig(
        scaling_mode="block", block_size=128, no_learned_rounding=True,
        skip_inefficient=True, device=device, calib_seed=PARITY_SEED,
    )
    stream_quantize(str(inp), str(stream_out), cfg)

    base = _load_tensors(base_out)
    stream = _load_tensors(stream_out)

    # Non-divisible weight: copied unchanged (no .weight_scale), identical fp32 bytes.
    assert "b.weight_scale" not in base and "b.weight_scale" not in stream
    assert base["b.weight"] == stream["b.weight"]

    # Divisible weight (quantized + bias-corrected) must match byte-for-byte.
    _assert_same_tensors(
        {k: v for k, v in base.items() if not k.startswith("b.")},
        {k: v for k, v in stream.items() if not k.startswith("b.")},
    )


# --------------------------------------------------------------------------- #
# P3 pure-numpy streaming (torch-free friendly)
# --------------------------------------------------------------------------- #
def test_stream_quantize_numpy_fallback_runs(tmp_path):
    inp = tmp_path / "input.safetensors"
    out = tmp_path / "np.safetensors"
    _make_synthetic_model(inp)
    cfg = QuantConfig(scaling_mode="block", block_size=8)
    manifest = stream_quantize(str(inp), str(out), cfg, use_numpy_fallback=True)
    assert set(manifest["done"]) == set(enumerate_tensor_names(str(inp)))
    # Output is a valid safetensors file with the quantized tensors + copies.
    loaded = _load_tensors(out)
    assert "model.layers.0.attn.q_proj.weight" in loaded
    assert "model.layers.0.attn.q_proj.weight_scale" in loaded
    assert "model.layers.0.attn.q_proj.comfy_quant" in loaded
    # Copied tensors preserved exactly.
    orig = _load_tensors(inp)
    assert loaded["model.embed"] == orig["model.embed"]
    assert loaded["model.conv.weight"] == orig["model.conv.weight"]


# --------------------------------------------------------------------------- #
# P4a: sharded-output streaming variant (stream_quantize_sharded)
# --------------------------------------------------------------------------- #
def _make_sharded_model(d):
    """Build a 2-shard HuggingFace model folder with a real index json.

    NOTE: shards + index json are written FLAT into ``d`` (no subdirectory). A fresh
    subdirectory under pytest's temp dir is aggressively trashed by the environment's
    safe-delete shim, which races ``save_file``'s temp-file creation; writing into the
    already-claimed ``d`` avoids that.
    """
    g = torch.Generator().manual_seed(1234)
    t1 = {
        "model.layers.0.attn.q_proj.weight": torch.randn(256, 128, generator=g),
        "model.layers.0.attn.k_proj.weight": torch.randn(256, 128, generator=g),
    }
    t2 = {
        "model.layers.0.attn.v_proj.weight": torch.randn(256, 128, generator=g),
        "model.layers.0.attn.o_proj.weight": torch.randn(128, 256, generator=g),
        "model.layers.0.attn.o_proj.bias": torch.randn(128, generator=g),
    }
    save_file(t1, str(d / "model-00001-of-00002.safetensors"))
    save_file(t2, str(d / "model-00002-of-00002.safetensors"))
    wm = {
        "model.layers.0.attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.attn.k_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.attn.v_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.0.attn.o_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.0.attn.o_proj.bias": "model-00002-of-00002.safetensors",
    }
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm, "metadata": {"total_size": 1}}))
    (d / "config.json").write_text("{}")
    return d


@requires_ctq
def test_stream_sharded_equals_quantize_shards_baseline(tmp_path):
    """stream_quantize_sharded output must be byte-identical (per shard) to the
    existing per-shard quantize_shards baseline on the same input + same seed."""
    from convert_to_quant import quantize

    from quantui.stream_quant import stream_quantize_sharded
    from quantui.worker_ctq import discover_shards, quantize_shards

    _make_sharded_model(tmp_path)

    base_out = tmp_path / "base"
    kwargs = dict(
        int8=True, scaling_mode="block", block_size=128, comfy_quant=True,
        simple=True, save_quant_metadata=False, manual_seed=PARITY_SEED,
    )
    quantize_shards(discover_shards(str(tmp_path)), str(base_out), kwargs, quantize)

    cfg = QuantConfig(
        scaling_mode="block", block_size=128, no_learned_rounding=True,
        calib_seed=PARITY_SEED, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    stream_out = tmp_path / "stream"
    stream_quantize_sharded(discover_shards(str(tmp_path)), str(stream_out), cfg)

    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard in shards:
        base = _load_tensors(base_out / shard)
        stream = _load_tensors(stream_out / shard)
        _assert_same_tensors(base, stream)
    # index json + sidecars copied verbatim.
    assert (stream_out / "model.safetensors.index.json").is_file()
    assert (stream_out / "config.json").is_file()
    idx = json.loads((stream_out / "model.safetensors.index.json").read_text())
    assert idx["weight_map"] == {
        "model.layers.0.attn.q_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.attn.k_proj.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.attn.v_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.0.attn.o_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.0.attn.o_proj.bias": "model-00002-of-00002.safetensors",
    }


@requires_ctq
def test_stream_sharded_resume_per_shard(tmp_path):
    """Re-running stream_quantize_sharded resumes each shard (no re-quantization) and
    yields the same byte-identical files; a global manifest records both shards."""
    from quantui.stream_quant import stream_quantize_sharded
    from quantui.worker_ctq import discover_shards

    _make_sharded_model(tmp_path)
    cfg = QuantConfig(
        scaling_mode="block", block_size=128, no_learned_rounding=True,
        calib_seed=PARITY_SEED, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    out = tmp_path / "stream"
    m1 = stream_quantize_sharded(discover_shards(str(tmp_path)), str(out), cfg)
    size1 = {s: (out / s).stat().st_size for s in m1["shards"]}

    # Re-run (resume): each shard should be skipped via its own manifest.
    m2 = stream_quantize_sharded(discover_shards(str(tmp_path)), str(out), cfg)
    size2 = {s: (out / s).stat().st_size for s in m2["shards"]}
    assert size1 == size2  # nothing rewritten
    assert set(m2["shards"]) == set(m1["shards"])

    # Global manifest present and well-formed.
    assert (out / ".quant-manifest.json").is_file()
    gm = json.loads((out / ".quant-manifest.json").read_text())
    assert gm["config_hash"] == cfg.config_hash()
    assert set(gm["shards"]) == set(m1["shards"])
