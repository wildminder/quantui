"""GGUF conformance checker: our quantizer output vs unsloth reference.

Structural conformance (spec-level):
  * valid GGUF header (magic/version), architecture KVs present
  * same tensor inventory as the reference (names + shapes)
  * per-tensor ggml quantization types
  * required tokenizer KVs present
Numerical conformance (two independent axes):
  * vs reference: our dequantized tensor vs unsloth's dequantized tensor
    (both quantize the same BF16 source -> values must agree within
    quantization noise; bit-exactness is NOT expected unless scale choices
    coincide, e.g. Q8_0)
  * vs original: quantization error of ours vs error of unsloth, both
    against the same original BF16 tensor (proves ours is a faithful
    quantization of THE SAME source weights)

Usage:
    python check_gguf_conformance.py <ours.gguf> <ref.gguf> <original.safetensors> \
        [--hf-prefix model.language_model.] [--max-tensors N] [--json OUT.json]

HF->GGUF name mapping for LFM2.5-VL (language tower only; vision lives in
mmproj): model.language_model.layers.N.X -> blk.N.X etc. The prefix maps:
    model.language_model.            ->  (nothing, then rule-based strip)
    model.language_model.layers.     -> blk.
    model.language_model.            -> (root norms / embed)
    embed_tokens.weight              -> token_embd.weight
    embedding_norm.weight            -> token_embd_norm.weight
    lm_head.weight                   -> output.weight
"""

import argparse
import json
import re
import struct as _struct
import sys

import gguf
import numpy as np

_STRUCT_U64 = _struct.Struct("<Q")

# --------------------------------------------------------------------------- #
# HF name -> GGUF name mapping (lfm2 language tower)
# --------------------------------------------------------------------------- #
def hf_to_gguf_name(hf_name: str) -> str:
    """Map an HF lfm2_vl language-tower tensor name to its GGUF name."""
    name = hf_name
    # vision tower + projector live in mmproj, skip them entirely
    if ".vision_tower." in name or ".multi_modal_projector." in name:
        return ""
    name = name.replace("model.language_model.", "", 1)
    name = re.sub(r"^layers\.(\d+)\.", r"blk.\1.", name)
    name = name.replace("embed_tokens.weight", "token_embd.weight")
    name = name.replace("embedding_norm.weight", "token_embd_norm.weight")
    name = name.replace("lm_head.weight", "output.weight")
    name = re.sub(r"^(\d+)\.", r"blk.\1.", name)  # safety net
    return name


# --------------------------------------------------------------------------- #
# GGUF name -> HF name inverse map (lfm2 language tower, verified against the
# LFM2.5-VL-3B checkpoint: gguf stores [out, in] like torch .T for 2-D weights)
# --------------------------------------------------------------------------- #
def gguf_to_hf_name(gguf_name: str) -> str:
    if gguf_name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if gguf_name == "token_embd_norm.weight":
        return "model.language_model.embedding_norm.weight"
    if gguf_name == "output.weight":
        return "model.language_model.lm_head.weight"  # (not present: tied? checked)
    m = re.match(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        return ""
    idx, rest = m.groups()
    suffix_map = {
        "ffn_down.weight": "feed_forward.w2.weight",
        "ffn_gate.weight": "feed_forward.w1.weight",
        "ffn_up.weight": "feed_forward.w3.weight",
        "ffn_norm.weight": "ffn_norm.weight",
        "attn_norm.weight": "operator_norm.weight",
        "shortconv.conv.weight": "conv.conv.weight",
        "shortconv.in_proj.weight": "conv.in_proj.weight",
        "shortconv.out_proj.weight": "conv.out_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.out_proj.weight",
        "attn_q_norm.weight": "self_attn.q_layernorm.weight",
        "attn_k_norm.weight": "self_attn.k_layernorm.weight",
    }
    hf_suffix = suffix_map.get(rest)
    if hf_suffix is None:
        return ""
    return f"model.language_model.layers.{idx}.{hf_suffix}"


# --------------------------------------------------------------------------- #
# Structural checks
# --------------------------------------------------------------------------- #
REQUIRED_TOKENIZER_KV_PREFIXES = (
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.model",
)

def structural_report(ours: gguf.GGUFReader, ref: gguf.GGUFReader) -> dict:
    issues: list[str] = []
    # magic/version are validated implicitly by GGUFReader construction
    our_tensors = {t.name: t for t in ours.tensors}
    ref_tensors = {t.name: t for t in ref.tensors}
    missing = sorted(set(ref_tensors) - set(our_tensors))
    extra = sorted(set(our_tensors) - set(ref_tensors))
    if missing:
        issues.append(f"missing tensors vs reference: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if extra:
        issues.append(f"extra tensors vs reference: {extra[:8]}{' ...' if len(extra) > 8 else ''}")

    shape_mismatch, dtype_mismatch = [], []
    for name in sorted(set(our_tensors) & set(ref_tensors)):
        o, r = our_tensors[name], ref_tensors[name]
        if tuple(int(x) for x in o.shape) != tuple(int(x) for x in r.shape):
            shape_mismatch.append((name, list(map(int, o.shape)), list(map(int, r.shape))))
        if o.tensor_type != r.tensor_type:
            dtype_mismatch.append((name, o.tensor_type.name, r.tensor_type.name))

    if shape_mismatch:
        issues.append(f"shape mismatches: {shape_mismatch[:6]}{' ...' if len(shape_mismatch) > 6 else ''}")
    if dtype_mismatch:
        issues.append(f"quant-type mismatches: {dtype_mismatch[:6]}{' ...' if len(dtype_mismatch) > 6 else ''}")

    for pref in REQUIRED_TOKENIZER_KV_PREFIXES:
        if not any(k.startswith(pref) for k in ours.fields):
            issues.append(f"missing required KV: {pref}*")
    if "general.architecture" not in ours.fields:
        issues.append("missing KV: general.architecture")

    return {
        "our_tensor_count": len(our_tensors),
        "ref_tensor_count": len(ref_tensors),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "shape_mismatch_count": len(shape_mismatch),
        "dtype_mismatch_count": len(dtype_mismatch),
        "dtype_mismatch_sample": dtype_mismatch[:12],
        "issues": issues,
    }


# --------------------------------------------------------------------------- #
# Numerical checks
# --------------------------------------------------------------------------- #
def tensor_rms(a: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    return float(np.sqrt(np.mean(a64 * a64))) or 1e-30


def compare_axis(ours_dq: np.ndarray, ref_dq: np.ndarray, orig: np.ndarray) -> dict:
    """Relative-error metrics for both comparison axes."""
    o64 = ours_dq.astype(np.float64)
    r64 = ref_dq.astype(np.float64)
    g64 = orig.astype(np.float64)
    rms_g = tensor_rms(orig)
    # ours vs reference (both are quantizations of the same source)
    rel_ref = float(np.sqrt(np.mean((o64 - r64) ** 2)) / rms_g)
    # quantization error vs original, per implementation
    err_ours = float(np.sqrt(np.mean((o64 - g64) ** 2)) / rms_g)
    err_ref = float(np.sqrt(np.mean((r64 - g64) ** 2)) / rms_g)
    return {
        "rel_vs_ref": rel_ref,
        "qerr_ours": err_ours,
        "qerr_ref": err_ref,
        "err_ratio": (err_ours / err_ref) if err_ref > 0 else float("inf"),
    }


def load_original_tensor(st_path: str, hf_name: str):
    # framework="numpy" chokes on bf16; torch-free path: read the header manually.
    # Simplest robust approach: use safe_open with framework="np" on a per-slice
    # basis is still dtype-bound, so parse the raw file bytes for this tensor.
    # The safetensors format: 8-byte LE u64 header_len, JSON header, aligned data.
    import json as _json
    with open(st_path, "rb") as f:
        (hlen,) = _STRUCT_U64.unpack(f.read(8))
        header = _json.loads(f.read(hlen))
        if hf_name not in header or hf_name == "__metadata__":
            return None
        info = header[hf_name]
        dtype, shape, begin, end = info["dtype"], info["shape"], info["data_offsets"][0], info["data_offsets"][1]
        f.seek(8 + hlen + begin)
        raw = f.read(end - begin)
    import numpy as _np
    dt_map = {"BF16": _np.uint16, "F16": _np.float16, "F32": _np.float32, "F64": _np.float64}
    if dtype not in dt_map:
        raise ValueError(f"unsupported original dtype {dtype} for {hf_name}")
    arr = _np.frombuffer(raw, dtype=dt_map[dtype]).reshape(shape)
    return arr


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="GGUF conformance checker")
    ap.add_argument("ours", help="our .gguf output")
    ap.add_argument("ref", help="unsloth reference .gguf")
    ap.add_argument("original", help="original .safetensors (single file)")
    ap.add_argument("--max-tensors", type=int, default=0, help="limit compared tensors (0=all)")
    ap.add_argument("--json", default="", help="write full report JSON here")
    args = ap.parse_args()

    ours = gguf.GGUFReader(args.ours)
    ref = gguf.GGUFReader(args.ref)

    struct = structural_report(ours, ref)
    print("=" * 72)
    print(f"OURS : {args.ours}  ({struct['our_tensor_count']} tensors)")
    print(f"REF  : {args.ref}  ({struct['ref_tensor_count']} tensors)")
    print("=" * 72)
    for it in struct["issues"]:
        print("STRUCT ISSUE:", it)
    if not struct["issues"]:
        print("STRUCT: OK (same inventory, shapes, quant types, tokenizer KVs)")

    our_t = {t.name: t for t in ours.tensors}
    ref_t = {t.name: t for t in ref.tensors}
    common = sorted(set(our_t) & set(ref_t))
    if args.max_tensors:
        common = common[: args.max_tensors]

    rows = []
    fails = 0
    for name in common:
        ot, rt = our_t[name], ref_t[name]
        n = int(np.prod([int(x) for x in ot.shape]))
        # HF name: inverse map (verified against the actual checkpoint naming)
        hf = gguf_to_hf_name(name)
        if not hf:
            rows.append({"name": name, "status": "unmapped-name"})
            continue
        orig = load_original_tensor(args.original, hf)
        if orig is None:
            rows.append({"name": name, "status": "no-original"})
            continue
        # uint16 raw bits (bf16/f16 loaded bit-exact) -> float32 for math
        orig_bits = np.asarray(orig)
        if orig_bits.dtype == np.uint16:
            # bf16: widen into the high 16 bits of an f32
            orig = (orig_bits.astype(np.uint32) << 16).view(np.float32)
        elif orig_bits.dtype in (np.float32, np.float16, np.float64):
            orig = orig_bits.astype(np.float32)
        else:
            rows.append({"name": name, "status": f"unsupported-original-dtype {orig_bits.dtype}"})
            continue
        ours_dq = gguf.dequantize(np.asarray(ot.data), ot.tensor_type)
        ref_dq = gguf.dequantize(np.asarray(rt.data), rt.tensor_type)
        orig_f = np.asarray(orig, dtype=np.float32)
        # gguf stores 2-D weights as [out, in]; HF/torch as [in, out] -> transpose.
        # Square matrices are ambiguous: pick the orientation with lower q-error
        # (both files quantize the same source, so the correct orientation wins).
        if orig_f.ndim == 2 and ours_dq.shape == tuple(reversed(orig_f.shape)) and ours_dq.shape[0] != ours_dq.shape[1]:
            orig_f = orig_f.T
        if orig_f.ndim == 2 and orig_f.shape == ours_dq.shape and orig_f.shape[0] == orig_f.shape[1]:
            # square: keep the orientation whose dequantized output hugs the original
            if float(np.sqrt(np.mean((ours_dq - orig_f) ** 2))) > float(np.sqrt(np.mean((ours_dq - orig_f.T) ** 2))):
                orig_f = orig_f.T
        ours_dq = ours_dq.reshape(-1)
        ref_dq = ref_dq.reshape(-1)
        orig_f = orig_f.reshape(-1)
        if ours_dq.shape != orig_f.shape or ref_dq.shape != orig_f.shape:
            rows.append({"name": name, "status": f"shape-mismatch dq {ours_dq.shape} vs {orig_f.shape}"})
            fails += 1
            continue
        m = compare_axis(ours_dq, ref_dq, orig_f)
        m.update({"name": name, "qtype": ot.tensor_type.name, "elems": n, "status": "ok"})
        rows.append(m)
        flag = ""
        # thresholds: same-quantizer pair should be far closer than raw q-error
        if m["rel_vs_ref"] > max(0.35 * m["qerr_ref"], 0.004):
            flag = "  <-- DIVERGES from reference"
            fails += 1
        # err_ratio is only meaningful when the reference has nonzero q-error
        # (F32/copy tensors carry zero q-error by construction)
        if m["qerr_ref"] > 0 and m["err_ratio"] > 1.60:
            flag += "  <-- OUR Q-ERROR MUCH WORSE"
            fails += 1
        print(f"{name:58s} {ot.tensor_type.name:7s} rel_vs_ref={m['rel_vs_ref']:.5f} "
              f"qerr ours={m['qerr_ours']:.5f} ref={m['qerr_ref']:.5f} ratio={m['err_ratio']:.3f}{flag}")

    print("-" * 72)
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"tensors compared: {len(rows)} (ok {n_ok}, issues {fails})")
    verdict = "PASS" if (fails == 0 and not struct["issues"]) else "FAIL"
    print(f"CONFORMANCE VERDICT: {verdict}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"structural": struct, "rows": rows, "verdict": verdict}, f, indent=1)
        print("report:", args.json)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
