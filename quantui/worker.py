"""Backend worker for Unsloth GGUF quantization.

This script is invoked by the TUI as a subprocess so the UI stays responsive
and Unsloth (torch/CUDA) lives in its own heavy process. It validates the
model input, imports Unsloth, and runs `save_pretrained_gguf` (or
`push_to_hub_gguf`). All progress is printed to stdout and streamed to the TUI.

Usage:
    python worker.py --model PATH --output DIR --method q4_k_m
    python worker.py --model PATH --output DIR --method "q4_k_m, q5_k_m"
    python worker.py --model PATH --output DIR --method iq2_xs --imatrix auto
    python worker.py --list-methods
"""

import argparse
import os
import shutil
import sys


# Module-level snapshot of the native surface (S5.2): the dispatch check must
# not import gguf_qkernels (which pulls numpy) before --list-methods can run.
def _native_methods() -> frozenset:
    try:
        from .gguf_qkernels import NATIVE_METHODS
        return frozenset(NATIVE_METHODS)
    except Exception:  # boundary: snapshot must never break worker startup
        return frozenset()

NATIVE_METHOD_SET = _native_methods()


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Method / imatrix helpers (T7, plan 2026-08-31-gguf-unsloth-parity).
# parse_methods is imported from run_config -- the SAME splitter the UI and
# validation use; never re-implement the comma semantics here.
# ---------------------------------------------------------------------------
def method_or_methods(raw: str):
    """`--method` -> unsloth's str-or-list API value.

    1 id  -> the plain string (the overwhelmingly common case).
    >=2 ids -> a list (unsloth loops the quant pipeline per id).
    0 ids -> ValueError: argparse allowed empty; a silent no-op run is worse
             than a loud failure.
    """
    from .run_config import parse_methods

    ids = parse_methods(raw)
    if not ids:
        raise ValueError("--method must name at least one quantization method.")
    return ids[0] if len(ids) == 1 else ids


def imatrix_value(raw: str):
    """`--imatrix` -> unsloth's imatrix_file= value.

    ""    -> None (most quants need no imatrix)
    auto  -> True (unsloth's own contract: fetch the UPSTREAM imatrix)
    path  -> the path string verbatim (unsloth loads THAT file)
    """
    if raw == "":
        return None
    if raw == "auto":
        return True
    return raw


def _gate_methods_imatrix(methods: list, imatrix: str) -> None:
    """Pre-import re-validation (fires BEFORE `from unsloth import ...`).

    unsloth raises RuntimeError for a missing imatrix only AFTER a full model
    load -- minutes in. This gate turns that into an instant, friendly error.
    The UI (validate_gguf, T5) runs the same check; this is the backstop for
    direct worker invocations.
    """
    from .quant_methods import IMATRIX_QUANT_IDS

    for mid in methods:
        if mid in IMATRIX_QUANT_IDS:
            if not imatrix:
                fail(
                    f"{mid} requires an imatrix (set a path or 'auto'). "
                    "unsloth refuses to quantize IQ* ids without imatrix_file=."
                )
    if imatrix and imatrix != "auto" and not os.path.isfile(imatrix):
        fail(f"imatrix file not found: {imatrix}")


# Structured progress envelope (same protocol as worker_ctq.py): the TUI parses
# "CTQ_PROGRESS {...}" lines into a determinate bar. Emitted here so the GGUF
# header bar advances too (user report: main progress showed 0 the whole run --
# the GGUF worker used to print only plain text, which carries no fraction).
_CTQ_PROGRESS_PREFIX = "CTQ_PROGRESS "


def progress(phase: str, cur=None, total=None, pct=None, label: str = "") -> None:
    import json

    payload: dict = {"phase": phase}
    if cur is not None:
        payload["cur"] = int(cur)
    if total is not None:
        payload["total"] = int(total)
    if pct is not None:
        payload["pct"] = float(pct)
    if label:
        payload["label"] = label
    print(f"{_CTQ_PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


def _run_with_output_progress(fn, output_dir: str, expected_bytes: int, label: str) -> None:
    """Run ``fn()`` in a thread and emit file-growth progress for the .gguf files.

    The GGUF export grows one or more ``.gguf`` files inside ``output_dir``; polling
    their combined size against the expected f16 model size gives a genuine overall
        progress signal. Falls back to a bare call on any instrumentation error so
    progress can never block the actual export.

    Exceptions from the export thread are RE-RAISED on the calling thread: the
    export failed, so reporting DONE/100% would be a lie (the TUI showed
    success while no file was written -- found by the LFM2.5-VL imatrix run).
    """
    import threading

    outcome: dict = {}

    def _target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - propagate ANY thread failure
            outcome["exc"] = exc

    def _emit_progress():
        sz = _gguf_bytes()
        if sz > 0:
            pct = min(99.0, 100.0 * sz / expected)
            if abs(pct - last[0]) >= 1.0 or last[0] < 0:
                progress("quantize", pct=pct, label=label)
                last[0] = pct

    def _gguf_bytes() -> int:
        try:
            return sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if f.endswith(".gguf")
            )
        except OSError:
            return 0

    expected = max(1, int(expected_bytes * 0.6))  # Q-quant is well under f16 size
    last = [-1.0]
    try:
        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        while thread.is_alive():
            _emit_progress()
            thread.join(timeout=1.0)
        thread.join()
    except Exception:
        # Instrumentation failed (never the export itself, which runs in the
        # thread); fall back to a bare synchronous call so the run proceeds.
        if "exc" not in outcome and not thread.is_alive():
            raise
        fn()
    if "exc" in outcome:
        raise outcome["exc"]
    progress("quantize", pct=100.0, label="GGUF written")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def resolve_model_dir(model_path: str) -> str:
    """Return the HF-format model directory to pass to from_pretrained.

    Accepts either a HuggingFace folder (must contain config.json) or a single
    .safetensors file (which must sit next to a config.json + tokenizer).
    """
    if not os.path.exists(model_path):
        fail(f"Model path does not exist: {model_path}")

    if os.path.isfile(model_path):
        if not model_path.endswith(".safetensors"):
            fail("Model file must be a .safetensors file or a HuggingFace folder.")
        parent = os.path.dirname(os.path.abspath(model_path))
        if not os.path.exists(os.path.join(parent, "config.json")):
            fail(
                "A single .safetensors needs a config.json (and tokenizer files) in the "
                "same folder. Put it in a HuggingFace-format folder, or pass the folder path."
            )
        return parent

    if not os.path.isdir(model_path):
        fail(f"Model path is neither a file nor a folder: {model_path}")

    if not os.path.exists(os.path.join(model_path, "config.json")):
        fail(
            f"No config.json found in {model_path}. Expected a HuggingFace-format model "
            "folder (config.json + *.safetensors + tokenizer files)."
        )
    return model_path


def check_supported_architecture(model_dir: str) -> None:
    """Pre-flight guard: reject models Unsloth GGUF cannot quantize, with a
    friendly message instead of a raw transformers/Unsloth traceback."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return  # No config to inspect; let Unsloth attempt the load.
    try:
        from transformers import AutoConfig
    except Exception:  # boundary: optional import probe in the worker subprocess;
        return         # the later unsloth import reports the real problem
    try:
        try:
            cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
        except Exception as exc:
            msg = str(exc).lower()
            if "trust_remote_code" in msg or "remote code" in msg:
                cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            else:
                raise
    except Exception as exc:
        fail(
            "Could not read this model's configuration with the installed transformers "
            f"({type(exc).__name__}: {exc}).\n"
            "The architecture may not be recognized by your transformers version. "
            "Models like VibeVoice need a recent transformers (the VibeVoice class was "
            "merged into transformers ~Sept 2025, PR #40546). Try upgrading:\n"
            "    pip install -U transformers\n"
            "Note: even after upgrading, the UNSLOTH backend only supports models "
            "whose weights llama.cpp can represent, so TTS / Seq2Seq models still "
            "won't convert that way.\n"
            "FIX: use the NATIVE backend — it converts ANY architecture without "
            "transformers (generic name mapping), e.g.:\n"
            f"    ... --method native_q8_0 --backend native   "
            f"(surface: {', '.join(sorted(NATIVE_METHOD_SET))})\n"
            "Or in the TUI: GGUF panel -> method 'native_q8_0' -> Run."
        )
        return
    archs = list(getattr(cfg, "architectures", []) or [])
    model_type = getattr(cfg, "model_type", None)
    # Unsloth 2026.9+ converts BOTH plain causal LMs and VLMs whose text tower
    # llama.cpp supports -- Lfm2VlForConditionalGeneration (LFM2.5-VL) exports
    # a language GGUF + an mmproj GGUF, and Qwen3-VL / Gemma3 style
    # ForConditionalGeneration archs are handled the same way. Blocking every
    # non-"ForCausalLM" arch locked users out of supported models (found by the
    # LFM2.5-VL-3B conformance run: unsloth+converter accept it, we refused it).
    # Blocklist instead: TTS / seq2seq / diffusion families that genuinely fail.
    blocked_model_types = {
        "vibevoice", "musicgen", "bark", "speecht5", "whisper",
        "t5", "m2m_100", "marian", "seamless_m4t", "studio_omni",
    }
    blocked = str(model_type or "").lower() in blocked_model_types
    if blocked:
        name = ", ".join(archs) if archs else (str(model_type) if model_type else "unknown")
        fail(
            f"Unsupported architecture for the UNSLOTH backend: {name}.\n"
            "Unsloth GGUF quantization only supports models whose weights llama.cpp "
            "can represent (causal LMs and VLMs with a causal-LM text tower).\n"
            "This model type is TTS / Seq2Seq / generative-audio.\n"
            "FIX: re-run with a NATIVE method id — it converts ANY architecture "
            "(generic name mapping, no transformers), e.g.:\n"
            f"    ... --method native_q8_0 --backend native   "
            f"(surface: {', '.join(sorted(NATIVE_METHOD_SET))})\n"
            "Or in the TUI: GGUF panel -> method 'native_q8_0' -> Run.\n"
            "(Alternatively run the model in PyTorch with bf16/fp16 or weight-only "
            "int4/int8 — torchao / quanto / bitsandbytes.)"
        )


def estimate_disk(model_dir: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith((".safetensors", ".bin", ".gguf", ".pt", ".pth")):
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total


def _run_native_backend(args) -> None:
    """S4.2: native GGUF export — NO transformers / unsloth / torch.

    Runs BEFORE any heavy import and BEFORE check_supported_architecture:
    the arch blocklist describes the *unsloth* backend's limits; the native
    exporter converts any HF checkpoint (TTS / unknown archs included) via
    generic name mapping (plan 2026-09-07).
    """
    from .gguf_export import GgufExportError, export_gguf
    from .gguf_qkernels import NATIVE_METHODS
    from .run_config import parse_methods

    methods = parse_methods(args.method)
    if len(methods) != 1:
        fail("Native backend supports exactly ONE method per run (comma lists are an unsloth-backend feature).")
    method = methods[0]
    if method not in NATIVE_METHODS:
        fail(
            f"'{method}' is not a native method. Native surface: {', '.join(NATIVE_METHODS)}. "
            "Official unsloth ids (q4_k_m, iq2_xs, ...) require --backend unsloth."
        )
    if args.imatrix:
        fail("The native GGUF backend does not support imatrix quants (v1). Use an unsloth IQ* id instead.")

    model_path = resolve_model_dir(args.model)
    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(model_path.rstrip("/\\"))
    out_path = os.path.join(out_dir, f"{base}-{method}.gguf")

    def _progress(done, total, name):
        # Counts go in cur/total (the stats line renders them as [cur/total]);
        # embedding them in the label duplicated the text in the footer.
        progress("quantize", cur=done, total=total,
                 pct=99.0 * done / max(total, 1),
                 label=f"Exporting GGUF ({method})")

    log(f"Native GGUF export: {model_path} -> {out_path} ({method})")
    try:
        report = export_gguf(model_path, out_path, method, progress=_progress)
    except GgufExportError as exc:
        fail(f"Native GGUF export failed: {exc}")

    log(f"Native export complete: {report.tensors_total} tensors "
        f"({report.tensors_quantized} quantized, {len(report.demoted)} demoted to F16, "
        f"{len(report.skipped)} skipped) -> {out_path}")
    for warning in report.warnings:
        log(f"WARNING: {warning}")
    log("DONE: GGUF written.")


def main() -> None:
    p = argparse.ArgumentParser(description="Unsloth GGUF quantizer worker")
    p.add_argument("--model", required=False, help="HF folder or .safetensors file")
    p.add_argument("--output", required=False, help="Output directory for the .gguf")
    p.add_argument(
        "--method", required=False,
        help="quantization_method id, or comma-joined ids for multi-quant",
    )
    p.add_argument(
        "--backend", choices=("unsloth", "native"), default="unsloth",
        help="worker engine: 'unsloth' (transformers+unsloth GGUF, default) or "
             "'native' (numpy-only exporter, no transformers; any architecture)",
    )
    p.add_argument(
        "--imatrix", default="",
        help="path to an imatrix file, or 'auto' to fetch the upstream one",
    )
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--push-to-hub", default="", help="HF repo id to push to")
    p.add_argument("--hf-token", default="", help="HF write token")
    p.add_argument("--list-methods", action="store_true", help="Print methods and exit")
    args = p.parse_args()

    if args.list_methods:
        from .quant_methods import METHODS
        for m in METHODS:
            marker = " [IMATRIX]" if m.needs_imatrix else ""
            print(f"{m.id}{marker}")
        return

    if not (args.model and args.output and args.method):
        fail("--model, --output and --method are required (unless --list-methods).")

    if args.backend == "native":
        _run_native_backend(args)
        return

    # S5.2 UX: a native_* method id implies the native backend even when the
    # user (or an older TUI profile) omitted --backend — the id IS the intent.
    if args.method and args.method.strip() in NATIVE_METHOD_SET:
        _run_native_backend(args)
        return

    _run_unsloth_backend(args)


def _run_unsloth_backend(args) -> None:
    """S4.2 dispatch target: the original unsloth flow (T7 et al.)."""
    from .run_config import parse_methods

    methods = parse_methods(args.method)
    if not methods:
        fail("--method must name at least one quantization method.")

    from .quant_methods import METHODS_BY_ID
    for mid in methods:
        if mid not in METHODS_BY_ID:
            # Don't hard-fail on unknown ids: Unsloth's API surface changes.
            log(f"WARNING: '{mid}' is not in the known list; passing it through to Unsloth anyway.")

    # T7: gate IQ*/imatrix BEFORE the heavy unsloth import + model load.
    _gate_methods_imatrix(methods, args.imatrix)

    # unsloth API: single id -> string, multiple ids -> list.
    method_arg = methods[0] if len(methods) == 1 else methods
    imatrix_arg = imatrix_value(args.imatrix)

    model_dir = resolve_model_dir(args.model)

    check_supported_architecture(model_dir)

    os.makedirs(args.output, exist_ok=True)
    if not os.access(args.output, os.W_OK):
        fail(f"Output directory not writable: {args.output}")

    model_bytes = estimate_disk(model_dir)
    free = shutil.disk_usage(args.output).free
    # Need ~2.5x: f16 intermediate + quantized output + headroom.
    # UQT_DISK_HEADROOM lets a power user lower the factor on a tight disk
    # (default 2.5; e.g. 1.6 is enough when the BF16 intermediate is the only
    # large byproduct). Values below 1.05 are ignored (would never fit even a
    # bare copy of the output).
    try:
        headroom = float(os.environ.get("UQT_DISK_HEADROOM", "2.5"))
    except ValueError:
        headroom = 2.5
    needed = int(model_bytes * max(headroom, 1.05))
    if free < needed:
        fail(
            f"Not enough free disk space in {args.output}. Need ~{needed/1e9:.1f} GB, "
            f"have ~{free/1e9:.1f} GB."
        )
    log(f"Model size ~{model_bytes/1e9:.2f} GB; output needs ~{needed/1e9:.1f} GB free.")

    log("Importing unsloth (this can take 10-30s)...")
    try:
        from unsloth import FastLanguageModel
    except Exception as exc:  # noqa: BLE001
        fail(
            f"Could not import unsloth: {exc}\n"
            "Install unsloth + a CUDA torch in THIS interpreter's environment. "
            "If your GPU env uses another python, point the TUI's 'Worker Python' field at it."
        )

    log(f"Loading model from: {model_dir}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=None,  # auto-detect
    )

    if args.push_to_hub:
        log(f"Exporting + pushing to HF ({args.push_to_hub}) with method={method_arg} ...")
        model.push_to_hub_gguf(
            args.push_to_hub,
            tokenizer,
            quantization_method=method_arg,
            token=args.hf_token or None,
            imatrix_file=imatrix_arg,
        )
    else:
        log(f"Exporting GGUF to {args.output} with method={method_arg} ...")
        _run_with_output_progress(
            lambda: model.save_pretrained_gguf(
                args.output,
                tokenizer,
                quantization_method=method_arg,
                maximum_memory_usage=0.5 if args.load_in_4bit else 0.75,
                imatrix_file=imatrix_arg,
            ),
            args.output,
            model_bytes,
            f"Exporting GGUF ({args.method})",
        )

    log("DONE: GGUF written.")


if __name__ == "__main__":
    main()
