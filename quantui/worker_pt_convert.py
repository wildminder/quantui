"""Backend worker for .pt -> .safetensors checkpoint conversion.

Subprocess twin of :mod:`quantui.pt_convert` (same pattern as ``worker_ctq``):
the TUI launches this module so torch loading happens OUTSIDE the UI process.
Emits the standard ``CTQ_PROGRESS`` envelopes so the header bar advances.

Usage:
    python -m quantui.worker_pt_convert -i model_225000.pt [-o out.safetensors]
        [--dtype keep|bf16]
"""

from __future__ import annotations

import argparse
import sys

from .stream_parser import CTQ_PROGRESS_PREFIX


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
    print(f"{CTQ_PROGRESS_PREFIX}{json.dumps(payload)}", flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=".pt checkpoint -> safetensors converter")
    p.add_argument("-i", "--input", required=True, help="Source .pt/.pth/.ckpt checkpoint.")
    p.add_argument("-o", "--output", default=None,
                   help="Destination .safetensors (default: <input dir>/<stem>-safetensors.safetensors).")
    p.add_argument("--dtype", choices=["keep", "bf16"], default="keep",
                   help="'keep' preserves source dtypes; 'bf16' casts floats to bfloat16.")
    args = p.parse_args(argv)

    from . import pt_convert

    log(f"Converting {args.input} -> {args.output or '(auto name)'} ...")
    try:
        report = pt_convert.convert_pt_to_safetensors(
            args.input,
            args.output,
            dtype=args.dtype,
            progress=lambda done, total, label="":
                progress("convert", cur=done, total=total, label=label),
        )
    except ImportError as exc:
        fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        fail(f"Conversion failed: {exc}")

    log(f"Candidate state dict: {report.selected}")
    log(f"Wrote {report.n_tensors} tensors ({report.n_params/1e6:.1f}M params, "
        f"{report.bytes_written/1e9:.2f} GB)"
        + (f", cast {report.cast_tensors} tensors to bf16" if report.cast_tensors else ""))
    if report.dropped_keys:
        log(f"Dropped non-inference keys: {', '.join(report.dropped_keys)}")
    log(f"DONE -> {report.output}")


if __name__ == "__main__":
    main()
