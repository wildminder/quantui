"""Capability detection for the ComfyUI / ``convert_to_quant`` (ctq) family.

The heavy dependencies (``torch``, ``triton``, ``convert_to_quant``) live in a
*separate* interpreter (``pybin_ctq``). Rather than import them into the TUI, we
probe that interpreter by running a tiny inline script with ``pybin -c "..."`` and
parsing the JSON it prints.

All results here are **advisory only** -- capability mismatches produce soft warning
strings and never block ``Run`` (see ``check_ctq_requirements``). The TUI shows them
in a badge + status line.

This module is dependency-free (stdlib only) so it imports cleanly in any env,
including sandboxes where the probed interpreter (or torch) is absent.
"""

import json
import subprocess
from dataclasses import dataclass, field

from .quant_methods import comfy_format

# Inline probe: prints a single JSON line describing the target interpreter's env.
# Runs inside ``pybin_ctq`` so it sees that interpreter's installed packages.
_PROBE_SCRIPT = r'''
import sys, json
info = {
    "python_version": ".".join(map(str, sys.version_info[:3])),
    "torch_version": None,
    "cuda_version": None,
    "triton": False,
    "comfy_kitchen": False,
    "convert_to_quant": False,
}
try:
    import torch
    info["torch_version"] = torch.__version__
    try:
        info["cuda_version"] = torch.version.cuda
    except Exception:
        pass
except Exception:
    pass
try:
    import triton
    info["triton"] = True
except Exception:
    pass
try:
    import comfy_kitchen
    info["comfy_kitchen"] = True
except Exception:
    pass
try:
    import convert_to_quant
    info["convert_to_quant"] = True
except Exception:
    pass
print(json.dumps(info))
'''


@dataclass
class CapabilityReport:
    """Snapshot of a probed worker interpreter's capabilities."""

    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    triton: bool = False
    comfy_kitchen: bool = False
    convert_to_quant: bool = False
    warnings: list[str] = field(default_factory=list)


def _version_tuple(version: str | None) -> tuple[int, ...]:
    """Best-effort parse of ``"2.10"`` / ``"13.0"`` into a comparable int tuple."""
    if not version:
        return ()
    parts: list[int] = []
    for piece in version.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def probe_worker_env(pybin: str) -> CapabilityReport:
    """Run ``pybin -c <probe>`` and parse the result into a ``CapabilityReport``.

    Any failure (interpreter missing, import errors, unparseable output) yields a
    report with ``None``/``False`` fields plus a ``warnings`` entry -- it never raises.
    """
    try:
        proc = subprocess.run(
            [pybin, "-c", _PROBE_SCRIPT],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        return CapabilityReport(warnings=[f"Could not run capability probe: {exc}"])

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"exit {proc.returncode}"
        return CapabilityReport(warnings=[f"Capability probe failed: {tail}"])

    payload: dict | None = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            # Non-JSON line (progress/warning noise on stdout): try the next one.
            continue

    if not isinstance(payload, dict):
        return CapabilityReport(warnings=["Could not parse capability probe output."])

    return CapabilityReport(
        python_version=payload.get("python_version"),
        torch_version=payload.get("torch_version"),
        cuda_version=payload.get("cuda_version"),
        triton=bool(payload.get("triton", False)),
        comfy_kitchen=bool(payload.get("comfy_kitchen", False)),
        convert_to_quant=bool(payload.get("convert_to_quant", False)),
        warnings=[],
    )


def check_ctq_requirements(report: CapabilityReport, format_id: str) -> list[str]:
    """Return soft, advisory warning strings for running ``format_id`` in ``report``.

    Never blocks Run -- the caller decides how (or whether) to surface these. An
    unknown ``format_id`` returns ``[]`` (no capability warnings).
    """
    warnings: list[str] = []
    try:
        fmt = comfy_format(format_id)
    except KeyError:
        return warnings

    if report.torch_version is None:
        warnings.append("PyTorch is not importable in this interpreter - install a CUDA build of torch.")
    elif fmt.requires_torch and _version_tuple(report.torch_version) < _version_tuple(fmt.requires_torch):
        warnings.append(
            f"{fmt.label} needs torch>={fmt.requires_torch} (found {report.torch_version})."
        )

    if fmt.requires_cuda and (
        report.cuda_version is None
        or _version_tuple(report.cuda_version) < _version_tuple(fmt.requires_cuda)
    ):
        warnings.append(
            f"{fmt.label} needs CUDA>={fmt.requires_cuda} (found {report.cuda_version or 'n/a'})."
        )

    if fmt.requires_py and (
        report.python_version is None
        or _version_tuple(report.python_version) < _version_tuple(fmt.requires_py)
    ):
        warnings.append(
            f"{fmt.label} needs Python>={fmt.requires_py} (found {report.python_version or 'n/a'})."
        )

    if "triton" in fmt.needs and not report.triton:
        warnings.append(
            "INT8 kernels need Triton - install 'triton' on Linux / 'triton-windows' on Windows."
        )

    if "blackwell" in fmt.needs and not report.comfy_kitchen:
        warnings.append("Blackwell formats (NVFP4/MXFP8) also need 'comfy-kitchen'.")

    if "comfy_kitchen" in fmt.needs and not report.comfy_kitchen:
        warnings.append(
            "W4A4/W4A8 formats need 'comfy-kitchen' + ComfyUI - point 'Worker Python (ctq)' "
            "at a ComfyUI interpreter that has comfy-kitchen installed."
        )

    if not report.convert_to_quant:
        warnings.append("'convert_to_quant' is not installed in this interpreter - pip install convert-to-quant.")

    return warnings
