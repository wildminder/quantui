"""Pure .pt -> .safetensors checkpoint-conversion logic (Textual-free).

Extracts an inference state dict from a PyTorch training checkpoint and saves it
as a clean ``.safetensors`` file that the ComfyUI family can quantize.

Motivation (user request): quantization input requires ``.safetensors``; raw
training checkpoints (``model_225000.pt`` etc.) are rejected. This module makes
the conversion an in-app one-click step.

Checkpoint shapes handled (mirrors the verified Raon-OpenTTS recipe):
- ``{"model_state_dict": {...}}``            -- raw weights
- ``{"ema_model_state_dict": {...}}``        -- EMA weights with "ema_model." prefix
- flat ``{tensor-name: tensor}``             -- already-clean checkpoints
Any other non-tensor entries (optimizer/scheduler state, counters) are ignored.

Candidate selection without a reference model: prefer EMA (inference best
practice), fall back to raw, then flat. Key normalization strips the
``ema_model.`` prefix and drops known-non-inference keys.

Textual-free and torch-lazy: torch / safetensors are imported inside functions so
importing this module never pulls heavy deps (keeps tests and TUI startup fast).
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

# Keys the official F5-TTS-style inference loaders explicitly drop.
EMA_SKIP = {"initted", "step", "update"}
STALE_KEYS = {
    "mel_spec.mel_stft.mel_scale.fb",
    "mel_spec.mel_stft.spectrogram.window",
}
EMA_PREFIX = "ema_model."


def is_pt_file(path: str) -> bool:
    """True when ``path`` exists and looks like a PyTorch checkpoint."""
    if not path or not os.path.isfile(path):
        return False
    return path.lower().endswith((".pt", ".pth", ".ckpt"))


def default_output_path(pt_path: str) -> str:
    """``dir/model_225000.pt`` -> ``dir/model_225000-safetensors.safetensors``."""
    base = os.path.splitext(os.path.basename(pt_path))[0]
    return os.path.join(os.path.dirname(os.path.abspath(pt_path)), f"{base}-safetensors.safetensors")


@dataclass
class ConvertReport:
    """Summary of one conversion (pure data; the UI just prints fields)."""

    source: str = ""
    output: str = ""
    selected: str = ""          # which candidate state dict won
    n_tensors: int = 0
    n_params: int = 0
    bytes_written: int = 0
    dtype: str = "keep"
    cast_tensors: int = 0
    dropped_keys: list[str] | None = None

    def ok(self) -> bool:
        return self.n_tensors > 0 and bool(self.output)


def describe_state_dict(sd: dict) -> str:
    """One-line summary: tensor count, dtypes, top prefixes (for the log)."""
    tensors = {k: v for k, v in sd.items() if hasattr(v, "shape") and hasattr(v, "dtype")}
    dtypes = Counter(str(v.dtype) for v in tensors.values())
    prefixes = Counter(k.split(".")[0] for k in tensors)
    top = ", ".join(f"{p}x{n}" for p, n in prefixes.most_common(4))
    nbytes = sum(
        getattr(v, "numel", lambda: 0)() * getattr(v, "element_size", lambda: 0)()
        for v in tensors.values()
    )
    return (f"tensors={len(tensors)} bytes={nbytes/1e9:.2f}GB "
            f"dtypes={dict(dtypes)} prefixes=[{top}]")


def _strip_prefix(sd: dict) -> tuple[dict, list[str]]:
    """Strip the ``ema_model.`` prefix; drop known skip/stale keys (matched
    against BOTH the raw and the stripped key name). Returns
    ``(clean_dict, dropped_key_names)``."""
    clean: dict = {}
    dropped: list[str] = []
    for k, v in sd.items():
        nk = k[len(EMA_PREFIX):] if k.startswith(EMA_PREFIX) else k
        if k in EMA_SKIP or k in STALE_KEYS or nk in EMA_SKIP or nk in STALE_KEYS:
            dropped.append(k)
            continue
        clean[nk] = v
    return clean, dropped


def extract_candidates(ckpt: dict) -> dict[str, dict]:
    """All plausible inference state dicts from a loaded checkpoint.

    Returns ``{candidate_name: {key: tensor}}`` ordered by preference:
    EMA-stripped first (inference best practice), raw second, flat third.
    """
    candidates: dict[str, dict] = {}
    ema = ckpt.get("ema_model_state_dict")
    if isinstance(ema, dict):
        clean, _ = _strip_prefix(ema)
        clean = {k: v for k, v in clean.items() if hasattr(v, "shape")}
        if clean:
            candidates["ema_model_state_dict(stripped)"] = clean
    raw = ckpt.get("model_state_dict")
    if isinstance(raw, dict):
        clean_raw = {k: v for k, v in raw.items() if hasattr(v, "shape")}
        for stale in STALE_KEYS:
            clean_raw.pop(stale, None)
        if clean_raw:
            candidates["model_state_dict(raw)"] = clean_raw
    flat = {k: v for k, v in ckpt.items() if hasattr(v, "shape")}
    if flat:
        candidates["toplevel-flat"] = flat
    return candidates


def score_candidate(cand: dict) -> tuple:
    """Preference score without a reference model: more tensors win; fewer
    suspiciously-named keys (optimizer moments etc.) win."""
    bad = sum(1 for k in cand if "exp_avg" in k or "_state" in k)
    return (len(cand), -bad)


def pick_candidate(ckpt: dict) -> tuple[str, dict]:
    """Choose the best inference state dict from a checkpoint dict."""
    candidates = extract_candidates(ckpt)
    if not candidates:
        raise ValueError(
            "No tensor state found in the checkpoint (looked for "
            "ema_model_state_dict / model_state_dict / top-level tensors)."
        )
    best = max(candidates.items(), key=lambda kv: score_candidate(kv[1]))
    return best


def convert_pt_to_safetensors(
    pt_path: str,
    out_path: str | None = None,
    dtype: str = "keep",
    progress=None,
) -> ConvertReport:
    """Convert one ``.pt`` checkpoint to a clean ``.safetensors`` file.

    Args:
        pt_path: source checkpoint (``.pt`` / ``.pth`` / ``.ckpt``).
        out_path: destination; defaults to :func:`default_output_path`.
        dtype: ``"keep"`` preserves source dtypes; ``"bf16"`` casts floating
            tensors to bfloat16 (except ``inv_freq``, matching the verified
            Raon recipe).
        progress: optional ``callable(done: int, total: int, label: str)``.

    Returns:
        :class:`ConvertReport` describing what was written.

    Raises:
        FileNotFoundError, ValueError, RuntimeError on unusable inputs;
        ImportError when torch/safetensors are missing in this interpreter.
    """
    if progress is None:
        progress = lambda done, total, label="": None  # noqa: E731
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {pt_path}")
    out = out_path or default_output_path(pt_path)

    import gc

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            f"torch is required to read .pt checkpoints: {exc}. Point the "
            "'Worker Python' field at an interpreter that has torch installed."
        ) from exc
    try:
        from safetensors.torch import save_file
    except Exception as exc:  # noqa: BLE001
        raise ImportError(f"safetensors is required: {exc}") from exc

    size_gb = os.path.getsize(pt_path) / 1e9
    progress(0, 100, f"Loading {os.path.basename(pt_path)} ({size_gb:.2f} GB)")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unsupported checkpoint format: top-level {type(ckpt).__name__} "
            "(expected a state-dict container)."
        )

    name, sd = pick_candidate(ckpt)
    dropped = [k for k in sd if k in STALE_KEYS]
    del ckpt
    gc.collect()

    report = ConvertReport(source=pt_path, output=out, selected=name, dtype=dtype)
    total = len(sd)
    out_sd: dict = {}
    for i, (key, tensor) in enumerate(sorted(sd.items()), start=1):
        t = tensor.detach().cpu().contiguous()
        if (
            dtype == "bf16"
            and t.is_floating_point()
            and not key.endswith("inv_freq")
        ):
            t = t.to(torch.bfloat16)
            report.cast_tensors += 1
        out_sd[key] = t
        report.n_params += t.numel()
        if i % 64 == 0 or i == total:
            progress(i, total, f"Converting tensors {i}/{total}")
    report.dropped_keys = dropped
    report.n_tensors = len(out_sd)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    progress(total, total, f"Saving {out}")
    save_file(out_sd, out, metadata={
        "format": "pt",
        "source": os.path.basename(pt_path),
        "state_dict": name,
        "dtype": dtype,
    })
    report.bytes_written = os.path.getsize(out)
    progress(total, total, "Conversion done")
    return report
