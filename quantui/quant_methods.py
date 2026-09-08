"""Quantization method registry for the multi-family quantization TUI.

This module is the single source of truth for *all* quantization families:

- **GGUF** (Unsloth, causal-LM LLMs) -- the original registry.
- **COMFY** (ComfyUI / ``convert_to_quant``, diffusion models) -- added in M1.

It declares the data model (``Family`` / ``Backend`` / ``OptionField`` /
``QuantMethod`` / ``ComfyFormat`` / ``Preset``) and the registry helpers used by
``app.py`` and ``worker_ctq.py``. The GGUF ``METHODS`` entries are kept byte-for-
byte intact except for the added ``family`` / ``backend`` / ``options`` fields
(``options=[]`` for GGUF -- its advanced options stay as the static GGUF panel).

The module is dependency-free (stdlib only) so it can be imported in any env,
including sandboxes where ``convert_to_quant`` / ``torch`` are not installed.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Family(StrEnum):
    """Top-level quantization family. This is the single seam the TUI branches on."""

    GGUF = "gguf"  # Unsloth GGUF (causal-LM LLMs)
    COMFY = "comfy"  # ComfyUI / convert_to_quant (diffusion)


class Backend(StrEnum):
    """Concrete worker backend behind a family."""

    UNSLOTH = "unsloth"
    CTQ = "convert_to_quant"
    # comfy-kitchen + comfy.quant_ops (a ComfyUI-python interpreter). Required for
    # W4A4 (convrot_w4a4) and W4A8 (asym_w4a8_int8), which convert_to_quant cannot emit.
    COMFY_KITCHEN = "comfy_kitchen"
    # Native GGUF exporter (plan 2026-09-07): numpy-only, no transformers /
    # unsloth / torch — converts ANY HF safetensors checkpoint (unknown or
    # TTS archs included) to spec-conformant GGUF via generic name mapping.
    NATIVE = "native"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class OptionField:
    """Declarative description of ONE dynamic form control (data-driven UI).

    A ``ComfyFormat`` carries a list of these (``extra_options``); the TUI renders
    one widget per field and emits the matching CLI flag when the user runs.
    """

    key: str  # widget id + CLI mapping base
    label: str
    widget: Literal["select", "checkbox", "input", "int"]
    default: Any = None
    choices: list[tuple[str, str]] = field(default_factory=list)  # (label, value)
    help: str = ""
    visible_when: str | None = None  # predicate over sibling option values
    cli_flag: str | None = None  # emit "--flag <value>"
    cli_when_true: str | None = None  # emit literal flag when value is truthy


@dataclass
class QuantMethod:
    """A single GGUF quantization method (rendered as a dropdown entry)."""

    id: str
    label: str
    family: Family
    backend: Backend
    # True for the 11 official IQ* ids: unsloth REQUIRES imatrix_file= (path
    # or True = fetch upstream) before it will quantize, else RuntimeError
    # after a full model load. The TUI gates on this flag pre-run.
    needs_imatrix: bool = False
    approx_bpw: float | None = None
    description: str = ""
    options: list[OptionField] = field(default_factory=list)  # GGUF keeps this empty


@dataclass
class ComfyFormat:
    """One ComfyUI / convert_to_quant output format."""

    id: str
    label: str
    base_flags: list[str] = field(default_factory=list)  # always emitted for format
    extra_options: list[OptionField] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)  # capability tags
    backend: "Backend" = field(default=Backend.CTQ)  # which worker produces this format
    quant_format: str | None = None  # on-disk .comfy_quant "format" string (kitchen formats)
    requires_py: str | None = None
    requires_torch: str | None = None
    requires_cuda: str | None = None


@dataclass
class Preset:
    """A named ComfyUI preset (emits a single ``--flag``)."""

    id: str  # flux2 / wan / t5xxl / hunyuan / zimage
    label: str
    flag: str  # "--flux2" etc.
    recommended_format: str  # default format id when this preset is chosen
    # Option values applied on top of the recommended format (key -> value).
    # Lets a preset pin e.g. scaling_mode=row + convrot=True for flux2.
    recommended_options: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# GGUF registry (original entries, now tagged with family/backend/options)
# --------------------------------------------------------------------------- #
# Canonical official unsloth quant ids — copied from ref/unsloth/unsloth/save.py
# (ALLOWED_QUANTS + IMATRIX_QUANTS dicts, same order). ORDER IS LOAD-BEARING:
# the UI iterates these, and unsloth's docs list them in this order. The 11
# IMATRIX ids additionally REQUIRE imatrix_file= (path or True = fetch the
# upstream imatrix) or unsloth refuses to quantize. Single source of truth for
# tests + run_config + worker — never re-list ids elsewhere.
ALLOWED_QUANT_IDS: tuple[str, ...] = (
    "not_quantized", "fast_quantized", "quantized", "f32", "bf16", "f16",
    "q8_0", "q4_k_m", "q5_k_m", "q2_k", "q2_k_l", "q3_k_l", "q3_k_m",
    "q3_k_s", "q4_0", "q4_1", "q4_k_s", "q4_k", "q5_k", "q5_0", "q5_1",
    "q5_k_s", "q6_k", "q3_k_xs",
)
IMATRIX_QUANT_IDS: tuple[str, ...] = (
    "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m",
    "iq3_xxs", "iq3_s", "iq3_m", "iq4_nl", "iq4_xs",
)
# Legacy note: the former dynamic_v2 flag and the q4_k_xl/q3_k_xl/q2_k_xl
# ("UD-* Dynamic 2.0") ids were REMOVED (plan 2026-08-31) —
# save_pretrained_gguf cannot produce them (the UD mixes are proprietary
# download-only). UD_INFO_FOOTER (added in T3) explains this in the UI.

# Single home for the GGUF default (T3b). unsloth's own recommended default
# for the "quantized" path is q4_k_m. app.py and panels.py RE-EXPORT this
# as DEFAULT_METHOD -- they must never redefine it (the old duplicated
# "q4_k_xl if present else METHODS[0]" guard degraded to not_quantized).
DEFAULT_GGUF_METHOD = "q4_k_m"

# approx_bpw = approximate bits-per-weight (helps estimate size vs quality).
# 35 official unsloth quant ids (plan 2026-08-31): the 24 ALLOWED_QUANTS in save.py
# dict order, then the 11 IMATRIX_QUANTS in save.py dict order. approx_bpw is a
# positional arg here, but needs_imatrix is ALWAYS passed by keyword -- that flag
# must never again share a positional slot with a removed field (see git history:
# it briefly occupied the old dynamic_v2 slot and broke every app mount).
# 35 official unsloth quant ids (plan 2026-08-31): the 24 ALLOWED_QUANTS in save.py
# dict order, then the 11 IMATRIX_QUANTS in save.py dict order. approx_bpw is a
# positional arg here, but needs_imatrix is ALWAYS passed by keyword -- that flag
# must never again share a positional slot with a removed field (see git history:
# it briefly occupied the old dynamic_v2 slot and broke every app mount).
# 35 official unsloth quant ids (plan 2026-08-31): the 24 ALLOWED_QUANTS in save.py
# dict order, then the 11 IMATRIX_QUANTS in save.py dict order. approx_bpw is a
# positional arg here, but needs_imatrix is ALWAYS passed by keyword -- that flag
# must never again share a positional slot with a removed field (see git history:
# it briefly occupied the old dynamic_v2 slot and broke every app mount).
# 35 official unsloth quant ids (plan 2026-08-31): the 24 ALLOWED_QUANTS in save.py
# dict order, then the 11 IMATRIX_QUANTS in save.py dict order. EVERY field after
# `backend` is passed BY KEYWORD -- a positional slot here is how needs_imatrix once
# silently inherited a removed field's position and crashed every app mount.
METHODS: list[QuantMethod] = [
    QuantMethod("not_quantized", "Not quantized (keep dtype)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=16.0,
                description="Recommended. Fast conversion. Slow inference, big files. Keeps the "
                             "model dtype (bf16/f16)."),
    QuantMethod("fast_quantized", "Fast quantized (-> Q8_0)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=8.5,
                description="Recommended. Fast conversion. OK inference, OK file size. Maps to "
                             "q8_0."),
    QuantMethod("quantized", "Quantized (-> Q4_K_M)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.85,
                description="Recommended. Slow conversion. Fast inference, small files. Maps to "
                             "q4_k_m."),
    QuantMethod("f32", "F32 (32-bit, lossless)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=32.0,
                description="Full 32-bit. Retains 100% accuracy; very slow and memory hungry."),
    QuantMethod("bf16", "BF16 (bfloat16, lossless)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=16.0,
                description="Bfloat16 - fastest conversion, retains 100% accuracy. Slow and "
                             "memory hungry."),
    QuantMethod("f16", "F16 (16-bit, lossless)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=16.0,
                description="Full 16-bit. Largest, lossless. Best as an intermediate before "
                             "manual quant."),
    QuantMethod("q8_0", "Q8_0 (8-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=8.5,
                description="Fast conversion. High resource use, but generally acceptable."),
    QuantMethod("q4_k_m", "Q4_K_M (4-bit, recommended)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.85,
                description="Recommended. Uses Q6_K for half of the attention.wv and "
                             "feed_forward.w2 tensors, else Q4_K."),
    QuantMethod("q5_k_m", "Q5_K_M (5-bit, recommended)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=5.5,
                description="Recommended. Uses Q6_K for half of the attention.wv and "
                             "feed_forward.w2 tensors, else Q5_K."),
    QuantMethod("q2_k", "Q2_K (2-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.35,
                description="Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K "
                             "for the other tensors."),
    QuantMethod("q2_k_l", "Q2_K_L (2-bit large, Unsloth preset)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.4,
                description="Unsloth preset: Q2_K body with Q8_0 output/token embeddings - "
                             "higher quality than plain Q2_K."),
    QuantMethod("q3_k_l", "Q3_K_L (3-bit large)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.0,
                description="Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 "
                             "tensors, else Q3_K."),
    QuantMethod("q3_k_m", "Q3_K_M (3-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.9,
                description="Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 "
                             "tensors, else Q3_K."),
    QuantMethod("q3_k_s", "Q3_K_S (3-bit small)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.5,
                description="Uses Q3_K for all tensors."),
    QuantMethod("q4_0", "Q4_0 (4-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.55,
                description="Original 4-bit quant method."),
    QuantMethod("q4_1", "Q4_1 (4-bit, +bias)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.8,
                description="Higher accuracy than q4_0 but not as high as q5_0; quicker "
                             "inference than q5 models."),
    QuantMethod("q4_k_s", "Q4_K_S (4-bit small)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.5,
                description="Uses Q4_K for all tensors."),
    QuantMethod("q4_k", "Q4_K (alias of Q4_K_M)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.85,
                description="Alias for q4_k_m."),
    QuantMethod("q5_k", "Q5_K (alias of Q5_K_M)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=5.5,
                description="Alias for q5_k_m."),
    QuantMethod("q5_0", "Q5_0 (5-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=5.5,
                description="Higher accuracy, higher resource usage and slower inference."),
    QuantMethod("q5_1", "Q5_1 (5-bit, +bias)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=5.5,
                description="Even higher accuracy, resource usage and slower inference."),
    QuantMethod("q5_k_s", "Q5_K_S (5-bit small)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=5.5,
                description="Uses Q5_K for all tensors."),
    QuantMethod("q6_k", "Q6_K (6-bit)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=6.6,
                description="Uses Q8_K for all tensors. Very good quality, fairly large."),
    QuantMethod("q3_k_xs", "Q3_K_XS (3-bit XS)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.3,
                description="3-bit extra-small quantization."),
    QuantMethod("iq1_s", "IQ1_S (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=1.56,
                description="1.56 bpw. Smallest, lowest quality. Needs an "
                             "imatrix.", needs_imatrix=True),
    QuantMethod("iq1_m", "IQ1_M (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=1.75,
                description="1.75 bpw. Very small. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq2_xxs", "IQ2_XXS (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=2.06,
                description="2.06 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq2_xs", "IQ2_XS (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=2.31,
                description="2.31 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq2_s", "IQ2_S (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=2.5,
                description="2.5 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq2_m", "IQ2_M (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=2.7,
                description="2.7 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq3_xxs", "IQ3_XXS (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.06,
                description="3.06 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq3_s", "IQ3_S (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.44,
                description="3.44 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq3_m", "IQ3_M (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=3.66,
                description="3.66 bpw. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq4_nl", "IQ4_NL (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.5,
                description="4.5 bpw non-linear. Needs an imatrix.", needs_imatrix=True),
    QuantMethod("iq4_xs", "IQ4_XS (imatrix)", Family.GGUF, Backend.UNSLOTH,
                approx_bpw=4.25,
                description="4.25 bpw. Needs an imatrix.", needs_imatrix=True),
    # --- Native backend (plan 2026-09-07): no transformers/unsloth/torch. ----
    # Appended AFTER the official 35 so the unsloth list order stays pinned.
    QuantMethod("native_q8_0", "Q8_0 (native)", Family.GGUF, Backend.NATIVE,
                approx_bpw=8.5,
                description="Q8_0 without transformers/unsloth; any architecture."),
    QuantMethod("native_q4_0", "Q4_0 (native)", Family.GGUF, Backend.NATIVE,
                approx_bpw=4.55,
                description="Q4_0 without transformers/unsloth; any architecture."),
    QuantMethod("native_f16", "F16 (native)", Family.GGUF, Backend.NATIVE,
                approx_bpw=16.0,
                description="Half-precision GGUF; lossless for f16 sources."),
    QuantMethod("native_bf16", "BF16 (native, lossless)", Family.GGUF, Backend.NATIVE,
                approx_bpw=16.0,
                description="BF16 GGUF; lossless for bf16 sources (no f16 overflow risk)."),
    QuantMethod("native_f32", "F32 (native)", Family.GGUF, Backend.NATIVE,
                approx_bpw=32.0,
                description="Full-precision GGUF; no quality loss."),
]

METHODS_BY_ID: dict[str, QuantMethod] = {m.id: m for m in METHODS}

# The 4 native-backend method ids (Backend.NATIVE, plan 2026-09-07). A tuple
# (ordered), consistent with ALLOWED_QUANT_IDS / IMATRIX_QUANT_IDS usage.
NATIVE_QUANT_IDS: tuple[str, ...] = tuple(
    m.id for m in METHODS if m.backend == Backend.NATIVE
)

# Shown under the GGUF method list. The UD-* ("Dynamic 2.0") mixes used to be
# registry entries (q4_k_xl / q3_k_xl / q2_k_xl) but were removed:
# save_pretrained_gguf cannot produce them, because the per-layer qtype recipe
# is proprietary and Unsloth ships the mixes as downloads only. Pointing the
# user at the downloads beats letting the ids silently disappear.
UD_INFO_FOOTER = (
    "Unsloth Dynamic (UD-*) mixes are proprietary download-only - download them "
    "from HuggingFace (unsloth/<model>-GGUF); an open dynamic-mix feature is on "
    "the roadmap."
)


# --------------------------------------------------------------------------- #
# ComfyUI / convert_to_quant registry data
# --------------------------------------------------------------------------- #
def _int8_common_options() -> list[OptionField]:
    """Extra quantization options shared by the unified INT8 format.

    - ``heur`` (a.k.a. ``--skip_inefficient_layers``): copy 2D weights whose
      dimensions are NOT divisible by ``block_size`` *unchanged* instead of
      quantizing them -- this is the safe escape hatch for the
      ``dimensions divisible by block_size`` crash.
    - ``manual_seed``: fixed seed for the simulated bias-correction calibration
      data (the streaming path defaults to a reproducible seed when omitted).
    - ``exclude_layers``: regex of layer-name keys to keep at original precision
      (e.g. ``attn_norm|text_embed`` -- the verified Raon-OpenTTS recipe keeps
      the AdaLN modulation + text-conditioning path in fp32/bf16).
    """
    return [
        OptionField(
            "heur", "Skip inefficient layers (--heur)", "checkbox",
            default=False,
            visible_when="format == 'int8'",
            cli_when_true="--heur",
            help="Copy 2D weights whose dims aren't divisible by block_size unchanged "
                 "instead of quantizing them. Use this to avoid the "
                 "'dimensions divisible by block_size' error.",
        ),
        OptionField(
            "manual_seed", "Manual seed (optional)", "input",
            default="",
            visible_when="format == 'int8'",
            cli_flag="--manual_seed",
            help="Fixed seed for the simulated calibration data used in bias correction. "
                 "Leave empty for the streaming default (reproducible across runs).",
        ),
        OptionField(
            "exclude_layers", "Exclude layers (regex, optional)", "input",
            default="",
            visible_when="format == 'int8'",
            cli_flag="--exclude_layers",
            help="Regex of tensor names kept at original precision. Example for "
                 "Raon-OpenTTS int8-convrot: attn_norm|text_embed",
        ),
        OptionField(
            "output_dtype", "Output dtype for unquantized weights", "select",
            default="bfloat16", choices=[("bfloat16", "bfloat16"), ("float16", "float16")],
            visible_when="format == 'int8'",
            cli_flag="--output_dtype",
            help="Downcasts fp32 passthrough weights (bfloat16 default == the "
                 "upstream quantize_raon_int8_convrot.py --downcast-fp32 behavior). "
                 "Legacy whole-file path only.",
        ),
    ]


COMFY_FORMATS: list[ComfyFormat] = [
    ComfyFormat("fp8_e4m3", "FP8 E4M3 (default)", base_flags=["--comfy_quant"]),
    # Unified INT8 entry (v0.4.0): replaces the former int8_block / int8_tensor /
    # int8_convrot trio, which exposed dead combinations (e.g. "tensor scaling +
    # convrot" did nothing). Scaling is now a first-class option and the UI shows
    # only valid combinations: block_size appears ONLY for block scaling; the
    # ConvRot toggle + group size appear ONLY for row scaling (ConvRot
    # mathematically requires row scales). base_flags carry only the
    # unconditional flag -- --scaling_mode is emitted from the OptionField below.
    ComfyFormat(
        "int8", "INT8 (W8A8)",
        base_flags=["--int8"],
        extra_options=[
            OptionField(
                "scaling_mode", "Scaling mode", "select",
                default="block",
                choices=[("block", "block"), ("tensor", "tensor"), ("row", "row")],
                cli_flag="--scaling_mode",
                help="INT8 scale granularity: block (finest, needs dims divisible "
                     "by block_size), tensor (one scale per tensor), row (per "
                     "output row; required for ConvRot).",
            ),
            OptionField(
                "block_size", "Block size", "select",
                default="128", choices=[("64", "64"), ("128", "128"), ("256", "256")],
                visible_when="format == 'int8' and scaling_mode == 'block'",
                cli_flag="--block_size",
                help="Block dimension for blockwise INT8 scaling. Each tensor dimension "
                     "must be divisible by block_size. If you hit 'dimensions divisible by "
                     "block_size', lower this to 64 (or enable 'Skip inefficient layers').",
            ),
            OptionField(
                "convrot", "Apply ConvRot rotation (W8A8)", "checkbox",
                default=False,
                visible_when="format == 'int8' and scaling_mode == 'row'",
                cli_when_true="--convrot",
                help="Rotate weights before row-scaled INT8 quantization (needs triton; "
                     "runs the learned-rounding path -- lower num_iter to speed it up).",
            ),
            OptionField(
                "convrot_group_size", "ConvRot group size", "select",
                default="256", choices=[("64", "64"), ("256", "256"), ("1024", "1024")],
                visible_when=("format == 'int8' and scaling_mode == 'row' "
                              "and convrot"),
                cli_flag="--convrot_group_size",
            ),
            *_int8_common_options(),
        ],
        needs=[],  # triton is needed only when the convrot toggle is ON (dynamic check)
    ),
    ComfyFormat(
        "nvfp4", "NVFP4 (Blackwell)", base_flags=["--nvfp4"], needs=["blackwell"],
        requires_py="3.12", requires_torch="2.10", requires_cuda="13.0",
    ),
    ComfyFormat(
        "mxfp8", "MXFP8 (Blackwell)", base_flags=["--mxfp8"], needs=["blackwell"],
        requires_py="3.12", requires_torch="2.10", requires_cuda="13.0",
    ),
    # P3: ConvRot W4A4 -- requires comfy-kitchen (a ComfyUI-python interpreter).
    ComfyFormat(
        "w4a4_convrot", "ConvRot W4A4", backend=Backend.COMFY_KITCHEN,
        base_flags=["--w4a4"], needs=["comfy_kitchen"], quant_format="convrot_w4a4",
    ),
    # P4: Asymmetric W4A8 -- requires comfy-kitchen.
    ComfyFormat(
        "w4a8_asym", "Asymmetric W4A8", backend=Backend.COMFY_KITCHEN,
        base_flags=["--w4a8"], needs=["comfy_kitchen"], quant_format="asym_w4a8_int8",
    ),
    # Combine (STEP 1.2 rename of the former on-the-fly passthrough): merges a
    # sharded input into ONE .safetensors without quantization (output mode is
    # ignored -- merging is the format's whole purpose); a single-file input is
    # copied byte-identical. No .comfy_quant baked.
    ComfyFormat(
        "combine", "Combine (merge shards, no quant)", backend=Backend.CTQ,
        base_flags=["--combine"],
    ),
    # BF16 / FP16 cast-only formats (STEP 3.1): pure dtype conversion via the
    # worker's --cast_dtype short-circuit (checked BEFORE any convert_to_quant
    # import, same pattern as --combine). Floating tensors are RTNE-cast;
    # integer/bool tensors pass through unchanged. No .comfy_quant baked.
    ComfyFormat("bf16", "BF16 (bfloat16, cast only)",
                base_flags=["--cast_dtype", "bfloat16"]),
    ComfyFormat("fp16", "FP16 (half, cast only)",
                base_flags=["--cast_dtype", "float16"]),
]

COMFY_PRESETS: list[Preset] = [
    # flux2 keeps the former int8_convrot behavior: row scaling + rotation.
    Preset("flux2", "FLUX.2", "--flux2", "int8",
           recommended_options={"scaling_mode": "row", "convrot": True}),
    Preset("wan", "WAN", "--wan", "int8",
           recommended_options={"scaling_mode": "block"}),
    Preset("t5xxl", "T5-XXL", "--t5xxl", "fp8_e4m3"),
    Preset("hunyuan", "Hunyuan", "--hunyuan", "int8",
           recommended_options={"scaling_mode": "block"}),
    Preset("zimage", "Z-Image", "--zimage", "fp8_e4m3"),
]


_COMFY_FORMATS_BY_ID: dict[str, ComfyFormat] = {f.id: f for f in COMFY_FORMATS}
_COMFY_PRESETS_BY_ID: dict[str, Preset] = {p.id: p for p in COMFY_PRESETS}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def methods_for_family(family: Family) -> list[QuantMethod]:
    """Return all ``QuantMethod`` entries belonging to ``family``."""
    return [m for m in METHODS if m.family == family]


def get_method(family: Family, mid: str) -> QuantMethod | None:
    """Return the ``QuantMethod`` for ``family`` + id, or ``None``."""
    for m in METHODS:
        if m.family == family and m.id == mid:
            return m
    return None


def comfy_format(fid: str) -> ComfyFormat:
    """Return the ``ComfyFormat`` for ``fid`` (raises ``KeyError`` if unknown)."""
    return _COMFY_FORMATS_BY_ID[fid]


def comfy_preset(pid: str) -> Preset | None:
    """Return the ``Preset`` for ``pid``, or ``None`` if unknown."""
    return _COMFY_PRESETS_BY_ID.get(pid)


def format_options() -> list[tuple[str, str]]:
    """``(label, id)`` tuples for the Format ``Select``."""
    return [(f.label, f.id) for f in COMFY_FORMATS]


def preset_options() -> list[tuple[str, str]]:
    """``(label, id)`` tuples for the Preset ``Select``."""
    return [(p.label, p.id) for p in COMFY_PRESETS]


def eval_visible_when(predicate: str | None, context: dict[str, Any]) -> bool:
    """Safely evaluate a small visibility predicate over ``context``.

    Supports ``key == value``, ``key != value``, ``key in (a, b, c)`` and
    ``key not in (...)`` combined with ``and`` / ``or``. Anything unexpected is
    treated as *visible=False* (so a mis-configured predicate hides the widget
    rather than crashing the UI). All names resolve from ``context`` only.
    """
    if not predicate:
        return True

    try:
        tree = ast.parse(predicate, mode="eval")
    except SyntaxError:
        return False

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.BoolOp):
            results = [visit(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(results)
            return any(results)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not visit(node.operand)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            # strict=True: ast.Compare always carries one comparator per op.
            for op, comp in zip(node.ops, node.comparators, strict=True):
                right = visit(comp)
                if isinstance(op, ast.Eq):
                    if left != right:
                        return False
                elif isinstance(op, ast.NotEq):
                    if left == right:
                        return False
                elif isinstance(op, ast.In):
                    if left not in right:
                        return False
                elif isinstance(op, ast.NotIn):
                    if left in right:
                        return False
                else:
                    return False
            return True
        if isinstance(node, ast.Name):
            return context.get(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return [visit(e) for e in node.elts]
        return False

    try:
        return bool(visit(tree))
    except Exception:  # boundary: schema-declared expressions are untrusted input;
        return False   # any parse/eval failure means "option not visible" by design


# --------------------------------------------------------------------------- #
# Display helper (used by the GGUF panel + list view)
# --------------------------------------------------------------------------- #
def list_line(method: QuantMethod) -> str:
    # "[IMATRIX] " replaces the old "[DYNAMIC 2.0] " badge: it marks the 11
    # official IQ* ids, which unsloth refuses to quantize without imatrix_file=.
    badge = "[IMATRIX] " if method.needs_imatrix else ""
    bpw = f"{method.approx_bpw:>4}bpw " if method.approx_bpw else "     "
    return f"{method.id:<12} {bpw} {badge}{method.description}"


# --------------------------------------------------------------------------- #
# Shared input classification (ctq COMFY family)
# --------------------------------------------------------------------------- #
# HuggingFace sharded-model index filename.
INDEX_NAME = "model.safetensors.index.json"

# A bare HuggingFace shard marker (e.g. ``model-00001-of-00003``) carries no
# semantic name of its own; when one is used as a single-file input we fall back
# to the parent folder name for auto-naming so the output is meaningful.
_SHARD_NAME_RE = re.compile(r"^model-\d+-of-\d+$")


def classify_input(path: str) -> tuple[str | None, str | None]:
    """Classify a ctq input path and return ``(kind, base_name)``.

    ``kind`` is one of ``"single_file"`` (a ``.safetensors`` file, or a folder
    holding exactly one ``.safetensors``), ``"sharded_folder"`` (a folder with
    ``model.safetensors.index.json``), or ``None`` (unusable). ``base_name`` is
    the meaningful name used for auto-naming outputs:
    - a single ``.safetensors`` file uses its own stem, **unless** that stem is
      just a HuggingFace shard marker (``model-00001-of-00003``) -- then the
      PARENT folder name is used instead;
    - folders (single-shard folder or sharded folder) use the folder name.
    """
    if not path:
        return (None, None)
    if os.path.isfile(path):
        if not path.endswith(".safetensors"):
            return (None, None)
        stem = os.path.splitext(os.path.basename(path))[0]
        if _SHARD_NAME_RE.match(stem):
            # Shard marker -> use the parent folder name (more meaningful).
            base = os.path.basename(os.path.dirname(os.path.abspath(path)))
        else:
            base = stem
        return ("single_file", base)
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, INDEX_NAME)):
            return ("sharded_folder", os.path.basename(path.rstrip(os.sep)))
        sts = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if len(sts) == 1:
            return ("single_file", os.path.basename(path.rstrip(os.sep)))
        return (None, None)
    return (None, None)


def is_sharded_folder(path: str) -> bool:
    """True iff ``path`` is a directory containing ``model.safetensors.index.json``."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, INDEX_NAME))
