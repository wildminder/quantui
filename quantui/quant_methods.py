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
from enum import Enum
from typing import Any, Literal


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Family(str, Enum):
    """Top-level quantization family. This is the single seam the TUI branches on."""

    GGUF = "gguf"  # Unsloth GGUF (causal-LM LLMs)
    COMFY = "comfy"  # ComfyUI / convert_to_quant (diffusion)


class Backend(str, Enum):
    """Concrete worker backend behind a family."""

    UNSLOTH = "unsloth"
    CTQ = "convert_to_quant"
    # comfy-kitchen + comfy.quant_ops (a ComfyUI-python interpreter). Required for
    # W4A4 (convrot_w4a4) and W4A8 (asym_w4a8_int8), which convert_to_quant cannot emit.
    COMFY_KITCHEN = "comfy_kitchen"


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
    dynamic_v2: bool = False
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


# --------------------------------------------------------------------------- #
# GGUF registry (original entries, now tagged with family/backend/options)
# --------------------------------------------------------------------------- #
# approx_bpw = approximate bits-per-weight (helps estimate size vs quality).
METHODS: list[QuantMethod] = [
    QuantMethod("f16", "F16 (16-bit, lossless)", Family.GGUF, Backend.UNSLOTH, False, 16.0,
                "Full 16-bit. Largest, lossless. Best as an intermediate before manual quant."),
    QuantMethod("q8_0", "Q8_0 (8-bit)", Family.GGUF, Backend.UNSLOTH, False, 8.6,
                "8-bit. Near-lossless, high memory use. Fast conversion."),
    QuantMethod("q6_k", "Q6_K (6-bit)", Family.GGUF, Backend.UNSLOTH, False, 6.6,
                "6-bit K-quant. Very good quality, fairly large."),
    QuantMethod("q5_k_m", "Q5_K_M (5-bit, recommended)", Family.GGUF, Backend.UNSLOTH, False, 5.5,
                "Recommended 5-bit. Near-lossless quality with good size."),
    QuantMethod("q5_k_s", "Q5_K_S (5-bit small)", Family.GGUF, Backend.UNSLOTH, False, 5.5,
                "5-bit small. Uses Q5_K for all tensors."),
    QuantMethod("q5_0", "Q5_0", Family.GGUF, Backend.UNSLOTH, False, 5.5, "Higher accuracy, slower inference."),
    QuantMethod("q5_1", "Q5_1 (Dynamic 2.0 format)", Family.GGUF, Backend.UNSLOTH, True, 5.5,
                "New Dynamic 2.0 efficiency format (ARM / Apple Silicon)."),
    QuantMethod("q4_k_m", "Q4_K_M (4-bit, recommended)", Family.GGUF, Backend.UNSLOTH, False, 4.85,
                "Recommended 4-bit. Good balance of size and quality."),
    QuantMethod("q4_k_s", "Q4_K_S (4-bit small)", Family.GGUF, Backend.UNSLOTH, False, 4.5,
                "4-bit small. Q4_K for all tensors."),
    QuantMethod("q4_0", "Q4_0", Family.GGUF, Backend.UNSLOTH, False, 4.55, "Original 4-bit method."),
    QuantMethod("q4_1", "Q4_1 (Dynamic 2.0 format)", Family.GGUF, Backend.UNSLOTH, True, 4.8,
                "New Dynamic 2.0 format. Higher accuracy than Q4_0."),
    QuantMethod("q4_nl", "Q4_NL (Dynamic 2.0 format)", Family.GGUF, Backend.UNSLOTH, True, 4.5,
                "New Dynamic 2.0 efficiency format for Apple Silicon / ARM."),
    QuantMethod("q3_k_m", "Q3_K_M (3-bit)", Family.GGUF, Backend.UNSLOTH, False, 3.9,
                "3-bit. Q4_K for key tensors."),
    QuantMethod("q3_k_l", "Q3_K_L (3-bit large)", Family.GGUF, Backend.UNSLOTH, False, 4.0, "3-bit large."),
    QuantMethod("q3_k_s", "Q3_K_S (3-bit small)", Family.GGUF, Backend.UNSLOTH, False, 3.5,
                "3-bit small. Q3_K for all tensors."),
    QuantMethod("q3_k_xs", "Q3_K_XS (3-bit XS)", Family.GGUF, Backend.UNSLOTH, False, 3.3, "3-bit extra-small."),
    QuantMethod("q2_k", "Q2_K (2-bit)", Family.GGUF, Backend.UNSLOTH, False, 2.96,
                "2-bit. Q4_K for key tensors."),
    QuantMethod("iq4_nl", "IQ4_NL (imatrix)", Family.GGUF, Backend.UNSLOTH, False, 4.5,
                "Importance-matrix 4-bit (needs an imatrix file)."),
    QuantMethod("iq3_xxs", "IQ3_XXS (imatrix)", Family.GGUF, Backend.UNSLOTH, False, 3.06, "Importance quant, very small."),
    QuantMethod("iq2_xxs", "IQ2_XXS (imatrix)", Family.GGUF, Backend.UNSLOTH, False, 2.06, "Importance quant, tiny."),
    QuantMethod("iq2_xs", "IQ2_XS (imatrix)", Family.GGUF, Backend.UNSLOTH, False, 2.31, "Importance quant."),
    # Dynamic 2.0 per-layer selective variants (the headline feature)
    QuantMethod("q4_k_xl", "UD-Q4_K_XL (Dynamic 2.0)", Family.GGUF, Backend.UNSLOTH, True, 4.5,
                "Dynamic 2.0 per-layer selective. Best quality at ~Q4 size. Output: UD-Q4_K_XL."),
    QuantMethod("q3_k_xl", "UD-Q3_K_XL (Dynamic 2.0)", Family.GGUF, Backend.UNSLOTH, True, 3.5,
                "Dynamic 2.0 per-layer selective, smaller. Output: UD-Q3_K_XL."),
    QuantMethod("q2_k_xl", "UD-Q2_K_XL (Dynamic 2.0)", Family.GGUF, Backend.UNSLOTH, True, 2.7,
                "Dynamic 2.0 per-layer selective, smallest. Output: UD-Q2_K_XL."),
]

METHODS_BY_ID: dict[str, QuantMethod] = {m.id: m for m in METHODS}


# --------------------------------------------------------------------------- #
# ComfyUI / convert_to_quant registry data
# --------------------------------------------------------------------------- #
def _int8_common_options() -> list[OptionField]:
    """Extra quantization options shared by EVERY INT8 format.

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
            visible_when="format in ('int8_block', 'int8_tensor', 'int8_convrot')",
            cli_when_true="--heur",
            help="Copy 2D weights whose dims aren't divisible by block_size unchanged "
                 "instead of quantizing them. Use this to avoid the "
                 "'dimensions divisible by block_size' error.",
        ),
        OptionField(
            "manual_seed", "Manual seed (optional)", "input",
            default="",
            visible_when="format in ('int8_block', 'int8_tensor', 'int8_convrot')",
            cli_flag="--manual_seed",
            help="Fixed seed for the simulated calibration data used in bias correction. "
                 "Leave empty for the streaming default (reproducible across runs).",
        ),
        OptionField(
            "exclude_layers", "Exclude layers (regex, optional)", "input",
            default="",
            visible_when="format in ('int8_block', 'int8_tensor', 'int8_convrot')",
            cli_flag="--exclude_layers",
            help="Regex of tensor names kept at original precision. Example for "
                 "Raon-OpenTTS int8-convrot: attn_norm|text_embed",
        ),
        OptionField(
            "output_dtype", "Output dtype for unquantized weights", "select",
            default="bfloat16", choices=[("bfloat16", "bfloat16"), ("float16", "float16")],
            visible_when="format in ('int8_block', 'int8_tensor', 'int8_convrot')",
            cli_flag="--output_dtype",
            help="Downcasts fp32 passthrough weights (bfloat16 default == the "
                 "upstream quantize_raon_int8_convrot.py --downcast-fp32 behavior). "
                 "Legacy whole-file path only.",
        ),
    ]


COMFY_FORMATS: list[ComfyFormat] = [
    ComfyFormat("fp8_e4m3", "FP8 E4M3 (default)", base_flags=["--comfy_quant"]),
    # P1.1: true blockwise INT8 (previously emitted only --int8 -> library-default
    # tensorwise). Now explicitly requests block scaling with a configurable block size.
    ComfyFormat(
        "int8_block", "INT8 blockwise",
        base_flags=["--int8", "--scaling_mode", "block"],
        extra_options=[
            OptionField(
                "block_size", "Block size", "select",
                default="128", choices=[("64", "64"), ("128", "128"), ("256", "256")],
                visible_when="format == 'int8_block'",
                cli_flag="--block_size",
                help="Block dimension for blockwise INT8 scaling. Each tensor dimension "
                     "must be divisible by block_size. If you hit 'dimensions divisible by "
                     "block_size', lower this to 64 (or enable 'Skip inefficient layers').",
            ),
            *_int8_common_options(),
        ],
    ),
    ComfyFormat(
        "int8_tensor", "INT8 tensor", base_flags=["--int8", "--scaling_mode", "tensor"],
        extra_options=[*_int8_common_options()],
    ),
    # P1.2: W8A8 ConvRot output. Maps 1:1 onto the toolkit's "int8_convrot" mode
    # (format=int8_tensorwise + convrot). The former "int8_row" duplicate entry was
    # removed -- both emitted identical flags/artifacts and only confused users.
    ComfyFormat(
        "int8_convrot", "INT8 ConvRot (W8A8)",
        base_flags=["--int8", "--scaling_mode", "row", "--convrot"],
        extra_options=[
            OptionField(
                "convrot_group_size", "ConvRot group size", "select",
                default="256", choices=[("64", "64"), ("256", "256"), ("1024", "1024")],
                visible_when="format == 'int8_convrot'",
                cli_flag="--convrot_group_size",
            ),
            *_int8_common_options(),
        ],
        needs=["triton"],
        quant_format="int8_tensorwise",
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
    # P5: On-the-fly passthrough -- copies the input unchanged so ComfyUI's
    # on_the_fly_quantization loader can quantize at load time. No .comfy_quant baked.
    ComfyFormat(
        "onthefly", "On-the-fly passthrough", backend=Backend.CTQ,
        base_flags=["--passthrough"],
    ),
]

COMFY_PRESETS: list[Preset] = [
    Preset("flux2", "FLUX.2", "--flux2", "int8_convrot"),
    Preset("wan", "WAN", "--wan", "int8_block"),
    Preset("t5xxl", "T5-XXL", "--t5xxl", "fp8_e4m3"),
    Preset("hunyuan", "Hunyuan", "--hunyuan", "int8_block"),
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
            for op, comp in zip(node.ops, node.comparators):
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
    badge = "[DYNAMIC 2.0] " if method.dynamic_v2 else ""
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
