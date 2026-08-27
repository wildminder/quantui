"""Model audit: header-only tensor inventory + exclusion advisor.

Pure stdlib (plus the stdlib-only :mod:`quantui.comfy_quant_schema`) -- no
torch / safetensors / numpy -- so the module runs in the headless test env and
in any worker env, and scans a 7 GiB checkpoint in milliseconds because only
the safetensors *header* (and the tiny ``.comfy_quant`` JSON blobs) are read.

The tool automates **stage 1** of the quantization-exclusion funnel: a
mechanical, deterministic scan that classifies every tensor, aggregates
per-module statistics, detects already-quantized layers, and proposes a
starting ``exclude_layers`` regex. Stage 2 (hot loops / skinny GEMMs) requires
reading the pipeline code and is deliberately out of scope -- the tool
suggests, the human decides.

Classification ground truth (frozen 2026-08-27, validated against the real
checkpoints): ``Breeze-TTS-2-bf16.safetensors`` (1115 tensors) classifies to
``linear 558 / vector 424 / bias 60 / other 67 / embedding 4 / head 1 /
linear_review 1`` (the review tensor is ``text_encoder_proj.weight``).
Per-module linears: ``backbone_model 196 / text_encoder 182 / depth_decoder
84 / codec_model 96``. The official ``int8-hybrid`` build quantized exactly
378 tensors (``backbone_model`` + ``text_encoder``), so the classifier has
zero false negatives against the official build: every tensor a human chose to
quantize is in ``linear``, and every tensor beyond that is a legitimate
stage-2 decision, not a classification error. The rule table below is therefore
evidence-based; changing it is a deliberate, reviewed act (the frozen-table
tests in ``tests/test_model_audit_classify.py`` are tripwires).
"""

from __future__ import annotations

import math
import re

# --------------------------------------------------------------------------- #
# Category vocabulary -- every tensor gets exactly one of these.
# --------------------------------------------------------------------------- #
CATEGORIES: tuple[str, ...] = (
    "linear",
    "embedding",
    "head",
    "linear_review",
    "vector",
    "bias",
    "quant_meta",
    "quant_scale",
    "other",
)

# 2D ``.weight`` segments that are plain matmul weights -- the quantizable
# population (validated set from the plan §0.2 table, rule 7).
LINEAR_SEGMENTS: frozenset[str] = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "in_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "dense",
        "linear",
        "mlp",
    }
)

# safetensors dtype -> element size in bytes (complete vocabulary of the spec).
DTYPE_ITEMSIZE: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "BF16": 2,
    "F16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

# --------------------------------------------------------------------------- #
# Name regexes -- compiled once at module level (plan §0.2 rules 3/5/6).
# --------------------------------------------------------------------------- #
# Kitchen companion tensors that accompany a quantized layer (rule 3).
_QUANT_SCALE_RE = re.compile(
    r"\.(?:weight_scale|weight_s_rel|weight_s_channel|weight_codebook|weight_correction)$"
)
# Embedding lookup tables: substring match on the segment before ".weight"
# (rule 5). Substring semantics are intentional and validated: Breeze's
# ``inputs_embeds_projector`` is an embedding-side adapter and must be kept.
_EMBEDDING_RE = re.compile(r"embed|wte|wpe|tokens")
# Output heads (rule 6): anchored so e.g. ``overhead`` does NOT match.
_HEAD_RE = re.compile(r"^(?:lm_head|[a-z0-9_]*_head)$")


def _weight_segment(name: str) -> str:
    """Return the dot-segment immediately before a trailing ``.weight``.

    ``"a.b.q_proj.weight" -> "q_proj"``; ``"lm_head.weight" -> "lm_head"``.
    Callers must have verified ``name.endswith(".weight")`` first.
    """
    stem = name[: -len(".weight")]
    return stem.rsplit(".", 1)[-1]


def classify_tensor(name: str, shape: list[int]) -> str:
    """Classify one tensor into exactly one of :data:`CATEGORIES`.

    Rules are evaluated in the frozen plan §0.2 order (first match wins) on
    ``(name, ndim(shape))``:

    1. ``.bias`` suffix                      -> ``bias`` (never quantized by ctq)
    2. ``.comfy_quant`` suffix               -> ``quant_meta`` (descriptor blob)
    3. kitchen companion suffix              -> ``quant_scale``
    4. ndim == 1                             -> ``vector`` (norms etc. -- keep)
    5. ndim == 2 ``.weight`` + embed/wte/wpe/tokens segment -> ``embedding``
    6. ndim == 2 ``.weight`` + ``lm_head``/``*_head`` segment -> ``head``
    7. ndim == 2 ``.weight`` + segment in :data:`LINEAR_SEGMENTS` -> ``linear``
    8. ndim == 2 ``.weight`` (anything else) -> ``linear_review`` (keep, review)
    9. everything else (3D+ convs, scalars)  -> ``other`` (keep, listed)

    Pure function, no I/O.
    """
    if name.endswith(".bias"):
        return "bias"
    if name.endswith(".comfy_quant"):
        return "quant_meta"
    if _QUANT_SCALE_RE.search(name):
        return "quant_scale"
    ndim = len(shape)
    if ndim == 1:
        return "vector"
    if ndim == 2 and name.endswith(".weight"):
        segment = _weight_segment(name)
        if _EMBEDDING_RE.search(segment):
            return "embedding"
        if _HEAD_RE.match(segment):
            return "head"
        if segment in LINEAR_SEGMENTS:
            return "linear"
        return "linear_review"
    return "other"


def tensor_bytes(dtype: str, shape: list[int]) -> int:
    """Return the payload size in bytes: ``itemsize * prod(shape)``.

    Unknown dtypes return ``0`` (callers count such tensors separately); a
    scalar shape ``[]`` counts one element (``math.prod([]) == 1``).
    """
    itemsize = DTYPE_ITEMSIZE.get(dtype)
    if itemsize is None:
        return 0
    return itemsize * math.prod(shape)
