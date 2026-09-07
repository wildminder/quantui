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

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass

from quantui import comfy_quant_schema

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
        "linear1",
        "linear2",
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


# --------------------------------------------------------------------------- #
# Module grouping + aggregation (plan STEP 1.2).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TensorInfo:
    """One classified tensor from a safetensors header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    category: str
    module: str
    nbytes: int


def module_of(name: str) -> str:
    """Return the top-level module: the prefix before the first ``"."``.

    Names without a dot (and the empty name) belong to ``"(root)"``.
    """
    if "." not in name:
        return "(root)"
    return name.split(".", 1)[0]


def collect_tensors(header: dict) -> list[TensorInfo]:
    """Classify every tensor in a parsed safetensors header.

    Skips the ``__metadata__`` pseudo-entry and returns the infos in
    deterministic (name-sorted) order regardless of header insertion order.
    """
    infos: list[TensorInfo] = []
    for name in sorted(header):
        if name == "__metadata__":
            continue
        spec = header[name]
        dtype = spec.get("dtype", "")
        shape = tuple(spec.get("shape", []))
        infos.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                category=classify_tensor(name, list(shape)),
                module=module_of(name),
                nbytes=tensor_bytes(dtype, list(shape)),
            )
        )
    return infos


@dataclass
class ModuleSummary:
    """Aggregated statistics for one top-level module."""

    module: str
    tensors: int
    params: int
    nbytes: int
    category_counts: dict[str, int]


def summarize_modules(infos: list[TensorInfo]) -> list[ModuleSummary]:
    """Aggregate :class:`TensorInfo` rows per top-level module.

    Returns summaries sorted by byte total descending, then module name
    ascending (deterministic tie-break).
    """
    per_module: dict[str, ModuleSummary] = {}
    for info in infos:
        summary = per_module.get(info.module)
        if summary is None:
            summary = ModuleSummary(
                module=info.module, tensors=0, params=0, nbytes=0, category_counts={}
            )
            per_module[info.module] = summary
        summary.tensors += 1
        summary.params += math.prod(info.shape)
        summary.nbytes += info.nbytes
        summary.category_counts[info.category] = (
            summary.category_counts.get(info.category, 0) + 1
        )
    return sorted(per_module.values(), key=lambda s: (-s.nbytes, s.module))


# --------------------------------------------------------------------------- #
# File-level audit engine (plan STEP 2.1).
# --------------------------------------------------------------------------- #
class AuditError(ValueError):
    """Raised for malformed / unreadable safetensors input (message names the path)."""


@dataclass
class AuditReport:
    """Complete, deterministic inventory of one safetensors file."""

    path: str
    metadata: dict  # __metadata__ verbatim ({} if absent)
    tensors: list[TensorInfo]
    category_counts: dict[str, int]
    category_bytes: dict[str, int]
    modules: list[ModuleSummary]
    quantized_layers: list[tuple[str, dict]]  # (prefix, comfy_quant config)
    quant_format_histogram: dict[str, int]
    total_bytes: int


def audit_file(path: str) -> AuditReport:
    """Scan a ``.safetensors`` file header-only and build the full inventory.

    Reuses :mod:`quantui.comfy_quant_schema` for all on-disk parsing (no
    duplicate parser): ``read_safetensors_header`` for the header and
    ``read_comfy_quant_configs`` for the ``.comfy_quant`` JSON blobs. Only the
    header and the tiny descriptor blobs are read -- never tensor payloads --
    so a 7 GiB checkpoint is scanned in milliseconds without loading it.

    Raises :class:`AuditError` (message contains ``path``) for missing files,
    non-safetensors bytes, truncated headers, or malformed ``.comfy_quant``
    JSON.
    """
    try:
        header, _data_start = comfy_quant_schema.read_safetensors_header(path)
    except OSError as exc:
        raise AuditError(f"{path}: cannot read file: {exc}") from exc
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError, MemoryError, OverflowError) as exc:
        # MemoryError/OverflowError: garbage first 8 bytes decode as an absurd
        # header length before the truncation check can fire.
        raise AuditError(f"{path}: not a valid safetensors file: {exc}") from exc

    tensors = collect_tensors(header)

    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    total_bytes = 0
    for info in tensors:
        category_counts[info.category] = category_counts.get(info.category, 0) + 1
        category_bytes[info.category] = category_bytes.get(info.category, 0) + info.nbytes
        total_bytes += info.nbytes

    try:
        quantized_layers = comfy_quant_schema.read_comfy_quant_configs(path)
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditError(f"{path}: malformed .comfy_quant descriptor: {exc}") from exc

    histogram: dict[str, int] = {}
    for _prefix, config in quantized_layers:
        fmt = str(config.get("format", "<missing>"))
        histogram[fmt] = histogram.get(fmt, 0) + 1

    return AuditReport(
        path=path,
        metadata=dict(header.get("__metadata__") or {}),
        tensors=tensors,
        category_counts=category_counts,
        category_bytes=category_bytes,
        modules=summarize_modules(tensors),
        quantized_layers=quantized_layers,
        quant_format_histogram=histogram,
        total_bytes=total_bytes,
    )


# --------------------------------------------------------------------------- #
# Sharded model folder audit (HuggingFace model.safetensors.index.json).
# --------------------------------------------------------------------------- #
SHARDED_INDEX_NAME = "model.safetensors.index.json"


def audit_sharded_folder(path: str) -> AuditReport:
    """Scan a HuggingFace sharded model folder header-only and merge the shards.

    A sharded model is a folder containing ``model.safetensors.index.json``
    (the index: ``{"weight_map": {tensor_name: shard_filename, ...}}`` plus an
    optional ``"metadata"`` object) and one or more shard ``.safetensors``
    files. Every shard is scanned exactly like :func:`audit_file` (header +
    ``.comfy_quant`` blobs only -- never tensor payloads) and the results are
    merged into a single :class:`AuditReport` whose ``path`` is the folder:

    * tensors are merged in sorted-shard order, deduplicated by name (if a
      tensor appears in multiple shards, the first occurrence wins);
    * ``quantized_layers`` are concatenated in sorted-shard order;
    * ``metadata`` comes from the index's ``"metadata"`` key when present,
      else from the first shard (sorted order) carrying ``__metadata__``.

    Raises :class:`AuditError` (message names the offending path) for a
    missing index file, malformed index JSON, a shard file referenced by the
    index but absent on disk, or a malformed safetensors header /
    ``.comfy_quant`` descriptor in any shard.
    """
    index_path = os.path.join(path, SHARDED_INDEX_NAME)
    if not os.path.isfile(index_path):
        raise AuditError(
            f"{path}: missing {SHARDED_INDEX_NAME} (not a sharded model folder)"
        )
    try:
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except OSError as exc:
        raise AuditError(f"{index_path}: cannot read index file: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditError(f"{index_path}: malformed index JSON: {exc}") from exc
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise AuditError(f"{index_path}: malformed index (missing 'weight_map' object)")

    shards = sorted({str(shard) for shard in index["weight_map"].values()})

    merged_tensors: list[TensorInfo] = []
    seen_names: set[str] = set()
    quantized_layers: list[tuple[str, dict]] = []
    shard_metadata: dict = {}
    for shard in shards:
        shard_path = os.path.join(path, shard)
        if not os.path.isfile(shard_path):
            raise AuditError(f"{path}: index references missing shard file: {shard}")
        try:
            header, _data_start = comfy_quant_schema.read_safetensors_header(shard_path)
        except OSError as exc:
            raise AuditError(f"{shard_path}: cannot read file: {exc}") from exc
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError, MemoryError, OverflowError) as exc:
            raise AuditError(f"{shard_path}: not a valid safetensors file: {exc}") from exc
        for info in collect_tensors(header):
            if info.name not in seen_names:
                seen_names.add(info.name)
                merged_tensors.append(info)
        if not shard_metadata:
            meta = header.get("__metadata__")
            if meta:
                shard_metadata = dict(meta)
        try:
            quantized_layers.extend(comfy_quant_schema.read_comfy_quant_configs(shard_path))
        except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuditError(f"{shard_path}: malformed .comfy_quant descriptor: {exc}") from exc

    if "metadata" in index and isinstance(index["metadata"], dict):
        metadata = dict(index["metadata"])
    else:
        metadata = shard_metadata

    category_counts: dict[str, int] = {}
    category_bytes: dict[str, int] = {}
    total_bytes = 0
    for info in merged_tensors:
        category_counts[info.category] = category_counts.get(info.category, 0) + 1
        category_bytes[info.category] = category_bytes.get(info.category, 0) + info.nbytes
        total_bytes += info.nbytes

    histogram: dict[str, int] = {}
    for _prefix, config in quantized_layers:
        fmt = str(config.get("format", "<missing>"))
        histogram[fmt] = histogram.get(fmt, 0) + 1

    return AuditReport(
        path=path,
        metadata=metadata,
        tensors=merged_tensors,
        category_counts=category_counts,
        category_bytes=category_bytes,
        modules=summarize_modules(merged_tensors),
        quantized_layers=quantized_layers,
        quant_format_histogram=histogram,
        total_bytes=total_bytes,
    )


def audit(path: str) -> AuditReport:
    """Audit dispatcher: single ``.safetensors`` file or sharded model folder.

    * existing file -> :func:`audit_file`;
    * existing folder holding ``model.safetensors.index.json`` ->
      :func:`audit_sharded_folder`;
    * anything else -> :class:`AuditError`.
    """
    if os.path.isfile(path):
        return audit_file(path)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, SHARDED_INDEX_NAME)):
        return audit_sharded_folder(path)
    raise AuditError(f"{path}: not a .safetensors file or sharded model folder")


# --------------------------------------------------------------------------- #
# Exclusion suggestion (plan STEP 2.2).
# --------------------------------------------------------------------------- #
# ctq quantizes only 2D ``.weight`` tensors, so the suggested regex only needs
# to cover the 2D keep-set: embeddings, heads, and unknown-role 2D weights.
_KEEP_CATEGORIES: frozenset[str] = frozenset({"embedding", "head", "linear_review"})


@dataclass(frozen=True)
class ExclusionSuggestion:
    """A starting ``exclude_layers`` regex covering the 2D keep-set."""

    regex: str  # single alternation, anchored, re.escape'd, sorted; "" if empty
    names: tuple[str, ...]  # the keep-set the regex covers (sorted)
    rationale: dict[str, int]  # category -> count that motivated the regex
    # NTH-007 (2026-09-04): whitelist-candidate heuristic hints. Each tuple is
    # (last_segment, occurrence_count, shape_repr). The frozen LINEAR_SEGMENTS
    # table stays authoritative; hints only SUGGEST that a repeated unknown
    # 2D-weight naming might belong in the whitelist (a whitelist PR, not an
    # automatic exclusion).
    hints: tuple[tuple[str, int, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.hints:
            object.__setattr__(self, "hints", ())


def _whitelist_hints(report: AuditReport) -> tuple[tuple[str, int, str], ...]:
    """Find repeated ``linear_review`` last-segments worth a whitelist look.

    A candidate is a last dot-segment (e.g. ``linear1`` from
    ``*.layers.N.ffn.linear1.weight``) that appears on >= 8 ``linear_review``
    2D tensors ALL with the same shape -- a strong naming-convention signal
    (the VibeVoice ``linear1``/``linear2`` case). Below threshold or shape
    variance -> no hint. Sorted by (-count, segment) for determinism.
    """
    _MIN_REPEATS = 8
    per_segment: dict[str, dict[tuple, int]] = {}
    for info in report.tensors:
        if info.category != "linear_review" or len(info.shape) != 2:
            continue
        last = _weight_segment(info.name)
        per_segment.setdefault(last, {}).setdefault(tuple(info.shape), 0)
        per_segment[last][tuple(info.shape)] += 1
    hints: list[tuple[str, int, str]] = []
    for segment, shapes in per_segment.items():
        for shape, count in shapes.items():
            if count >= _MIN_REPEATS:
                hints.append((segment, count, str(shape)))
    return tuple(sorted(hints, key=lambda h: (-h[1], h[0])))


def suggest_exclusions(report: AuditReport) -> ExclusionSuggestion:
    """Propose a starting ``exclude_layers`` regex from an :class:`AuditReport`.

    Includes exactly the 2D ``.weight`` tensors in the keep categories
    (``embedding`` / ``head`` / ``linear_review``); vectors, biases, ``other``
    and the kitchen companions are never touched by ctq and need no entry.

    The regex is ONE alternation of ``re.escape``d names, anchored ``^…$`` and
    sorted for determinism, so it reproduces the keep-set under the
    ``re.search`` semantics of ``QuantConfig.excluded`` (tensor_quant.py:113).
    Empty keep-set -> ``regex == ""``. The tool suggests, the human decides:
    stage-2 exclusions (hot loops / skinny GEMMs) are the user's call.
    """
    keep = sorted(
        info.name
        for info in report.tensors
        if info.category in _KEEP_CATEGORIES and len(info.shape) == 2
    )
    rationale: dict[str, int] = {}
    for info in report.tensors:
        if info.category in _KEEP_CATEGORIES and len(info.shape) == 2:
            rationale[info.category] = rationale.get(info.category, 0) + 1
    if not keep:
        return ExclusionSuggestion(regex="", names=(), rationale={},
                                   hints=_whitelist_hints(report))
    regex = "^(" + "|".join(re.escape(name) for name in keep) + ")$"
    return ExclusionSuggestion(regex=regex, names=tuple(keep), rationale=rationale,
                               hints=_whitelist_hints(report))


# --------------------------------------------------------------------------- #
# Report renderers (plan STEP 3.1).
# --------------------------------------------------------------------------- #
def _human_bytes(nbytes: int) -> str:
    """Render a byte count human-readable with 2 decimals (B / KiB / MiB / GiB / TiB)."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def _total_params(report: AuditReport) -> int:
    return sum(math.prod(info.shape) for info in report.tensors)


def _module_quantized_params(report: AuditReport) -> dict[str, int]:
    """Per-module count of params living in already-quantized matrices (NTH-011).

    A quantized layer's ``<base>.weight`` carries ``q_params``; the module is
    the first dot-separated segment of ``<base>`` (``module_of`` semantics).
    Modules with no quantized layers are absent from the result.
    """
    out: dict[str, int] = {}
    for prefix, _config in report.quantized_layers:
        w_shape = None
        for info in report.tensors:
            if info.name == f"{prefix}.weight":
                w_shape = info.shape
                break
        params = math.prod(w_shape) if w_shape else 0
        module = prefix.split(".")[0] if prefix else prefix
        out[module] = out.get(module, 0) + params
    return out


def render_text(report: AuditReport, suggestion: ExclusionSuggestion) -> str:
    """Render the fixed-layout plain-text audit report.

    Layout (asserted by tests): title line with the file name; totals line
    (tensors / params / bytes human-readable); category table; per-module
    table in bytes-desc order; a "Quantized layers" section only when the file
    carries ``.comfy_quant`` descriptors; and the suggestion section with the
    regex on its own line prefixed ``exclude_layers: ``. The closing note
    states the stage-1/stage-2 boundary explicitly.
    """
    lines: list[str] = []
    lines.append(f"Model audit: {os.path.basename(report.path)}")
    lines.append(
        f"Tensors: {len(report.tensors)} | Params: {_total_params(report):,} | "
        f"Bytes: {_human_bytes(report.total_bytes)}"
    )
    lines.append("")
    lines.append("Categories:")
    for category in CATEGORIES:
        count = report.category_counts.get(category, 0)
        if count == 0:
            continue
        lines.append(
            f"  {category:<14} {count:>6} tensors  {_human_bytes(report.category_bytes.get(category, 0)):>12}"
        )
    lines.append("")
    lines.append("Modules:")
    q_by_module = _module_quantized_params(report)
    lin_by_module: dict[str, int] = {}
    for summary in report.modules:
        lin_params = sum(
            math.prod(info.shape)
            for info in report.tensors
            if info.module == summary.module and info.category in ("linear", "linear_review")
        )
        lin_by_module[summary.module] = lin_params
    lines.append(
        f"  {'module':<24} {'tensors':>8} {'params':>14} {'bytes':>12} {'linears':>8} {'q%':>7}"
    )
    for summary in report.modules:
        linears = summary.category_counts.get("linear", 0)
        qp = q_by_module.get(summary.module, 0)
        denom = lin_by_module.get(summary.module, 0)
        ratio = 100.0 * qp / denom if denom else 0.0
        lines.append(
            f"  {summary.module:<24} {summary.tensors:>8} {summary.params:>14,} "
            f"{_human_bytes(summary.nbytes):>12} {linears:>8} {ratio:>6.1f}%"
        )
    if report.quantized_layers:
        lines.append("")
        lines.append(f"Quantized layers: {len(report.quantized_layers)}")
        for fmt in sorted(report.quant_format_histogram):
            lines.append(f"  {fmt}: {report.quant_format_histogram[fmt]}")
    lines.append("")
    lines.append("Suggested exclude_layers (stage-1 keep-set only):")
    if suggestion.regex:
        lines.append(f"exclude_layers: {suggestion.regex}")
        for category in sorted(suggestion.rationale):
            lines.append(f"  {category}: {suggestion.rationale[category]}")
    else:
        lines.append("exclude_layers: (none needed — no 2D keep-set tensors found)")
    # NTH-007: whitelist-candidate heuristic hints (suggest-only, never applied).
    if suggestion.hints:
        lines.append("")
        lines.append("Whitelist candidates (repeated unknown 2D-weight naming):")
        for segment, count, shape in suggestion.hints:
            lines.append(
                f"  candidate whitelist segment: {segment} ({count} occurrences, shape {shape})"
            )
    lines.append("")
    lines.append(
        "Note: this regex covers header-level keeps only (embeddings / heads / "
        "unknown-role 2D weights). Hot-loop and skinny-GEMM exclusions are "
        "stage-2 decisions — read the pipeline code; the tool suggests, the human decides."
    )
    return "\n".join(lines) + "\n"


def render_json(report: AuditReport, suggestion: ExclusionSuggestion) -> str:
    """Render the audit report as stable JSON (``indent=2, sort_keys=True``)."""
    payload = {
        "path": report.path,
        "metadata": report.metadata,
        "totals": {
            "tensors": len(report.tensors),
            "params": _total_params(report),
            "bytes": report.total_bytes,
        },
        "category_counts": report.category_counts,
        "category_bytes": report.category_bytes,
        "modules": [
            {
                "module": summary.module,
                "tensors": summary.tensors,
                "params": summary.params,
                "bytes": summary.nbytes,
                "category_counts": summary.category_counts,
            }
            for summary in report.modules
        ],
        "quantized_layers": [
            {"prefix": prefix, "config": config}
            for prefix, config in report.quantized_layers
        ],
        "quant_format_histogram": report.quant_format_histogram,
        "suggestion": {
            "regex": suggestion.regex,
            "names": list(suggestion.names),
            "rationale": suggestion.rationale,
            "hints": [
                {"segment": segment, "count": count, "shape": shape}
                for segment, count, shape in suggestion.hints
            ],
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# --------------------------------------------------------------------------- #
# CLI entry point (plan STEP 3.2): python -m quantui.model_audit
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantui.model_audit",
        description=(
            "Header-only audit of a .safetensors checkpoint: classify every "
            "tensor, aggregate per-module stats, detect already-quantized "
            "layers, and propose a starting exclude_layers regex."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="path to a .safetensors file or a sharded model folder",
    )
    parser.add_argument("--json", action="store_true", help="emit the JSON report instead of text")
    parser.add_argument(
        "--out",
        default=None,
        help="also write the report to this path (parents created); still echoed to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code (0 ok, 2 audit error)."""
    args = _build_parser().parse_args(argv)
    try:
        report = audit(args.input)
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    suggestion = suggest_exclusions(report)
    payload = render_json(report, suggestion) if args.json else render_text(report, suggestion)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
