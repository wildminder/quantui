"""Pure (Textual-free) byte-stream parsing + progress classification.

Extracted verbatim from ``app.py`` during the modularization refactor (T01). This
module imports ONLY the stdlib, so it is unit-testable without a UI or a real
subprocess. ``worker_runner`` / ``ui_bridge`` consume it; ``app.py`` delegates to it
via compatibility aliases (removed in T09).

**Regexes live ONLY here (see plan §8)** -- never re-define a progress regex elsewhere.
"""

import json
import re
from dataclasses import dataclass

# Structured progress envelope emitted by OUR workers (worker_ctq.py / worker_ctq_kitchen.py).
# A line of the exact form:
#     CTQ_PROGRESS {"phase":"shard","cur":1,"total":3,"label":"Quantizing shard ...","pct":33.3}
# is our OWN reliable progress signal. Unlike third-party tqdm / ctq "(N/M) Processing"
# headers, this envelope is unambiguously classified as progress and carries a structured
# (cur, total, pct) payload that drives a real Textual ProgressBar. See live_progress.py.
CTQ_PROGRESS_PREFIX = "CTQ_PROGRESS "

# tqdm / optimizer progress bar: "<pct>%|...|" (percent glued to a pipe, the bar,
# then a second pipe). We deliberately do NOT require a trailing '[' so ?%-bars and
# bracket-less bars still collapse into the live-progress widget.
_PROGRESS_RE = re.compile(r"(?:\d+|[?])\s*%\s*\|.*?\|")
# Bare rate line without a bar, e.g. "1234/5678 [00:01<?, ?it/s]" (or "?/? [?it/s]").
_RATE_RE = re.compile(r"\b(?:\d+|[?])/(?:\d+|[?])\b.*?(?:it/s|s/it)")

# ctq (convert_to_quant) per-tensor progress header, e.g.
#   "(1/211) Processing (INT8): model.embed_tokens.weight"
# Each such header is followed by indented "    - " detail lines and the next
# "(N/M) Processing ..." header. We collapse every header into ONE live-progress
# slot so the widget shows only the latest step.
# The pattern REQUIRES PARENTHESES so it does NOT match the worker's own
# "[1/3] Quantizing shard" (square brackets) nor "Epoch 3/10" (no parens).
_CTQ_PROGRESS_RE = re.compile(r"\((?:\d+|[?])/(?:\d+|[?])\)\s+\w")
# Indented "- " continuation line under a (N/M) Processing header, e.g.
#   "    - Tensor shape: [576, 1536], Max rank: 576. Using k=256 components."
# These must NOT clear the live-progress widget.
_PROGRESS_DETAIL_RE = re.compile(r"^\s+-\s")
# Pure horizontal rule emitted by convert_to_quant between a (N/M) Processing header
# and its "    - " detail lines (e.g. "----"). MUST NOT match a real status line like
# "=== Quantization finished successfully ===" (which contains letters), so that the
# latter still clears the live-progress widget at a phase boundary.
_SEPARATOR_RE = re.compile(r"^\s*[-=_*·─━│]{3,}\s*$")

# tqdm / optimizer bar line (third-party calibration/loading bar), e.g.
#   "Optimizing INT8 (Prodigy-plateau):  50%|#####| 2000/4000 [00:01<?, ?it/s]"
#   "Loading tensors:  12%|###| 30/250 [00:01<?, ?it/s]"
# We parse the REAL (cur, total, pct) out of these so the TUI can drive a determinate
# ProgressBar instead of freezing on the raw text. The patterns deliberately require a
# '%|' bar form OR an 'it/s' rate so they NEVER match a ctq "(N/M) Processing" header
# (parenthesized counter, no bar) or a plain "Epoch 3/10" log line.
_TQDM_BAR_RE = re.compile(r"(?P<pct>\d+|[?])\s*%\s*\|.*?\|")
_TQDM_COUNTER_RE = re.compile(r"\|\s*(?P<cur>\d+|[?])\s*/\s*(?P<total>\d+|[?])\s*\[")
_TQDM_RATE_RE = re.compile(r"\b(?:\d+|[?])/(?:\d+|[?])\b[^\n]*?(?:it/s|s/it)")


Category = str  # one of: "progress" | "detail" | "separator" | "line" | "blank"


@dataclass
class StreamSegment:
    """One parsed chunk of worker stdout, ready for routing by ``LogRouter``."""

    content: str
    category: Category
    is_blank: bool = False


def parse_ctq_progress(msg: str) -> dict | None:
    """Return the parsed JSON payload of a ``CTQ_PROGRESS`` envelope, or ``None``.

    Only a line that begins (after stripping) with :data:`CTQ_PROGRESS_PREFIX` and
    contains valid JSON is recognized. A malformed / non-envelope line returns ``None``
    so the classifier falls through to the regex-based path. Pure stdlib (json + str),
    so it is safe to call on every line without importing ``textual``.
    """
    s = msg.strip()
    if not s.startswith(CTQ_PROGRESS_PREFIX):
        return None
    body = s[len(CTQ_PROGRESS_PREFIX):].strip()
    if not body:
        return None
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def parse_tqdm_progress(msg: str) -> dict | None:
    """Extract the REAL ``(cur, total, pct)`` from a third-party tqdm bar line.

    Examples handled (convert_to_quant's own calibration/loading bars):

    * ``Optimizing INT8 (Prodigy-plateau):  50%|#####| 2000/4000 [00:01<?, ?it/s]``
    * ``Loading tensors:  12%|###| 30/250 [00:01<?, ?it/s]``
    * ``1234/5678 [00:01<?, ?it/s]`` (a bare rate line, no bar)

    Unlike scraping the raw text, this returns a structured ``{"label", "cur", "total",
    "pct"}`` the store turns into a *determinate* :class:`ProgressState` (a real bar),
    and it is the complement to the worker's file-size polling (the "overall length"
    signal). Returns ``None`` when the line is not a usable tqdm bar.

    Deliberately does NOT match:

    * ctq ``(N/M) Processing (INT8): ...`` headers -- parenthesized counter, no ``%|``
      bar, no ``it/s`` (guarded by both the has_bar/has_rate gate and ``_CTQ_PROGRESS_RE``).
    * ``Epoch 3/10 loss=0.1`` log lines -- no ``%|`` and no ``it/s``.
    * Unknown-total bars (``?%|###| ?/? [?it/s]``) -- nothing usable to draw, so we
      return ``None`` and the caller falls back to the legacy text path.
    """
    s = msg.strip()
    if not s:
        return None
    has_bar = bool(_TQDM_BAR_RE.search(s))
    has_rate = bool(_TQDM_RATE_RE.search(s))
    if not (has_bar or has_rate):
        return None
    # Never treat a ctq "(N/M) Processing" header as a tqdm bar.
    if _CTQ_PROGRESS_RE.search(s):
        return None

    # Description before the first ": " (e.g. "Optimizing INT8 (Prodigy-plateau)").
    label = ""
    mcol = re.search(r":\s", s)
    if mcol:
        label = s[: mcol.start()].strip()

    cur = total = None
    pct = None
    mc = _TQDM_COUNTER_RE.search(s)
    if mc:
        cur_s, total_s = mc.group("cur"), mc.group("total")
        cur = None if cur_s == "?" else int(cur_s)
        total = None if total_s == "?" else int(total_s)
    mp = _TQDM_BAR_RE.search(s)
    if mp:
        p = mp.group("pct")
        pct = None if p == "?" else float(p)
    # Derive pct from cur/total when the bar omitted the leading "N%" token.
    if pct is None and cur is not None and total:
        pct = 100.0 * cur / total
    # Nothing usable (e.g. unknown-total "?/?") -> fall back to the legacy text path.
    if pct is None and cur is None and total is None:
        return None
    return {"label": label, "cur": cur, "total": total, "pct": pct}


def split_frames(text: str, partial: str) -> tuple[list[tuple[str, str]], str]:
    """Split a chunk of worker stdout into (content, kind) segments.

    tqdm rewrites its line with '\\r' (in-place overwrite) and only ends a bar with
    '\\n'. We treat '\\r'-terminated segments as *progress* frames (the latest of a
    bar) and '\\n'-terminated segments as *line* log entries. ``partial`` carries an
    unfinished segment across chunk boundaries.

    Returns ``(segments, new_partial)`` where each segment is ``(content, kind)`` with
    ``kind`` in ``{"progress", "line"}``.
    """
    data = partial + text
    segs = re.split(r"([\r\n])", data)
    out: list[tuple[str, str]] = []
    cur = segs[0] if segs else ""
    idx = 1
    while idx < len(segs):
        delim = segs[idx]
        kind = "line" if delim == "\n" else "progress"
        out.append((cur, kind))
        cur = segs[idx + 1] if idx + 1 < len(segs) else ""
        idx += 2
    return out, cur


class ProgressClassifier:
    """Single source of truth for recognizing progress / detail lines."""

    @staticmethod
    def is_progress_line(msg: str) -> bool:
        """True if ``msg`` looks like a tqdm/optimizer/ctq progress-bar line.

        Also matches the ctq per-tensor ``(N/M) Processing ...`` header so the MAIN
        quantization phase shows a live bar in the ComfyUI tab.
        """
        m = msg.strip()
        if not m:
            return False
        return bool(
            _PROGRESS_RE.search(m)
            or _RATE_RE.search(m)
            or _CTQ_PROGRESS_RE.search(m)
        )

    @staticmethod
    def is_progress_detail(msg: str) -> bool:
        """True if ``msg`` is an indented ``- `` continuation of a (N/M) Processing step.

        These detail lines must keep the live-progress widget on screen rather than
        clearing it (otherwise the per-tensor header flashes then disappears). The raw
        ``msg`` is searched so the leading indentation is preserved.
        """
        return bool(_PROGRESS_DETAIL_RE.search(msg))

    @staticmethod
    def is_separator(msg: str) -> bool:
        """True if ``msg`` is a pure horizontal rule (e.g. ``----`` / ``===``).

        Used to echo the convert_to_quant separator between a ``(N/M) Processing``
        header and its details to the log WITHOUT clearing the live-progress widget
        (the root cause of the "no bar + fewer strings" ComfyUI regression). A line
        containing any letters (e.g. ``=== Quantization finished successfully ===``)
        is intentionally NOT a separator, so phase-boundary lines still clear.
        """
        return bool(_SEPARATOR_RE.search(msg.strip()))

    @classmethod
    def classify(cls, msg: str) -> Category:
        """Single source of truth for the 4-way routing decision.

        Order matters: blank -> 'blank'; a ``CTQ_PROGRESS`` envelope -> 'progress'
        (reliable, structured); separator -> 'separator'; detail -> 'detail';
        progress -> 'progress'; else 'line'.
        """
        if not msg.strip():
            return "blank"
        if parse_ctq_progress(msg) is not None:
            return "progress"
        if cls.is_separator(msg):
            return "separator"
        if cls.is_progress_detail(msg):
            return "detail"
        if cls.is_progress_line(msg):
            return "progress"
        return "line"

    @staticmethod
    def progress_key(msg: str) -> str:
        """Stable key so consecutive updates overwrite the same live-progress slot.

        For ctq ``(N/M) Processing`` headers it collates every step to ONE slot by
        normalizing the varying first number (``(1/211)`` -> ``(#/211)``) and dropping
        the per-step layer name, so the widget shows only the latest step instead of
        211 stacked lines.
        """
        parts = re.split(r"\d+%", msg, maxsplit=1)
        if len(parts) > 1:
            return parts[0].strip() or "progress"
        m = re.search(r"\((?:\d+|[?])/(?:\d+|[?])\)", msg)
        if m:
            head = msg[: m.start()].strip()
            counter = re.sub(r"\d+", "#", m.group(0), count=1)  # (1/211) -> (#/211)
            return f"{head} {counter}".strip() or "ctq_progress"
        return msg.strip() or "progress"
