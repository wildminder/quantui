"""Pure (Textual-free) live-progress store.

Collapses streaming progress lines into one in-place widget keyed by a stable slot,
so a tqdm / convert_to_quant bar shows only the *latest* frame instead of flooding
the log. No ``textual`` import -- unit-testable without a UI (T04).

Two kinds of progress converge here:

* **Structured** -- a ``CTQ_PROGRESS {"phase":..., "cur":..., "total":..., "pct":...,
  "label":...}`` envelope emitted by OUR workers (worker_ctq.py / worker_ctq_kitchen.py).
  These carry a real (cur, total, pct) payload and drive a determinate ProgressBar.
* **Legacy text** -- a third-party tqdm / ctq ``(N/M) Processing`` header or any other
  progress line. It is stored as a text-only ProgressState (no determinate total), so the
  widget shows the raw text without a numeric bar.

The collapse key for legacy lines is derived by ``ProgressClassifier.progress_key``
(e.g. every ctq ``(N/M) Processing`` header collapses to ``(#/M)``), so consecutive
steps overwrite the same slot rather than stacking. Structured lines key by ``phase``.
"""

from dataclasses import dataclass, field

from .stream_parser import ProgressClassifier, parse_ctq_progress, parse_tqdm_progress


@dataclass
class ProgressState:
    """One collapsed progress slot: either a structured (determinate) step or plain text.

    ``rate`` (F2-S2.2, footer-v2) carries the tqdm speed token (e.g. "66.7it/s")
    when the bar reported one; ``None`` otherwise. Consumed by the footer stats line.
    """

    phase: str
    cur: int | None = None
    total: int | None = None
    pct: float | None = None
    label: str = ""
    rate: str | None = None

    @property
    def determinate(self) -> bool:
        """True when we have a real total OR a percentage to fill a bar with."""
        return (self.total is not None and self.total > 0) or (self.pct is not None)

    @property
    def text(self) -> str:
        """Human-readable single line (used by the on-screen label + the run-log snapshot)."""
        if self.total:
            return f"{self.label} [{self.cur or 0}/{self.total}]"
        if self.pct is not None:
            return f"{self.label} [{self.pct:.0f}%]"
        return self.label or self.phase

    @staticmethod
    def from_dict(d: dict) -> "ProgressState":
        """Build a structured state from a ``CTQ_PROGRESS`` JSON payload (coerces types)."""

        def _int(v):
            try:
                return int(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _float(v):
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return ProgressState(
            phase=str(d.get("phase", "progress")),
            cur=_int(d.get("cur")),
            total=_int(d.get("total")),
            pct=_float(d.get("pct")),
            label=str(d.get("label", "")),
            rate=str(d["rate"]) if d.get("rate") is not None else None,
        )


@dataclass
class LiveProgressStore:
    """Collapses progress lines into one in-place widget keyed by a stable slot."""

    _states: dict[str, ProgressState] = field(default_factory=dict)

    def update(self, msg: str, key: str | None = None) -> None:
        s = msg.replace("\r", "").rstrip()
        # 1) Structured envelope first -- reliable, determinate progress from our own workers.
        payload = parse_ctq_progress(s)
        if payload is not None:
            st = ProgressState.from_dict(payload)
            self._states[st.phase] = st
            return
        # 2) Third-party tqdm bar (convert_to_quant calibration/loading) -- parse the REAL
        #    (cur, total, pct) into a determinate state that MERGES into the main
        #    "quantize" slot, so the bar advances instead of freezing on raw text. This is
        #    the complement to the worker's output-file-size polling (overall-length signal).
        td = parse_tqdm_progress(s)
        if td is not None:
            st = ProgressState(
                phase="quantize",
                cur=td["cur"],
                total=td["total"],
                pct=td["pct"],
                label=td["label"] or "Quantizing",
                rate=td.get("rate"),
            )
            self._states["quantize"] = st
            return
        # 3) Legacy text path (the ctq "(N/M) Processing" headers / unknown-total bars /
        #    any other progress line that carries no structured signal).
        k = key if key is not None else ProgressClassifier.progress_key(s)
        self._states[k] = ProgressState(phase=k, label=s)

    def clear(self) -> None:
        self._states.clear()

    def states(self) -> list[ProgressState]:
        """Ordered list of the current collapsed progress states (structured-first)."""
        return list(self._states.values())

    def render(self) -> str:
        """``'\\n'.join(text)``; ``' '`` when empty (keeps the bordered box height constant)."""
        if not self._states:
            return " "
        return "\n".join(st.text for st in self._states.values())

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the current slots as ``{key: human-readable-text}``.

        Kept as ``str`` values for backward compatibility with existing tests asserting on
        the collapsed progress text.
        """
        return {k: st.text for k, st in self._states.items()}

    def __len__(self) -> int:
        return len(self._states)
