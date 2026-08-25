"""Pure (Textual-free) bridge between parsed stream segments and widget writes.

``LogRouter`` turns a :class:`~quantui.stream_parser.StreamSegment` into the
right sequence of :class:`LogSink` calls: it writes the **authoritative** full log line
to the temp file first, then routes by category -- progress updates the
:class:`LiveProgressStore` and refreshes the live widget; ``detail``/``separator`` lines
are echoed to the log **without** clearing the live bar (the T02/T03 fix for the ComfyUI
"no bar + fewer strings" regression); a real ``line`` clears the bar and appends.

``QuantApp`` is the concrete :class:`LogSink` / :class:`LogObserver` -- it is the only
module here that touches Textual widgets. This module imports neither ``textual`` nor any
widget, so it is unit-testable without a UI.
"""

from typing import Protocol

from .live_progress import LiveProgressStore
from .stream_parser import ProgressClassifier, StreamSegment


class LogObserver(Protocol):
    """Implemented (structurally) by ``QuantApp``."""

    def on_segment(self, seg: StreamSegment) -> None: ...
    def on_raw(self, raw: bytes) -> None: ...
    def on_verdict(self, raw: bytes, category: str, live_count: int, content: str) -> None: ...
    def live_count(self) -> int: ...


class LogSink(Protocol):
    """Concrete implementation lives in ``QuantApp`` (touches widgets)."""

    def write_run_log(self, line: str) -> None: ...
    def write_log(self, line: str) -> None: ...
    def update_live(self, text: str) -> None: ...
    def update_progress(self, states: list) -> None: ...
    def clear_live(self) -> None: ...


class LogRouter:
    """Dispatches a parsed ``StreamSegment`` to the concrete ``LogSink``."""

    def __init__(
        self,
        sink: LogSink,
        store: LiveProgressStore,
        classifier: type[ProgressClassifier] = ProgressClassifier,
    ) -> None:
        self._sink = sink
        self._store = store
        self._classifier = classifier

    def dispatch(self, seg: StreamSegment) -> None:
        content = seg.content.replace("\r", "")
        if not content.strip():
            return  # blank / tqdm-only lines add no information (and are never logged)

        # (1) Authoritative full log -> temp file (every line, always).
        self._sink.write_run_log(content)

        # (2) Route by category. seg.category is set when the segment is built; fall
        # back to re-classifying in case a raw segment arrives without one.
        category = seg.category or self._classifier.classify(content)
        if category == "progress":
            # Collapse every frame of the same bar into one slot; refresh the widget.
            self._store.update(content)
            self._sink.update_progress(self._store.states())
        elif category in ("detail", "separator"):
            # Keep the live bar on screen; just echo the line to the log.
            self._sink.write_log(content)
        else:  # line / blank-with-content
            # A real log line ends the current (collapsed) bar; then append to the log.
            self._store.clear()
            self._sink.clear_live()
            self._sink.write_log(content)
