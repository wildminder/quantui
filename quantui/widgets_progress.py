"""BlockBar — chunky multi-row progress widget (plan 2026-09-09-control-panel
S1.2).

The native ProgressBar's inner Bar pins ``width: 32; height: 1`` and renders a
single centered ``━`` strip: it cannot look like a control-panel readout, and
even forced to ``height: 3`` it fills only the centered row. BlockBar instead
fills its WHOLE widget area with ``█`` (filled) / ``░`` (track) blocks — full
width comes from CSS (``width: 1fr``), row count is a constructor arg.

ProgressBar compatibility (the #footer_bar contract, footer-v2 S2.1):
``update(total=, progress=)`` keeps the exact call shape used by
``QuantApp._render_footer_bar`` (progress on the 0-``total`` scale), and the
``progress`` property aliases ``percentage * 100`` so existing tests reading
``bar.progress`` keep working unchanged.

The row math lives in the pure :func:`build_bar_text` so tests exercise it
without any app boot (``Widget.size`` is only meaningful after layout).
"""

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget

FILL = "█"
TRACK = "░"
# Muted theme-family colors: success green when complete, steel cyan in flight
# (plan S1.2 allows widget-internal literals short-term; palette v2 values).
_FILL_STYLE = "bold #5fd7a0"
_TRACK_STYLE = "bold #4dc3ff"


def build_bar_text(pct: float | None, width: int, rows: int) -> Text:
    """Pure render math: ``rows`` identical lines of block glyphs.

    ``pct`` is a 0-1 fraction (values outside are clamped); ``width`` is the
    content width in cells (guarded to >= 1 for the pre-layout zero-size case).
    """
    p = pct or 0.0
    w = max(width, 1)
    # Half-up fill count: round() alone would banker-round exact .5 fractions
    # (25% of 10 cells -> 2), which reads wrong on a progress readout.
    filled = int(w * min(p, 1.0) + 0.5)
    style = _FILL_STYLE if p >= 1.0 else _TRACK_STYLE
    row = Text(f"{FILL * filled}{TRACK * (w - filled)}", style=style)
    return Text("\n").join([row] * rows)


class BlockBar(Widget):
    """Chunky multi-row progress bar filling its full widget area.

    State is the ``percentage`` reactive (0.0-1.0, like Textual's ProgressBar);
    ``update(total=, progress=)`` mirrors the native API for drop-in use.
    """

    DEFAULT_CSS = "BlockBar { width: 1fr; height: auto; }"

    percentage: reactive[float | None] = reactive(None)

    def __init__(self, total: float = 100, rows: int = 3, **kwargs) -> None:
        super().__init__(**kwargs)
        self._total = total
        self.rows = rows

    @property
    def progress(self) -> float:
        """0-``total`` scale alias of ``percentage`` (ProgressBar-compatible).

        Rounded to 6 decimals so round-trip float noise (``0.4 * 100`` ->
        ``40.00000000000001``) never breaks ``bar.progress == 40`` reads.
        """
        return round((self.percentage or 0.0) * 100, 6)

    def update(self, total: float | None = None, progress: float | None = None) -> None:
        """ProgressBar-compatible update (drop-in for #footer_bar callers).

        ``progress`` is on the 0-``total`` scale (native ProgressBar semantics);
        it is clamped to [0, 1] after normalization so stray over/underflows
        render as full/empty instead of corrupting the row math.
        """
        if total is not None:
            self._total = total
        if progress is not None:
            frac = progress / self._total if self._total else 0.0
            self.percentage = max(0.0, min(1.0, frac))

    def render(self) -> Text:
        """Render the bar across the widget's content area (``rows`` rows)."""
        return build_bar_text(self.percentage, self.size.width, self.rows)
