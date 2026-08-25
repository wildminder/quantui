"""Pure header-progress hold [IMP-001 S3B.3].

Extracted from the ``_last_header_pct`` bookkeeping that lived inline in
``QuantApp._render_header_strip`` / ``_clear_header_strip`` (S1.7 user-report
fix). Textual-free so it can be unit-tested headlessly.
"""


class HeaderProgressHold:
    """Monotonic progress hold: never lets displayed percentage dip within one run (S1.7).

    A mid-run log line can clear the live-progress store, making
    ``run_monitor.aggregate`` return ``None`` or a lower fraction; feeding every
    observed percentage through :meth:`next` guarantees the displayed value only
    ever rises until :meth:`reset` is called at run boundaries.
    """

    def __init__(self) -> None:
        self._held: float | None = None

    def next(self, pct: float) -> float:
        """Feed one observed percentage; returns the value to display.

        The first observation passes through; afterwards any value lower than
        the held one is clamped to it (monotonic within one run).
        """
        if self._held is None or pct >= self._held:
            self._held = pct
        return self._held

    def held(self) -> float | None:
        """Read the currently held value WITHOUT advancing the hold.

        ``None`` means nothing observed since the last :meth:`reset` -- the app
        renders the idle 0% bar in that case.
        """
        return self._held

    def reset(self) -> None:
        """Clear the hold at a run boundary (finished / new run starting)."""
        self._held = None
