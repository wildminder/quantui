"""Pure run-monitor math (plan S1.7): aggregate progress + ETA across phases.

``aggregate`` consumes the ordered :class:`quantui.live_progress.ProgressState`
list from ``LiveProgressStore.states()`` plus an elapsed-seconds value and
returns ``(pct, eta_s)``:

* ``pct`` -- mean of determinate states' completion fraction (cur/total or
  pct/100), weighted equally; indeterminate states are IGNORED (not counted as
  0). ``None`` when no determinate state exists -> the caller shows an
  indeterminate bar.
* ``eta_s`` -- ``elapsed * (1 - p) / p`` for a single aggregate fraction ``p``;
  ``None`` when unknown/incomplete. Never negative.

Textual-free: unit-testable without a UI. ETA accuracy is best-effort by
design (plan risk note) -- it must never block or crash on odd inputs.
"""

from __future__ import annotations


def _state_fraction(st) -> float | None:
    """Completion fraction in [0, 1] for one ProgressState, or None if indeterminate."""
    if getattr(st, "total", None):
        cur = st.cur or 0
        return max(0.0, min(1.0, cur / st.total))
    if getattr(st, "pct", None) is not None:
        return max(0.0, min(1.0, st.pct / 100.0))
    return None


def aggregate(states: list, elapsed_s: float = 0.0) -> tuple[float | None, float | None]:
    """Return ``(pct, eta_s)`` for the current collapsed progress states.

    Args:
        states: ordered list of ``ProgressState`` from ``LiveProgressStore.states()``.
        elapsed_s: seconds since the run started (for the ETA projection).

    Returns:
        ``(pct, eta_s)`` -- both ``None`` when there is no determinate signal;
        ``eta_s`` is additionally ``None`` while ``pct`` is 0 (division guard).
        ``pct`` is rounded to 1 decimal; ``eta_s`` to whole seconds.
    """
    fractions = [f for f in (_state_fraction(s) for s in states) if f is not None]
    if not fractions:
        return (None, None)
    p = sum(fractions) / len(fractions)
    eta_s: float | None = None
    if p > 0.0 and elapsed_s > 0.0:
        eta_s = max(0.0, round(elapsed_s * (1.0 - p) / p))
    return (round(p * 100.0, 1), eta_s)


def format_eta(eta_s: float | int | None) -> str:
    """Human-readable ETA cell: ``MM:SS`` or ``H:MM:SS``; ``--`` when unknown."""
    if eta_s is None:
        return "--"
    total = int(max(0, eta_s))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def footer_stats(
    states: list, elapsed_s: float = 0.0, held_pct: float | None = None
) -> dict:
    """Derive the run-footer stats payload (plan 2026-09-08-footer-v2 S2.1).

    Args:
        states: ordered ``ProgressState`` list from ``LiveProgressStore.states()``.
        elapsed_s: seconds since the run started.
        held_pct: last monotonic aggregate pct (from ``HeaderProgressHold``) —
            wins over a fresh LOWER aggregate so the footer bar never dips when
            a log line temporarily clears the store (same rule as the header).

    Returns:
        ``{"pct": float | None, "eta_s": float | None, "elapsed_s": float,
        "counts": str, "rate": str | None}`` — ``counts`` joins the determinate
        states' single-line texts (``label [cur/total]``) with "; " ("" when
        none); ``rate`` is the speed token of the LAST determinate state with
        one (F2-S2.2), e.g. "66.7it/s".
    """
    pct, eta_s = aggregate(states, elapsed_s)
    if pct is None:
        pct = held_pct  # hold fallback: store momentarily empty
    elif held_pct is not None and held_pct > pct:
        pct = held_pct  # monotonic: never dip below the held value
    counts = "; ".join(
        st.text for st in states
        if getattr(st, "determinate", False) and (getattr(st, "text", "") or "").strip()
    )
    rate = None
    for st in states:  # last determinate state with a rate wins
        if getattr(st, "determinate", False) and getattr(st, "rate", None):
            rate = st.rate
    return {
        "pct": pct,
        "eta_s": eta_s,
        "elapsed_s": float(elapsed_s),
        "counts": counts,
        "rate": rate,
    }
