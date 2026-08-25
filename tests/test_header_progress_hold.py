"""Unit tests for the pure header-progress hold [IMP-001 S3B.3].

``HeaderProgressHold`` implements the S1.7 user-report fix as a pure class:
the displayed percentage must never dip within one run, even when a real log
line clears the live-progress store mid-run.
"""

from quantui.header_progress import HeaderProgressHold


def test_first_value_passthrough():
    """A fresh hold adopts the first value unchanged."""
    hold = HeaderProgressHold()
    assert hold.next(40.0) == 40.0


def test_monotonic_hold_dips_are_clamped():
    """next(40) then next(30) -> 40: the bar never dips within one run."""
    hold = HeaderProgressHold()
    assert hold.next(40.0) == 40.0
    assert hold.next(30.0) == 40.0
    # Rising values still pass through.
    assert hold.next(55.0) == 55.0


def test_reset_clears_hold():
    """reset() lets the next run start from its own (lower) value."""
    hold = HeaderProgressHold()
    hold.next(80.0)
    hold.reset()
    # After reset the very next value is passthrough again...
    assert hold.next(10.0) == 10.0
    # ...and monotonicity applies to the NEW run only.
    assert hold.next(5.0) == 10.0


def test_none_signal_holds_last_value():
    """The app passes the held value when aggregate() returns None; the pure
    contract is that a held value survives repeated reads untouched."""
    hold = HeaderProgressHold()
    assert hold.next(25.0) == 25.0
    # Re-reading the same held value is idempotent (no drift).
    assert hold.next(25.0) == 25.0


def test_float_equality_edge_cases():
    """Equal values and exact-zero boundaries behave predictably."""
    hold = HeaderProgressHold()
    assert hold.next(0.0) == 0.0          # zero is a legal starting value
    assert hold.next(0.0) == 0.0          # equal -> keep (>= branch)
    assert hold.next(100.0) == 100.0      # full bar
    assert hold.next(99.999999) == 100.0  # any dip after full stays full
