"""Unit tests for quantui.run_monitor (plan S1.7)."""

from quantui.live_progress import ProgressState
from quantui.run_monitor import aggregate, format_eta


def test_aggregate_mixed_states():
    """Determinate states are averaged; indeterminate ones ignored."""
    det1 = ProgressState(phase="a", cur=40, total=4000, label="A")   # 1%
    det2 = ProgressState(phase="b", cur=50, total=100, label="B")    # 50%
    indet = ProgressState(phase="c", label="text only")              # ignored
    pct, eta = aggregate([det1, det2, indet], elapsed_s=10.0)
    assert pct == round((0.01 + 0.5) / 2 * 100, 1) == 25.5
    # ETA from the single aggregate fraction p=0.255
    p = (0.01 + 0.5) / 2
    assert eta == round(10.0 * (1 - p) / p)
    # No determinate state at all -> both unknown.
    assert aggregate([indet]) == (None, None)
    assert aggregate([]) == (None, None)


def test_eta_monotonic():
    """ETA decreases as progress advances at constant elapsed time."""
    states = [ProgressState(phase="q", label="Q")]
    etas = []
    for cur in (100, 2000, 3900):
        states[0].cur = cur
        states[0].total = 4000
        _, eta = aggregate(states, elapsed_s=30.0)
        etas.append(eta)
    assert all(e is not None for e in etas)
    assert etas[0] > etas[1] > etas[2] >= 0
    # Complete -> ETA 0.
    states[0].cur = 4000
    _, eta = aggregate(states, elapsed_s=30.0)
    assert eta == 0


def test_format_eta():
    assert format_eta(None) == "--"
    assert format_eta(0) == "00:00"
    assert format_eta(65) == "01:05"
    assert format_eta(3671) == "1:01:11"
