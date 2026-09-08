"""Unit tests for run_monitor.footer_stats (plan 2026-09-08-footer-v2 S2.1).

Pure math — no UI boot. footer_stats derives the footer stats-line payload
(pct with hold fallback, ETA, elapsed, per-phase counts) from the same inputs
the header strip already consumes.
"""

from quantui.live_progress import ProgressState
from quantui.run_monitor import footer_stats


def _det(phase, cur, total, label=""):
    return ProgressState(phase=phase, cur=cur, total=total, label=label or phase)


def test_no_determinate_states_yields_none_pct():
    st = ProgressState(phase="prep", label="Preparing")
    out = footer_stats([st], elapsed_s=5.0)
    assert out["pct"] is None
    assert out["eta_s"] is None
    assert out["elapsed_s"] == 5.0
    assert out["counts"] == ""


def test_single_determinate_pct_and_counts():
    out = footer_stats([_det("quantize", 2600, 4000, "Optimizing INT8")], 10.0)
    assert out["pct"] == 65.0
    assert out["counts"] == "Optimizing INT8 [2600/4000]"
    assert out["eta_s"] is not None and out["eta_s"] > 0


def test_hold_fallback_when_store_cleared():
    """Sparse log lines clear the store: the held pct must win over None."""
    out = footer_stats([], elapsed_s=10.0, held_pct=42.0)
    assert out["pct"] == 42.0
    assert out["counts"] == ""


def test_hold_preferred_never_dips():
    """A fresh aggregate LOWER than the held value must not dip the bar."""
    states = [_det("a", 1, 100, "A")]
    out = footer_stats(states, elapsed_s=10.0, held_pct=50.0)
    assert out["pct"] == 50.0


def test_pct_is_rounded_to_one_decimal():
    out = footer_stats([_det("a", 1, 3, "A")], 0.0)
    assert out["pct"] == round(100 / 3, 1)
    # p == 0 -> no ETA (division guard, same as aggregate()).
    assert out["eta_s"] is None


def test_counts_joins_multiple_phases():
    out = footer_stats(
        [_det("load", 2, 3, "Loading shard"), _det("q", 40, 4000, "Optimizing INT8")],
        1.0,
    )
    assert out["counts"] == "Loading shard [2/3]; Optimizing INT8 [40/4000]"


def test_zero_elapsed_never_produces_eta():
    out = footer_stats([_det("a", 50, 100, "A")], elapsed_s=0.0)
    assert out["eta_s"] is None
