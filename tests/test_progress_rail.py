"""Headless tests for the stacked ProgressRail (plan S1.8).

Covers: two structured CTQ_PROGRESS envelopes -> 2 stacked determinate rows;
ctq (N/M) headers still collapse to ONE row; clicking a row posts
``ProgressRail.BarClicked`` with the phase key.
"""


from quantui import app as appmod
from quantui import panels


def _envelope(phase: str, cur: int, total: int, label: str) -> str:
    import json
    return "CTQ_PROGRESS " + json.dumps(
        {"phase": phase, "cur": cur, "total": total, "pct": round(100 * cur / total, 1),
         "label": label}
    )


async def test_progress_rail_stacks_determinate_bars(tmp_path, monkeypatch):
    """Two distinct structured phases -> two stacked label rows (phase-name
    chips: counts live in the stats line — dedup round 2)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg(_envelope("shard", 1, 3, "Quantizing shard"))
        a.log_msg(_envelope("quantize", 40, 4000, "Optimizing INT8"))
        await pilot.pause()
        rail = a.query_one("#progress_rail", panels.ProgressRail)
        rows = list(rail.query(".rail_row"))
        assert len(rows) == 2, rows
        # No row carries a ProgressBar anymore (single-bar rule).
        for row in rows:
            assert len(list(row.query("ProgressBar"))) == 0
        # Labels are phase-name chips WITHOUT counts (stats line owns counts).
        labels = [str(r.query_one(".rail_label").content) for r in rows]
        assert "Quantizing shard" in labels[0] and "[1/3]" not in labels[0]
        assert "Optimizing INT8" in labels[1] and "[40/4000]" not in labels[1]


async def test_rail_collapses_nm_headers(tmp_path, monkeypatch):
    """ctq '(N/M) Processing' headers collapse to ONE row (store already does)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("(1/211) Processing (INT8): a.weight")
        a.log_msg("(2/211) Processing (INT8): b.weight")
        await pilot.pause()
        rail = a.query_one("#progress_rail", panels.ProgressRail)
        rows = list(rail.query(".rail_row"))
        assert len(rows) == 1, rows
        label = rows[0].query_one(".rail_label")
        assert "(2/211)" in str(label.content)


async def test_bar_click_posts_message(tmp_path, monkeypatch):
    """Clicking a row posts BarClicked(phase) up to the app."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg(_envelope("shard", 2, 3, "Quantizing shard"))
        await pilot.pause()
        rail = a.query_one("#progress_rail", panels.ProgressRail)
        row = rail.query_one(".rail_row")

        clicked = []

        # Capture the message at the App level (bubbled from the row).
        orig = appmod.QuantApp.on_progress_rail_bar_clicked

        def rec(self, event):
            clicked.append(event.phase)
            orig(self, event)

        monkeypatch.setattr(appmod.QuantApp, "on_progress_rail_bar_clicked", rec)
        # Simulate a click on the row.
        row.post_message(panels.ProgressRail.BarClicked("shard"))
        await pilot.pause()
        assert clicked == ["shard"], clicked


async def test_click_row_label_posts_bar_clicked(tmp_path, monkeypatch):
    """Regression (user report, quantization crash): a REAL click on a row's
    chip Label must post BarClicked.

    The old code read ``event.target`` (not a Textual attribute) -> the click
    raised ``AttributeError: 'Click' object has no attribute 'target'``. The
    event arrives with ``event.widget`` = the Label; the rail's parent walk
    must resolve it to the row.
    """
    from textual.widgets import Label as _Label  # noqa: F401  (type clarity)

    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # Reveal the footer WITHOUT action_run: no run means no completion
        # toast, so the pilot click cannot be intercepted.
        a._show_run_footer()
        a.log_msg(_envelope("shard", 1, 3, "Quantizing shard"))
        await pilot.pause()

        rail = a.query_one("#progress_rail", panels.ProgressRail)
        label = rail.query_one(".rail_label", _Label)

        clicked = []
        orig = appmod.QuantApp.on_progress_rail_bar_clicked

        def rec(self, event):
            clicked.append(event.phase)
            orig(self, event)

        monkeypatch.setattr(appmod.QuantApp, "on_progress_rail_bar_clicked", rec)
        await pilot.click(label)
        await pilot.pause()

        assert clicked == ["shard"], clicked
        # Observable effect (same contract as tests/test_log_filter.py):
        # the drawer opens, filtered to the clicked phase.
        assert a.query_one("#log_drawer").display is True
        assert a._log_filter == "shard"


async def test_click_rail_body_no_crash(tmp_path, monkeypatch):
    """A click on the rail itself (or a non-row child) must not crash and must
    not post BarClicked."""
    from textual import events

    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test(size=(120, 40)) as pilot:
        a._show_run_footer()
        # Overflow setup: 6 phases -> rows + a '+2 more' non-row child.
        for i in range(6):
            a.log_msg(_envelope(f"phase{i}", i + 1, 10, f"Phase {i}"))
        await pilot.pause()
        rail = a.query_one("#progress_rail", panels.ProgressRail)

        clicked = []
        orig = appmod.QuantApp.on_progress_rail_bar_clicked

        def rec(self, event):
            clicked.append(event.phase)
            orig(self, event)

        monkeypatch.setattr(appmod.QuantApp, "on_progress_rail_bar_clicked", rec)

        # (a) Real click on the '+N more' Label (a rail child, NOT a row).
        more = rail.query_one(".rail_more")
        await pilot.click(more)
        await pilot.pause()
        assert clicked == [], clicked

        # (b) A Click event posted at the rail body itself (widget = rail).
        rail.post_message(
            events.Click(
                widget=rail, x=0, y=0, delta_x=0, delta_y=0, button=1,
                shift=False, meta=False, ctrl=False,
            )
        )
        await pilot.pause()
        assert clicked == [], clicked
        # No BarClicked -> drawer stays hidden with no filter applied.
        assert a.query_one("#log_drawer").display is False
        assert a._log_filter == ""


async def test_rail_overflow_caps_at_four_rows(tmp_path, monkeypatch):
    """More than MAX_ROWS phases -> exactly MAX_ROWS rows + a '+N more' label."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        for i in range(6):
            a.log_msg(_envelope(f"phase{i}", i + 1, 10, f"Phase {i}"))
        await pilot.pause()
        rail = a.query_one("#progress_rail", panels.ProgressRail)
        rows = list(rail.query(".rail_row"))
        more = list(rail.query(".rail_more"))
        assert len(rows) == panels.ProgressRail.MAX_ROWS
        assert len(more) == 1
        assert "2 more" in str(more[0].content)
