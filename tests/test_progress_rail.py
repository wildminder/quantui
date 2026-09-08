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
    """Two distinct structured phases -> two stacked label rows (label-only:
    the footer's wide aggregate bar is the ONE progress bar — redundancy fix)."""
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
        # The labels carry the per-phase counts, not the aggregate.
        labels = [str(r.query_one(".rail_label").content) for r in rows]
        assert "[1/3]" in labels[0]
        assert "[40/4000]" in labels[1]


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
