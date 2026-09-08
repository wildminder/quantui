"""Post-v0.9.1 dedup: header strip removed — behavior migrated to the footer.

The old header tests pinned tqdm-driven bar advancement and reset semantics;
the same behavior now lives in #footer_bar (own monotonic hold) and is tested
in test_run_footer_headless.py. This file keeps the dedup absence pins.
"""

from textual.css.query import NoMatches

from quantui import app as appmod


async def test_header_widgets_are_gone(tmp_path, monkeypatch):
    """Absence pin: querying the old header ids must raise NoMatches."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        for sel in ("#header_progress", "#eta_label"):
            try:
                a.query_one(sel)
                raise AssertionError(f"{sel} still mounted — dedup regression")
            except NoMatches:
                pass  # expected


async def test_unknown_total_bar_still_drives_footer_labels(tmp_path, monkeypatch):
    """An unknown-total bar (?%|... ?/?) yields no determinate signal:
    the footer stats line shows '--' and the bar stays at 0 (was header ETA test)."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        a._run_start_ts = __import__("time").monotonic() - 3.0
        a.log_msg("Optimizing INT8 (x):  ?%|          | ?/? [00:00<?, ?it/s]")
        await pilot.pause()
        from textual.widgets import Label

        stats = str(a.query_one("#footer_stats", Label).content)
        assert "--" in stats
