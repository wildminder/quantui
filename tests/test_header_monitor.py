"""Post-v0.9.1 dedup: the S1.7 header progress strip is REMOVED.

These are absence pins — the footer's wide aggregate bar + stats line are the
single progress surface; the strip used to duplicate them during runs and
show "0% / --" at idle. The pure HeaderProgressHold class survives (used by
the footer hold) and is tested in test_header_progress_hold.py.
"""

from textual.css.query import NoMatches

from quantui import app as appmod


async def test_header_strip_is_gone(tmp_path, monkeypatch):
    """No #header_strip / #header_progress / #eta_label anywhere in the app."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        for sel in ("#header_strip", "#header_progress", "#eta_label"):
            try:
                a.query_one(sel)
                raise AssertionError(f"{sel} still mounted — dedup regression")
            except NoMatches:
                pass  # expected: removed


async def test_progress_drives_footer_only(tmp_path, monkeypatch):
    """A determinate tqdm frame advances the FOOTER bar — the only visible one."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._show_run_footer()
        await pilot.pause()
        for i in range(5):
            a.log_msg(
                f"Optimizing INT8 (Prodigy-plateau):   {i}%|          | {i*10}/4000 "
                f"[00:00<?, ?it/s, loss=0.{i}]"
            )
        await pilot.pause()
        from textual.widgets import ProgressBar

        from quantui.panels import RunFooter

        footer = a.query_one("#run_footer", RunFooter)
        footer_bar = footer.query_one("#footer_bar", ProgressBar)
        assert footer_bar.progress > 0
        # Single-surface invariant: the footer bar is the ONLY visible bar.
        visible = [
            b
            for b in a.query("ProgressBar")
            if b.display and b.region.width > 0 and b.region.height > 0
        ]
        assert visible == [footer_bar]
