"""Headless tests for the global header progress strip (plan S1.7)."""

from textual.widgets import Label, ProgressBar

from quantui import app as appmod


async def test_header_updates_from_tqdm_frame(tmp_path, monkeypatch):
    """A determinate tqdm frame drives #header_progress; a real line resets it."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # The 5-frame tqdm sequence (same as the pinned collapse test).
        for i in range(5):
            a.log_msg(
                f"Optimizing INT8 (Prodigy-plateau):   {i}%|          | {i*10}/4000 "
                f"[00:00<?, ?it/s, loss=0.{i}]"
            )
        await pilot.pause()
        bar = a.query_one("#header_progress", ProgressBar)
        assert bar.progress > 0  # last frame is 40/4000 -> ~1%
        # A real log line clears the bars: progress back to 0 and ETA blank.
        a.log_msg("=== Quantization finished successfully ===")
        await pilot.pause()
        assert bar.progress == 0
        assert a.query_one("#eta_label", Label).content == "--"


async def test_header_indeterminate_shows_blank_eta(tmp_path, monkeypatch):
    """No determinate state -> bar stays at 0 and ETA shows '--'."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # A legacy text-only progress line carries no cur/total/pct signal.
        a._update_live_progress("(1/3) Processing something")
        await pilot.pause()
        bar = a.query_one("#header_progress", ProgressBar)
        assert bar.progress == 0
        assert a.query_one("#eta_label", Label).content == "--"
