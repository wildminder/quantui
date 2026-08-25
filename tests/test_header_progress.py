"""Headless tests for the header global progress strip (plan S1.7)."""

from textual.widgets import Label, ProgressBar

from quantui import app as appmod


async def test_header_updates_from_tqdm_frame(tmp_path, monkeypatch):
    """The 5-frame tqdm sequence drives the header bar; a real line clears it."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # Simulate an active run so the ETA clock ticks.
        a._run_start_ts = __import__("time").monotonic() - 5.0
        for i in range(5):
            a.log_msg(
                f"Optimizing INT8 (Prodigy-plateau):   {i}%|          | {i*10}/4000 "
                f"[00:00<?, ?it/s, loss=0.{i}]"
            )
        await pilot.pause()
        bar = a.query_one("#header_progress", ProgressBar)
        assert bar.progress > 0, "determinate tqdm frame must advance the header bar"

        # A real (non-progress) line clears the bars -> header resets.
        a.log_msg("=== Quantization finished successfully ===")
        await pilot.pause()
        assert bar.progress == 0
        eta = a.query_one("#eta_label", Label).content
        assert eta == "--"


async def test_header_indeterminate_shows_blank_eta(tmp_path, monkeypatch):
    """An unknown-total bar leaves the aggregate indeterminate -> ETA '--'."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a._run_start_ts = __import__("time").monotonic() - 3.0
        # Unknown-total tqdm bar (?%|... ?/?): no usable determinate signal.
        a.log_msg("Optimizing INT8 (x):  ?%|          | ?/? [00:00<?, ?it/s]")
        await pilot.pause()
        bar = a.query_one("#header_progress", ProgressBar)
        assert bar.progress == 0
        assert a.query_one("#eta_label", Label).content == "--"
