"""Headless tests for click-bar -> filtered log drawer (plan S1.9)."""

import json

from textual.widgets import RichLog

from quantui import app as appmod


def _envelope(phase: str, cur: int, total: int, label: str) -> str:
    return "CTQ_PROGRESS " + json.dumps(
        {"phase": phase, "cur": cur, "total": total,
         "pct": round(100 * cur / total, 1), "label": label}
    )


def _drawer_lines(a) -> list[str]:
    log = a.query_one("#log", RichLog)
    return ["".join(str(seg) for seg in line) for line in log.lines]


async def test_click_bar_opens_filtered_drawer(tmp_path, monkeypatch):
    """Posting BarClicked('Optimizing INT8') opens the drawer and shows only
    matching lines from the authoritative temp file."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # tqdm frames for the phase + unrelated detail lines.
        for i in range(5):
            a.log_msg(f"Optimizing INT8 (Prodigy-plateau):   {i}%|  | {i*10}/4000 "
                      f"[00:00<?, ?it/s]")
        a.log_msg("unrelated detail line")
        await pilot.pause()

        assert a.query_one("#log_drawer").display is False
        # Click a rail row via the BarClicked message (the label term the plan
        # names in its example; tqdm bars collapse into the 'quantize' slot).
        from quantui.panels import ProgressRail
        a.post_message(ProgressRail.BarClicked("Optimizing INT8"))
        await pilot.pause()
        drawer = a.query_one("#log_drawer")
        assert drawer.display is True
        assert a._log_filter == "Optimizing INT8"
        lines = _drawer_lines(a)
        assert lines, "filtered view must show matching lines"
        assert all("Optimizing INT8" in ln for ln in lines), lines
        # The border title advertises the active filter.
        assert "Optimizing INT8" in drawer.border_title
        assert "unrelated" not in "\n".join(lines)


async def test_esc_clears_filter(tmp_path, monkeypatch):
    """Esc clears an active filter; the normal stream resumes."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("Optimizing INT8 frame one")
        a.log_msg("another Optimizing INT8 line")
        a.log_msg("plain line after")
        await pilot.pause()
        # Activate the filter programmatically.
        a.query_one("#log_drawer").display = True
        a._apply_log_filter("Optimizing INT8")
        await pilot.pause()
        assert a._log_filter == "Optimizing INT8"
        # Esc clears it.
        await pilot.press("escape")
        await pilot.pause()
        assert a._log_filter == ""
        # New writes flow unfiltered again.
        n_before = len(_drawer_lines(a))
        a.log_msg("post-filter line plain")
        await pilot.pause()
        lines = _drawer_lines(a)
        assert len(lines) > n_before
        assert any("post-filter line plain" in ln for ln in lines)


async def test_filtered_writes_only_matching_lines(tmp_path, monkeypatch):
    """While filtered, live writes append ONLY matching lines."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("Optimizing INT8 seed line")
        a.query_one("#log_drawer").display = True
        a._apply_log_filter("Optimizing INT8")
        a.log_msg("Optimizing INT8 second frame")
        a.log_msg("noise that must not show")
        await pilot.pause()
        lines = _drawer_lines(a)
        assert any("seed line" in ln for ln in lines)
        assert any("second frame" in ln for ln in lines)
        assert not any("noise" in ln for ln in lines)
