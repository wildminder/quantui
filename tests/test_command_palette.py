"""Headless tests for the command palette (plan S3.1)."""

from quantui import app as appmod
from quantui.app import QuantCommands


def test_palette_lists_core_commands():
    """Provider search finds core commands by fuzzy term."""
    provider = QuantCommands(None)  # screen=None: matches() doesn't need it
    hits = []
    import asyncio

    async def collect():
        async for hit in provider.search("log"):
            hits.append(hit)

    asyncio.run(collect())
    texts = [h.text or "" for h in hits]
    assert any("Toggle log" in t for t in texts), texts
    assert any("Open log file" in t for t in texts), texts


async def test_palette_action_toggles_drawer():
    """Executing the Toggle-log command shows #log_drawer."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        drawer = a.query_one("#log_drawer")
        assert drawer.display is False
        a.action_toggle_log()
        await pilot.pause()
        assert drawer.display is True


def test_palette_focus_jumps_present():
    provider = QuantCommands(None)
    assert any("#model" in j[1] for j in provider._JUMPS)
    assert any("#ctq_input" in j[1] for j in provider._JUMPS)


async def test_app_has_quant_commands_registered():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        from quantui.app import QuantCommands as QC

        assert any(
            p is QC for p in (type(a).COMMANDS or ())
        ), type(a).COMMANDS
