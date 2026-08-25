"""Widget-contract tests (plan Q4b / S0.2) -- the refactor tripwire.

These boot the real ``QuantApp`` headlessly and assert that every id exported by
``quantui.ids`` exists and is queryable, plus a few type/cap assertions on the
key widgets (``#log`` is a capped ``RichLog``, ``#progress_rail`` is a
``ProgressRail``, both family radios are present). Any layout refactor in
Phase 1+ MUST keep this file green unchanged.
"""

from textual.widgets import RadioButton, RadioSet, RichLog

from quantui import app as appmod
from quantui import ids


async def test_all_contract_ids_exist():
    """Every contract id must be present and queryable in the composed app."""
    a = appmod.QuantApp()
    async with a.run_test():
        # Conditional ctq widgets only become visible after refresh_ctq_visibility();
        # they are MOUNTED from the start, so plain query_one works regardless.
        for wid in ids.all_ids():
            assert a.query_one(wid) is not None, f"missing contract widget {wid}"


async def test_conditional_ctq_widgets_queryable_after_visibility_refresh():
    a = appmod.QuantApp()
    async with a.run_test():
        a.refresh_ctq_visibility()
        for wid in ("#ctq_scaling_mode", "#convrot_group_size", "#block_size",
                    "#heur", "#manual_seed"):
            assert a.query_one(wid) is not None, f"missing conditional widget {wid}"


async def test_log_is_richlog_with_cap():
    """#log stays a RichLog with the RUN_LOG_MAX_DISPLAY_LINES cap (plan Q2)."""
    a = appmod.QuantApp()
    async with a.run_test():
        log = a.query_one(ids.LOG)
        assert isinstance(log, RichLog)
        assert log.max_lines == appmod.RUN_LOG_MAX_DISPLAY_LINES


async def test_progress_rail_is_progressrail():
    """S1.8: #progress_rail (the stacked mini-bar rail) is mounted in the app."""
    from quantui import panels

    a = appmod.QuantApp()
    async with a.run_test():
        rail = a.query_one("#progress_rail")
        assert isinstance(rail, panels.ProgressRail)


async def test_family_radios_present():
    """Both family radios (#fam_gguf / #fam_comfy) exist inside #family."""
    a = appmod.QuantApp()
    async with a.run_test():
        fam = a.query_one(ids.FAMILY, RadioSet)
        radios = list(fam.query(RadioButton))
        assert [r.id for r in radios] == ["fam_gguf", "fam_comfy"]


async def test_rail_exists_and_params_primary():
    """S1.1: params-primary split -- a right #rail exists and #log is still queryable."""
    from textual.containers import Vertical, VerticalScroll

    a = appmod.QuantApp()
    async with a.run_test():
        rail = a.query_one("#rail")
        assert isinstance(rail, Vertical)
        params = a.query_one("#params")
        assert isinstance(params, VerticalScroll)
        # params is the primary (left) column: rail comes after it in the DOM order.
        body_children = [c.id for c in a.query_one("#body").children]
        assert body_children == ["params", "rail"]
        assert a.query_one(ids.LOG) is not None


async def test_collapsible_advanced_default_collapsed():
    """S1.2: GGUF + CTQ Advanced sections are collapsed by default; expanding
    reveals inputs and values are still read by validate()."""
    from textual.widgets import Collapsible, Input

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        gguf_adv = a.query_one("#gguf_panel Collapsible")
        comfy_adv = a.query_one("#comfy_panel Collapsible")
        assert isinstance(gguf_adv, Collapsible) and gguf_adv.collapsed is True
        assert isinstance(comfy_adv, Collapsible) and comfy_adv.collapsed is True
        # Expand and set an advanced value; _read_config must still see it.
        gguf_adv.collapsed = False
        await pilot.pause()
        a.query_one("#maxseq", Input).value = "2048"
        cfg = a._read_config()
        assert cfg.gguf.max_seq_length == "2048"
