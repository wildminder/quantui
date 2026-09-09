"""Unit tests for ``quantui.ids`` (plan S0.1)."""

from quantui import ids


def test_ids_are_unique_and_prefixed():
    """Every contract id is a '#'-prefixed selector and all values are unique."""
    all_ids = ids.all_ids()
    assert len(all_ids) > 0
    for wid in all_ids:
        assert isinstance(wid, str)
        assert wid.startswith("#"), wid
        assert " " not in wid, wid  # a single simple selector, no compound
    assert len(set(all_ids)) == len(all_ids), "duplicate ids in ids.py"


def test_pinned_literals_match_current_app():
    """The constants must equal the literals the pinned tests already rely on."""
    assert ids.FAMILY == "#family"
    assert ids.GGUF_PANEL == "#gguf_panel"
    assert ids.COMFY_PANEL == "#comfy_panel"
    assert ids.RUN == "#run"
    assert ids.RUN_CTQ == "#run_ctq"
    assert ids.STATUS == "#status"
    assert ids.LOG == "#log"
    assert ids.RESULT_BUTTONS == "#result_buttons"
