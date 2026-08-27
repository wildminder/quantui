"""STEP 4.1: headless TUI tests for the model-audit screen + palette command.

Follows the ``run_test()`` async pattern from ``tests/test_app_headless.py``
(pytest.ini sets ``asyncio_mode = auto``). The audit itself runs in a worker
thread inside :class:`AuditScreen`; tests wait on ``app.workers`` before
asserting so the thread-posted result is deterministically applied.
"""

from textual.widgets import DataTable, Input

from quantui import app as appmod
from quantui.app import QuantCommands
from quantui.comfy_quant_schema import write_safetensors
from quantui.model_audit import audit_file, suggest_exclusions
from quantui.quant_methods import Family
from quantui.screens import AuditScreen, PathModal


def _write_fixture(tmp_path, name="audit.safetensors"):
    specs = {
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.norm.weight": ("BF16", [4], b"\x00" * 8),
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
        "lm_head.weight": ("BF16", [10, 4], b"\x00" * 80),
    }
    path = tmp_path / name
    path.write_bytes(write_safetensors(specs))
    return str(path)


def _switch_family(app, fam: Family):
    app.family = fam
    app.query_one("#gguf_panel").display = fam == Family.GGUF
    app.query_one("#comfy_panel").display = fam == Family.COMFY
    app.refresh_ctq_visibility()


def test_palette_contains_audit_command():
    provider = QuantCommands(None)
    titles = [title for title, _action, _help in provider._ACTIONS]
    assert "Audit model file" in titles
    # And it is discoverable through the palette's discovery stream.
    import asyncio

    hits = []

    async def collect():
        async for hit in provider.discover():
            hits.append(hit)

    asyncio.run(collect())
    texts = [(h.text or "") + " " + str(getattr(h, "display", "")) for h in hits]
    assert any("Audit model file" in t for t in texts), texts


async def test_audit_screen_shows_module_table(tmp_path):
    fixture = _write_fixture(tmp_path)
    expected_report = audit_file(fixture)
    expected_regex = suggest_exclusions(expected_report).regex

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        screen = AuditScreen(fixture)
        a.push_screen(screen)
        await pilot.pause()
        await a.workers.wait_for_complete()
        await pilot.pause()

        table = screen.query_one("#audit_table", DataTable)
        assert table.row_count == len(expected_report.modules)
        suggestion = screen.query_one("#audit_suggestion", Input)
        assert suggestion.value == expected_regex


async def test_audit_screen_error_state(tmp_path):
    missing = str(tmp_path / "does_not_exist.safetensors")
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        screen = AuditScreen(missing)
        a.push_screen(screen)
        await pilot.pause()
        await a.workers.wait_for_complete()
        await pilot.pause()

        table = screen.query_one("#audit_table", DataTable)
        assert table.row_count == 0
        assert table.display is False
        error_label = screen.query_one("#audit_error")
        assert error_label.display is True
        assert missing in str(error_label.content)


async def test_action_audit_uses_ctq_input(tmp_path):
    fixture = _write_fixture(tmp_path)
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        _switch_family(a, Family.COMFY)
        a.query_one("#ctq_input", Input).value = fixture
        a.action_audit_model()
        await pilot.pause()

        top = a.screen_stack[-1]
        assert isinstance(top, AuditScreen)
        assert top.audit_path == fixture


async def test_action_audit_empty_input_opens_path_modal():
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        _switch_family(a, Family.COMFY)
        a.query_one("#ctq_input", Input).value = ""
        a.action_audit_model()
        await pilot.pause()

        top = a.screen_stack[-1]
        assert isinstance(top, PathModal)
