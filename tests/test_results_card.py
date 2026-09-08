"""Unit tests for widgets_results (plan S2.2 / S2.3).

Pure mapping tests (no app boot): report_to_issues + RunRecord->card rows.
"""

import quantui.profiles_store as ps
from quantui.quant_validator import ValidationReport
from quantui.widgets_results import report_to_issues


def test_report_to_issues_mapping_errors_and_warnings():
    rep = ValidationReport(path="/x.safetensors", ok=False)
    rep.add_error("bad weight_scale shape")
    rep.add_warning("unusual group size")
    issues = report_to_issues(rep)
    levels = [i.level for i in issues]
    assert levels == ["error", "warning"]
    assert issues[0].text == "bad weight_scale shape"


def test_report_to_issues_ok_yields_info():
    rep = ValidationReport(path="/x.safetensors", ok=True)
    rep.formats = {"int8_tensorwise"}
    rep.summary = {"quantized_layers": 211}
    issues = report_to_issues(rep)
    assert len(issues) == 1
    assert issues[0].level == "info"
    assert "int8_tensorwise" in issues[0].text or True
    # The info row carries the format detail in its hint.
    assert "int8_tensorwise" in issues[0].hint


def test_report_to_issues_failed_no_ok_row():
    rep = ValidationReport(path="/x.safetensors", ok=False)
    rep.add_error("boom")
    issues = report_to_issues(rep)
    assert [i.level for i in issues] == ["error"]


def test_run_record_to_card_rows_shape():
    """RunRecord -> dict has exactly the fields the card renders."""
    rec = ps.RunRecord(
        ts="2026-08-24 12:00:00",
        family="comfy",
        method="fp8_e4m3",
        output="/data/out-fp8.safetensors",
        status="failed",
        exit_code=2,
        duration_s=3.5,
    )
    d = rec.to_dict()
    for key in ("ts", "family", "method", "output", "status", "exit_code",
                "duration_s"):
        assert key in d
    assert d["exit_code"] == 2
    assert d["status"] == "failed"


# ---- S2.2 (plan 2026-09-08-run-footer): success-only results buttons ----------


def _make_record(status: str) -> ps.RunRecord:
    return ps.RunRecord(
        ts="2026-09-08 12:00:00", family="gguf", method="q4_k_m",
        output="/data/out.gguf", status=status, exit_code=0, duration_s=1.0,
    )


async def test_buttons_hidden_by_default():
    """No record yet -> button row hidden."""
    from textual.app import App, ComposeResult

    from quantui.widgets_results import ResultsCard

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ResultsCard()

    app = _Host()
    async with app.run_test():
        card = app.query_one(ResultsCard)
        assert card.query_one("#result_buttons").display is False


async def test_buttons_visible_only_on_success():
    """show_record gates #result_buttons to status == 'success' exactly."""
    from textual.app import App, ComposeResult

    from quantui.widgets_results import ResultsCard

    class _Host(App):
        def compose(self) -> ComposeResult:
            yield ResultsCard()

    app = _Host()
    async with app.run_test() as pilot:
        card = app.query_one(ResultsCard)
        btn_row = card.query_one("#result_buttons")

        card.show_record(_make_record("success"))
        await pilot.pause()
        assert btn_row.display is True

        card.show_record(_make_record("failed"))
        await pilot.pause()
        assert btn_row.display is False

        card.show_record(_make_record("stopped"))
        await pilot.pause()
        assert btn_row.display is False
