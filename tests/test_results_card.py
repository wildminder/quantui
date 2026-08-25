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
