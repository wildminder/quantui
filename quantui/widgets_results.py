"""Results card + structured validation issue rows (plan S2.2 / S2.3).

``ResultsCard`` is a rail-side summary of the LAST finished run: outcome line,
output path, duration, exit code, plus ``[Copy path] [Open folder]`` buttons.
It lives in the right-hand rail under the ProgressRail. When idle it renders a
muted placeholder so the rail layout does not jump.

``show_issues`` renders structured validation issues (level/text/hint rows)
from :meth:`QuantApp.action_validate_comfy` -- replacing log-only output while
keeping the log lines intact.

Widget ids go through :mod:`quantui.ids` (contract Q4a).
"""

import os

from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Label

from . import ids


class Issue:
    """One structured validation issue row (S2.3)."""

    __slots__ = ("level", "text", "hint")

    def __init__(self, level: str, text: str, hint: str = "") -> None:
        self.level = level  # "error" | "warning" | "info"
        self.text = text
        self.hint = hint


def report_to_issues(report) -> list[Issue]:
    """Map a :class:`quantui.quant_validator.ValidationReport` into Issue rows."""
    issues: list[Issue] = []
    for e in getattr(report, "errors", []) or []:
        issues.append(Issue("error", str(e)))
    for w in getattr(report, "warnings", []) or []:
        issues.append(Issue("warning", str(w)))
    if not issues and bool(getattr(report, "ok", False)):
        fmts = ", ".join(sorted(getattr(report, "formats", []) or [])) or "unknown"
        layers = getattr(report, "summary", {}).get("quantized_layers")
        detail = f"format: {fmts}"
        if layers is not None:
            detail += f" | quantized layers: {layers}"
        issues.append(Issue("info", "File looks valid", detail))
    return issues


def _build_results_card_shared():
    """Shared construction seam for the ResultsCard (plan 2026-09-08-run-footer).

    Lazy import inside :mod:`quantui.panels` historically avoided a cycle; the
    footer now needs the same seam, so it lives here next to the widget."""
    return ResultsCard()


class ResultsCard(Vertical):
    """Run-outcome summary card (plan S2.2).

    Since the 2026-09-08-run-footer plan the card lives in the run footer and
    its ``[Copy path] / [Open folder]`` button row (``#result_buttons``) is
    revealed ONLY when the displayed record's status is ``success`` — hidden
    while idle, running, failed or stopped (user request 2026-09-08).
    """

    DEFAULT_CSS = """
    ResultsCard { height: auto; margin-top: 1; padding: 0 1;
                  border: round $panel-darken-1; }
    ResultsCard > Label { height: auto; margin-top: 0; text-style: none; width: 1fr; }
    ResultsCard .results_outcome { text-style: bold; }
    ResultsCard .results_error { color: $error; }
    ResultsCard .results_ok { color: $success; }
    ResultsCard .results_stopped { color: $warning; }
    ResultsCard .results_muted { color: $text-muted; }
    ResultsCard .issue_row { color: $text; }
    ResultsCard .issue_error { color: $error; }
    ResultsCard .issue_warning { color: $warning; }
    ResultsCard .issue_hint { color: $text-muted; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(id=ids.RESULTS_CARD.lstrip("#"), **kwargs)

    def compose(self):
        yield Label("No runs yet.", classes="results_outcome results_muted",
                    id=ids.RESULT_OUTCOME.lstrip("#"))
        yield Label("", classes="results_muted", id=ids.RESULT_PATH.lstrip("#"))
        yield Label("", classes="results_muted", id=ids.RESULT_META.lstrip("#"))
        from textual.containers import Horizontal

        yield Horizontal(
            Button("Copy path", id=ids.COPY_OUT_PATH.lstrip("#"), variant="default"),
            Button("Open folder", id=ids.OPEN_OUT_FOLDER.lstrip("#"), variant="default"),
            Button("Close ✕", id=ids.CLOSE_FOOTER.lstrip("#"), variant="default"),
            id=ids.RESULT_BUTTONS.lstrip("#"),
            classes="log_buttons",
        )

    # ---- record rendering (S2.2) ---------------------------------------------
    def show_record(self, record) -> None:
        """Render one :class:`profiles_store.RunRecord` into the card.

        Best-effort on freshly mounted cards (children filled via _pending).
        """
        self._pending_record = record
        try:
            outcome = self.query_one(f"#{ids.RESULT_OUTCOME.lstrip('#')}", Label)
            path_lbl = self.query_one(f"#{ids.RESULT_PATH.lstrip('#')}", Label)
            meta_lbl = self.query_one(f"#{ids.RESULT_META.lstrip('#')}", Label)
        except NoMatches:
            return  # not composed yet; on_mount replays _pending_record
        status = getattr(record, "status", "")
        cls = {"success": "results_ok", "failed": "results_error",
               "stopped": "results_stopped"}.get(status, "results_muted")
        icon = {"success": "Done \u2713", "failed": "Failed \u2717",
                "stopped": "Stopped"}.get(status, status or "--")
        outcome.update(f"{icon}")
        outcome.set_class(False, "results_muted")
        for c in ("results_ok", "results_error", "results_stopped"):
            outcome.set_class(cls == c, c)
        out = getattr(record, "output", "") or "(no output path)"
        path_lbl.update(out)
        bits = []
        dur = getattr(record, "duration_s", 0.0)
        if dur:
            bits.append(f"{dur:.1f}s")
        bits.append(f"exit {getattr(record, 'exit_code', '?')}")
        bits.append(str(getattr(record, "ts", "")))
        meta_lbl.update(" | ".join(bits))
        # S2.2 (plan 2026-09-08-run-footer): [Copy path]/[Open folder] appear
        # ONLY on a successful quantization (user request 2026-09-08).
        try:
            self.query_one(f"#{ids.RESULT_BUTTONS.lstrip('#')}").display = (
                status == "success"
            )
        except NoMatches:
            pass  # not composed yet

    def on_mount(self) -> None:
        # S2.2 (plan 2026-09-08-run-footer): the Copy/Open button row is hidden
        # until a SUCCESSFUL record lands (show_record gates it).
        try:
            self.query_one(f"#{ids.RESULT_BUTTONS.lstrip('#')}").display = False
        except NoMatches:
            pass  # not composed yet
        pending = getattr(self, "_pending_record", None)
        if pending is not None:
            self.show_record(pending)

    @property
    def output_path(self) -> str:
        """The currently displayed output path ('' when none)."""
        try:
            lbl = self.query_one(f"#{ids.RESULT_PATH.lstrip('#')}", Label)
            return str(lbl.content or "")
        except NoMatches:
            return ""

    def clear_issues(self) -> None:
        """Remove all issue rows (before a new validation pass)."""
        for row in list(self.query(".issue_row, .issue_error, .issue_warning, .issue_hint")):
            row.remove()

    def show_issues(self, issues: list[Issue]) -> None:
        """Render structured validation issue rows (S2.3), newest section at top."""
        self.clear_issues()
        anchor = self.query_one(f"#{ids.RESULT_OUTCOME.lstrip('#')}", Label)
        for issue in reversed(issues):  # insert above the outcome line
            hint_text = f" — {issue.hint}" if issue.hint else ""
            lbl = Label(
                f"[{issue.level.upper()}] {issue.text}{hint_text}",
                classes=f"issue_row issue_{issue.level}",
                markup=False,
            )
            self.mount(lbl, before=anchor)

    class OpenFolder(Message):
        """Posted when the user clicks [Open folder]; carries the folder path."""

        def __init__(self, folder: str) -> None:
            super().__init__()
            self.folder = folder

    class CloseRequested(Message):
        """Posted when the user clicks [Close ✕]: the app hides the run footer."""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        path = self.output_path
        if bid == ids.CLOSE_FOOTER.lstrip("#"):
            self.post_message(self.CloseRequested())
        elif bid == ids.COPY_OUT_PATH.lstrip("#"):
            if path:
                self.app.copy_to_clipboard(path)
                self.notify("Output path copied.")
        elif bid == ids.OPEN_OUT_FOLDER.lstrip("#"):
            if path:
                folder = (
                    os.path.dirname(path)
                    if os.path.splitext(path)[1]
                    else path
                )
                self.post_message(self.OpenFolder(folder))
