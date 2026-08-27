"""Modal screens (T06).

``PathModal`` is moved verbatim from ``app.py`` -- a folder/file picker built on
``DirectoryTree``. It is a self-contained ``ModalScreen`` (no app state), so it lives
here as ``screens.PathModal``. The browse handlers in ``app.py`` push ``screens.PathModal``.
Widget ids inside the modal (#tree / #cancel / #use) are preserved exactly;
newer ids: #path_entry (path-entry Input) and #drive_<d> (Windows drive
quick-jump buttons).
"""

import os
import string
from pathlib import Path

from textual import work
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Input,
    Label,
)

from .model_audit import AuditError, _human_bytes, audit, suggest_exclusions


class PathModal(ModalScreen):
    """Folder/file picker built on DirectoryTree.

    ``DirectoryTree`` cannot navigate above its root, which on Windows traps a
    picker opened with an empty/relative start inside one drive. Two escape
    hatches are provided:

    * a path-entry ``Input`` (``#path_entry``) pre-filled with the start path:
      Enter on an existing directory re-roots the tree there (Textual 8.2.8's
      ``DirectoryTree.path`` reactive reloads on assignment); in ``file_mode``
      Enter on an existing file selects it and enables ``#use``; nonexistent
      paths are ignored.
    * on Windows only, a row of drive quick-jump buttons (``#drive_<d>`` for
      every existing drive letter) that re-root the tree at ``<d>:\\``.

    Selecting a directory in the tree syncs its path back into ``#path_entry``.
    """

    def __init__(self, start: str, file_mode: bool = False) -> None:
        super().__init__()
        self.start = start or os.path.expanduser("~")
        self.file_mode = file_mode
        self.selected = ""

    def compose(self) -> "object":
        tree = DirectoryTree(self.start, id="tree")
        children: list = [
            Label("Select a model folder" if not self.file_mode else "Select a folder or .safetensors file"),
            Input(value=self.start, placeholder="type a path and press Enter", id="path_entry"),
        ]
        if os.name == "nt":
            # Windows drive quick-jump: DirectoryTree cannot navigate above its
            # root, so a button per existing drive letter re-roots the tree.
            drives = [
                Button(f"{d}:", id=f"drive_{d}", classes="drive")
                for d in string.ascii_uppercase
                if os.path.exists(f"{d}:\\")
            ]
            if drives:
                children.append(Horizontal(*drives, id="drive_row"))
        children.extend([
            tree,
            Horizontal(
                Button("Cancel", id="cancel"),
                Button("Use Selected", id="use", disabled=True),
                classes="buttons",
            ),
        ])
        yield Vertical(*children, classes="modal")

    def _reroot_tree(self, path: str) -> None:
        """Re-root the DirectoryTree at ``path`` (the ``path`` reactive reloads)."""
        tree = self.query_one("#tree", DirectoryTree)
        tree.path = Path(path)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in #path_entry: re-root on a directory, select a file in file_mode."""
        if event.input.id != "path_entry":
            return
        value = event.value.strip()
        if not value:
            return
        if os.path.isdir(value):
            self._reroot_tree(value)
        elif self.file_mode and os.path.isfile(value):
            self.selected = value
            self.query_one("#use", Button).disabled = False
        # Nonexistent paths are ignored (kept minimal by design).

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("drive_"):
            self._reroot_tree(f"{button_id[len('drive_')]}:\\")
        elif event.button.id == "cancel":
            self.dismiss("")
        else:
            self.dismiss(self.selected)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.NodeSelected) -> None:
        self.selected = event.node.data.path
        self.query_one("#path_entry", Input).value = str(event.node.data.path)
        self.query_one("#use", Button).disabled = False

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.selected = event.node.data.path
        self.query_one("#use", Button).disabled = False


class ConfirmModal(ModalScreen):
    """Yes/no confirmation modal (plan S1.10: stop-confirm above threshold).

    ``dismiss(True)`` on confirm, ``dismiss(False)`` on cancel/Escape.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, message: str, confirm_label: str = "Confirm",
                 cancel_label: str = "Cancel") -> None:
        super().__init__()
        self.message = message
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label

    def compose(self) -> "object":
        yield Vertical(
            Label(self.message),
            Horizontal(
                Button(self.cancel_label, id="confirm_cancel", variant="default"),
                Button(self.confirm_label, id="confirm_ok", variant="error"),
                classes="buttons",
            ),
            classes="modal confirm",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm_ok":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_cancel(self) -> None:
        self.dismiss(False)


class NameModal(ModalScreen):
    """Text-input modal (plan S2.4): ask for a profile name, dismiss(str|None).

    ``dismiss(None)`` on cancel/Escape; ``dismiss(name)`` with a stripped
    non-empty name on OK / Enter.
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, title: str = "Save profile as…") -> None:
        super().__init__()
        self.title_text = title

    def compose(self) -> "object":
        from textual.widgets import Input

        yield Vertical(
            Label(self.title_text),
            Input(placeholder="profile name", id="profile_name"),
            Horizontal(
                Button("Cancel", id="name_cancel"),
                Button("Save", id="name_ok", variant="success"),
                classes="buttons",
            ),
            classes="modal confirm",
        )

    def on_input_submitted(self, event) -> None:
        name = event.input.value.strip()
        if name:
            self.dismiss(name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "name_ok":
            inp = self.query_one("#profile_name")
            name = inp.value.strip()
            if name:
                self.dismiss(name)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class RecentJobsScreen(ModalScreen):
    """Recent-jobs table (plan S2.5): ts / family / method / status rows.

    Activating a row dismisses with the record dict; Escape dismisses None.
    """

    BINDINGS = [("escape", "cancel", "Close")]

    COLUMNS = ("ts", "family", "method", "status")

    def __init__(self, records: list[dict]) -> None:
        super().__init__()
        self.records = list(records)

    def compose(self) -> "object":
        from textual.containers import VerticalScroll
        from textual.widgets import DataTable

        table = DataTable(id="recents_table")
        table.cursor_type = "row"
        yield Vertical(
            Label("Recent jobs — Enter to re-run"),
            VerticalScroll(table),
            classes="modal",
        )

    def on_mount(self) -> None:
        from textual.widgets import DataTable

        table = self.query_one("#recents_table", DataTable)
        table.add_columns(*self.COLUMNS)
        for rec in self.records:
            table.add_row(
                str(rec.get("ts", "")),
                str(rec.get("family", "")),
                str(rec.get("method", "")),
                str(rec.get("status", "")),
                key=str(rec.get("ts", "")) + str(rec.get("output", "")),
            )
        table.focus()

    def on_data_table_row_selected(self, event) -> None:
        idx = getattr(event, "cursor_row", getattr(event, "row_key", None).row or 0)
        if 0 <= idx < len(self.records):
            self.dismiss(self.records[idx])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AuditScreen(ModalScreen):
    """Model-audit modal (plan 2026-08-27, STEP 4.1).

    Runs :func:`quantui.model_audit.audit` synchronously in a worker
    thread (header-only, so it is fast even for huge checkpoints; the
    dispatcher accepts a single ``.safetensors`` file or a HuggingFace
    sharded model folder) and posts
    the result back to the UI: a summary header line, a ``DataTable`` with the
    per-module table (module / tensors / params / bytes / linears), and the
    suggested ``exclude_layers`` regex in a read-only, selectable ``Input``
    (``#audit_suggestion``). On :class:`AuditError` the modal shows the error
    text instead of a table.
    """

    BINDINGS = [("escape", "cancel", "Close")]

    def __init__(self, path: str) -> None:
        super().__init__()
        self.audit_path = path

    def compose(self) -> "object":
        table = DataTable(id="audit_table")
        table.cursor_type = "row"
        yield Vertical(
            Label(f"Model audit: {os.path.basename(self.audit_path)}", id="audit_title"),
            Label("", id="audit_summary"),
            Label("", id="audit_error"),
            table,
            Label("Suggested exclude_layers (select to copy):"),
            # NOTE: Textual 8.2.8's Input has no read_only mode; a plain Input
            # is used so the regex stays selectable/copyable (the plan's core
            # requirement). Editing is harmless — the value is advisory.
            Input(value="", id="audit_suggestion"),
            Horizontal(
                Button("Close", id="audit_close"),
                classes="buttons",
            ),
            classes="modal",
        )

    def on_mount(self) -> None:
        error_label = self.query_one("#audit_error", Label)
        error_label.display = False
        self._run_audit()

    @work(thread=True, exclusive=True, group="audit")
    def _run_audit(self) -> None:
        """Worker-thread body: audit the file and post the result to the UI."""
        try:
            report = audit(self.audit_path)
            suggestion = suggest_exclusions(report)
        except AuditError as exc:
            self.app.call_from_thread(self._show_error, str(exc))
            return
        self.app.call_from_thread(self._show_report, report, suggestion)

    def _show_report(self, report, suggestion) -> None:
        """Main-thread: populate the summary, table, and suggestion box."""
        total_params = 0
        for info in report.tensors:
            params = 1
            for dim in info.shape:
                params *= dim
            total_params += params
        summary = self.query_one("#audit_summary", Label)
        summary.update(
            f"Tensors: {len(report.tensors)} | Params: {total_params:,} | "
            f"Bytes: {_human_bytes(report.total_bytes)}"
            + (
                f" | Quantized layers: {len(report.quantized_layers)}"
                if report.quantized_layers
                else ""
            )
        )
        table = self.query_one("#audit_table", DataTable)
        table.add_columns("module", "tensors", "params", "bytes", "linears")
        for module in report.modules:
            table.add_row(
                module.module,
                str(module.tensors),
                f"{module.params:,}",
                _human_bytes(module.nbytes),
                str(module.category_counts.get("linear", 0)),
            )
        self.query_one("#audit_suggestion", Input).value = suggestion.regex

    def _show_error(self, message: str) -> None:
        """Main-thread: error state — hide the table, show the error text."""
        self.query_one("#audit_table", DataTable).display = False
        error_label = self.query_one("#audit_error", Label)
        error_label.update(f"Audit failed: {message}")
        error_label.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "audit_close":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
