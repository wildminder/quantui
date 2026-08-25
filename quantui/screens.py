"""Modal screens (T06).

``PathModal`` is moved verbatim from ``app.py`` -- a folder/file picker built on
``DirectoryTree``. It is a self-contained ``ModalScreen`` (no app state), so it lives
here as ``screens.PathModal``. The browse handlers in ``app.py`` push ``screens.PathModal``.
Widget ids inside the modal (#tree / #cancel / #use) are preserved exactly.
"""

import os

from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Label,
)


class PathModal(ModalScreen):
    """Folder/file picker built on DirectoryTree."""

    def __init__(self, start: str, file_mode: bool = False) -> None:
        super().__init__()
        self.start = start or os.path.expanduser("~")
        self.file_mode = file_mode
        self.selected = ""

    def compose(self) -> "object":
        tree = DirectoryTree(self.start, id="tree")
        yield Vertical(
            Label("Select a model folder" if not self.file_mode else "Select a folder or .safetensors file"),
            tree,
            Horizontal(
                Button("Cancel", id="cancel"),
                Button("Use Selected", id="use", disabled=True),
                classes="buttons",
            ),
            classes="modal",
        )

    def on_directory_tree_directory_selected(self, event: DirectoryTree.NodeSelected) -> None:
        self.selected = event.node.data.path
        self.query_one("#use", Button).disabled = False

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.selected = event.node.data.path
        self.query_one("#use", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss("")
        else:
            self.dismiss(self.selected)


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
