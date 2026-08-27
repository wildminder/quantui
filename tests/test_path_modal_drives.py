"""PathModal path-entry + Windows drive quick-jump tests (headless run_test()).

Covers the fix for the Windows picker trap: ``DirectoryTree`` cannot navigate
above its root, so a picker opened with an empty/relative start could never
reach other drives (e.g. ``X:\\_Models`` or ``D:\\``). The modal now carries a
path-entry ``Input`` (Enter re-roots the tree / selects a file in file mode)
and, on Windows only, a row of drive quick-jump buttons.

Follows the ``run_test()`` async pattern from ``tests/test_app_headless.py``
(pytest.ini sets ``asyncio_mode = auto``).
"""

import os
import string
from pathlib import Path

import pytest
from textual.widgets import Button, DirectoryTree, Input

from quantui import app as appmod
from quantui.comfy_quant_schema import write_safetensors
from quantui.screens import PathModal


async def _push_modal(pilot, app, start: str, file_mode: bool = False) -> PathModal:
    """Push a PathModal and wait until it is mounted."""
    modal = PathModal(start, file_mode=file_mode)
    app.push_screen(modal)
    await pilot.pause()
    assert app.screen_stack[-1] is modal
    return modal


async def test_path_entry_reroots_tree(tmp_path):
    """Enter on an existing directory in #path_entry re-roots the tree there."""
    subdir = tmp_path / "nested" / "models"
    subdir.mkdir(parents=True)

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path))
        entry = modal.query_one("#path_entry", Input)
        tree = modal.query_one("#tree", DirectoryTree)

        entry.value = str(subdir)
        await entry.action_submit()
        await pilot.pause()

        assert Path(tree.path).resolve() == subdir.resolve()


async def test_path_entry_file_mode_selects_file(tmp_path):
    """file_mode: Enter on an existing file selects it and enables #use."""
    fixture = tmp_path / "model.safetensors"
    fixture.write_bytes(
        write_safetensors({"a.q_proj.weight": ("BF16", [2, 2], b"\x00" * 8)})
    )

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path), file_mode=True)
        entry = modal.query_one("#path_entry", Input)

        entry.value = str(fixture)
        await entry.action_submit()
        await pilot.pause()

        assert modal.selected == str(fixture)
        assert modal.query_one("#use", Button).disabled is False


async def test_path_entry_nonexistent_ignored(tmp_path):
    """Enter on a nonexistent path is a no-op (tree root and #use unchanged)."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path))
        entry = modal.query_one("#path_entry", Input)
        tree = modal.query_one("#tree", DirectoryTree)
        before = Path(tree.path)

        entry.value = str(tmp_path / "no" / "such" / "path")
        await entry.action_submit()
        await pilot.pause()

        assert Path(tree.path) == before
        assert modal.query_one("#use", Button).disabled is True


async def test_tree_selection_syncs_path_entry(tmp_path):
    """Selecting a directory in the tree syncs its path into #path_entry."""
    subdir = tmp_path / "child"
    subdir.mkdir()

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path))
        tree = modal.query_one("#tree", DirectoryTree)
        await pilot.pause()

        # Expand the root and select the child directory node.
        tree.root.expand()
        for _ in range(50):
            if tree.root.children:
                break
            await pilot.pause()
        target = None
        for node in tree.root.children:
            if Path(node.data.path) == subdir:
                target = node
                break
        assert target is not None
        tree.post_message(DirectoryTree.DirectorySelected(target, subdir))
        await pilot.pause()

        entry = modal.query_one("#path_entry", Input)
        assert Path(entry.value).resolve() == subdir.resolve()


@pytest.mark.skipif(os.name != "nt", reason="drive buttons are Windows-only")
async def test_drive_buttons_windows_only(tmp_path):
    """On Windows: one #drive_<d> button per existing drive; pressing re-roots."""
    existing = [d for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    assert existing, "test host must have at least one drive"

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path))
        for d in existing:
            assert modal.query_one(f"#drive_{d}", Button) is not None
        # No button for a drive that does not exist.
        missing = [d for d in string.ascii_uppercase if d not in existing]
        for d in missing[:3]:
            assert list(modal.query(f"#drive_{d}").results(Button)) == []

        # Pressing a drive button re-roots the tree at that drive.
        target = existing[0]
        modal.query_one(f"#drive_{target}", Button).press()
        await pilot.pause()
        tree = modal.query_one("#tree", DirectoryTree)
        assert Path(tree.path) == Path(f"{target}:\\")


@pytest.mark.skipif(os.name == "nt", reason="non-Windows hosts only")
async def test_no_drive_buttons_off_windows(tmp_path):
    """Off Windows the drive quick-jump row is not mounted at all."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(pilot, a, str(tmp_path))
        assert list(modal.query('[id^="drive_"]').results(Button)) == []
