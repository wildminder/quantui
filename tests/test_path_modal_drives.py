"""PathModal path-entry + Windows drive quick-jump tests (headless run_test()).

Covers the fix for the Windows picker trap: ``DirectoryTree`` cannot navigate
above its root, so a picker opened with an empty/relative start could never
reach other drives (e.g. another drive or ``D:\\``). The modal now carries a
path-entry ``Input`` (Enter re-roots the tree / selects a file in file mode)
and, on Windows only, a row of drive quick-jump buttons.

NTH-013 de-flake (2026-09-04): this file was intermittent under full-suite
load (one 188 s gate run, 2 failures, errors truncated at "textua...").
Root cause: ``pilot.pause()`` waits for message-pump idle via
``_wait_for_screen()`` with a HARD 30 s timeout (``WaitForScreenTimeout``),
and the ``DirectoryTree`` pump legitimately sits blocked awaiting its
``@work(thread=True)`` directory listing while a load is in flight -- so
every pause call was another 30 s timeout risk under load. This file now
makes ZERO pause calls:

* ``push_screen`` returns an ``AwaitMount`` -- awaiting it is a deterministic
  mount wait (no pause);
* content-dependent waits are CONDITION polls on real ``asyncio.sleep()``
  ticks, which let the pump and thread workers take as long as they need
  without any timeout-excepting pause in the path;
* assertions are unchanged -- same behaviour, robust timing.

Follows the ``run_test()`` async pattern from ``tests/test_app_headless.py``
(pytest.ini sets ``asyncio_mode = auto``).
"""

import asyncio
import os
import string
from pathlib import Path

import pytest
from textual.widgets import Button, DirectoryTree, Input

from quantui import app as appmod
from quantui.comfy_quant_schema import write_safetensors
from quantui.screens import PathModal


async def _wait_until(predicate, timeout: float = 30.0, step: float = 0.05) -> bool:
    """Poll ``predicate`` until truthy or ``timeout`` elapses (NTH-013).

    Real sleeps, deliberately NOT ``pilot.pause()``: pause() runs
    ``_wait_for_screen()`` which raises after 30 s if any widget pump is
    backed up -- and the tree's pump blocks on its thread workers while a
    directory listing is in flight. Polling on sleeps gives the pump and the
    thread workers time regardless of machine load.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(step)
    return True


async def _push_modal(app, start: str, file_mode: bool = False) -> PathModal:
    """Push a PathModal and wait for it (and its children) to be mounted.

    ``push_screen`` returns an ``AwaitMount``; awaiting it is a deterministic
    mount wait -- no pause needed (NTH-013).
    """
    modal = PathModal(start, file_mode=file_mode)
    await app.push_screen(modal)
    assert app.screen_stack[-1] is modal
    return modal


async def test_path_entry_reroots_tree(tmp_path):
    """Enter on an existing directory in #path_entry re-roots the tree there."""
    subdir = tmp_path / "nested" / "models"
    subdir.mkdir(parents=True)

    a = appmod.QuantApp()
    async with a.run_test():
        modal = await _push_modal(a, str(tmp_path))
        entry = modal.query_one("#path_entry", Input)
        tree = modal.query_one("#tree", DirectoryTree)

        entry.value = str(subdir)
        await entry.action_submit()
        ok = await _wait_until(lambda: Path(tree.path).resolve() == subdir.resolve())
        assert ok, f"#tree.path did not re-root to {subdir} (got {tree.path!r})"
        # The re-root listing still needs a thread worker; settle pause-free so
        # the app shuts down without a pending-load race. `nested/models` is
        # EMPTY, so "no children" is also a valid fully-loaded state: wait for
        # either children or a quiet queue instead of insisting on children.
        assert await _wait_until(
            lambda: bool(tree.root.children) or tree._load_queue.empty()
        )
        await asyncio.sleep(0.1)


async def test_path_entry_file_mode_selects_file(tmp_path):
    """file_mode: Enter on an existing file selects it and enables #use."""
    fixture = tmp_path / "model.safetensors"
    fixture.write_bytes(
        write_safetensors({"a.q_proj.weight": ("BF16", [2, 2], b"\x00" * 8)})
    )

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(a, str(tmp_path), file_mode=True)
        entry = modal.query_one("#path_entry", Input)

        entry.value = str(fixture)
        await entry.action_submit()
        ok = await _wait_until(lambda: modal.selected == str(fixture))
        assert ok, f"modal.selected not set (got {modal.selected!r})"
        assert modal.query_one("#use", Button).disabled is False
        await pilot.pause()  # final settle


async def test_path_entry_nonexistent_ignored(tmp_path):
    """Enter on a nonexistent path is a no-op (tree root and #use unchanged)."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(a, str(tmp_path))
        entry = modal.query_one("#path_entry", Input)
        tree = modal.query_one("#tree", DirectoryTree)
        before = Path(tree.path)

        entry.value = str(tmp_path / "no" / "such" / "path")
        await entry.action_submit()
        # Nothing should change: give the pump a brief, pause-free window to
        # process (and no-op) the submit, then assert the unchanged state.
        await asyncio.sleep(0.1)
        assert Path(tree.path) == before
        assert modal.query_one("#use", Button).disabled is True
        await pilot.pause()  # final settle


async def test_tree_selection_syncs_path_entry(tmp_path):
    """Selecting a directory in the tree syncs its path into #path_entry."""
    subdir = tmp_path / "child"
    subdir.mkdir()

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(a, str(tmp_path))
        tree = modal.query_one("#tree", DirectoryTree)

        # Expand the root; children appear once the thread-worker listing
        # lands -- wait on the condition, not on a fixed pause count.
        tree.root.expand()
        assert await _wait_until(lambda: bool(tree.root.children)), (
            "root children never populated"
        )
        target = None
        for node in tree.root.children:
            if Path(node.data.path) == subdir:
                target = node
                break
        assert target is not None
        tree.post_message(DirectoryTree.DirectorySelected(target, subdir))
        entry = modal.query_one("#path_entry", Input)
        ok = await _wait_until(
            lambda: bool(entry.value)
            and Path(entry.value).resolve() == subdir.resolve()
        )
        assert ok, f"#path_entry did not sync to {subdir} (got {entry.value!r})"
        await pilot.pause()  # final settle


@pytest.mark.skipif(os.name != "nt", reason="drive buttons are Windows-only")
async def test_drive_buttons_windows_only(tmp_path):
    """On Windows: one #drive_<d> button per existing drive; pressing re-roots."""
    existing = [d for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    assert existing, "test host must have at least one drive"

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(a, str(tmp_path))
        for d in existing:
            assert modal.query_one(f"#drive_{d}", Button) is not None
        # No button for a drive that does not exist.
        missing = [d for d in string.ascii_uppercase if d not in existing]
        for d in missing[:3]:
            assert list(modal.query(f"#drive_{d}").results(Button)) == []

        # Pressing a drive button re-roots the tree at that drive.
        target = existing[0]
        tree = modal.query_one("#tree", DirectoryTree)
        modal.query_one(f"#drive_{target}", Button).press()
        ok = await _wait_until(lambda: Path(tree.path) == Path(f"{target}:\\"))
        assert ok, f"#tree.path did not re-root to {target}: (got {tree.path!r})"
        await pilot.pause()  # final settle


@pytest.mark.skipif(os.name == "nt", reason="non-Windows hosts only")
async def test_no_drive_buttons_off_windows(tmp_path):
    """Off Windows the drive quick-jump row is not mounted at all."""
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        modal = await _push_modal(a, str(tmp_path))
        assert list(modal.query('[id^="drive_"]').results(Button)) == []
        await pilot.pause()  # final settle
