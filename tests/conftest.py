"""Shared helpers for the headless Textual test suite.

Panel widgets are created during ``compose``, and a single ``pilot.pause()``
after entering ``run_test()`` is not reliably enough to observe that mount on a
loaded or slow machine. Several tests failed on the GitHub Actions runner with
``NoMatches`` for exactly this reason, so the suite waits for the widget it
needs instead of assuming the mount already happened.
"""

from __future__ import annotations

import asyncio

MOUNT_ATTEMPTS = 20


async def wait_mounted(app, pilot, selector: str) -> None:
    """Poll until ``selector`` exists on the app's current screen.

    Silently gives up after ``MOUNT_ATTEMPTS`` pauses so the caller's own
    ``query_one`` raises the usual, more informative ``NoMatches``.
    """
    for _ in range(MOUNT_ATTEMPTS):
        await pilot.pause()
        try:
            app.screen.query_one(selector)
            return
        except Exception:
            continue


def run(coro):
    """Run an async test body on a fresh event loop."""
    return asyncio.run(coro)
