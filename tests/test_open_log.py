"""Headless tests for `o` open-authoritative-log-file (plan S1.6).

The launcher is factored into the module function ``quantui.app.open_in_editor``
so tests monkeypatch it -- no test ever spawns a real editor.
"""


from quantui import app as appmod


async def test_open_log_calls_editor(tmp_path, monkeypatch):
    """`o` opens the CURRENT per-run temp-file log via open_in_editor."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        a.log_msg("hello")
        called = []
        monkeypatch.setattr(appmod, "open_in_editor", lambda p: called.append(p))
        a.action_open_log_file()
        assert called == [a._run_log_path]


async def test_open_log_empty_notifies(tmp_path, monkeypatch):
    """Empty log -> notify only; the editor is NOT launched."""
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        notified = []
        monkeypatch.setattr(a, "notify", lambda *args, **kw: notified.append(args))
        launched = []
        monkeypatch.setattr(appmod, "open_in_editor", lambda p: launched.append(p))
        a.action_open_log_file()
        assert len(notified) == 1
        assert launched == []


async def test_open_log_no_editor_notifies(tmp_path, monkeypatch):
    """A failing launcher must degrade to a toast, never raise into the UI."""
    import os
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("something")
        if os.name != "nt":
            # On POSIX simulate "no editor configured".
            monkeypatch.delenv("EDITOR", raising=False)
            monkeypatch.delenv("PAGER", raising=False)
        else:
            def boom(p):
                raise RuntimeError("no startfile")
            monkeypatch.setattr(appmod, "open_in_editor", boom)
        notified = []
        monkeypatch.setattr(a, "notify", lambda *args, **kw: notified.append(args))
        a.action_open_log_file()
        await pilot.pause()
        assert len(notified) == 1
