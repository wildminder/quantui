"""Command-palette provider, extracted from ``quantui/app.py`` [IMP-001 S3B.2].

``QuantCommands`` is moved here verbatim; it stays duck-typed (accesses the
screen via ``self.screen``, no import of QuantApp) so the module has no
dependency on the app. ``quantui.app`` re-imports the class and keeps the
post-definition wiring so ``QuantApp.COMMANDS`` still contains it.
"""

from textual.command import DiscoveryHit, Hit, Provider
from textual.css.query import NoMatches


class QuantCommands(Provider):
    """Command-palette provider (plan S3.1): app actions + focus jumps.

    Built-in Ctrl+P opens the palette in Textual 8; this provider supplies the
    entries. ``matches`` uses the framework Matcher so fuzzy-finding works.
    """

    _ACTIONS = [
        ("Toggle log", "action_toggle_log", "Show/hide the bottom log drawer"),
        ("Open log file", "action_open_log_file", "Open the full per-run log"),
        ("Save profile", "action_save_profile", "Save current params (Ctrl+S)"),
        ("Show recents", "action_show_recents", "Recent jobs, re-run (Ctrl+R)"),
        ("Run quantization", "_palette_run", "Start a run with current params"),
        ("GGUF family", "action_family_gguf", "Switch to the GGUF family (1)"),
        ("ComfyUI family", "action_family_comfy", "Switch to ComfyUI family (2)"),
    ]

    _JUMPS = [
        ("Focus model path", "#model"),
        ("Focus output path", "#output"),
        ("Focus ctq input", "#ctq_input"),
        ("Focus method select", "#method"),
    ]

    async def discover(self):
        for title, action, help_ in self._ACTIONS:
            yield DiscoveryHit(title, lambda a=action: self._run(a), help=help_)
        for title, target in self._JUMPS:
            yield DiscoveryHit(
                title, lambda t=target: self._focus(t), help=f"jump to {target}"
            )

    async def search(self, query: str):
        matcher = self.matcher(query)
        for title, action, help_ in self._ACTIONS:
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title),
                          lambda a=action: self._run(a), text=title, help=help_)
        for title, target in self._JUMPS:
            score = matcher.match(title)
            if score > 0:
                yield Hit(score, matcher.highlight(title),
                          lambda t=target: self._focus(t), text=title)

    def _run(self, action: str) -> None:
        if action == "_palette_run":
            self._palette_run()
        else:
            getattr(self.screen, action)()

    def _palette_run(self) -> None:
        # Same morphing entry point as the Run buttons.
        screen = self.screen
        if getattr(screen, "_run_active", False):
            screen.action_stop_run()
        else:
            screen.action_run()

    def _focus(self, selector: str) -> None:
        try:
            widget = self.screen.query_one(selector)
            widget.focus()
        except NoMatches:
            pass  # target widget not mounted
