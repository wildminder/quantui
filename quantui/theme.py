"""The 'quantui-cyber' app theme (plan 2026-09-08-scifi-ui S2.1).

Pure module: builds the Textual Theme instance. Kept free of any App import
so tests can pin the exact color values without booting the TUI. The App
registers it in ``QuantApp.__init__`` and defaults to it.

Design language: deep-space blue surfaces + electric-cyan primary + magenta
accent. Dark-only (light terminals fall back through Textual's ANSI handling).
"""

from textual.theme import Theme


def build_cyber_theme() -> Theme:
    """Build the 'quantui-cyber' Theme (deterministic: fixed hex values).

    Colors are the single source of truth for the sci-fi look — the MAIN_CSS
    sci-fi rules (S2.2/S2.3) reference theme variables, so a retune here
    propagates everywhere. tests/test_theme_cyber.py pins these values.
    """
    return Theme(
        name="quantui-cyber",
        primary="#00e5ff",  # electric cyan — inputs, primary structure
        secondary="#7c4dff",  # violet — secondary emphasis
        accent="#ff2d95",  # magenta — focus "power-up" ring
        success="#00ffa3",  # neon green — Run button / success states
        warning="#ffb020",
        error="#ff3d5a",
        background="#050b14",  # deep space
        surface="#0a1424",
        panel="#101d33",
        boost="#16253f",
        dark=True,
    )
