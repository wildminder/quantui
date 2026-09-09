"""The 'quantui-cyber' app theme (plan 2026-09-08-scifi-ui S2.1; palette v2 in
plan 2026-09-09-control-panel S1.1).

Pure module: builds the Textual Theme instance. Kept free of any App import
so tests can pin the exact color values without booting the TUI. The App
registers it in ``QuantApp.__init__`` and defaults to it.

Design language (palette v2, "control room slate"): professional slate/navy
surfaces at 14-24% luminance (One Dark / Tokyo-Night family — no near-black),
muted steel-cyan primary, soft violet accent. Dark-only (light terminals fall
back through Textual's ANSI handling).
"""

from textual.theme import Theme


def build_cyber_theme() -> Theme:
    """Build the 'quantui-cyber' Theme (deterministic: fixed hex values).

    Colors are the single source of truth for the look — the MAIN_CSS rules
    reference theme variables, so a retune here propagates everywhere.
    tests/test_theme_cyber.py pins these values (and enforces the
    no-near-black luminance floor for the surface stack).
    """
    return Theme(
        name="quantui-cyber",
        primary="#4dc3ff",  # steel cyan — inputs, primary structure
        secondary="#7c4dff",  # violet — secondary emphasis
        accent="#c792ea",  # soft violet — focus ring
        success="#5fd7a0",  # muted terminal green — Run button / success states
        warning="#e5c07b",  # muted amber
        error="#e06c75",  # muted red (One Dark family)
        # Surface stack: plan §0.3 targets 14-24% perceived luminance with a
        # hard no-near-black floor at 12% (pinned by test_no_near_black). The
        # plan's draft hexes measured 10.6-17.9% (its own floor failed), so the
        # stack keeps the identical hue ramp shifted to actually hit the stated
        # targets: 14.2% / 19.0% / 22.3% / 25.8%.
        background="#1c2536",  # slate navy (~14% luminance)
        surface="#253246",  # raised field background (~19%)
        panel="#2d3a51",  # panel / section fills (~22%)
        boost="#35435c",  # hover / boost visible (~26%)
        dark=True,
    )
