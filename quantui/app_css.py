"""Application stylesheet, extracted from ``QuantApp.CSS`` verbatim [IMP-001 S3B.1].

Single source of truth for the TUI stylesheet. ``quantui.app`` imports
``MAIN_CSS`` and aliases it as ``QuantApp.CSS`` so existing references
(pinned tests included) keep working unchanged.
"""

MAIN_CSS = """
    #body { height: 1fr; }
    #params { width: 100%; height: 1fr; padding: 1 2; }
    # Run footer (plan 2026-09-08-run-footer): the master style — hidden until a
    # run starts (QuantApp._show_run_footer flips display on).
    #run_footer { height: auto; max-height: 25; display: none; padding: 0 1;
                 border-top: solid $panel-darken-1; background: $surface-darken-1; }
    #run_footer > #footer_left { width: 1fr; height: auto; padding: 0 1 0 0; }
    #run_footer > #results_card { width: 50; margin-top: 0; }
    # F2-S1.2 (plan 2026-09-08-footer-v2): two-state footer. Done mode hides the
    # left panel via display toggling and stretches the card to full width.
    #run_footer.mode-done > #results_card { width: 100%; }
    #footer_bar { width: 1fr; height: 3; margin-bottom: 1; }
    # Stats line as a boxed readout (plan 2026-09-09-control-panel S2.2):
    # round panel border + surface fill. The explicit height: 3 matters — a
    # border box at height 2 spends BOTH rows on the border and clips its text.
    #footer_stats { border: round $panel; background: $surface; padding: 0 1;
                   height: 3; margin-top: 0; text-style: none; color: $text-muted; }
    # Phase chips as pills (S2.2): the rail row grows to the 3-row box
    # (height: auto — the old fixed 1 clipped the border), the label hugs its
    # text with a max-width cap so long labels fill the row instead of
    # overflowing it.
    .rail_row { height: auto; }
    .rail_label { border: round $panel; background: $surface; padding: 0 1;
                  height: 3; width: auto; max-width: 100%; }
    # Radio buttons keep a readable, theme-driven label color (S2.2).
    RadioSet > RadioButton { color: $text; }
    .field { height: auto; }
    .field Input { width: 1fr; }
    .field Button { width: auto; }
    .modal { width: 70%; height: 70%; border: round $accent; background: $surface; }
    .modal DirectoryTree { height: 1fr; }
    .buttons { height: auto; align: right middle; }
    .log_buttons { height: auto; }
    .log_follow { height: 1; margin: 0 1 0 0; text-style: none; color: $text-muted; width: auto; }
    Input { margin-bottom: 1; height: 3; }
    Label { margin-top: 1; text-style: bold; }
    # Section headers (plan 2026-09-09-control-panel S2.1): the numbered
    # section labels read as instrument-group strips — panel-tinted, bold,
    # $text (not muted), full width. The bare Label rule above stays for
    # non-section labels.
    .section_header { background: $panel; color: $text; text-style: bold;
                      padding: 0 1; margin-top: 1; width: 1fr; }
    # Sci-fi inputs (plan 2026-09-08-scifi-ui S2.2): electric-cyan frame from
    # the theme's primary; focusing "powers up" the border to the accent hue.
    # The -invalid state stays error-red so validation remains unambiguous.
    Input { border: round $primary; background: $surface; }
    Input:focus { border: round $accent; text-style: bold; }
    Input.-invalid { border: round $error; }
    # F2-S1.1 (plan 2026-09-08-footer-v2): the collapsed Advanced section sits
    # directly above the Run button — Textual's Collapsible has no bottom
    # margin, so the button visually sticks to it. One line fixes both panels.
    Collapsible { margin-bottom: 1; }
    #method_info { height: 3; margin-bottom: 1; color: $text-muted; }
    #status { height: 1; background: $panel; }
    RichLog { height: 8; border: round $panel-darken-1; }
    Button { height: 3; }
    # Sci-fi buttons (plan 2026-09-08-scifi-ui S2.3): bold labels; the widget
    # that owns the focus ring "powers up" to the accent hue (same rule as
    # inputs). Variant colors (success/warning/default) come from the theme.
    Button { text-style: bold; }
    Button:focus, Button:hover { border: round $accent; }
    #family { height: auto; margin: 1 0; }
    .hidden { display: none; }
    #ctq_cap_warn { height: auto; margin-top: 1; color: $text-muted; }
    #ctq_cap_warn.warn { color: $warning; text-style: bold; }
    .capwarn { color: $text-muted; }
    # Log drawer: the drawer itself is the sized element (S/M/L presets); the log
    # fills ALL remaining space (height: 1fr) and the button row is docked to the
    # bottom, so resizing never leaves an empty gap or clipped buttons.
    #log_drawer { dock: bottom; height: 24; display: none; border: round $panel-darken-1; background: $surface; padding: 0 1; }
    #log_drawer > #log { height: 1fr; border: none; }
    #log_drawer > .log_buttons { dock: bottom; height: auto; }
    .field_hint_visible { height: 1; margin-top: 0; text-style: none; color: $error; }
    # The legacy tall ansi_red invalid rule was consolidated into the S2.2
    # sci-fi rule above (Input.-invalid -> round $error) when the cyber theme
    # landed: one invalid style, themed color.
    # S1.7 header strip rules removed (post-v0.9.1 dedup): the run footer is
    # the single progress surface.
    """
