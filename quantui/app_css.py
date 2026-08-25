"""Application stylesheet, extracted from ``QuantApp.CSS`` verbatim [IMP-001 S3B.1].

Single source of truth for the TUI stylesheet. ``quantui.app`` imports
``MAIN_CSS`` and aliases it as ``QuantApp.CSS`` so existing references
(pinned tests included) keep working unchanged.
"""

MAIN_CSS = """
    #body { height: 1fr; }
    #params { width: 70%; height: 1fr; border-right: solid $panel-darken-1; padding: 1 2; }
    #rail { width: 30%; height: 1fr; padding: 0 1; }
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
    #method_info { height: 3; margin-bottom: 1; color: $text-muted; }
    #status { height: 1; background: $panel; }
    RichLog { height: 8; border: round $panel-darken-1; }
    Button { height: 3; }
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
    Input.-invalid { border: tall ansi_red; }
    #header_strip { height: 1; background: $panel; }
    #header_strip > ProgressBar { width: 1fr; }
    #header_strip > Label { margin-top: 0; text-style: none; color: $text-muted; width: auto; padding: 0 1; }
    """
