"""GGUF + ComfyUI panel builders (T06).

These functions are moved verbatim from ``app.py`` (the pre-M1 GGUF block and the
data-driven ComfyUI block). They are pure widget construction -- no app state, no
``self`` -- so they live here as module-level builders. ``QuantApp.compose`` delegates
to them (see app.py). Widget ids are preserved exactly so the headless tests keep
passing.

The derived constants (DEFAULT_METHOD / DEFAULT_CTQ_FORMAT) mirror
``app.py``; they are computed from ``quant_methods`` data so this module does not
need to import the composition root (avoids a circular import).
"""

import sys

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import (
    Button,
    Checkbox,
    Collapsible,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
)

from .quant_methods import (
    COMFY_FORMATS,
    DEFAULT_GGUF_METHOD,
    UD_INFO_FOOTER,
    format_options,
    preset_options,
)

# --- derived constants (source of truth is quant_methods) ---------------------
# T8 (plan 2026-08-31-gguf-unsloth-parity): the (label, id) pairs are gone --
# the GGUF panel mounts a free-text #method Input, and the only remaining
# method Select is the wizard's own SELECT_METHOD_OPTIONS (screens_wizard.py).
DEFAULT_METHOD = DEFAULT_GGUF_METHOD
DEFAULT_CTQ_FORMAT = COMFY_FORMATS[0].id  # fp8_e4m3


class ProgressView(Vertical):
    """Live-progress widget: a label + a real (determinate) Textual ProgressBar.

    Composed as ``#live_progress`` (the id is preserved so the rest of the app and the
    tests keep working). It is always mounted at a constant height so the RichLog layout
    never jumps when progress appears / disappears. Legacy (text-only) progress shows the
    raw line in the label with the bar hidden; structured ``CTQ_PROGRESS`` envelopes fill
    the bar determinately from their (cur, total) / pct payload.

    Moved here verbatim from ``app.py`` (S0.3 compose-seam extraction): it is pure
    widget construction, and ``build_main_layout`` needs it without importing the
    composition root (which would be circular).
    """

    DEFAULT_CSS = """
    ProgressView { height: auto; min-height: 3; max-height: 9; }
    ProgressView > Label { height: 1; margin: 0 0 0 0; }
    ProgressView > ProgressBar { height: 1; margin: 0; }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._states: list = []

    def compose(self) -> ComposeResult:
        yield Label("", id="progress_label")
        yield ProgressBar(id="progress_bar", show_percentage=True)

    def on_mount(self) -> None:
        # Start empty: no progress yet, bar hidden. The bordered box stays (constant height).
        self.query_one("#progress_bar", ProgressBar).display = False

    def set_text(self, text: str) -> None:
        """Legacy path: show raw text only (no determinate bar)."""
        self._states = []
        self.query_one("#progress_label", Label).update(text)
        self.query_one("#progress_bar", ProgressBar).display = False

    def set_states(self, states: list) -> None:
        """Structured path: render the collapsed progress states (label + determinate bar)."""
        self._states = list(states)
        label = self.query_one("#progress_label", Label)
        bar = self.query_one("#progress_bar", ProgressBar)
        if not states:
            label.update("")
            bar.display = False
            return
        # Prefer the last *determinate* state to drive the bar (cur/total or pct).
        primary = None
        for st in states:
            if getattr(st, "determinate", False):
                primary = st
        summary = " | ".join(
            st.text for st in states if (getattr(st, "text", "") or "").strip()
        ) or states[-1].phase
        label.update(summary)
        if primary is not None:
            bar.display = True
            if primary.total:
                bar.update(total=primary.total, progress=min(primary.cur or 0, primary.total))
            elif primary.pct is not None:
                bar.update(total=100, progress=int(primary.pct))
        else:
            bar.display = False


class ProgressRailRow(Horizontal):
    """One stacked row of the ProgressRail: a phase label with counts (no bar).

    F2 fix (user report): the footer already has the wide aggregate bar
    (#footer_bar); a per-row ProgressBar rendered the SAME aggregate signal a
    second time — redundant. The row is now label-only: ``label [cur/total]``
    (the phase's own counts, NOT the aggregate pct). Clicking a row still
    posts :class:`ProgressRail.BarClicked` so the app can open the log drawer
    filtered to that phase (S1.9). The row fills its label in ``on_mount`` (a
    freshly ``mount()``-ed row has no composed children yet, so callers must
    NOT query into it before the mount completes).
    """

    DEFAULT_CSS = """
    ProgressRailRow { height: 1; margin-bottom: 0; }
    """

    def __init__(self, st) -> None:
        # NOTE: no widget id -- phase keys like "(#/211)" are not valid Textual ids.
        super().__init__(classes="rail_row")
        self.phase_key = st.phase
        self._state = st

    def compose(self) -> ComposeResult:
        yield Label("", classes="rail_label")

    def on_mount(self) -> None:
        self.update_state(self._state)

    def update_state(self, st) -> None:
        """Fill the label from one ProgressState.

        Best-effort on freshly mounted rows: a row's children only exist after
        its ``mount()`` completes, so a not-yet-composed row keeps ``_state``
        and fills itself in ``on_mount`` instead.
        """
        self._state = st
        try:
            lbl = self.query_one(".rail_label", Label)
        except NoMatches:
            return  # children not composed yet; on_mount will fill it
        lbl.update(st.text or st.phase)


class ProgressRail(Vertical):
    """Stacked phase labels, one per collapsed progress state (plan S1.8).

    Replaces the single ``#live_progress`` widget: every entry of
    ``LiveProgressStore.states()`` renders as its own label row (max
    ``MAX_ROWS`` visible; overflow collapses into a "+N more" label). Since
    the footer-v2 redundancy fix the rows are label-only — the wide aggregate
    bar (#footer_bar) is the single progress bar. Clicking a row posts
    :class:`BarClicked` with the phase key.
    """

    MAX_ROWS = 4

    class BarClicked(Message):
        """Posted when the user clicks one progress row."""

        def __init__(self, phase: str) -> None:
            super().__init__()
            self.phase = phase

    DEFAULT_CSS = """
    ProgressRail { height: auto; }
    ProgressRail .rail_label { height: 1; margin-top: 0; text-style: none;
                               width: 1fr; color: $text-muted; }
    ProgressRail .rail_more { height: 1; margin-top: 0; text-style: none;
                              color: $text-disabled; }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Phase-keyed row registry (phases are not always valid widget ids).
        self._rows: dict[str, ProgressRailRow] = {}

    def set_states(self, states: list) -> None:
        """Render one row per state (reconciled by phase key), max MAX_ROWS."""
        shown = states[: self.MAX_ROWS]
        seen: set[str] = set()
        for st in shown:
            seen.add(st.phase)
            row = self._rows.get(st.phase)
            if row is None:
                row = ProgressRailRow(st)
                self._rows[st.phase] = row
                self.mount(row)  # row fills itself in on_mount
            else:
                row.update_state(st)
        # Remove stale rows (phases no longer present).
        for key in [k for k in self._rows if k not in seen]:
            self._rows.pop(key).remove()
        # Overflow indicator.
        for more in list(self.query(".rail_more")):
            more.remove()
        if len(states) > self.MAX_ROWS:
            self.mount(
                Label(f"+{len(states) - self.MAX_ROWS} more", classes="rail_more")
            )

    def on_click(self, event) -> None:
        """Click any part of a row -> post BarClicked(phase)."""
        target = event.target
        while target is not None and not isinstance(target, ProgressRailRow):
            target = target.parent
        if isinstance(target, ProgressRailRow):
            self.post_message(self.BarClicked(target.phase_key))


class RunFooter(Horizontal):
    """Full-width run footer, TWO-MODE (plan 2026-09-08-footer-v2).

    Mode "progress" (while a run is active): the left panel (#footer_left)
    shows the wide aggregate bar (#footer_bar), the stats line (#footer_stats),
    the per-phase ProgressRail (#progress_rail) and the status Label (#status);
    the ResultsCard is HIDDEN. Mode "done" (after _finish_run_record lands the
    record): the ResultsCard REPLACES the progress panel entirely (full width)
    so the verdict appears exactly once. Widget ids are the historical rail ids
    (contract Q4a); master styling lives in MAIN_CSS (IMP-001 S3B.1).
    """

    MODES = ("progress", "done")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._mode = "progress"

    def set_mode(self, mode: str) -> None:
        """Switch the footer panel: 'progress' = left panel only; 'done' = card only."""
        if mode not in self.MODES:
            raise ValueError(f"unknown footer mode: {mode}")
        self._mode = mode
        self.set_class(mode == "done", "mode-done")
        try:
            self.query_one("#footer_left").display = mode == "progress"
            # ResultsCard import is lazy (cycle-avoidance, see compose); the id
            # selector reaches the same widget without the class reference.
            self.query_one("#results_card").display = mode == "done"
        except NoMatches:
            pass  # not composed yet; on_mount applies the initial state

    def compose(self) -> ComposeResult:
        from .widgets_results import _build_results_card_shared

        with Vertical(id="footer_left"):
            # The ONE progress bar. show_percentage=False: the stats line right
            # below renders the pct (a separate PercentageStatus sub-widget
            # overlapped the stats line otherwise).
            yield ProgressBar(id="footer_bar", show_percentage=False, show_eta=False)
            yield Label("--", id="footer_stats")
            yield ProgressRail(id="progress_rail")
            yield Label(id="status")
        yield _build_results_card_shared()

    def on_mount(self) -> None:
        # Initial state: progress panel only (card hidden until a run completes).
        self.set_mode("progress")


def build_main_layout(run_log_max_lines: int) -> Vertical:
    """The main body layout (plan 2026-09-08-run-footer, layout v2).

    #params (VerticalScroll) now spans the FULL width; the former right rail's
    widgets (ProgressRail #progress_rail, status #status, ResultsCard) live in
    the on-demand :class:`RunFooter` (#run_footer) stacked below it — hidden
    until a run starts. Since S1.4 the log lives in :func:`build_log_drawer`
    (hidden by default). All widget ids are preserved exactly (contract tests
    stay green; the layout-tree pin in test_widget_contract.py was rewritten
    for v2 in the same commit).
    """
    return Vertical(
        VerticalScroll(
            build_gguf_panel(),
            build_comfy_panel(),
            id="params",
        ),
        RunFooter(id="run_footer"),
        id="body",
    )


class PtSuggestBox(Horizontal):
    """".pt checkpoint detected" suggestion box (user feature).

    Shown above the ComfyUI input when ``#ctq_input`` points at a PyTorch
    checkpoint (.pt/.pth/.ckpt). The [Convert to safetensors] button launches
    the out-of-box converter worker; on success the input field is re-pointed
    at the produced .safetensors. Hidden by default; visibility is managed by
    ``HandlersMixin.update_pt_suggest``.
    """

    DEFAULT_CSS = """
    PtSuggestBox { height: auto; margin-bottom: 1; padding: 0 1;
                   border: round $warning; background: $surface-darken-1; }
    PtSuggestBox > Label { height: auto; margin-top: 0; text-style: none;
                           color: $text-muted; width: 1fr; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(id="pt_suggest", **kwargs)
        # Hidden until update_pt_suggest() detects a .pt input (avoids a flash
        # of the box on startup before the first visibility pass).
        self.display = False

    def compose(self) -> ComposeResult:
        yield Label("PyTorch checkpoint detected (.pt) — quantization needs "
                    "safetensors.", id="pt_suggest_label")
        yield Button("Convert to safetensors", id="pt_convert_btn", variant="warning")


# Drawer size presets S/M/L in rows (plan Q1): fixed presets, cycled with , / .
LOG_DRAWER_SIZES: dict[str, int] = {"S": 12, "M": 24, "L": 40}
LOG_DRAWER_ORDER: tuple[str, ...] = ("S", "M", "L")


class FollowRichLog(RichLog):
    """``#log`` RichLog with scroll-pause detection (plan S1.5).

    Textual's ``ScrollUp`` key events are consumed (and stopped) by
    ``RichLog._on_scroll_up``, so they never reach the app. This subclass posts a
    bubble-up :class:`ScrolledUp` message whenever the user scrolls the widget UP
    (scroll_y decreased), which ``QuantApp`` handles to pause tail-follow.
    Programmatic ``scroll_end`` from follow-mode writes never decreases
    scroll_y, so auto-follow is not self-interrupted.
    """

    class ScrolledUp(Message):
        """Posted when the user scrolls this log upward."""

        @property
        def control(self) -> "FollowRichLog":
            return self._sender  # type: ignore[attr-defined]

    @classmethod
    def _get_scrolled_up_handler_name(cls) -> str:
        return f"on_{cls.__name__.lower()}_scrolled_up"

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if new_value < old_value - 0.01:
            self.post_message(self.ScrolledUp())


def build_log_drawer(run_log_max_lines: int) -> Vertical:
    """The collapsible bottom log drawer (plan S1.4).

    Hidden (``display=False``) at startup; toggled with ``l``, resized with
    ``,`` / ``.`` through the S/M/L presets.

    Layout fix (user report): the Copy/Save button row is docked to the BOTTOM of
    the drawer and the ``#log`` RichLog gets ``height: 1fr`` so it absorbs ALL the
    remaining space. Previously both children stacked from the top, leaving a big
    empty gap under the log (and the S/M/L presets resized the whole drawer while
    the log stayed at its fixed height -- shrinking just clipped the buttons).
    """
    return Vertical(
        FollowRichLog(id="log", markup=False, highlight=True, max_lines=run_log_max_lines),
        Horizontal(
            Label(id="log_follow", classes="log_follow"),
            Button("Copy", id="copy_log", variant="default"),
            Button("Save to file", id="save_log", variant="default"),
            classes="log_buttons",
        ),
        id="log_drawer",
    )


def build_gguf_panel() -> VerticalScroll:
    """The original Unsloth GGUF widget block, extracted verbatim (C2 refactor).

    No logic changed vs the pre-M1 app -- this is a pure move into a standalone
    builder so the ComfyUI panel can live beside it. Keep edits to this block in
    lockstep with the GGUF happy path.

    S1.2: the advanced options (worker interpreter, max seq length, 4-bit load,
    Hub push fields) are wrapped in a collapsed ``Collapsible`` so the dashboard
    shows only the essential controls by default. Widget ids are unchanged, so
    ``_read_config`` still reads every value.
    """
    return VerticalScroll(
        Label("1. Model path (HuggingFace folder or single .safetensors)"),
        Horizontal(Input(id="model", placeholder="/path/to/model"), Button("Browse", id="browse_model"), classes="field"),
        Label("2. Output folder"),
        Horizontal(Input(id="output", placeholder="/path/to/output"), Button("Browse", id="browse_out"), classes="field"),
        Label("3. Quantization method"),
        # T8 (plan 2026-08-31-gguf-unsloth-parity): Select -> Input. A 35-entry
        # dropdown cannot express multi-method runs ("q4_k_m, q5_k_m"); free
        # text + validate_gguf's whitelist (T5) catches typos the Select used
        # to prevent. Same #method id: _read_config / profiles keep working.
        # The #pick_method button (method-picker feature) opens a modal list
        # of all 35 official ids so nobody types them by hand -- the Input
        # stays editable for comma lists (same classes="field" pattern as the
        # #model + #browse_model row above).
        Horizontal(
            Input(id="method", value=DEFAULT_METHOD,
                  placeholder="e.g. q4_k_m — comma-separate for multiple"),
            Button("Pick from list", id="pick_method"),
            classes="field",
        ),
        Static(id="method_info"),
        Static(UD_INFO_FOOTER, id="ud_footer"),
        Collapsible(
            Label("Imatrix (required for IQ* quants)"),
            Input(id="imatrix_path",
                  placeholder="imatrix .dat/.gguf path (required for IQ* quants)"),
            Checkbox("Auto (fetch upstream Unsloth imatrix)", id="imatrix_auto"),
            Label("Custom method (optional, overrides the field above)"),
            Input(id="custom", placeholder="e.g. q4_k_m, q5_k_m — comma-separate for multiple"),
            Label("Worker Python interpreter (must have unsloth + CUDA)"),
            Input(id="pybin", value=sys.executable),
            Label("Max sequence length"),
            Input(id="maxseq", value="4096"),
            Checkbox("Load in 4-bit (faster load, slightly lower quality)", id="load4"),
            Checkbox("Push result to Hugging Face Hub", id="push"),
            Input(id="hub", placeholder="your-user/model-Q4_K_XL-gguf"),
            Input(id="token", placeholder="hf_... (optional)"),
            title="Advanced",
            collapsed=True,
        ),
        Button("Run Quantization", id="run", variant="success"),
        Button("List all methods", id="listm", variant="default"),
        id="gguf_panel",
    )


def build_comfy_panel() -> VerticalScroll:
    """ComfyUI panel, rendered data-driven from the registry schema.

    Widget ids are stable for tests: #ctq_input, #audit_ctq_in, #ctq_output, #ctq_format,
    #ctq_preset, #ctq_comfy_quant, #ctq_save_quant_metadata, #ctq_simple,
    #ctq_low_memory, #ctq_calib_samples, #pybin_ctq, #ctq_cap_warn, plus the
    conditional option widgets (#scaling_mode / #block_size / #convrot /
    #convrot_group_size etc.) declared by the format registry.
    """
    convrot_choices = [("64", "64"), ("256", "256"), ("1024", "1024")]
    return VerticalScroll(
        Label("1. Input: a single .safetensors file, a folder with one, or a HuggingFace sharded folder (model.safetensors.index.json)"),
        Horizontal(Input(id="ctq_input", placeholder="/path/to/model.safetensors or /path/to/hf-model-folder"),
                   Button("Browse", id="browse_ctq_in"),
                   Button("Audit", id="audit_ctq_in", disabled=True), classes="field"),
        # .pt suggestion (user feature): hidden until the input is a .pt/.pth/.ckpt.
        PtSuggestBox(),
        Label("2. Output: a .safetensors file (single) OR a folder (HuggingFace sharded output)"),
        Horizontal(Input(id="ctq_output", placeholder="/path/to/MyModel-fp8_e4m3.safetensors OR /path/to/output-folder"),
                   Button("Browse", id="browse_ctq_out"), classes="field"),
        Label("Output mode (HuggingFace sharded input only)"),
        RadioSet(
            RadioButton("Sharded folder (no-merge)", value=True, id="om_sharded"),
            RadioButton("Single file (merge)", id="om_single"),
            id="ctq_output_mode",
        ),
        Label("3. Format"),
        Select(format_options(), id="ctq_format", value=DEFAULT_CTQ_FORMAT, allow_blank=False),
        # Unified INT8 option widgets (v0.4.0): scaling / block_size / convrot /
        # convrot_group_size are now plain OptionField-driven widgets (the old
        # special-cased #ctq_scaling_mode widget is gone -- the registry owns
        # both the values and the visibility predicates). All start hidden.
        Label("Scaling mode", id="scaling_mode_label", classes="hidden"),
        Select([("block", "block"), ("tensor", "tensor"), ("row", "row")],
               id="scaling_mode", value="block", allow_blank=False, classes="hidden"),
        Label("Block size", id="block_size_label", classes="hidden"),
        Select([("64", "64"), ("128", "128"), ("256", "256")], id="block_size",
               value="128", allow_blank=False, classes="hidden"),
        Label("ConvRot rotation (row only)", id="convrot_label", classes="hidden"),
        Checkbox("Apply ConvRot rotation (needs triton)", id="convrot",
                 value=False, classes="hidden"),
        Label("ConvRot group size", id="convrot_group_size_label", classes="hidden"),
        Select(convrot_choices, id="convrot_group_size", value="256", allow_blank=False, classes="hidden"),
        # P-expose: INT8 quantization parameters surfaced in the UI so users can avoid
        # the "dimensions divisible by block_size" crash (heur ->
        # copy non-divisible weights unchanged). Hidden until an INT8 format is selected.
        # (Scaling mode / Block size / ConvRot widgets above are registry-declared.)
        Label("Skip inefficient layers (--heur)", id="heur_label", classes="hidden"),
        Checkbox("Copy non-block-divisible weights unchanged (avoids block_size crash)",
                 id="heur", value=False, classes="hidden"),
        Label("Manual seed (optional; empty = streaming default)", id="manual_seed_label", classes="hidden"),
        Input(id="manual_seed", placeholder="e.g. 233983427", classes="hidden"),
        # Raon-recipe parity: keep matched layers at original precision
        # (e.g. "attn_norm|text_embed"). Declared as an INT8 OptionField too.
        Label("Exclude layers (regex, optional)", id="exclude_layers_label", classes="hidden"),
        Input(id="exclude_layers", placeholder="e.g. attn_norm|text_embed", classes="hidden"),
        # == upstream quantize_raon_int8_convrot.py --downcast-fp32 (bfloat16 default).
        Label("Output dtype (unquantized weights; bfloat16 = --downcast-fp32)", id="output_dtype_label", classes="hidden"),
        Select([("bfloat16", "bfloat16"), ("float16", "float16")], id="output_dtype",
               value="bfloat16", allow_blank=False, classes="hidden"),
        Label("4. Preset (optional)"),
        Select(preset_options(), id="ctq_preset", allow_blank=True),
        Collapsible(
            Label("Toggles"),
            Checkbox("comfy_quant", id="ctq_comfy_quant", value=True),
            Checkbox("save_quant_metadata", id="ctq_save_quant_metadata", value=True),
            Checkbox("simple", id="ctq_simple", value=False),
            Checkbox("low_memory", id="ctq_low_memory", value=False),
            Label("Calibration samples (optional; empty = ctq default)"),
            Input(id="ctq_calib_samples", placeholder="e.g. 8"),
            Label("Learned-rounding iterations per tensor (empty = ctq default 4000). "
                  "Lower (500-1000) = much faster convrot runs"),
            Input(id="ctq_num_iter", placeholder="e.g. 1000"),
            Label("Worker Python interpreter (convert_to_quant + CUDA torch)"),
            Input(id="pybin_ctq", value=sys.executable),
            title="Advanced",
            collapsed=True,
        ),
        Label(id="ctq_cap_warn", classes="capwarn"),
        Static("Tip: a HuggingFace sharded input quantizes each shard in place "
               "and writes the OUTPUT as a folder (index json + config/tokenizer copied).",
               classes="capwarn"),
        Button("Run Quantization", id="run_ctq", variant="success"),
        # Validation feature (Msg E): test a quantized .safetensors for structural
        # integrity without re-quantizing. Surfaced as its own small section so it is
        # discoverable but does not add buttons to the run/stop morph.
        Label("Validate a quantized .safetensors (structure + integrity check)"),
        Horizontal(Input(id="validate_path", placeholder="/path/to/quantized.safetensors"),
                   Button("Browse", id="browse_validate_in"), classes="field"),
        Button("Validate", id="validate", variant="default"),
        id="comfy_panel",
    )
