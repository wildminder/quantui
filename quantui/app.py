"""Terminal UI for multi-family model quantization.

Supports two families (see ``quant_methods.Family``), each with its own worker:

* **Unsloth GGUF** (causal-LM LLMs)  -> ``worker.py``   (output = folder)
* **ComfyUI / convert_to_quant**      -> ``worker_ctq.py`` (output = .safetensors)

The GGUF happy path is intentionally byte-for-byte equivalent to the pre-M1 app:
the GGUF widgets were extracted into ``build_gguf_panel()`` (a pure refactor, no
behavior change) and the GGUF branch of ``validate`` / ``auto_suggest_output`` /
``action_run`` is unchanged. The ComfyUI family is rendered data-driven from the
registry schema and branches through the same three seams.

The heavy work runs in a separate python process (configurable per family) so this
UI stays light and streams the worker's live log.
"""

import os
import subprocess
import sys
import tempfile
import time
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
)
from textual.widgets._select import InvalidSelectValueError

# Test-patch anchor: test_capability_badge_nonblocking monkeypatches
# ``appmod.capabilities.probe_worker_env``; the import keeps the module attribute
# reachable even though app.py itself does not reference it directly.
from . import (
    capabilities,  # noqa: F401
    handlers,
    panels,
    profiles_store,
    quant_validator,
    run_config,
    run_monitor,
    screens,
    screens_wizard,
    stream_parser,
    ui_bridge,
    worker_runner,
)
from .app_css import MAIN_CSS
from .header_progress import HeaderProgressHold
from .live_progress import LiveProgressStore
from .quant_methods import (
    COMFY_FORMATS,
    DEFAULT_GGUF_METHOD,
    METHODS_BY_ID,
    Family,
    comfy_format,
    comfy_preset,
    eval_visible_when,
)
from .quant_methods import (
    is_sharded_folder as quant_methods_is_sharded_folder,
)

# NTH-001: project constants are single-homed in run_config; re-exported here
# for backward compatibility with existing imports/tests.
from .run_config import (  # noqa: F401  (re-export)
    DEFAULT_CTQ_FORMAT,
    DEFAULT_CTQ_OUTPUT_MODE,
)
from .widgets_results import ResultsCard, report_to_issues

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def open_in_editor(path: str) -> None:
    """Open ``path`` in the user's editor / pager / file manager (plan S1.6).

    Factored as a module-level function so tests can monkeypatch it (tests must
    never spawn a real editor). Windows uses ``os.startfile``; elsewhere the
    ``$EDITOR`` / ``$PAGER`` is spawned in a NEW console-less subprocess. Raises
    ``RuntimeError`` when no launcher can be found.
    """
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 -- intentional shell-open on Windows
        return
    editor = os.environ.get("EDITOR") or os.environ.get("PAGER")
    if not editor:
        raise RuntimeError("no editor/pager configured ($EDITOR/$PAGER)")
    subprocess.run([editor, path], check=False)


def _notify_no_launcher(app: "QuantApp") -> None:
    app.emit_toast("No editor available to open the log.", "warning")


def _literal(text: str) -> str:
    r"""Escape square brackets so Textual markup does not eat a literal badge.

    Static renders Textual markup, and the parser accepts an UPPERCASE tag
    (``[IMATRIX]`` reads as a style name) even though ``textual.markup.escape``
    only escapes lowercase-leading tags -- so the badge silently vanished from
    the UI and applied a bogus style to the rest of the line.

    Only ``[`` is escaped: the parser consumes the backslash on ``\[`` and
    leaves a lone ``]`` alone, so escaping ``]`` too would paint a stray
    backslash into the rendered line.
    """
    return text.replace("[", r"\[")

# T3b (plan 2026-08-31-gguf-unsloth-parity): re-export, NOT a redefinition.
# The old duplicate ("q4_k_xl if present else METHODS[0]") degraded to
# not_quantized after the UD-* removal; the single home is quant_methods.
DEFAULT_METHOD = DEFAULT_GGUF_METHOD  # noqa: F811  (re-export alias)

# NTH-001: project constants are single-homed in run_config; re-exported here
# for backward compatibility with existing imports/tests.

# Feature A: on-screen log display cap (RichLog max_lines). The authoritative full
# log lives in a per-run temp file (see _open_run_log / _read_full_log).
RUN_LOG_MAX_DISPLAY_LINES = 100

# S1.10: stop-confirm threshold. A run younger than this many seconds stops
# instantly (quick-abort semantics); older runs ask for confirmation first.
STOP_CONFIRM_AFTER_S = 15

# --- Live progress: collapse tqdm-style progress lines into one in-place widget ---
# A tqdm/optimizer progress line looks like:
#   "Optimizing INT8 (Prodigy-plateau):   0%|          | 0/4000 [00:00<?, ?it/s]"
# tqdm/optimizer bars (e.g. "Optimizing INT8 (Prodigy-plateau): 0%|...| 0/4000 [...]")
# rewrite their line in place with a carriage return ('\r') and only emit a trailing
# '\n' when the bar finishes. If we read them as ordinary '\n'-delimited lines, every
# intermediate frame is held in the pipe until that final '\n', so the console either
# floods (thousands of lines) or shows a frozen 0% bar that blinks when the buffered
# frames burst through at the end. We therefore split the raw byte stream on BOTH '\r'
# (in-place overwrite -> live progress frame) and '\n' (newline -> real log line), and
# render the frames in a single live-updating widget. The full stream still goes to the
# authoritative temp log.
# Match a tqdm/optimizer bar line. Robust to the real quant libraries that emit:
#   * the usual known-total bar:  "Optimizing INT8 (...): 50%|#####| 2000/4000 [...]"
#   * an UNKNOWN-total bar:        "?%|###| ?/? [?it/s]"   (Unsloth/Prodigy often can't
#     predict the total up front, so the percent and count are the literal '?')
#   * a bar with no trailing metadata bracket ('[') -- some wrappers omit it.
# Progress regexes + parsing/classification now live in quantui/stream_parser.py

# ProgressView (the #live_progress widget class) moved verbatim into panels.py
# (S0.3 compose-seam extraction); re-exported here so existing
# ``appmod.ProgressView`` references (tests included) keep working.
from .panels import ProgressView  # noqa: F401,E402


class QuantApp(handlers.HandlersMixin, App):
    # IMP-001 S3B.1: stylesheet extracted verbatim to quantui/app_css.py; the
    # alias keeps the pinned CSS contract (``QuantApp.CSS``) intact.
    CSS = MAIN_CSS

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run", "Run"),
        ("1", "family_gguf", "GGUF family"),
        ("2", "family_comfy", "ComfyUI family"),
        ("l", "toggle_log", "Toggle log drawer"),
        ("o", "open_log_file", "Open full log file"),
        ("comma", "log_size_prev", "Log size -"),
        ("full_stop", "log_size_next", "Log size +"),
        ("escape", "clear_log_filter", "Clear log filter"),
        # S2.4: save current params as a named profile.
        ("ctrl+s", "save_profile", "Save profile"),
        # S2.5: show recent jobs (row-activate re-runs).
        ("ctrl+r", "show_recents", "Recent jobs"),
        # Placeholder help binding: the cheat-sheet overlay itself is Phase 2/3;
        # the Footer hint teaches the keymap now (plan S1.10).
        ("question_mark", "help_placeholder", "Help"),
        # S3.3: stepped wizard (disabled while a run is active).
        ("w", "open_wizard", "Wizard"),
    ]

    def action_help_placeholder(self) -> None:
        """`?`: Phase-3 placeholder -- shows a toast pointing at the Footer."""
        self.emit_toast("Keys: r run | l log | o open log | 1/2 family | , . log size | q quit",
                        "information")

    # ---- S3.3: wizard -----------------------------------------------------------
    def action_open_wizard(self) -> None:
        """Key ``w``: open the stepped wizard; disabled during an active run."""
        if self._run_active:
            self.emit_toast("A run is active — stop it before starting the wizard.", "warning")
            return
        self.push_screen(screens_wizard.WizardScreen(), self._on_wizard_done)

    def _on_wizard_done(self, values: dict | None) -> None:
        """Wizard finished: copy values into the MAIN form widgets and start the
        SAME action_run pipeline (determinism gate: identical inputs -> identical
        command vs the form path, since both go through _read_config)."""
        if not values:
            return
        self._apply_profile_fields(values)
        self.action_run()

    def action_clear_log_filter(self) -> None:
        """Esc: clear an active phase filter (S1.9); normal stream resumes."""
        self._clear_log_filter()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.family: Family = Family.GGUF
        # Feature A: per-run full-log temp file (authoritative; RichLog only shows a cap).
        self._run_log_path: str = ""
        self._run_log_fh = None  # type: ignore[var-annotated]
        # Live-progress diagnostic log (opened only when UNSLOTH_CTQ_DEBUG_PROGRESS=1).
        self._progress_debug_path: str = ""
        self._progress_debug_fh = None  # type: ignore[var-annotated]
        # Live-progress: pure collapse store (T04) replaces the raw dict. The router
        # dispatches parsed segments to the LogSink methods implemented below.
        self._live = LiveProgressStore()
        self.router = ui_bridge.LogRouter(self, self._live)
        # Textual-free subprocess runner (T05). Pumps a worker's stdout via os.read on
        # the raw pipe fd and emits StreamSegments to this app as the LogObserver.
        self.runner = worker_runner.WorkerRunner()
        # Run-monitor state (S1.7): wall-clock start of the current run (0 = idle).
        self._run_start_ts: float = 0.0
        # Run-state flags for the Stop button (set mostly from the main thread via
        # call_from_thread; read in the worker thread -- simple bools are GIL-safe).
        self._run_active = False
        self._stop_requested = False
        # Log drawer state (S1.4): starts hidden at size M; follow mode on (S1.5).
        self._log_drawer_size: str = "M"
        self._log_follow: bool = True
        # Phase filter for the drawer (S1.9): empty = normal capped stream.
        self._log_filter: str = ""
        # Wave 2 state: last-run duration (S2.1/S2.2), last RunRecord (S2.2),
        # lazily-loaded profiles store (S2.4/S2.5).
        self._last_run_duration: float = 0.0
        self._last_record = None
        self._profiles_store = None
        # Header progress stickiness (S1.7 user fix), extracted to a pure class
        # (IMP-001 S3B.3): the displayed % never dips within one run; reset at
        # run boundaries only (see _clear_header_strip).
        self._progress_hold = HeaderProgressHold()

    # ---- composition ---------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        # Global run-monitor strip (S1.7): determinate aggregate bar + ETA.
        yield Horizontal(
            ProgressBar(id="header_progress", show_percentage=False),
            Label("--", id="eta_label"),
            id="header_strip",
        )
        yield RadioSet(
            RadioButton("Unsloth GGUF", value=True, id="fam_gguf"),
            RadioButton("ComfyUI / convert_to_quant", id="fam_comfy"),
            id="family",
        )
        # Main layout delegated to panels.build_main_layout (S0.3 seam) -- all
        # Phase-1 layout surgery happens inside that builder.
        yield panels.build_main_layout(RUN_LOG_MAX_DISPLAY_LINES)
        # Collapsible bottom log drawer (S1.4): hidden at startup, `l` toggles.
        yield panels.build_log_drawer(RUN_LOG_MAX_DISPLAY_LINES)
        yield Footer()

    # ---- helpers -------------------------------------------------------------

    def log_msg(self, msg: str) -> None:
        # Delegate routing to LogRouter (T04) -- the single source of truth for the
        # 4-way classify/route decision. The router writes the authoritative temp-file
        # log first, then routes the classified segment to the LiveProgressStore /
        # RichLog via the LogSink methods implemented on this app. A pure separator
        # ("----" between a ctq header and its details) echoes to the RichLog WITHOUT
        # clearing the live-progress widget -- this is the fix for the "no bar + fewer
        # strings" ComfyUI regression (T02/T03).
        text = msg.replace("\r", "")
        if not text.strip():
            return  # skip blank/tqdm-only lines -- they add no information
        seg = stream_parser.StreamSegment(
            text, stream_parser.ProgressClassifier.classify(text), False
        )
        self.router.dispatch(seg)



    def _update_live_progress(self, msg: str) -> None:
        # Legacy per-message live update (kept for backward-compat / T09). The router
        # updates the store directly and calls the LogSink.update_live(text) below, so
        # this is no longer on the hot path, but action_run and tests may still call it.
        self._live.update(msg)
        self.update_live(self._live.render())

    def _clear_live_progress(self) -> None:
        if not len(self._live):
            return
        self._live.clear()
        self.clear_live()

    # --- LogSink (concrete widget writes; always runs on the main thread) --------
    def write_run_log(self, line: str) -> None:
        """Authoritative full log -> per-run temp file (every line, always)."""
        try:
            if self._run_log_fh is not None:
                self._run_log_fh.write(line + "\n")
                self._run_log_fh.flush()
        except Exception as exc:
            # Class-L: log-and-continue (log file I/O must not break the UI).
            self._debug_swallow(exc, "write_run_log")

    def write_log(self, line: str) -> None:
        """Append a line to the on-screen RichLog.

        While a phase filter is active (S1.9) only lines containing the filter
        term are appended; the authoritative temp file keeps EVERY line either
        way (written by ``write_run_log`` upstream of this method).
        """
        try:
            log = self.query_one(RichLog)
            if self._log_filter and self._log_filter not in line:
                return  # filtered out of the drawer view only
            if not log._size_known:
                # S1.4: the log drawer starts HIDDEN, and Textual defers every
                # RichLog.write until the widget gets its first real region
                # (which never happens while display=False). Force the size-known
                # flag so writes land in the capped buffer immediately -- pinned
                # regression tests read ``RichLog.lines`` while the drawer is
                # still closed, and the buffer is flushed visually as soon as
                # the drawer is opened. The cap (max_lines) still applies.
                log._size_known = True
            # S1.5: while paused (user scrolled up) we append WITHOUT auto-scroll
            # so the frozen view is not yanked to the tail.
            log.write(line, scroll_end=self._log_follow)
        except NoMatches:
            # Widget not mounted yet (early log lines before compose finishes).
            pass

    def _apply_log_filter(self, term: str) -> None:
        """Activate/clear the phase filter and re-render the drawer from the
        FULL temp-file log (last 500 matching lines; plan S1.9)."""
        self._log_filter = term or ""
        drawer = self._drawer()
        if drawer is None:
            return
        if self._log_filter:
            tail: list[str] = []
            for ln in self._read_full_log().splitlines():
                if self._log_filter in ln:
                    tail.append(ln)
            tail = tail[-500:]
            try:
                log = self.query_one(RichLog)
                log.clear()
                if not log._size_known:
                    log._size_known = True
                for ln in tail:
                    log.write(ln)
                log.scroll_end(animate=False)
            except NoMatches:
                pass  # drawer RichLog not mounted yet
        else:
            self._resume_follow()
        self._update_drawer_title()

    def on_follow_rich_log_scrolled_up(self, event) -> None:
        """S1.5: the user scrolled #log upward -> pause follow, show '(paused)'."""
        self._set_log_follow(False)

    def _set_log_follow(self, follow: bool) -> None:
        """Toggle follow state + refresh the follow indicator + drawer title."""
        changed = self._log_follow != follow
        self._log_follow = follow
        if not changed:
            return
        try:
            self.query_one("#log_follow", Label).update(
                "follow" if follow else "paused"
            )
        except NoMatches:
            pass  # indicator label not mounted yet
        self._update_drawer_title()

    def _resume_follow(self) -> None:
        """Resume tail-follow and jump to the bottom (S1.5)."""
        self._set_log_follow(True)
        try:
            self.query_one(RichLog).scroll_end(animate=False)
        except NoMatches:
            pass  # drawer closed / RichLog not mounted

    def update_live(self, text: str) -> None:
        """Legacy per-message live update: show the text-only progress (no determinate bar).

        Since S1.8 the render target is the stacked ``#progress_rail`` (the old
        single ``#live_progress`` widget was deleted in the same step).
        """
        try:
            self.query_one("#progress_rail", panels.ProgressRail).set_states(self._live.states())
        except NoMatches:
            pass  # rail not mounted yet

    def update_progress(self, states: list) -> None:
        """Render the structured/collapsed progress store into #progress_rail (and,
        since S1.7, into the global header strip).

        ``states`` is the ordered list of :class:`~quantui.live_progress.ProgressState`
        from :attr:`_live`. Each determinate state renders as its own stacked
        mini-bar row; text-only states get a label-only row (bar hidden).
        """
        try:
            self.query_one("#progress_rail", panels.ProgressRail).set_states(states)
        except NoMatches:
            pass  # rail not mounted yet
        self._render_header_strip(states)

    def on_progress_rail_bar_clicked(self, event) -> None:
        """S1.9: clicking a progress row opens the log drawer filtered to that
        phase (only matching lines from the authoritative temp file)."""
        event.stop()
        phase = getattr(event, "phase", "") or ""
        if not phase:
            return
        self._log_filter = ""  # reset first so toggle logic is simple
        drawer = self._drawer()
        if drawer is not None and not drawer.display:
            # open WITHOUT resuming follow overwriting the filter render
            drawer.display = True
        self._apply_log_filter(phase)

    def _clear_log_filter(self) -> None:
        """Esc in the drawer / any manual clear: back to the normal stream."""
        if not self._log_filter:
            return
        self._apply_log_filter("")
        try:
            log = self.query_one(RichLog)
            log.clear()
            for ln in self._read_full_log().splitlines()[-RUN_LOG_MAX_DISPLAY_LINES:]:
                log.write(ln)
            log.scroll_end(animate=False)
        except NoMatches:
            pass  # drawer RichLog not mounted

    def _render_header_strip(self, states: list) -> None:
        """Global header progress + ETA (S1.7): aggregate of determinate states.

        User-report fix: the bar must NOT reset to 0 every time a real log line
        clears the live-progress store mid-run. The pure
        :class:`HeaderProgressHold` keeps the last known fraction and it is only
        reset when the run actually finishes (:meth:`_clear_header_strip`), so
        the bar is monotonic within one run.
        """
        try:
            bar = self.query_one("#header_progress", ProgressBar)
            eta_lbl = self.query_one("#eta_label", Label)
        except NoMatches:
            return  # header strip not mounted yet
        elapsed = 0.0
        if self._run_start_ts > 0:
            elapsed = time.monotonic() - self._run_start_ts
        pct, eta_s = run_monitor.aggregate(states, elapsed)
        if pct is None:
            # No determinate signal right now: hold the last value instead of 0.
            held = self._progress_hold.held()
            if held is None:
                bar.update(total=100, progress=0)
                eta_lbl.update("--")
                return
            eta_s = None
            pct = held
        else:
            pct = self._progress_hold.next(pct)
        bar.update(total=100, progress=pct)
        eta_lbl.update(run_monitor.format_eta(eta_s))

    def _clear_header_strip(self) -> None:
        """Reset the header strip to idle (run finished / new run starting)."""
        self._progress_hold.reset()
        try:
            bar = self.query_one("#header_progress", ProgressBar)
            eta_lbl = self.query_one("#eta_label", Label)
            bar.update(total=100, progress=0)
            eta_lbl.update("--")
        except NoMatches:
            pass  # header strip not mounted yet

    def clear_live(self) -> None:
        """Blank the progress rail (keep it mounted so the layout doesn't jump)."""
        try:
            self.query_one("#progress_rail", panels.ProgressRail).set_states([])
        except NoMatches:
            pass  # rail not mounted yet
        self._clear_header_strip()

    # --- LogObserver (subject callback; marshals to the main thread) ------------
    def on_segment(self, seg: "stream_parser.StreamSegment") -> None:
        """WorkerRunner / _pump calls this from the worker thread; re-marshal the
        dispatch onto the main thread so widget writes stay single-threaded."""
        self.call_from_thread(self.router.dispatch, seg)

    def on_raw(self, raw: bytes) -> None:
        if self._progress_debug_fh is not None:
            try:
                self._progress_debug_fh.write(f"RAW\t{raw!r}\n")
                self._progress_debug_fh.flush()
            except Exception as exc:
                # Class-L: debug tracing must never break log streaming.
                self._debug_swallow(exc, "on_raw")

    def on_verdict(self, raw: bytes, category: str, live_count: int, content: str) -> None:
        if self._progress_debug_fh is not None:
            try:
                self._progress_debug_fh.write(
                    f"SEG\t{category}\t{live_count}\t{content!r}\n"
                )
                self._progress_debug_fh.flush()
            except Exception as exc:
                # Class-L: debug tracing must never break log streaming.
                self._debug_swallow(exc, "on_verdict")

    def live_count(self) -> int:
        return len(self._live)

    # Stream frame splitting now lives in quantui.stream_parser (T01); the

    def _read_full_log(self) -> str:
        """Return the FULL run log from the temp file (not the capped RichLog window)."""
        try:
            with open(self._run_log_path, encoding="utf-8") as fh:
                return fh.read()
        except OSError:
            # FileNotFoundError before the first run creates the temp file.
            return ""

    def copy_log(self) -> None:
        text = self._read_full_log().rstrip("\n")
        if not text:
            self.notify("Log is empty.")
            return
        self.copy_to_clipboard(text)
        self.notify("Log copied to clipboard.")

    def save_log(self) -> None:
        text = self._read_full_log()
        if not text:
            self.notify("Log is empty.")
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(tempfile.gettempdir(), f"unsloth-quant-tui-log-{ts}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        self.log_msg(f"Log dumped to {path}")
        self.notify(f"Log saved to {path}")

    def set_status(self, msg: str) -> None:
        try:
            self.query_one("#status", Label).update(msg)
        except NoMatches:
            pass  # status label not mounted yet

    def _enable_stop(self) -> None:
        """Main-thread helper: mark a run active and morph the Run buttons into Stop.

        Called via ``call_from_thread`` from the worker thread right before the worker
        subprocess is launched. No extra button is added -- the existing #run / #run_ctq
        buttons relabel to "Stop" (error variant) and stay enabled so they can stop the run.
        """
        self._run_active = True
        self._stop_requested = False
        for wid in ("#run", "#run_ctq"):
            try:
                btn = self.query_one(wid, Button)
                btn.label = "Stop"
                btn.variant = "error"
            except NoMatches:
                pass  # family panel for the other backend not mounted

    def _disable_stop(self, rc: int | None = None) -> None:
        """Main-thread helper: clear run-active state and morph the Stop buttons back to Run.

        Called via ``call_from_thread`` from the worker thread once the run completes
        (success, failure, or user-stop). When ``rc`` is provided (S2.1), a completion
        toast + terminal bell fire: success -> "information" toast + bell; failure ->
        "error" toast; user-stop -> no completion verdict toast (the stop branch logs).
        Also finalizes the results card (S2.2) and records the recent job (S2.5).
        """
        self._run_active = False
        self._run_start_ts = 0.0  # stop the ETA clock
        self._clear_header_strip()
        for wid in ("#run", "#run_ctq"):
            try:
                btn = self.query_one(wid, Button)
                btn.label = "Run Quantization"
                btn.variant = "success"
            except NoMatches:
                pass  # family panel for the other backend not mounted
        if rc is not None:
            self._finish_run_record(rc)

    def _finish_run_record(self, rc: int) -> None:
        """Shared run-finalization (S2.1 toast/bell + S2.2 results card + S2.5 recents).

        Runs on the main thread right after ``_disable_stop`` morphs the buttons back.
        A user-stop (``_stop_requested`` set) skips the success/error verdict.
        """
        # Duration is computed by the caller into _last_run_duration before this runs
        # (time.monotonic is not shared across threads reliably at this point).
        duration = float(getattr(self, "_last_run_duration", 0.0) or 0.0)
        outcome = "stopped"
        if not self._stop_requested:
            outcome = "success" if rc == 0 else "failed"
        try:
            cfg_snapshot = self._profile_fields()
        except NoMatches:
            cfg_snapshot = {}  # some widgets not mounted (headless / partial UI)
        record = profiles_store.RunRecord(
            ts=time.strftime("%Y-%m-%d %H:%M:%S"),
            family=self.family.value,
            method=self.selected_method(),
            output=self._current_output_path(),
            status=outcome,
            exit_code=rc,
            duration_s=round(duration, 1),
            config=cfg_snapshot,
        )
        self._last_record = record
        try:
            profiles_store.add_recent(record.to_dict(), store=self._store())
            profiles_store.save_store(self._store())
        except Exception as exc:  # noqa: BLE001
            # Class-L: persistence must never break the UI.
            self._debug_swallow(exc, "_finish_run_record.persist")
        # S2.2: populate the results card from the record.
        try:
            self.query_one(ResultsCard).show_record(record)
        except NoMatches:
            pass  # results card not mounted yet
        # S2.1: completion toast + terminal bell (never on user-stop).
        if outcome == "success":
            self.emit_toast("Quantization finished \u2713", "information")
            self._ring_bell()
        elif outcome == "failed":
            self.emit_toast(f"Quantization failed (exit {rc})", "error")

    def _ring_bell(self) -> None:
        """Write BEL (\\x07) to stdout so the terminal beeps/flash-alerts (plan S2.1)."""
        try:
            sys.stdout.write("\x07")
            sys.stdout.flush()
        except Exception as exc:
            # Class-L: a missing/broken stdout must never break run completion.
            self._debug_swallow(exc, "_ring_bell")

    def _current_output_path(self) -> str:
        """Best-effort output path of the just-finished run (for the results card).

        Reads the active family's output widget; falls back to ``_read_config``
        (which tests / headless flows may have replaced with a canned config).
        """
        try:
            cfg_field = (
                "#ctq_output" if self.family == Family.COMFY else "#output"
            )
            val = str(self.query_one(cfg_field, Input).value.strip())
            if val:
                return val
        except NoMatches:
            pass  # widget not mounted; fall through to _read_config
        try:
            cfg = self._read_config()
            if self.family == Family.COMFY and cfg.ctq is not None:
                return cfg.ctq.output
            if cfg.gguf is not None:
                return cfg.gguf.output
        except NoMatches:
            pass  # same widgets missing inside _read_config
        return ""

    def _store(self):
        """Lazy per-app profiles store handle (created on first use, S2.4/S2.5)."""
        store = getattr(self, "_profiles_store", None)
        if store is None:
            store = profiles_store.load_store()
            self._profiles_store = store
        return store

    # ---- family tab shortcuts (S1.3) ------------------------------------------
    def action_family_gguf(self) -> None:
        """Key ``1``: switch to the GGUF panel.

        Goes through ``_set_family`` so ``self.family`` is updated
        SYNCHRONOUSLY: pressing the radio alone only *posts* the change, and a
        caller that switches family then acts in the same block (wizard /
        profile apply -> Run) would otherwise validate the wrong panel.
        """
        self._set_family(Family.GGUF)

    def action_family_comfy(self) -> None:
        """Key ``2``: see action_family_gguf."""
        self._set_family(Family.COMFY)

    def emit_toast(self, msg: str, severity: str = "information") -> None:
        """Toast wrapper (plan Q5): route user feedback through App.notify."""
        try:
            self.notify(msg, severity=severity)
        except Exception as exc:
            # Class-L: notification failure must never break the calling action.
            self._debug_swallow(exc, "emit_toast")

    # ---- S2.2: results-card button routing ------------------------------------
    def on_results_card_open_folder(self, event) -> None:
        """[Open folder] on the ResultsCard: open the output's parent folder."""
        try:
            open_in_editor(event.folder)
        except Exception:
            # open_in_editor raises when no file manager/launcher exists — that is
            # a true environment boundary; surface it to the user instead.
            self.emit_toast("No file manager available to open the folder.", "warning")

    # ---- S2.4: profiles ---------------------------------------------------------
    def _profile_fields(self) -> dict:
        """Snapshot the current UI params into a plain dict (profile shape).

        Field names mirror ``_read_config`` so the applier is symmetric.
        """
        return {
            "family": self.family.value,
            "model": self.query_one("#model", Input).value,
            "output": self.query_one("#output", Input).value,
            "method": self.query_one("#method", Input).value,
            "custom": self.query_one("#custom", Input).value,
            "imatrix": self.imatrix_value(),
            "maxseq": self.query_one("#maxseq", Input).value,
            "ctq_input": self.query_one("#ctq_input", Input).value,
            "ctq_output": self.query_one("#ctq_output", Input).value,
            "ctq_num_iter": self.query_one("#ctq_num_iter", Input).value,
            "ctq_format": self.ctq_format(),
            "ctq_scaling_mode": self._safe_widget_value("#scaling_mode"),
            "ctq_convrot": bool(self._safe_checkbox_value("#convrot")),
            "ctq_block_size": self._safe_widget_value("#block_size"),
            "ctq_convrot_group_size": self._safe_widget_value("#convrot_group_size"),
            "ctq_preset": (
                lambda v: str(v) if v and v is not Select.BLANK else ""
            )(self.query_one("#ctq_preset", Select).value),
        }

    def _safe_widget_value(self, selector: str) -> str:
        """Best-effort Select value read for optional widgets (profile snapshot)."""
        try:
            v = self.query_one(selector, Select).value
            return str(v) if v and v is not Select.BLANK else ""
        except NoMatches:
            return ""

    def _safe_checkbox_value(self, selector: str) -> bool:
        try:
            return bool(self.query_one(selector, Checkbox).value)
        except NoMatches:
            return False

    def _apply_profile_fields(self, fields: dict) -> None:
        """Write a saved profile back into the widgets (same key map as
        :meth:`_profile_fields` -- form/review determinism seam for S3.3)."""
        fam = fields.get("family", "gguf")
        if fam == Family.COMFY.value:
            self.action_family_comfy()
        else:
            self.action_family_gguf()
        mapping = {
            "#model": fields.get("model", ""),
            "#output": fields.get("output", ""),
            "#custom": fields.get("custom", ""),
            "#imatrix_path": (
                "" if fields.get("imatrix") == "auto" else fields.get("imatrix", "")
            ),
            "#maxseq": fields.get("maxseq", ""),
            "#ctq_input": fields.get("ctq_input", ""),
            "#ctq_output": fields.get("ctq_output", ""),
            "#ctq_num_iter": fields.get("ctq_num_iter", ""),
        }
        for wid, val in mapping.items():
            try:
                self.query_one(wid, Input).value = val or ""
            except NoMatches:
                pass  # widget not mounted (family panel hidden)
        # T9: #method is an Input now -- any string (including comma lists and
        # ids from newer unsloth versions) applies cleanly; no option-value
        # errors possible, so the old InvalidSelectValueError dance is gone.
        try:
            if fields.get("method"):
                self.query_one("#method", Input).value = fields["method"]
        except NoMatches:
            pass  # widget not mounted (family panel hidden)
        try:
            self.query_one("#imatrix_auto", Checkbox).value = (
                fields.get("imatrix") == "auto"
            )
        except NoMatches:
            pass
        try:
            if fields.get("ctq_format"):
                self.query_one("#ctq_format", Select).value = fields["ctq_format"]
        except (NoMatches, InvalidSelectValueError):
            pass  # comfy panel not mounted / stale format id
        # Unified INT8 option widgets (stale/absent values are silently skipped).
        for key, wid in (("ctq_scaling_mode", "#scaling_mode"),
                         ("ctq_block_size", "#block_size"),
                         ("ctq_convrot_group_size", "#convrot_group_size")):
            if fields.get(key):
                try:
                    self.query_one(wid, Select).value = fields[key]
                except (NoMatches, InvalidSelectValueError):
                    pass
        if "ctq_convrot" in fields:
            try:
                self.query_one("#convrot", Checkbox).value = bool(fields["ctq_convrot"])
            except NoMatches:
                pass
        self.refresh_ctq_visibility()

    def action_save_profile(self) -> None:
        """Ctrl+S: push NameModal, persist current params under that name."""
        screen = screens.NameModal()

        def _on_name(name: str | None) -> None:
            if not name:
                return
            try:
                profiles_store.save_profile(name.strip(), self._profile_fields())
                self.emit_toast(f"Profile '{name.strip()}' saved.", "information")
            except Exception as exc:  # noqa: BLE001
                self.emit_toast(f"Failed to save profile: {exc}", "error")

        self.push_screen(screen, _on_name)

    def action_load_profile(self) -> None:
        """Load flow used by the palette / future Select overlay: pick from saved."""
        names = profiles_store.list_profiles()
        if not names:
            self.emit_toast("No saved profiles yet (Ctrl+S saves one).", "warning")
            return
        self._pending_profile_names = names
        # The interactive picker lands with the wizard; palette consumers call
        # apply_profile_by_name directly. Keep the toast informative meanwhile.
        self.emit_toast(
            f"Profiles: {', '.join(names[:5])}{' …' if len(names) > 5 else ''}",
            "information",
        )

    def apply_profile_by_name(self, name: str) -> bool:
        """Apply one saved profile to the widgets; False when it does not exist."""
        prof = profiles_store.get_profile(name)
        if prof is None:
            return False
        self._apply_profile_fields(prof)
        self.emit_toast(f"Profile '{name}' loaded.", "information")
        return True

    # ---- S2.5: recents + one-key re-run ----------------------------------------
    def action_show_recents(self) -> None:
        """Ctrl+R: list recent jobs; activating a row reloads its config."""
        recents = profiles_store.list_recents()
        if not recents:
            self.emit_toast("No recent jobs yet.", "warning")
            return
        self.push_screen(screens.RecentJobsScreen(recents), self._on_recent_selected)

    def _on_recent_selected(self, record: dict | None) -> None:
        """Row-activated on RecentJobsScreen: reload cfg into widgets.

        If the recorded output dir contains a partial-checkpoint marker, ask the
        user about resuming first (pre-fills paths either way).
        """
        if not record:
            return
        fields = record.get("config") or {}
        if fields:
            self._apply_profile_fields(fields)
        out = str(record.get("output", "") or "")
        marker = profiles_store.PARTIAL_MARKER
        if out and os.path.isdir(out) and os.path.exists(os.path.join(out, marker)):
            modal = screens.ConfirmModal(
                "Output contains a partial checkpoint — resume?",
                confirm_label="Resume",
                cancel_label="Fresh run",
            )

            def _on_resume(_resume: bool | None) -> None:
                # Both answers keep the pre-filled paths; resume semantics are
                # handled by worker.py/worker_ctq.py detecting existing outputs.
                pass

            self.push_screen(modal, _on_resume)

    # ---- S3.1: command palette ---------------------------------------------------
    COMMANDS = None  # set below after class QuantCommands is defined    # ---- log drawer (S1.4) ------------------------------------------------------
    def _drawer(self) -> "panels.Vertical | None":
        try:
            return self.query_one("#log_drawer")
        except NoMatches:
            return None

    def action_toggle_log(self) -> None:
        """Key ``l``: show/hide the bottom log drawer.

        The RichLog stays MOUNTED the whole time (plan Q2): hiding only sets
        ``display=False`` so the capped buffer keeps accumulating and no
        ``query_one("#log")`` call site ever breaks. Re-opening resumes follow
        (S1.5).
        """
        drawer = self._drawer()
        if drawer is None:
            return
        drawer.display = not drawer.display
        if drawer.display:
            self._resume_follow()

    # ---- open authoritative log file (S1.6) -------------------------------------
    def action_open_log_file(self) -> None:
        """Key ``o``: open the per-run temp-file log (the FULL authoritative log).

        Notifies instead of launching when the log is empty or no launcher is
        available. Never raises into the UI.
        """
        if not self._read_full_log().strip():
            self.emit_toast("Log is empty — nothing to open.", "warning")
            return
        try:
            open_in_editor(self._run_log_path)
        except Exception:
            self.emit_toast("No editor available to open the log.", "warning")

    def _cycle_log_size(self, direction: int) -> None:
        """Cycle the S/M/L height presets (`,`: smaller, `.`: larger)."""
        order = panels.LOG_DRAWER_ORDER
        if self._log_drawer_size not in order:
            self._log_drawer_size = "M"
        idx = (order.index(self._log_drawer_size) + direction) % len(order)
        self._log_drawer_size = order[idx]
        self._apply_drawer_size()

    def action_log_size_prev(self) -> None:
        self._cycle_log_size(-1)

    def action_log_size_next(self) -> None:
        self._cycle_log_size(+1)

    def _apply_drawer_size(self) -> None:
        drawer = self._drawer()
        if drawer is None:
            return
        drawer.styles.height = panels.LOG_DRAWER_SIZES.get(
            self._log_drawer_size, panels.LOG_DRAWER_SIZES["M"]
        )
        self._update_drawer_title()

    def _update_drawer_title(self) -> None:
        drawer = self._drawer()
        if drawer is None:
            return
        state = "follow" if getattr(self, "_log_follow", True) else "paused"
        filt = f" | filter: {self._log_filter}" if getattr(self, "_log_filter", "") else ""
        drawer.border_title = f"Log [{self._log_drawer_size}] ({state}){filt}"
        try:
            self.query_one("#log_follow", Label).update(
                "follow" if self._log_follow else "paused"
            )
        except NoMatches:
            pass  # follow indicator not mounted yet

    def on_mount(self) -> None:
        self._open_run_log()  # authoritative per-run temp log (Feature A)
        self.family = Family.GGUF
        self.query_one("#comfy_panel").display = False
        self.refresh_ctq_visibility()
        self.update_method_info()
        self.set_status("Ready.")
        # Log drawer (S1.4): hidden by default, size preset M.
        drawer = self._drawer()
        if drawer is not None:
            drawer.display = False
            self._apply_drawer_size()

    def _debug_swallow(self, exc: BaseException, context: str) -> None:
        """Record an intentionally-swallowed error into the authoritative run log.

        CRIT-002 Class-L mechanism: call sites that keep a broad ``except
        Exception`` (non-boundary) must route through this so silent failures
        become diagnosable. Never raises; no-op when no run log is open.
        """
        try:
            fh = getattr(self, "_run_log_fh", None)
            if fh is not None:
                fh.write(f"[swallowed] {context}: {type(exc).__name__}: {exc}\n")
        except Exception:  # boundary: logging the logger must never raise
            pass

    def on_unmount(self) -> None:
        """Close the per-run run-log handle (Feature A)."""
        try:
            if self._run_log_fh is not None:
                self._run_log_fh.close()
        except Exception as exc:
            # Class-L: teardown must never raise during app shutdown.
            self._debug_swallow(exc, "on_unmount.close_run_log")
        try:
            if self._progress_debug_fh is not None:
                self._progress_debug_fh.close()
        except Exception as exc:
            # Class-L: teardown must never raise during app shutdown.
            self._debug_swallow(exc, "on_unmount.close_debug_fh")

    def _open_run_log(self) -> None:
        """Open <tempdir>/unsloth-quant-tui-<ts>.log for this run (authoritative full log).

        Honors ``UNSLOTH_CTQ_LOG_DIR`` to redirect the run log (useful for CI capture
        / debug). The file is the single source of truth for the FULL log; ``RichLog``
        only renders a capped window.
        """
        ts = time.strftime("%Y%m%d-%H%M%S")
        logdir = os.environ.get("UNSLOTH_CTQ_LOG_DIR") or tempfile.gettempdir()
        self._run_log_path = os.path.join(logdir, f"unsloth-quant-tui-{ts}.log")
        self._run_log_fh = open(self._run_log_path, "a", encoding="utf-8", buffering=1)
        # Optional diagnostic: capture the EXACT bytes / verdict each frame receives so
        # a user can paste back the real third-party library output if the bar still
        # misbehaves (set UNSLOTH_CTQ_DEBUG_PROGRESS=1). Off by default (zero overhead).
        if os.environ.get("UNSLOTH_CTQ_DEBUG_PROGRESS") == "1":
            self._progress_debug_path = self._run_log_path[:-4] + ".progress-debug.log"
            self._progress_debug_fh = open(
                self._progress_debug_path, "a", encoding="utf-8", buffering=1
            )

    def update_method_info(self) -> None:
        mid = self.selected_method()
        m = METHODS_BY_ID.get(mid)
        if m:
            badge = "[IMATRIX] " if m.needs_imatrix else ""
            hint = " (needs an imatrix — set a path or 'auto')" if m.needs_imatrix else ""
            # The badge is emphasised with real markup; the literal brackets it
            # displays are escaped (see _literal) so they survive rendering.
            head = f"[b yellow]{_literal(badge.strip())}[/b yellow] " if badge else ""
            self.query_one("#method_info", Static).update(
                head + _literal(m.description) + _literal(hint)
            )
        else:
            self.query_one("#method_info", Static).update(
                "Custom method — passed through to Unsloth."
            )

    def refresh_ctq_visibility(self) -> None:
        """Show/hide ctq widgets based on the selected format (data-driven).

        Every format-specific option widget is driven by its own ``visible_when``
        predicate (declared in ``quant_methods``), so adding a new option there
        automatically gets UI show/hide handling here.

        All option widgets are reset on every call: a widget declared by the current
        format is shown only if its predicate passes, and any widget NOT declared by the
        current format is explicitly hidden -- otherwise a widget left visible by a
        previously-selected format (e.g. ``convrot_group_size`` from a prior selection)
        would linger after switching to ``fp8_e4m3``.
        """
        fmt = self.ctq_format()
        # User-report fix: the "Output mode" radio set only matters for a sharded
        # (HuggingFace folder with model.safetensors.index.json) input -- for a
        # single .safetensors there is nothing to merge, so hide it.
        try:
            inp = self.query_one("#ctq_input", Input).value.strip()
            is_sharded = bool(inp) and quant_methods_is_sharded_folder(inp)
        except Exception:
            is_sharded = False
        self._set_option_display("ctq_output_mode", is_sharded)
        cf = comfy_format(fmt)
        # Every option widget (across all formats) is reset on every call. A widget is
        # visible only if the CURRENT format declares it AND its predicate passes; any
        # widget not declared by the current format is hidden. The predicate context
        # carries "format" PLUS the live values of already-resolved sibling options,
        # so chained predicates work (e.g. block_size requires scaling_mode == 'block';
        # convrot_group_size requires scaling_mode == 'row' and convrot on). Options
        # are evaluated in declaration order so dependencies resolve naturally.
        all_keys: set[str] = set()
        for f in COMFY_FORMATS:
            for opt in f.extra_options:
                all_keys.add(opt.key)
        # Declaration-order pass FIRST so dependency chains resolve deterministically
        # (e.g. block_size's predicate reads scaling_mode's live value); widgets from
        # other formats are then explicitly hidden.
        ctx: dict[str, Any] = {"format": fmt}
        for opt in cf.extra_options:
            visible = eval_visible_when(opt.visible_when, ctx)
            ctx[opt.key] = self._option_live_value(opt)
            self._set_option_display(opt.key, visible)
            all_keys.discard(opt.key)
        for key in all_keys:
            self._set_option_display(key, False)

    def _option_live_value(self, opt) -> Any:
        """Best-effort read of an option's current UI value (for predicate contexts)."""
        try:
            w = self.query_one(f"#{opt.key}")
        except NoMatches:
            return opt.default
        if opt.widget == "checkbox":
            return bool(w.value)
        return str(w.value) if w.value is not None else opt.default

    def _set_option_display(self, key: str, visible: bool) -> None:
        """Set the display of an option widget and its label (best-effort)."""
        for wid in (f"#{key}", f"#{key}_label"):
            try:
                self.query_one(wid).display = visible
            except NoMatches:
                pass  # option not rendered for the current format

    def apply_preset(self) -> None:
        val = self.query_one("#ctq_preset", Select).value
        if val and val is not Select.BLANK:
            p = comfy_preset(str(val))
            if p:
                self.query_one("#ctq_format", Select).value = p.recommended_format
                # Apply the preset's recommended option values (e.g. flux2 pins
                # scaling_mode=row + convrot on) before the visibility refresh so
                # dependent widgets (convrot_group_size etc.) appear correctly.
                for key, value in p.recommended_options.items():
                    try:
                        w = self.query_one(f"#{key}")
                    except NoMatches:
                        continue
                    if isinstance(w, Checkbox):
                        w.value = bool(value)
                    elif isinstance(w, Input):
                        w.value = str(value)
                    else:  # Select and any other value widget
                        try:
                            w.value = str(value)
                        except Exception:  # boundary: preset value not among select choices
                            pass
                self.refresh_ctq_visibility()

    # ---- input handlers ------------------------------------------------------

    @work(thread=True, exclusive=True)
    def action_run(self) -> None:
        self._clear_live_progress()  # start each run with a clean live-progress widget
        self._run_start_ts = time.monotonic()  # S1.7: ETA clock starts now
        self.set_status("Validating...")
        cfg = self._read_config()
        errors = run_config.validate(cfg)
        if errors:
            for e in errors:
                self.log_msg(f"VALIDATION ERROR: {e}")
            self.set_status("Validation failed.")
            # S2.1: surface the first error as an error toast (log lines kept).
            self.emit_toast(errors[0], "error")
            return

        # Arm the Stop button (main thread) before launching the worker subprocess.
        self.call_from_thread(self._enable_stop)

        if self.family == Family.GGUF:
            self._run_gguf(cfg)
        else:
            self._run_comfy(cfg)

    def _run_proc(self, cmd: list[str], cwd: str) -> int:
        """Launch a worker via the (Textual-free) :class:`WorkerRunner` and stream its
        stdout to this app as the ``LogObserver``.

        The runner injects ``-u`` (unbuffered) as ``argv[1]`` and pumps the child's
        stdout with ``os.read`` on the raw pipe fd so tqdm / convert_to_quant frames
        stream live instead of bursting at EOF. See
        ``quantui.worker_runner`` for the byte-path details.
        """
        return self.runner.run(
            cmd,
            cwd,
            observer=self,
            progress_debug_fh=self._progress_debug_fh,
        )

    def _run_gguf(self, cfg: "run_config.RunConfig") -> None:
        cmd = run_config.build_gguf_cmd(cfg.gguf)
        method = cfg.gguf.method

        self.app.call_from_thread(self.set_status, f"Running ({method})...")

        t0 = time.monotonic()
        rc = self._run_proc(cmd, REPO_ROOT)

        # Disarm Stop regardless of outcome (runs on the main thread).
        self.app.call_from_thread(
            self._note_duration, time.monotonic() - t0
        )
        self.app.call_from_thread(self._disable_stop, rc)

        if self._stop_requested:
            # User pressed Stop: report the stop and do NOT show a "Failed" verdict.
            self.app.call_from_thread(self.set_status, "Stopped")
            self.app.call_from_thread(self.log_msg, "=== Quantization stopped by user ===")
        elif rc == 0:
            self.app.call_from_thread(self.set_status, "Done \u2713")
            self.app.call_from_thread(self.log_msg, "=== Quantization finished successfully ===")
        else:
            self.app.call_from_thread(self.set_status, f"Failed (exit {rc})")
            self.app.call_from_thread(self.log_msg, f"=== Worker exited with code {rc} ===")

    def _run_comfy(self, cfg: "run_config.RunConfig") -> None:
        cmd = run_config.build_ctq_cmd(cfg.ctq)

        self.app.call_from_thread(self.set_status, "Running (ComfyUI)...")

        t0 = time.monotonic()
        rc = self._run_proc(cmd, REPO_ROOT)

        # Disarm Stop regardless of outcome (runs on the main thread).
        self.app.call_from_thread(
            self._note_duration, time.monotonic() - t0
        )
        self.app.call_from_thread(self._disable_stop, rc)

        if self._stop_requested:
            # User pressed Stop: report the stop and do NOT show a "Failed" verdict.
            self.app.call_from_thread(self.set_status, "Stopped")
            self.app.call_from_thread(self.log_msg, "=== Quantization stopped by user ===")
        elif rc == 0:
            self.app.call_from_thread(self.set_status, "Done \u2713")
            self.app.call_from_thread(self.log_msg, "=== ComfyUI quantization finished successfully ===")
        else:
            self.app.call_from_thread(self.set_status, f"Failed (exit {rc})")
            self.app.call_from_thread(self.log_msg, f"=== Worker exited with code {rc} ===")

    def _note_duration(self, seconds: float) -> None:
        """Capture the just-finished run's wall-clock duration on the main thread
        (consumed by _finish_run_record; S2.1/S2.2)."""
        self._last_run_duration = float(seconds)

    @work(thread=True, exclusive=True)
    def action_validate_comfy(self) -> None:
        """Validate a quantized ``.safetensors`` (structural integrity, no re-quant).

        Runs on a worker thread (the structural pass is pure-stdlib and fast even for
        large files since only the safetensors *header* is read). Results are marshaled
        back to the main thread via ``call_from_thread`` and printed to the RichLog.
        """
        path = self.query_one("#validate_path").value.strip()
        self.app.call_from_thread(self.set_status, "Validating...")

        if not path:
            self.app.call_from_thread(self.log_msg, "VALIDATION ERROR: no file path provided")
            self.app.call_from_thread(self.set_status, "Validation failed.")
            return

        self.app.call_from_thread(self.log_msg, f"=== Validating {path} ===")
        try:
            report = quant_validator.validate_comfy_quant(path, numeric=False)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self.log_msg, f"VALIDATION ERROR: {exc}")
            self.app.call_from_thread(self.set_status, "Validation failed.")
            return

        for line in quant_validator.format_report(report).splitlines():
            self.app.call_from_thread(self.log_msg, line)

        # S2.3: structured issue rows on the ResultsCard (log lines kept above).
        def _show_rows():
            try:
                card = self.query_one(ResultsCard)
                card.show_issues(report_to_issues(report))
            except NoMatches:
                pass  # results card not mounted yet

        self.app.call_from_thread(_show_rows)
        self.app.call_from_thread(
            self.set_status, "Valid \u2713" if report.ok else "Invalid \u2717"
        )
        self.app.call_from_thread(
            self.emit_toast,
            "Validation passed \u2713" if report.ok else "Validation found problems",
            "information" if report.ok else "error",
        )


# IMP-001 S3B.2: the command-palette provider moved verbatim to
# quantui/palette.py (duck-typed, no app import). Re-imported here so the
# pinned ``from quantui.app import QuantCommands`` contract keeps working,
# then wired into COMMANDS below.
from .palette import QuantCommands  # noqa: E402  (must follow the app class)

# Wire the provider into the app AFTER import (S3.1).
QuantApp.COMMANDS = set(QuantApp.COMMANDS or ()) | {QuantCommands}



if __name__ == "__main__":
    QuantApp().run()
