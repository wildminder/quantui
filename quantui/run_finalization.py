"""Run-finalization cluster: run-record persistence + completion feedback.

IMP-006 extraction (2026-09-04): the output-ownership / run-finalization state
machine that accumulated in ``app.py``'s composition root:

* ``_finish_run_record`` -- shared S2.1/S2.2/S2.5 finalizer (toast + bell +
  results card + recent-job persistence) with the output-ownership read
  (``_current_output_path``);
* ``_note_duration`` -- main-thread capture of the wall-clock duration;
* ``_ring_bell`` -- terminal BEL (S2.1).

Moved VERBATIM into ``RunFinalizationMixin``; ``QuantApp`` mixes it in, so
every ``QuantApp.method`` path (incl. test monkeypatch targets like
``a._ring_bell``) is unchanged. Duck-typed app surface: ``_profile_fields``,
``selected_method``, ``_read_config``, ``_store``, ``_debug_swallow``,
``emit_toast``, ``query_one`` (all provided by QuantApp / HandlersMixin).
"""

from __future__ import annotations

import sys
import time

from textual.css.query import NoMatches
from textual.widgets import Input

from . import profiles_store
from .quant_methods import Family
from .widgets_results import ResultsCard


class RunFinalizationMixin:
    """Run-finalization cluster of QuantApp. See module docstring."""

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

    def _note_duration(self, seconds: float) -> None:
        """Capture the just-finished run's wall-clock duration on the main thread
        (consumed by _finish_run_record; S2.1/S2.2)."""
        self._last_run_duration = float(seconds)

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
