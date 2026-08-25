"""Event handlers + validation / command-build logic (T07).

These methods are moved verbatim from ``app.py`` into a ``HandlersMixin`` so the
composition root (``QuantApp``) stays focused on wiring. ``QuantApp`` inherits
``HandlersMixin`` (``class QuantApp(HandlersMixin, App)``); every method here uses
``self.`` exactly as before, so there is zero behaviour change.

To keep this module self-contained and free of a circular import back into ``app.py``,
the project constants (DEFAULT_CTQ_FORMAT / DEFAULT_CTQ_OUTPUT_MODE /
WORKER_CTQ_MODULE) are single-homed in ``run_config.py`` (NTH-001) and re-exported
here.
"""

import os
import sys
import time

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, Label, RadioSet, Select

from . import capabilities, run_config, screens
from .quant_methods import (
    METHODS,
    Family,
    comfy_format,
    list_line,
)

# --- project constants (NTH-001: single home in run_config; re-exported here
# for backward compatibility with existing imports/tests) ----------------------
from .run_config import (  # noqa: E402, F401  (WORKER_CTQ_MODULE is a re-export)
    DEFAULT_CTQ_FORMAT,
    DEFAULT_CTQ_OUTPUT_MODE,
    WORKER_CTQ_MODULE,
)


class HandlersMixin:
    # ---- ctq helpers ---------------------------------------------------------

    def selected_method(self) -> str:
        custom = self.query_one("#custom", Input).value.strip()
        if custom:
            return custom
        return str(self.query_one("#method", Select).value)

    def ctq_format(self) -> str:
        val = self.query_one("#ctq_format", Select).value
        return str(val) if val else DEFAULT_CTQ_FORMAT

    def ctq_option_value(self, opt) -> object:
        # The ComfyUI panel renders only a fixed set of option widgets (#ctq_scaling_mode,
        # #convrot_group_size). Any option declared in the format registry but not rendered
        # as a widget (e.g. #block_size for int8_block) has no node to query. Fall back to
        # the option's declared default so reading config never crashes the run.
        try:
            widget = self.query_one(f"#{opt.key}")
        except NoMatches:
            return opt.default
        if opt.widget == "checkbox":
            return bool(widget.value)
        return widget.value

    def ctq_output_mode(self) -> str:
        """Return the ctq output mode for a sharded input: ``"single"`` or ``"sharded"``.

        Default is ``"sharded"`` (the existing per-shard directory output). ``"single"``
        merges the input shards and quantizes once into a single ``.safetensors`` file.
        """
        try:
            w = self.query_one("#ctq_output_mode", RadioSet)
            if w.query_one("#om_single").value:
                return "single"
            return "sharded"
        except NoMatches:
            return DEFAULT_CTQ_OUTPUT_MODE

    def ctq_quant_tags(self) -> list[str]:
        """Short, filesystem-safe tags describing the current ComfyUI/ctq configuration.

        Used to build a *meaningful* output filename (e.g. ``fp8_e4m3`` or
        ``int8_convrot-gs256`` or ``int8_block-simple-lowmem``) instead of the opaque
        ``-CTQ`` suffix. The format id is the primary descriptor; extra tags only
        capture options that change the emitted artifact:
        - ``gs<group_size>`` for the ConvRot group size (only when relevant);
        - ``simple`` / ``lowmem`` when those toggles are on;
        - ``calib<N>`` when a custom calibration-sample count is set.
        ``comfy_quant`` is always on and ``save_quant_metadata`` only adds a sidecar
        file, so neither is encoded in the name.

        The actual tag computation lives in :func:`run_config.ctq_quant_tags`; this
        method only reads the widgets and delegates so there is one source of truth.
        """
        fmt = self.ctq_format()
        gs = None
        if fmt == "int8_convrot":
            try:
                gs = self.query_one("#convrot_group_size", Select).value
            except NoMatches:
                gs = None
        simple = lowmem = False
        try:
            simple = bool(self.query_one("#ctq_simple", Checkbox).value)
        except NoMatches:
            pass
        try:
            lowmem = bool(self.query_one("#ctq_low_memory", Checkbox).value)
        except NoMatches:
            pass
        calib = ""
        try:
            calib = self.query_one("#ctq_calib_samples", Input).value.strip()
        except NoMatches:
            pass
        heur = False
        try:
            heur = bool(self.query_one("#heur", Checkbox).value)
        except NoMatches:
            pass
        return run_config.ctq_quant_tags(fmt, gs, simple, lowmem, calib, heur)

    def ctq_output_stem(self, base: str) -> str:
        """Build a meaningful output filename stem: ``<base>-<quant_tags>``."""
        return f"{base}-" + "-".join(self.ctq_quant_tags())

    # ---- input handlers ------------------------------------------------------

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        # Only the top-level family RadioSet switches the whole UI. Other RadioSets
        # inside a panel (e.g. #ctq_output_mode) must NOT trigger a family change,
        # otherwise selecting "Single file (merge)" flipped the tab back to GGUF.
        if event.radio_set.id != "family":
            return
        rid = event.pressed.id
        self.family = Family.COMFY if rid == "fam_comfy" else Family.GGUF
        is_comfy = self.family == Family.COMFY
        self.query_one("#gguf_panel").display = not is_comfy
        self.query_one("#comfy_panel").display = is_comfy
        self.refresh_ctq_visibility()
        if is_comfy:
            try:
                pybin = self.query_one("#pybin_ctq", Input).value.strip() or sys.executable
            except Exception:
                pybin = sys.executable
            self.refresh_capabilities(pybin)
        self.set_status("Ready." if not is_comfy else "ComfyUI mode.")

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id
        if sid == "method":
            self.update_method_info()
            self.auto_suggest_output(Family.GGUF)
        elif sid == "ctq_format":
            self.refresh_ctq_visibility()
        elif sid == "ctq_preset":
            self.apply_preset()

    def on_input_changed(self, event) -> None:
        """Live .pt detection while the user types/pastes into #ctq_input."""
        if getattr(event.input, "id", "") == "ctq_input":
            self.update_pt_suggest()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        iid = event.input.id
        if iid == "model":
            self.auto_suggest_output(Family.GGUF)
        elif iid == "ctq_input":
            self.auto_suggest_output(Family.COMFY)
            # The output-mode radio only applies to sharded inputs; re-evaluate.
            self.refresh_ctq_visibility()
            self.update_pt_suggest()
        elif iid == "ctq_output":
            self.auto_suggest_output(Family.COMFY)
        elif iid == "pybin_ctq":
            self.refresh_capabilities(event.input.value.strip() or sys.executable)

    # ---- S3.2: inline blur validation (wizard part 1) ---------------------------
    REQUIRED_FIELDS: dict[str, str] = {
        "model": "Model path is required.",
        "output": "Output folder is required.",
        "ctq_input": "Input path is required.",
        "ctq_output": "Output path is required.",
    }

    def on_input_blurred(self, event: Input.Blurred) -> None:
        """S3.2: empty REQUIRED fields get a red border + hint label on blur.

        Never blocks typing (validation only on blur); a filled field clears
        the hint and the ``-invalid`` class. Non-required fields are ignored.

        User-report fix: blurring ``#ctq_input`` also re-evaluates the
        output-mode visibility -- a pasted/typed sharded folder path only
        becomes a *checkable* directory once the field loses focus, and the
        "Output mode" radio must appear right then (not just after Enter).
        """
        iid = event.input.id or ""
        if iid == "ctq_input":
            self.refresh_ctq_visibility()
            self.update_pt_suggest()
        message = self.REQUIRED_FIELDS.get(iid)
        if message is None:
            return  # not a required field
        empty = not (event.input.value or "").strip()
        try:
            event.input.set_class(empty, "-invalid")
        except NoMatches:
            pass
        hint_id = f"field_hint_{iid}"
        try:
            hint = self.query_one(f"#{hint_id}")
        except NoMatches:
            if not empty:
                return  # nothing to clear
            # Create the hint lazily right below the field.
            try:
                event.input.parent.mount(
                    Label(message, id=hint_id, classes="field_hint_visible"),
                    after=event.input,
                )
            except NoMatches:  # boundary: parent gone mid-event; polish only
                return
            return
        try:
            hint.update(message if empty else "")
            hint.display = bool(empty)
        except NoMatches:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        b = event.button.id
        if b == "browse_model":
            self.push_screen(screens.PathModal(self.query_one("#model", Input).value, file_mode=True), self.set_model)
        elif b == "browse_out":
            self.push_screen(screens.PathModal(self.query_one("#output", Input).value), self.set_out)
        elif b == "browse_ctq_in":
            self.push_screen(screens.PathModal(self.query_one("#ctq_input", Input).value, file_mode=True), self.set_ctq_model)
        elif b == "browse_ctq_out":
            self.push_screen(screens.PathModal(self.query_one("#ctq_output", Input).value), self.set_ctq_out)
        elif b == "browse_validate_in":
            self.push_screen(screens.PathModal(self.query_one("#validate_path", Input).value, file_mode=True), self.set_validate_path)
        elif b == "validate":
            self.action_validate_comfy()
        elif b in ("run", "run_ctq"):
            # The Run button morphs into Stop while a run is active (no extra button).
            if self._run_active:
                self.action_stop_run()
            else:
                self.action_run()
        elif b == "listm":
            self.list_methods()
        elif b == "pt_convert_btn":
            self.action_convert_pt()
        elif b == "copy_log":
            self.copy_log()
        elif b == "save_log":
            self.save_log()

    def action_stop_run(self) -> None:
        """Stop handler (main thread, via on_button_pressed on the morphed Run/Stop button).

        Threshold-gated (plan S1.10): if the run has been going for less than
        ``STOP_CONFIRM_AFTER_S`` seconds, terminate immediately (keeps the
        existing quick-abort semantics); otherwise push a ConfirmModal first and
        only terminate when the user confirms. Idempotent while stopping.
        """
        if self._stop_requested:
            return  # already stopping; ignore repeat clicks
        if not self.runner.is_running():
            return  # no live run -> no-op

        from . import app as app_mod  # local: avoid a module-level cycle

        elapsed = 0.0
        start = getattr(self, "_run_start_ts", 0.0)
        if start > 0:
            elapsed = time.monotonic() - start
        if elapsed < app_mod.STOP_CONFIRM_AFTER_S:
            self._terminate_run()
            return
        # Long-running job -> confirm before terminating.
        screen = screens.ConfirmModal(
            "Terminate running quantization?",
            confirm_label="Terminate",
            cancel_label="Keep running",
        )

        def _on_confirm(do_stop: bool | None) -> None:
            if do_stop:
                self._terminate_run()

        self.app.push_screen(screen, _on_confirm)

    def _terminate_run(self) -> None:
        """Flag the stop and kill the worker subprocess (shared by both paths)."""
        self._stop_requested = True
        self.runner.terminate()
        self.log_msg("Stop requested — terminating worker process...")

    def set_model(self, result: str) -> None:
        if result:
            self.query_one("#model", Input).value = result
            self.auto_suggest_output(Family.GGUF)

    def set_out(self, result: str) -> None:
        if result:
            self.query_one("#output", Input).value = result

    def set_ctq_model(self, result: str) -> None:
        if result:
            self.query_one("#ctq_input", Input).value = result
            self.auto_suggest_output(Family.COMFY)
            # The output-mode radio only applies to sharded inputs; re-evaluate.
            self.refresh_ctq_visibility()
            self.update_pt_suggest()

    # ---- .pt suggestion box (user feature) ------------------------------------

    def update_pt_suggest(self) -> None:
        """Show the .pt suggestion box iff #ctq_input points at a checkpoint.

        Called after any change to ``#ctq_input`` (submit, browse, profile apply).
        """
        try:
            from . import pt_convert

            box = self.query_one("#pt_suggest")
            inp = self.query_one("#ctq_input", Input).value.strip()
            box.display = pt_convert.is_pt_file(inp)
        except NoMatches:  # boundary: panel not mounted (e.g. GGUF family)
            pass

    def action_convert_pt(self) -> None:
        """[Convert to safetensors] button: run the converter worker, then
        re-point #ctq_input at the produced .safetensors."""
        from . import pt_convert

        src = self.query_one("#ctq_input", Input).value.strip()
        if not pt_convert.is_pt_file(src):
            self.emit_toast("Input is not a .pt/.pth/.ckpt file.", "warning")
            return
        out = pt_convert.default_output_path(src)

        self.set_status("Converting .pt ...")

        def _on_done() -> None:
            self.query_one("#ctq_input", Input).value = out
            self.auto_suggest_output(Family.COMFY)
            self.update_pt_suggest()
            self.refresh_ctq_visibility()
            self.set_status("Ready.")
            self.emit_toast("Checkpoint converted to safetensors.", "information")

        self.run_worker(
            lambda: self._convert_pt_thread(src, out, _on_done),
            thread=True,
            exclusive=True,
            group="pt_convert",
        )

    def _convert_pt_thread(self, src: str, out: str, on_done) -> None:
        """Worker-thread body: run the conversion in-process via pt_convert.

        The TUI process itself has torch available whenever it can run GGUF jobs;
        for heavy checkpoints users can also point 'Worker Python' elsewhere later.
        """
        from . import pt_convert

        try:
            report = pt_convert.convert_pt_to_safetensors(
                src, out,
                progress=lambda done, total, label="":
                    None,  # header bar is driven by the store; keep it simple
            )
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(
                self.log_msg, f"PT CONVERT ERROR: {exc}"
            )
            self.app.call_from_thread(self.set_status, "Conversion failed.")
            self.app.call_from_thread(self.emit_toast, f"Conversion failed: {exc}", "error")
            return
        for line in (
            f"=== Converted {os.path.basename(report.source)} ===",
            f"candidate: {report.selected}",
            f"tensors: {report.n_tensors} ({report.n_params/1e6:.1f}M params)",
            f"output: {report.output} ({report.bytes_written/1e9:.2f} GB)",
        ):
            self.app.call_from_thread(self.log_msg, line)
        self.app.call_from_thread(on_done)

    def set_validate_path(self, result: str) -> None:
        if result:
            self.query_one("#validate_path", Input).value = result

    def set_ctq_out(self, result: str) -> None:
        if result:
            self.query_one("#ctq_output", Input).value = result
            self.auto_suggest_output(Family.COMFY)

    def _output_state(self, out: str) -> str:
        """Classify the output field: empty | file_path | dir_path.

        Used by ``auto_suggest_output`` so it only fills the output when the user
        has not already typed an explicit file (Cases B/E) or when the field is a
        directory that should receive the artifact (Cases C/F).
        """
        if not out:
            return "empty"
        if os.path.isdir(out):
            return "dir_path"
        if out.endswith(os.sep):
            return "dir_path"
        if os.path.splitext(out)[1]:
            return "file_path"
        return "dir_path"

    def auto_suggest_output(self, family: Family | None = None) -> None:
        family = family or self.family
        cfg = self._read_config()
        if family == Family.GGUF:
            suggested = run_config.suggest_gguf_output(
                cfg.gguf.model, cfg.gguf.output, cfg.gguf.method
            )
            if suggested:
                self.query_one("#output", Input).value = suggested
        else:  # ComfyUI / convert_to_quant -- 6-combo auto-naming (plan §5.2)
            suggested = run_config.suggest_comfy_output(
                cfg.ctq.input,
                cfg.ctq.output,
                cfg.ctq.quant_tags,
                cfg.ctq.output_mode,
            )
            if suggested:
                self.query_one("#ctq_output", Input).value = suggested

    def list_methods(self) -> None:
        self.log_msg("=== Available quantization methods ===")
        for m in METHODS:
            self.log_msg(list_line(m))
        self.log_msg("======================================")

    # ---- capability badge (non-blocking) -------------------------------------

    @work(thread=True)
    def refresh_capabilities(self, pybin: str | None = None) -> None:
        """Probe the ctq interpreter and render advisory warnings in the badge."""
        if pybin is None:
            try:
                pybin = self.query_one("#pybin_ctq", Input).value.strip() or sys.executable
            except Exception:
                pybin = sys.executable
        try:
            report = capabilities.probe_worker_env(pybin)
        except Exception as exc:  # noqa: BLE001
            self.app.call_from_thread(self.set_capability, f"Capability probe failed: {exc}", True)
            return
        warns = capabilities.check_ctq_requirements(report, self.ctq_format())
        if warns:
            self.app.call_from_thread(self.set_capability, " | ".join(warns), True)
        else:
            self.app.call_from_thread(self.set_capability, "Capabilities OK for this format.", False)

    def set_capability(self, msg: str, warn: bool) -> None:
        try:
            w = self.query_one("#ctq_cap_warn", Label)
            w.update(msg)
            w.set_class(warn, "warn")
        except NoMatches:  # boundary: badge not mounted (GGUF family active)
            pass

    # ---- validation + run ----------------------------------------------------

    def _read_config(self) -> "run_config.RunConfig":
        """Read every widget once into a :class:`run_config.RunConfig`.

        This is the single dependency-injection point: pure downstream logic
        (``run_config.validate`` / ``build_ctq_cmd`` / ``auto_suggest_output``)
        consumes the dataclass instead of ``self.query_one``, so it is unit-testable
        without booting the Textual app.
        """
        gguf = run_config.GgufConfig(
            model=self.query_one("#model", Input).value.strip(),
            output=self.query_one("#output", Input).value.strip(),
            method=self.selected_method(),
            pybin=self.query_one("#pybin", Input).value.strip(),
            max_seq_length=self.query_one("#maxseq", Input).value.strip() or "4096",
            load_in_4bit=self.query_one("#load4", Checkbox).value,
            push_to_hub=self.query_one("#push", Checkbox).value,
            hub_repo=self.query_one("#hub", Input).value.strip(),
            hf_token=self.query_one("#token", Input).value.strip(),
        )
        preset_raw = self.query_one("#ctq_preset", Select).value
        preset = str(preset_raw) if (preset_raw and preset_raw is not Select.BLANK) else None
        fmt = self.ctq_format()
        cf = comfy_format(fmt)
        option_values = {opt.key: self.ctq_option_value(opt) for opt in cf.extra_options}
        ctq = run_config.CtqConfig(
            input=self.query_one("#ctq_input", Input).value.strip(),
            output=self.query_one("#ctq_output", Input).value.strip(),
            pybin=self.query_one("#pybin_ctq", Input).value.strip() or sys.executable,
            format=fmt,
            output_mode=self.ctq_output_mode(),
            preset=preset,
            option_values=option_values,
            comfy_quant=self.query_one("#ctq_comfy_quant", Checkbox).value,
            save_quant_metadata=self.query_one("#ctq_save_quant_metadata", Checkbox).value,
            simple=self.query_one("#ctq_simple", Checkbox).value,
            low_memory=self.query_one("#ctq_low_memory", Checkbox).value,
            calib_samples=self.query_one("#ctq_calib_samples", Input).value.strip(),
            num_iter=self.query_one("#ctq_num_iter", Input).value.strip(),
            quant_tags=self.ctq_quant_tags(),
        )
        return run_config.RunConfig(family=self.family, gguf=gguf, ctq=ctq)

    def validate(self) -> list[str]:
        """Validate the current UI state by delegating to :func:`run_config.validate`."""
        return run_config.validate(self._read_config())

    def build_ctq_cmd(self) -> list[str]:
        """Build the ``worker_ctq.py`` argument vector by delegating to
        :func:`run_config.build_ctq_cmd` over the read-only :class:`RunConfig`."""
        return run_config.build_ctq_cmd(self._read_config().ctq)
