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
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
)

from . import capabilities, run_config, screens
from .quant_methods import (
    Family,
    comfy_format,
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
        # T9 (plan 2026-08-31-gguf-unsloth-parity): #method is an Input now
        # (multi-method free text, T8). Custom override still wins; BOTH paths
        # strip whitespace. A comma list is returned verbatim -- the same
        # parse_methods downstream (validate_gguf / worker) splits it.
        custom = self.query_one("#custom", Input).value.strip()
        if custom:
            return custom
        return self.query_one("#method", Input).value.strip()

    def imatrix_value(self) -> str:
        """The GGUF imatrix setting: "" | "auto" | local path.

        "auto" (the #imatrix_auto checkbox) wins over the path field --
        checking "fetch upstream" while a stale path sits in the field must
        not silently load the stale file.
        """
        try:
            if self.query_one("#imatrix_auto", Checkbox).value:
                return "auto"
        except NoMatches:
            return ""
        return self.query_one("#imatrix_path", Input).value.strip()

    def ctq_format(self) -> str:
        val = self.query_one("#ctq_format", Select).value
        return str(val) if val else DEFAULT_CTQ_FORMAT

    def ctq_option_value(self, opt) -> object:
        # Every registry OptionField has a matching widget in the ComfyUI panel
        # (#scaling_mode, #block_size, #convrot, #convrot_group_size, ...). The
        # fallback to opt.default remains for robustness (widget not mounted).
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
        ``int8-row-convrot-gs256`` or ``int8-block-simple-lowmem``) instead of the
        opaque ``-CTQ`` suffix. The format id is the primary descriptor; extra tags
        only capture options that change the emitted artifact:
        - ``<scaling>`` for INT8 scaling mode (block/tensor/row);
        - ``convrot`` + ``gs<group_size>`` when ConvRot is on;
        - ``simple`` / ``lowmem`` when those toggles are on;
        - ``calib<N>`` when a custom calibration-sample count is set.
        ``comfy_quant`` is always on and ``save_quant_metadata`` only adds a sidecar
        file, so neither is encoded in the name.

        The actual tag computation lives in :func:`run_config.ctq_quant_tags`; this
        method only reads the widgets and delegates so there is one source of truth.
        """
        fmt = self.ctq_format()
        scaling = None
        convrot = False
        gs = None
        if fmt == "int8":
            try:
                scaling = str(self.query_one("#scaling_mode", Select).value)
            except NoMatches:
                scaling = None
            try:
                convrot = bool(self.query_one("#convrot", Checkbox).value)
            except NoMatches:
                convrot = False
            if convrot:
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
        return run_config.ctq_quant_tags(
            fmt, gs, simple, lowmem, calib, heur, scaling=scaling, convrot=convrot
        )

    def ctq_output_stem(self, base: str) -> str:
        """Build a meaningful output filename stem: ``<base>-<quant_tags>``."""
        return f"{base}-" + "-".join(self.ctq_quant_tags())

    # ---- input handlers ------------------------------------------------------

    def _set_family(self, family: Family, *, press_radio: bool = True) -> None:
        """Switch the active family **now** -- not after the radio message pumps.

        ``RadioButton.value = True`` only *posts* ``RadioSet.Changed``; the
        handler that assigns ``self.family`` runs on the NEXT message pump
        cycle. Any caller that switches family and then acts inside the SAME
        synchronous block (``_on_wizard_done`` -> ``action_run``) therefore
        read a STALE family in ``_read_config`` and validated the wrong panel.
        That latent bug was masked while ``#method`` was a Select (a comfy
        wizard run validated as GGUF but happened to pass); free-text method
        entry (T8) made it fatal. Both paths now converge here.

        The work is idempotent: the radio press re-enters this method via the
        ``RadioSet.Changed`` message, and that second call is a no-op.
        """
        changed = self.family != family
        self.family = family
        is_comfy = family == Family.COMFY
        if press_radio:
            try:
                self.query_one(
                    "#fam_comfy" if is_comfy else "#fam_gguf", RadioButton
                ).value = True
            except NoMatches:
                pass  # family radio not mounted
        for wid, visible in (("#gguf_panel", not is_comfy), ("#comfy_panel", is_comfy)):
            try:
                self.query_one(wid).display = visible
            except NoMatches:
                pass  # panel not mounted
        self.refresh_ctq_visibility()
        if is_comfy and changed:
            try:
                pybin = self.query_one("#pybin_ctq", Input).value.strip() or sys.executable
            except Exception:
                pybin = sys.executable
            self.refresh_capabilities(pybin)
        self.set_status("Ready." if not is_comfy else "ComfyUI mode.")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        # Only the top-level family RadioSet switches the whole UI. Other RadioSets
        # inside a panel (e.g. #ctq_output_mode) must NOT trigger a family change,
        # otherwise selecting "Single file (merge)" flipped the tab back to GGUF.
        if event.radio_set.id != "family":
            return
        rid = event.pressed.id
        self._set_family(
            Family.COMFY if rid == "fam_comfy" else Family.GGUF, press_radio=False
        )

    def on_select_changed(self, event: Select.Changed) -> None:
        sid = event.select.id
        # NOTE: there is no "method" Select anymore (T8) -- #method is an Input,
        # handled by on_input_changed / on_input_submitted below.
        if sid in ("ctq_format", "scaling_mode"):
            # scaling_mode drives chained visibility (block_size for block;
            # convrot + group size for row), so re-evaluate like a format change.
            self.refresh_ctq_visibility()
            self.refresh_output_name()  # tags include fmt + scaling
        elif sid == "convrot_group_size":
            self.refresh_output_name()  # tags include gs<N>
        elif sid == "ctq_preset":
            self.apply_preset()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if getattr(event.checkbox, "id", "") == "convrot":
            # Toggling ConvRot shows/hides the group size (row scaling only).
            self.refresh_ctq_visibility()
            self.refresh_output_name()  # tags gain/lose convrot+gs

    def on_input_changed(self, event) -> None:
        """Live .pt detection (#ctq_input) + live method info (#method).

        T9: #method is free text now, so the description line must refresh as
        the user types -- including the "[IMATRIX]" marker for IQ* quants.
        Output auto-suggest is deliberately NOT fired per keystroke (noisy);
        it runs on Enter via :meth:`on_input_submitted`.
        """
        iid = getattr(event.input, "id", "")
        if iid == "ctq_input":
            self.update_pt_suggest()
            self.update_audit_button()
        elif iid == "method":
            self.update_method_info()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        iid = event.input.id
        if iid == "method":
            # Enter = "committed": now it is worth re-deriving the output name.
            self.update_method_info()
            self.auto_suggest_output(Family.GGUF)
        elif iid == "model":
            self.auto_suggest_output(Family.GGUF)
        elif iid == "ctq_input":
            self.auto_suggest_output(Family.COMFY)
            # The output-mode radio only applies to sharded inputs; re-evaluate.
            self.refresh_ctq_visibility()
            self.update_pt_suggest()
            self.update_audit_button()
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
            self.update_audit_button()
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
        elif b == "audit_ctq_in":
            self.action_audit_model()
        elif b == "browse_ctq_out":
            self.push_screen(screens.PathModal(self.query_one("#ctq_output", Input).value), self.set_ctq_out)
        elif b == "browse_validate_in":
            self.push_screen(screens.PathModal(self.query_one("#validate_path", Input).value, file_mode=True), self.set_validate_path)
        elif b == "pick_method":
            # Method-picker feature: open the 35-method list modal with the
            # current #method value preselected (a comma list preselects every
            # entry it names). The Input itself stays untouched until the user
            # Confirms -- Cancel/Escape dismiss None (no change).
            current = self.query_one("#method", Input).value
            self.push_screen(
                screens.MethodPickerScreen(initial=current),
                self._on_method_picked,
            )
        elif b == "validate":
            self.action_validate_comfy()
        elif b in ("run", "run_ctq"):
            # The Run button morphs into Stop while a run is active (no extra button).
            if self._run_active:
                self.action_stop_run()
            else:
                self.action_run()
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
            self.update_audit_button()

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

    def update_audit_button(self) -> None:
        """Enable the inline Audit button iff #ctq_input holds an auditable path.

        Mirrors the resolution rule of :meth:`action_audit_model`: the button
        is enabled exactly when :meth:`_resolve_audit_path` finds a concrete
        target for the typed value. Called at every site that reacts to
        ``#ctq_input`` changes (same sites as ``update_pt_suggest``).
        """
        try:
            raw = self.query_one("#ctq_input", Input).value.strip()
        except NoMatches:  # boundary: panel not mounted (e.g. GGUF family)
            return
        enabled = bool(self._resolve_audit_path(raw))
        try:
            self.query_one("#audit_ctq_in", Button).disabled = not enabled
        except NoMatches:  # boundary: button not mounted
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
            self.update_audit_button()
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

    def _on_method_picked(self, value) -> None:
        """MethodPickerScreen callback: apply the picked ids to #method.

        ``value`` is the comma-joined id string (registry order) on Confirm,
        or ``None`` on Cancel/Escape. An empty string (nothing selected +
        Confirm) also leaves the field alone -- clearing the method would
        break the run gate's default-method expectations.
        """
        if isinstance(value, str) and value:
            self.query_one("#method", Input).value = value
            self.update_method_info()
            self.auto_suggest_output(Family.GGUF)

    def set_ctq_out(self, result: str) -> None:
        if result:
            self.query_one("#ctq_output", Input).value = result
            self.auto_suggest_output(Family.COMFY)

    # ---- model audit (plan 2026-08-27, STEP 4.1) -----------------------------

    def _resolve_audit_path(self, raw: str) -> str:
        """Return a concrete auditable path for audit, or ``""``.

        Accepts a ``.safetensors`` file directly, a folder holding exactly one
        ``.safetensors`` (resolved to that file), or a HuggingFace sharded
        model folder (``model.safetensors.index.json`` present -- the folder
        itself is the audit target). Anything else is unusable.
        """
        if not raw:
            return ""
        if os.path.isfile(raw):
            return raw if raw.endswith(".safetensors") else ""
        if os.path.isdir(raw):
            sts = [f for f in os.listdir(raw) if f.endswith(".safetensors")]
            if len(sts) == 1:
                return os.path.join(raw, sts[0])
            if os.path.isfile(os.path.join(raw, "model.safetensors.index.json")):
                return raw
        return ""

    def action_audit_model(self) -> None:
        """Palette: audit a ``.safetensors`` checkpoint (classify + suggest).

        Resolves the target from ``#ctq_input`` when the ComfyUI family tab is
        active and the input is a ``.safetensors`` file or single-file folder;
        otherwise opens ``PathModal(file_mode=True)`` first. Then pushes the
        :class:`screens.AuditScreen` modal.
        """
        path = ""
        if self.family == Family.COMFY:
            try:
                raw = self.query_one("#ctq_input", Input).value.strip()
            except NoMatches:
                raw = ""
            path = self._resolve_audit_path(raw)
        if path:
            self.push_screen(screens.AuditScreen(path))
            return
        try:
            start = self.query_one("#ctq_input", Input).value.strip()
        except NoMatches:
            start = ""
        self.push_screen(screens.PathModal(start, file_mode=True), self._on_audit_path)

    def _on_audit_path(self, result: str) -> None:
        """PathModal callback: push the AuditScreen for the chosen file."""
        if not result:
            return
        path = self._resolve_audit_path(result)
        if not path and result.endswith(".safetensors"):
            path = result
        if path:
            self.push_screen(screens.AuditScreen(path))

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
                out_widget = self.query_one("#ctq_output", Input)
                out_widget.value = suggested
                # Track exactly what we wrote so refresh_output_name can tell
                # our own suggestion from a user-typed path.
                self._ctq_output_owned_value = suggested

    def refresh_output_name(self) -> None:
        """Re-suggest the output FILENAME while preserving the user's folder.

        User-report fix: changing an option (e.g. scaling mode) used to either
        leave a stale filename behind or, once the field held a .safetensors
        path, never update again (explicit-file rule). Expected behavior: the
        DIRECTORY part of the field stays untouched; only the auto-generated
        file name is refreshed to reflect current options.

        Ownership model:

        * Directory-shaped value (``...\\out``, with or without trailing
          separator, existing or not): always treated as a destination FOLDER.
          The auto-generated ``<stem>.safetensors`` is refreshed inside it --
          even when the user typed the folder by hand, because the user chose
          the *where*, not the *what*.
        * Explicit ``.safetensors`` value: only rewritten while its content
          equals what WE last suggested (``self._ctq_output_owned_value``).
          A hand-typed or browsed filename is the user's own choice and is
          left strictly alone.
        """
        out_widget = self.query_one("#ctq_output", Input)
        current = out_widget.value.strip()
        if not current:
            # Nothing chosen yet -- fall back to the standard full-path suggest.
            self.auto_suggest_output(Family.COMFY)
            return
        cfg = self._read_config()
        kind, base = run_config.classify_input(cfg.ctq.input)
        if base is None:
            return  # unusable input; nothing sensible to suggest
        state = run_config.output_state(current)
        if state == "file_path" and current != getattr(
            self, "_ctq_output_owned_value", None
        ):
            return  # explicit filename chosen by the user -> hands off
        stem = run_config.ctq_output_stem(base, cfg.ctq.quant_tags)
        if state == "file_path":
            candidate = os.path.join(
                os.path.dirname(os.path.abspath(current)), f"{stem}.safetensors"
            )
        else:
            target_dir = current.rstrip("\\/") or current
            candidate = os.path.join(target_dir, f"{stem}.safetensors")
        if candidate != current:
            out_widget.value = candidate
            self._ctq_output_owned_value = candidate

    # The old methods-dump handler was removed (plan 2026-09-08-scifi-ui
    # S1.2): its button is gone; the method picker modal lists all methods
    # interactively (see the removal tripwire test file).

    # ---- capability badge (non-blocking) -------------------------------------

    # Own group + exclusive: a run (``action_run`` is exclusive in the DEFAULT
    # group) must never cancel a capability probe, and a new probe cancels the
    # previous one so two probes cannot race the badge out of order.
    @work(thread=True, group="capabilities", exclusive=True)
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
        fmt = self.ctq_format()
        try:
            cf = comfy_format(fmt)
            option_values = {o.key: self.ctq_option_value(o) for o in cf.extra_options}
        except KeyError:
            option_values = {}
        warns = capabilities.check_ctq_requirements(report, fmt, option_values)
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
            imatrix=self.imatrix_value(),
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
