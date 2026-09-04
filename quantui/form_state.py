"""Form-state cluster: profile snapshot/apply + method-info + ctq visibility.

IMP-006 extraction (2026-09-04): these methods accumulated in ``app.py``'s
composition root after v0.2.1 even though they form one coherent cluster --
"read/write the FORM STATE of the two family panels":

* profile snapshot/apply (S2.4): ``_profile_fields`` / ``_apply_profile_fields``
  (+ the safe widget readers they share);
* method-info line: ``update_method_info`` (T9: #method is a free-text Input);
* data-driven ctq visibility: ``refresh_ctq_visibility`` with its
  ``_option_live_value`` / ``_set_option_display`` helpers;
* preset application: ``apply_preset``.

Moved VERBATIM into ``FormStateMixin``; ``QuantApp`` mixes it in, so every
``QuantApp.method`` attribute path and monkeypatch target is unchanged. The
mixin duck-types the app (``self.query_one`` / ``self.family`` / the
``imatrix_value`` / ``ctq_format`` / ``selected_method`` helpers from
``handlers.HandlersMixin``), so it stays Textual-friendly but testable.
"""

from __future__ import annotations

from typing import Any

from textual.css.query import NoMatches
from textual.widgets import Checkbox, Input, Select, Static
from textual.widgets._select import InvalidSelectValueError

from . import profiles_store  # noqa: F401  (public re-export surface kept)
from .quant_methods import (
    COMFY_FORMATS,
    METHODS_BY_ID,
    Family,
    comfy_format,
    comfy_preset,
    eval_visible_when,
)
from .quant_methods import (
    is_sharded_folder as quant_methods_is_sharded_folder,
)
from .screens import _literal


class FormStateMixin:
    """Form-state cluster of QuantApp (profile snapshot/apply, method info,
    ctq visibility, presets). See module docstring."""

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

    # ---- method info line -------------------------------------------------------
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

    # ---- data-driven ctq visibility ---------------------------------------------
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
