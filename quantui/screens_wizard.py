"""Wizard mode (plan S3.3): a 4-step guided flow over the same config seam.

Steps: 1 Paths -> 2 Method/Format -> 3 Options (advanced review) -> 4 Review
(``_read_config()``-style summary + [Start]). Start copies the collected values
into the MAIN form widgets via ``QuantApp._apply_profile_fields`` and then calls
the SAME ``action_run`` path -- so the wizard produces a byte-identical command
vs the form for identical inputs (determinism gate).
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static

from quantui.quant_methods import COMFY_FORMATS, DEFAULT_GGUF_METHOD, METHODS

# (label, id) pairs for the wizard's own method Select. The MAIN form uses a
# free-text #method Input (T8), so these pairs are wizard-only.
SELECT_METHOD_OPTIONS = [(m.label, m.id) for m in METHODS]


class WizardScreen(ModalScreen):
    """Stepped quantization wizard; dismisses None on cancel."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self) -> None:
        super().__init__()
        self.step = 0
        self.values: dict = {"family": "gguf"}

    def compose(self) -> ComposeResult:
        yield Vertical(
            Label("Quantization wizard", id="wizard_title"),
            Vertical(id="wizard_body"),
            Horizontal(
                Button("Back", id="wiz_back"),
                Button("Next", id="wiz_next", variant="success"),
                classes="buttons",
            ),
            classes="modal",
        )

    def on_mount(self) -> None:
        self._render_step()

    # ---- steps ---------------------------------------------------------------
    def _render_step(self) -> None:
        body = self.query_one("#wizard_body", Vertical)
        body.remove_children()
        title = self.query_one("#wizard_title", Label)
        if self.step == 0:
            title.update("Step 1/4 — Paths & family")
            body.mount(
                Label("Model / input path"),
                Input(self.values.get("model", ""), id="wiz_model"),
                Label("Output path"),
                Input(self.values.get("output", ""), id="wiz_output"),
                Static("Family: gguf = Unsloth GGUF | comfy = ComfyUI convert_to_quant"),
            )
        elif self.step == 1:
            title.update("Step 2/4 — Family & method")
            body.mount(
                Label("family"),
                Select(
                    [("gguf", "gguf"), ("comfy", "comfy")],
                    value=self.values.get("family", "gguf"),
                    id="wiz_family",
                    allow_blank=False,
                ),
                Label("method"),
                # T3b: pin the default -- an unpinned Select lands on
                # METHODS[0] (not_quantized), which wastes the run.
                Select(
                    SELECT_METHOD_OPTIONS, id="wiz_method", allow_blank=False,
                    value=self.values.get("method", DEFAULT_GGUF_METHOD),
                ),
            )
        elif self.step == 2:
            title.update("Step 3/4 — Format options")
            body.mount(
                Label("ctq format (comfy family only)"),
                Select(
                    [(f.id, f.id) for f in COMFY_FORMATS],
                    value=self.values.get("ctq_format") or COMFY_FORMATS[0].id,
                    id="wiz_ctq_format",
                    allow_blank=False,
                ),
            )
        else:
            title.update("Step 4/4 — Review")
            rows = [
                f"model/input : {self.values.get('model', '')}",
                f"output      : {self.values.get('output', '')}",
                f"family      : {self.values.get('family', 'gguf')}",
                f"method      : {self.values.get('method', '')}",
                f"ctq format  : {self.values.get('ctq_format', '')}",
            ]
            from textual.containers import VerticalScroll

            body.mount(VerticalScroll(Static("\n".join(rows), markup=False)))

    def _harvest(self) -> None:
        """Copy the current step's widget values into self.values."""
        ids_by_step = {
            0: ("wiz_model", "wiz_output"),
            1: ("wiz_family", "wiz_method"),
            2: ("wiz_ctq_format",),
        }
        keymap = {
            "wiz_model": "model",
            "wiz_output": "output",
            "wiz_family": "family",
            "wiz_method": "method",
            "wiz_ctq_format": "ctq_format",
        }
        for wid in ids_by_step.get(self.step, ()):
            try:
                w = self.query_one(f"#{wid}")
                val = getattr(w, "value", "")
                val = str(val) if val is not Select.BLANK else ""
                if wid == "wiz_output":
                    # Output maps to the right field per family.
                    if self.values.get("family") == "comfy":
                        self.values["ctq_output"] = val
                    else:
                        self.values["output"] = val
                elif wid == "wiz_model":
                    if self.values.get("family") == "comfy":
                        self.values["ctq_input"] = val
                    self.values["model"] = val
                else:
                    self.values[keymap[wid]] = val
            except NoMatches:
                pass  # step widget not mounted

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "wiz_back":
            self._harvest()
            if self.step > 0:
                self.step -= 1
                self._render_step()
        elif bid == "wiz_next":
            self._harvest()
            if self.step < 3:
                self.step += 1
                self._render_step()
            else:
                self.dismiss(dict(self.values))

    def action_cancel(self) -> None:
        self.dismiss(None)
