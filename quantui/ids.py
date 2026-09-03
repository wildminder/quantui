"""Widget-id contract constants (plan Q4a).

Single source of truth for every Textual widget id the app and its pinned tests
rely on. Layout refactors (the dashboard-first redesign) MUST keep these ids
identical; ``tests/test_widget_contract.py`` boots the app and asserts every one
of them stays queryable -- it is the refactor tripwire.

Naming convention: every constant value is a Textual selector string starting
with ``#`` so call sites can do ``self.query_one(ids.LOG, RichLog)`` directly.
"""

# Family selector -----------------------------------------------------------
FAMILY = "#family"
FAM_GGUF = "#fam_gguf"
FAM_COMFY = "#fam_comfy"

# Parameter panels ------------------------------------------------------------
GGUF_PANEL = "#gguf_panel"
COMFY_PANEL = "#comfy_panel"

# Run buttons (morph Run <-> Stop) ---------------------------------------------
RUN = "#run"
RUN_CTQ = "#run_ctq"

# Status / capability badge ----------------------------------------------------
STATUS = "#status"
CAP_WARN = "#ctq_cap_warn"

# Log (kept mounted, display-toggled; plan Q2) ----------------------------------
LOG = "#log"
LOG_DRAWER = "#log_drawer"

# Stacked progress rail (S1.8) -- replaced the single #live_progress widget,
# which was DELETED in the same commit as this constant (plan Q4c).
PROGRESS_RAIL = "#progress_rail"

# Results card (S2.2/S2.3) ------------------------------------------------------
RESULTS_CARD = "#results_card"
RESULT_OUTCOME = "#result_outcome"
RESULT_PATH = "#result_path"
RESULT_META = "#result_meta"
COPY_OUT_PATH = "#copy_out_path"
OPEN_OUT_FOLDER = "#open_out_folder"

# .pt -> .safetensors converter (user feature): a suggestion box shown in the
# ComfyUI panel when #ctq_input points at a PyTorch checkpoint (.pt/.pth/.ckpt).
PT_SUGGEST = "#pt_suggest"
PT_CONVERT_BTN = "#pt_convert_btn"

# GGUF imatrix + UD footer (T8, plan 2026-08-31-gguf-unsloth-parity) -------------
IMATRIX_PATH = "#imatrix_path"
IMATRIX_AUTO = "#imatrix_auto"
UD_FOOTER = "#ud_footer"

# Method picker (feature: pick-from-list modal) -- the button beside the
# free-text #method Input that opens MethodPickerScreen with all 35 ids.
PICK_METHOD = "#pick_method"


def all_ids() -> tuple[str, ...]:
    """Every contract id that must exist in the composed app *right now*.

    Grown step-by-step with the redesign: a step that introduces a new required
    widget adds its constant AND registers it here in the same commit, so the
    contract test always reflects exactly the widgets the plan pins down.
    """
    return (
        FAMILY,
        FAM_GGUF,
        FAM_COMFY,
        GGUF_PANEL,
        COMFY_PANEL,
        RUN,
        RUN_CTQ,
        STATUS,
        CAP_WARN,
        LOG_DRAWER,
        PROGRESS_RAIL,
        RESULTS_CARD,
        RESULT_OUTCOME,
        RESULT_PATH,
        RESULT_META,
        COPY_OUT_PATH,
        OPEN_OUT_FOLDER,
        PT_SUGGEST,
        PT_CONVERT_BTN,
        IMATRIX_PATH,
        IMATRIX_AUTO,
        UD_FOOTER,
        PICK_METHOD,
    )
