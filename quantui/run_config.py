"""Run configuration dataclasses + pure (Textual-free) logic (T08).

This module is the single source of truth for *what* a quantization run needs:

- ``GgufConfig`` / ``CtqConfig`` capture every input the UI gathers (read once from
  the widgets by ``HandlersMixin._read_config``).
- ``RunConfig`` is the discriminated union (``family`` + the matching sub-config).

All the *logic* that used to live inline in ``app.py``/``HandlersMixin`` — validation,
ComfyUI command building, and the 6-combination auto-naming — is moved here as pure
functions that consume the dataclasses instead of ``self.query_one(...)``. That makes
them trivially unit-testable without booting the Textual app.

This module imports **no** Textual code so it can run anywhere (tests, workers, CI).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .quant_methods import (
    ALLOWED_QUANT_IDS,
    COMFY_FORMATS,
    IMATRIX_QUANT_IDS,
    Backend,
    Family,
    classify_input,
    comfy_format,
    comfy_preset,
    eval_visible_when,
)

# The complete official unsloth quant-id whitelist (single source: the
# registry constants). validate_gguf checks every method id against this.
_ALL_QUANT_IDS = frozenset(ALLOWED_QUANT_IDS) | frozenset(IMATRIX_QUANT_IDS)

# ---------------------------------------------------------------------------
# Project constants (single home — NTH-001). app.py / handlers.py re-export
# these names for backward compatibility; do NOT redefine them elsewhere.
# ---------------------------------------------------------------------------
WORKER_CTQ_MODULE = "quantui.worker_ctq"
WORKER_CTQ_KITCHEN_MODULE = "quantui.worker_ctq_kitchen"
WORKER_GGUF_MODULE = "quantui.worker"
WORKER_PT_CONVERT_MODULE = "quantui.worker_pt_convert"
DEFAULT_CTQ_FORMAT = COMFY_FORMATS[0].id  # fp8_e4m3
DEFAULT_CTQ_OUTPUT_MODE = "sharded"

# Formats whose worker path ALWAYS merges into ONE .safetensors regardless of
# output mode (combine = merge-only; bf16/fp16 cast-only merge-cast). For these
# a .safetensors output is valid even for a sharded input in sharded mode, and
# the builder appends <stem>.safetensors to a directory output.
MERGE_TO_ONE_FORMATS = frozenset({"combine", "bf16", "fp16"})


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class GgufConfig:
    """All inputs for an Unsloth/GGUF run."""

    model: str = ""
    output: str = ""
    # Comma-joined method ids ("q4_k_m, q5_k_m") or a single custom string.
    # Multi-method runs loop the worker per id (unsloth's own
    # save_pretrained_gguf accepts quantization_method as str OR list).
    method: str = ""
    # "" = no imatrix (most quants); "auto" = fetch the upstream Unsloth
    # imatrix at run time; otherwise a local path to an imatrix file. The 11
    # IQ* ids REQUIRE a non-empty value (validate_gguf, T5).
    imatrix: str = ""
    pybin: str = ""
    max_seq_length: str = "4096"
    load_in_4bit: bool = False
    push_to_hub: bool = False
    hub_repo: str = ""
    hf_token: str = ""

    @property
    def method_list(self) -> list[str]:
        """`method` split into de-duplicated ids (order-preserving)."""
        return parse_methods(self.method)


def parse_methods(raw: str) -> list[str]:
    """Split a (possibly comma-joined) method string into clean ids.

    - splits on ``,``; strips whitespace
    - drops empty segments (stray separators never become "" methods)
    - de-duplicates, preserving first-seen order (a repeated id runs once)
    """
    seen: list[str] = []
    for part in (raw or "").split(","):
        m = part.strip()
        if m and m not in seen:
            seen.append(m)
    return seen


@dataclass
class CtqConfig:
    """All inputs for a ComfyUI / convert_to_quant run."""

    input: str = ""
    output: str = ""
    pybin: str = ""
    format: str = "fp8_e4m3"
    output_mode: str = "sharded"  # "single" | "sharded"
    preset: str | None = None
    option_values: dict[str, Any] = field(default_factory=dict)
    comfy_quant: bool = False
    save_quant_metadata: bool = False
    simple: bool = False
    low_memory: bool = False
    calib_samples: str = ""
    # Learned-rounding iterations per tensor (ctq default 4000). Empty => ctq default.
    # Only meaningful when NOT --simple. Lowering (500-1000) speeds up convrot runs.
    num_iter: str = ""
    quant_tags: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """Discriminated run configuration keyed on the family."""

    family: Family = Family.GGUF
    gguf: GgufConfig | None = None
    ctq: CtqConfig | None = None


# --------------------------------------------------------------------------- #
# Naming helpers (pure)
# --------------------------------------------------------------------------- #
def ctq_quant_tags(
    fmt: str,
    convrot_group_size: Any = None,
    simple: bool = False,
    low_memory: bool = False,
    calib_samples: str = "",
    heur: bool = False,
    scaling: str | None = None,
    convrot: bool = False,
) -> list[str]:
    """Short, filesystem-safe tags describing a ComfyUI/ctq configuration.

    Mirrors the widget-reading ``HandlersMixin.ctq_quant_tags`` but consumes plain
    values. The format id is the primary descriptor; extra tags only capture options
    that change the emitted artifact. For the unified ``int8`` format the scaling
    mode becomes a tag (``int8-block`` / ``int8-tensor`` / ``int8-row``) and
    ``convrot`` + ``gs<N>`` appear only for row scaling WITH rotation on (a stale
    convrot checkbox under another scaling mode is ignored).
    """
    tags = [fmt]
    if fmt == "int8" and scaling:
        tags.append(scaling)
        # ConvRot requires row scaling; a stale ticked checkbox under another
        # scaling mode must NOT leak convrot/gs tags into the filename.
        if convrot and scaling == "row":
            tags.append("convrot")
            if convrot_group_size:
                tags.append(f"gs{convrot_group_size}")
    if simple:
        tags.append("simple")
    if low_memory:
        tags.append("lowmem")
    if calib_samples:
        tags.append(f"calib{calib_samples}")
    if heur:
        tags.append("heur")
    return tags


def ctq_output_stem(base: str, quant_tags: list[str]) -> str:
    """Build a meaningful output filename stem: ``<base>-<quant_tags>``."""
    return f"{base}-" + "-".join(quant_tags)


def output_state(out: str) -> str:
    """Classify the output field: ``empty`` | ``file_path`` | ``dir_path``."""
    if not out:
        return "empty"
    if os.path.isdir(out):
        return "dir_path"
    if out.endswith(os.sep):
        return "dir_path"
    if os.path.splitext(out)[1]:
        return "file_path"
    return "dir_path"


# --------------------------------------------------------------------------- #
# Auto-suggest (pure) -- return the suggested output path, or None to leave as-is
# --------------------------------------------------------------------------- #
def suggest_gguf_output(model: str, output: str, method: str) -> str | None:
    """GGUF auto-naming: ``<parent>/<modelbase>-<METHOD>``.

    Returns ``None`` when there is no model or when the user already typed an output.
    """
    model = (model or "").strip()
    output = (output or "").strip()
    if not model or output:
        return None
    if os.path.isfile(model):
        base = os.path.basename(os.path.dirname(os.path.abspath(model)))
    else:
        base = os.path.basename(os.path.abspath(model.rstrip(os.sep)))
    mid = (method or "").upper()
    return os.path.join(os.path.dirname(os.path.abspath(model)), f"{base}-{mid}")


def suggest_comfy_output(
    inp: str, output: str, quant_tags: list[str], output_mode: str
) -> str | None:
    """ComfyUI/ctq 6-combination auto-naming (plan §5.2).

    Returns the suggested output path or ``None`` when the field should be left
    untouched (explicit ``.safetensors`` file).

    User-report fix: a directory-shaped output (extension-less, e.g.
    ``...\\out``) is a DESTINATION FOLDER whether or not it exists on disk yet
    -- the artifact name is placed inside it (Case C). Previously such a path
    was classified ``dir_path`` and treated as "leave unchanged", so changing
    quant options never refreshed the filename inside the chosen folder.
    """
    inp = (inp or "").strip()
    output = (output or "").strip()
    if not inp:
        return None
    kind, base = classify_input(inp)
    if base is None:
        return None  # unusable input
    state = output_state(output)
    stem = ctq_output_stem(base, quant_tags)
    if kind == "single_file":
        if state == "file_path":
            return None  # explicit .safetensors -> user's own name, leave as-is
        target_dir = (
            os.path.dirname(os.path.abspath(inp)) if state == "empty" else output
        )
        # dir_path OR empty-with-no-input-dir fallback -> build <dir>/<stem>.safetensors.
        return os.path.join(target_dir, f"{stem}.safetensors")
    # sharded_folder
    if output_mode == "single":
        # Mirror build_ctq_cmd: an explicit .safetensors is kept as-is (None);
        # a directory-shaped or empty destination gets <stem>.safetensors
        # inside it (or next to the input when empty).
        if state == "file_path":
            return None
        parent = os.path.dirname(os.path.abspath(inp))
        target_dir = output if output else parent
        return os.path.join(target_dir, f"{stem}.safetensors")
    # sharded (default): output is a directory destination.
    if state == "empty":
        parent = os.path.dirname(os.path.abspath(inp))
        return os.path.join(parent, stem)
    # dir_path: place the merged single .safetensors inside the chosen folder.
    # An explicit .safetensors under sharded mode is invalid anyway (validate_ctq
    # flags it) -- hands off rather than nest a path inside a filename.
    if state == "file_path":
        return None
    return os.path.join(output, f"{stem}.safetensors")


# --------------------------------------------------------------------------- #
# Validation (pure)
# --------------------------------------------------------------------------- #
def validate_gguf(g: GgufConfig) -> list[str]:
    errors: list[str] = []
    if not g.model:
        errors.append("Model path is required.")
    elif not os.path.exists(g.model):
        errors.append(f"Model path does not exist: {g.model}")
    if not g.output:
        errors.append("Output folder is required.")
    if not g.pybin:
        errors.append("Worker Python interpreter is required.")
    elif not os.path.exists(g.pybin) and shutil_which(g.pybin) is None:
        errors.append(f"Worker python not found: {g.pybin}")
    if not g.method:
        errors.append("Quantization method is required.")
    else:
        # T5 (plan 2026-08-31-gguf-unsloth-parity): gate methods BEFORE the
        # run. unsloth only accepts the 35 official ids, and it fails AFTER
        # a full model load -- catching typos and missing imatrices here
        # turns a multi-minute failure into an instant one.
        _valid_ids = _ALL_QUANT_IDS  # (allowed | imatrix) single source
        for mid in g.method_list:
            if mid not in _valid_ids:
                errors.append(
                    f"Unknown quantization method '{mid}' -- pick one from the dropdown."
                )
            elif mid in IMATRIX_QUANT_IDS:
                if g.imatrix == "":
                    errors.append(
                        f"{mid} requires an imatrix (set a path or 'auto')."
                    )
                elif g.imatrix != "auto" and not os.path.isfile(g.imatrix):
                    errors.append(f"imatrix file not found: {g.imatrix}")
    return errors


def validate_ctq(c: CtqConfig) -> list[str]:
    errors: list[str] = []
    kind: str | None = None
    if not c.input:
        errors.append("Input .safetensors (or HuggingFace sharded folder) is required.")
    elif not os.path.exists(c.input):
        errors.append(f"Input does not exist: {c.input}")
    else:
        kind, _ = classify_input(c.input)
        if kind is None:
            if os.path.isdir(c.input):
                errors.append(
                    "Folder must contain model.safetensors.index.json "
                    "(HuggingFace sharded model) or exactly one .safetensors file."
                )
            else:
                errors.append(
                    "Input must be a .safetensors file (or a HuggingFace sharded folder). "
                    "Tip: .pt checkpoints can be converted in-app via the "
                    "'Convert to safetensors' button above the input field."
                )

    if not c.output:
        errors.append("Output is required.")
    elif kind == "sharded_folder":
        # sharded+single accepts a .safetensors file (or a directory/bare stem,
        # builder appends -CTQ.safetensors); only the default sharded mode keeps
        # the directory-only rule. EXCEPT for merge-to-one formats (combine /
        # bf16 / fp16): they ALWAYS write one merged .safetensors file, so a
        # file output is correct even in sharded mode.
        if (
            c.output_mode == "sharded"
            and c.format not in MERGE_TO_ONE_FORMATS
            and c.output.endswith(".safetensors")
        ):
            errors.append("Sharded output must be a directory (omit the .safetensors filename).")
    elif kind == "single_file" and not c.output.endswith(".safetensors") and not os.path.isdir(c.output):
        errors.append("Output must be a .safetensors file or a directory.")
    # (a directory output for single-file is accepted; build_ctq_cmd appends the filename)

    if not c.pybin:
        errors.append("Worker Python interpreter (ctq) is required.")
    # NOTE: capability warnings NEVER block Run (advisory only).
    return errors


def validate(cfg: RunConfig) -> list[str]:
    """Validate a run config, dispatching on its family."""
    if cfg.family == Family.GGUF:
        if cfg.gguf is None:
            raise ValueError("GGUF run config requires a GgufConfig payload.")
        return validate_gguf(cfg.gguf)
    if cfg.ctq is None:
        raise ValueError("ctq run config requires a CtqConfig payload.")
    return validate_ctq(cfg.ctq)


# --------------------------------------------------------------------------- #
# Command building (pure)
# --------------------------------------------------------------------------- #
def build_gguf_cmd(g: GgufConfig) -> list[str]:
    """Build the ``worker.py`` argument vector for a GGUF run."""
    cmd: list[str] = [
        g.pybin or sys.executable,
        "-m",
        WORKER_GGUF_MODULE,
        "--model",
        g.model,
        "--output",
        g.output,
        "--method",
        g.method,
        "--max-seq-length",
        g.max_seq_length or "4096",
    ]
    if g.load_in_4bit:
        cmd.append("--load-in-4bit")
    if g.push_to_hub and g.hub_repo:
        cmd += ["--push-to-hub", g.hub_repo]
        if g.hf_token:
            cmd += ["--hf-token", g.hf_token]
    return cmd


def build_ctq_cmd(c: CtqConfig) -> list[str]:
    """Build the ``worker_ctq.py`` argument vector for a ComfyUI run."""
    inp = c.input.strip()
    out = c.output.strip()
    pybin = c.pybin or sys.executable
    fmt = c.format
    cf = comfy_format(fmt)

    # Defensive filename append (mirror of the original logic): for a single-file
    # input (or a sharded input in single-file mode) whose output is a directory
    # (or a bare stem), build the .safetensors name. Sharded output in the default
    # sharded mode is passed through unchanged as the destination directory.
    kind, base = classify_input(inp)
    mode = c.output_mode
    out_final = out
    merge_to_one = c.format in MERGE_TO_ONE_FORMATS
    if kind == "single_file" and out_final and not out_final.endswith(".safetensors"):
        b = base or "model"
        stem = ctq_output_stem(b, c.quant_tags)
        out_final = os.path.join(out_final, f"{stem}.safetensors")
    elif kind == "sharded_folder" and (mode == "single" or merge_to_one):
        # Merge-to-one formats (combine / bf16 / fp16) ALWAYS write one merged
        # .safetensors regardless of output mode, so the stem is appended in
        # sharded mode too.
        b = base or "model"
        stem = ctq_output_stem(b, c.quant_tags)
        if not out_final:
            parent = os.path.dirname(os.path.abspath(inp))
            out_final = os.path.join(parent, f"{stem}.safetensors")
        elif not out_final.endswith(".safetensors"):
            out_final = os.path.join(out_final, f"{stem}.safetensors")

    # Route W4A4/W4A8 (comfy-kitchen) formats to the dedicated kitchen worker.
    module = (
        WORKER_CTQ_KITCHEN_MODULE
        if cf.backend == Backend.COMFY_KITCHEN
        else WORKER_CTQ_MODULE
    )
    cmd: list[str] = [pybin, "-m", module, "-i", inp, "-o", out_final]
    cmd += list(cf.base_flags)
    if cf.backend == Backend.COMFY_KITCHEN:
        # Tell the kitchen worker which on-disk .comfy_quant format to serialize.
        cmd += ["--format-id", fmt]

    # extra options declared by the format (e.g. scaling_mode, block_size,
    # convrot, convrot_group_size). Predicates can reference sibling option
    # values (block_size requires scaling_mode == 'block'; convrot_group_size
    # requires row + convrot), so the context accumulates resolved values in
    # declaration order.
    context: dict[str, Any] = {"format": fmt}
    for opt in cf.extra_options:
        if opt.visible_when and not eval_visible_when(opt.visible_when, context):
            continue
        val = c.option_values.get(opt.key, opt.default)
        context[opt.key] = val
        # Skip unset values: None, or an empty string (e.g. a blank manual_seed input),
        # so we never emit a dangling "--manual_seed" with no value.
        if val is None or val == "":
            continue
        if opt.cli_flag:
            cmd += [opt.cli_flag, str(val)]
        elif opt.cli_when_true and val:
            cmd.append(opt.cli_when_true)

    # shared toggles -- CTQ worker ONLY. The comfy-kitchen worker's argparse has
    # none of these options (nor --calib_samples / --num_iter / preset flags);
    # emitting them there crashes with "unrecognized arguments". The kitchen
    # path quantizes natively per-tensor and needs no calibration/optimizer.
    is_kitchen = cf.backend == Backend.COMFY_KITCHEN
    if not is_kitchen:
        if c.comfy_quant:
            cmd.append("--comfy_quant")
        if c.save_quant_metadata:
            cmd.append("--save_quant_metadata")
        if c.simple:
            cmd.append("--simple")
        if c.low_memory:
            cmd.append("--low_memory")

        if c.calib_samples:
            cmd += ["--calib_samples", c.calib_samples]

        if c.num_iter and not c.simple:
            cmd += ["--num_iter", str(c.num_iter)]

        # preset flag is a ctq quantize() kwarg -- meaningless for the kitchen worker.
        if c.preset:
            p = comfy_preset(c.preset)
            if p:
                cmd.append(p.flag)

    # Feature B: tell the worker the output mode for a sharded input.
    cmd += ["--output-mode", mode]
    return cmd


def build_pt_convert_cmd(pt_path: str, out_path: str | None = None,
                         dtype: str = "keep", pybin: str = "") -> list[str]:
    """Build the ``worker_pt_convert.py`` argument vector for a .pt conversion."""
    cmd: list[str] = [
        pybin or sys.executable,
        "-m",
        WORKER_PT_CONVERT_MODULE,
        "-i",
        pt_path,
    ]
    if out_path:
        cmd += ["-o", out_path]
    if dtype and dtype != "keep":
        cmd += ["--dtype", dtype]
    return cmd


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)
