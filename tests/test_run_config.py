"""Tests for the Textual-free run config (T08).

These exercise ``quantui.run_config`` directly -- no Textual app is
booted -- proving the validation / command-build / auto-naming logic is pure and
unit-testable once it consumes dataclasses instead of ``self.query_one``.
"""

import os
import sys
import tempfile

from quantui import run_config as rc
from quantui.quant_methods import Family


def test_validate_gguf_and_ctq():
    # GGUF: every required field empty -> all four required errors.
    g = rc.GgufConfig(model="", output="", pybin="", method="")
    errs = rc.validate_gguf(g)
    assert any("Model path is required" in e for e in errs)
    assert any("Output folder is required" in e for e in errs)
    assert any("Worker Python interpreter is required" in e for e in errs)
    assert any("Quantization method is required" in e for e in errs)

    # GGUF: valid model/pybin/method but no output -> only the output error.
    g2 = rc.GgufConfig(model=sys.executable, output="", pybin=sys.executable, method="q4_k_m")
    errs2 = rc.validate_gguf(g2)
    assert any("Output folder is required" in e for e in errs2)
    assert not any("method" in e.lower() for e in errs2)

    # CTQ: missing input -> input-required error.
    c = rc.CtqConfig(input="", output="", pybin=sys.executable, format="fp8_e4m3", output_mode="sharded")
    cerrs = rc.validate_ctq(c)
    assert any("required" in e for e in cerrs)

    # CTQ: sharded input + .safetensors output in sharded mode -> directory error.
    with tempfile.TemporaryDirectory() as d:
        sh = os.path.join(d, "mymodel")
        os.makedirs(sh)
        with open(os.path.join(sh, "model.safetensors.index.json"), "w") as fh:
            fh.write("{}")
        c2 = rc.CtqConfig(
            input=sh, output=os.path.join(d, "out.safetensors"),
            pybin=sys.executable, format="fp8_e4m3", output_mode="sharded",
        )
        cerrs2 = rc.validate_ctq(c2)
        assert any("Sharded output must be a directory" in e for e in cerrs2)

    # Dispatch via RunConfig must equal the family-specific validator.
    cfg = rc.RunConfig(family=Family.GGUF, gguf=g)
    assert rc.validate(cfg) == rc.validate_gguf(g)


def test_build_ctq_cmd_emits_flags(tmp_path):
    # Unified INT8 with row scaling + ConvRot (the flux2 recipe, formerly
    # the separate int8_convrot format entry).
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m),
        output=str(tmp_path / "model-int8-row-convrot.safetensors"),
        pybin=sys.executable,
        format="int8",
        output_mode="sharded",
        preset="flux2",
        option_values={
            "scaling_mode": "row", "convrot": True,
            "convrot_group_size": "256",
        },
        comfy_quant=False,
        save_quant_metadata=False,
        simple=False,
        low_memory=False,
        calib_samples="",
        quant_tags=rc.ctq_quant_tags("int8", "256", scaling="row", convrot=True),
    )
    cmd = rc.build_ctq_cmd(c)
    for tok in ["-i", "-o", "--int8", "--scaling_mode", "row",
                "--convrot", "--convrot_group_size", "256", "--flux2", "--output-mode"]:
        assert tok in cmd, tok
    i = cmd.index("--output-mode")
    assert cmd[i + 1] == "sharded"


def test_build_ctq_cmd_int8_block_emits_blockwise(tmp_path):
    # Block scaling: --block_size is emitted (default 128) but no convrot flag.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-int8-block.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        option_values={"scaling_mode": "block"},
        quant_tags=rc.ctq_quant_tags("int8", scaling="block"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert cmd[0] == sys.executable and cmd[1] == "-m" and cmd[2] == "quantui.worker_ctq"
    for tok in ["--int8", "--scaling_mode", "block", "--block_size", "128"]:
        assert tok in cmd, tok
    assert "--convrot" not in cmd, cmd


def test_build_ctq_cmd_int8_tensor_hides_block_size_and_convrot(tmp_path):
    # Ambiguity fix: tensor scaling must NOT leak block_size/convrot flags even if
    # stale values linger in option_values (visibility predicate filters them).
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-int8-tensor.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        option_values={"scaling_mode": "tensor", "block_size": "64", "convrot": True},
        quant_tags=rc.ctq_quant_tags("int8", scaling="tensor"),
    )
    cmd = rc.build_ctq_cmd(c)
    for tok in ["--int8", "--scaling_mode", "tensor"]:
        assert tok in cmd, tok
    assert "--block_size" not in cmd, cmd
    assert "--convrot" not in cmd, cmd
    assert "--convrot_group_size" not in cmd, cmd


def test_build_ctq_cmd_int8_block_exposes_heur_and_manual_seed(tmp_path):
    # UI-exposure (Msg 4): block_size select, --heur checkbox, --manual_seed input must
    # all flow into the worker command.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-int8-block.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        option_values={"scaling_mode": "block", "block_size": "64",
                       "heur": True, "manual_seed": "233983427"},
        quant_tags=rc.ctq_quant_tags("int8", heur=True, scaling="block"),
    )
    cmd = rc.build_ctq_cmd(c)
    # block_size override 64 (not the 128 default)
    i = cmd.index("--block_size")
    assert cmd[i + 1] == "64", cmd
    # --heur emitted (checkbox truthy)
    assert "--heur" in cmd, cmd
    # --manual_seed emitted with the entered value
    j = cmd.index("--manual_seed")
    assert cmd[j + 1] == "233983427", cmd


def test_build_ctq_cmd_skips_empty_manual_seed(tmp_path):
    # A blank manual_seed input must NOT emit a dangling "--manual_seed" with no value.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-int8-block.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        option_values={"scaling_mode": "block", "block_size": "128",
                       "heur": False, "manual_seed": ""},
        quant_tags=rc.ctq_quant_tags("int8", scaling="block"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert "--manual_seed" not in cmd, cmd
    # heur unchecked -> no --heur flag
    assert "--heur" not in cmd, cmd
    # block_size default still present
    assert "--block_size" in cmd and "128" in cmd


def test_build_ctq_cmd_emits_num_iter(tmp_path):
    # CPU-utilization fix: num_iter passthrough for learned-rounding runs. The
    # optimizer loop is GPU-latency-bound (single-core), so lowering iterations is
    # the main speed lever; the field must reach the worker command line.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-out.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        num_iter="1000",
        quant_tags=rc.ctq_quant_tags("int8", scaling="row", convrot=True),
    )
    cmd = rc.build_ctq_cmd(c)
    i = cmd.index("--num_iter")
    assert cmd[i + 1] == "1000", cmd


def test_build_ctq_cmd_num_iter_suppressed_when_simple_or_blank(tmp_path):
    # --simple skips the optimizer entirely -> --num_iter must NOT be emitted
    # (ctq would reject/ignore it); a blank value emits nothing either.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c_simple = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "a.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        simple=True, num_iter="1000",
        quant_tags=rc.ctq_quant_tags("int8", simple=True, scaling="row"),
    )
    assert "--num_iter" not in rc.build_ctq_cmd(c_simple), c_simple

    c_blank = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "b.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        num_iter="",
        quant_tags=rc.ctq_quant_tags("int8", scaling="row"),
    )
    assert "--num_iter" not in rc.build_ctq_cmd(c_blank), c_blank


def test_build_ctq_cmd_routes_kitchen_backend(tmp_path):
    # P3.3: w4a4_convrot routes to the kitchen worker + carries --format-id.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-w4a4_convrot.safetensors"),
        pybin=sys.executable, format="w4a4_convrot", output_mode="sharded",
        quant_tags=rc.ctq_quant_tags("w4a4_convrot"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert cmd[2] == "quantui.worker_ctq_kitchen", cmd
    assert "--w4a4" in cmd
    i = cmd.index("--format-id")
    assert cmd[i + 1] == "w4a4_convrot"


def test_build_ctq_cmd_routes_w4a8_kitchen(tmp_path):
    # P4.3: w4a8_asym routes to the kitchen worker.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "model-w4a8_asym.safetensors"),
        pybin=sys.executable, format="w4a8_asym", output_mode="sharded",
        quant_tags=rc.ctq_quant_tags("w4a8_asym"),
    )
    cmd = rc.build_ctq_cmd(c)
    assert cmd[2] == "quantui.worker_ctq_kitchen", cmd
    assert "--w4a8" in cmd
    i = cmd.index("--format-id")
    assert cmd[i + 1] == "w4a8_asym"


def test_build_ctq_cmd_kitchen_never_gets_ctq_only_flags(tmp_path):
    # User-report follow-up: the kitchen worker's argparse has NO --simple,
    # --low_memory, --comfy_quant, --save_quant_metadata, --calib_samples,
    # --num_iter or preset flags. Emitting any of them crashes the worker with
    # "unrecognized arguments". All CTQ-only toggles must be gated on the
    # backend: tick every checkbox and set a preset -> kitchen cmd stays clean.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "m-w4a4.safetensors"),
        pybin=sys.executable, format="w4a4_convrot", output_mode="sharded",
        quant_tags=rc.ctq_quant_tags("w4a4_convrot"),
        comfy_quant=True, save_quant_metadata=True,
        simple=True, low_memory=True, calib_samples="128", num_iter="500",
        preset="flux2",
    )
    cmd = rc.build_ctq_cmd(c)
    for tok in ["--simple", "--low_memory", "--comfy_quant",
                "--save_quant_metadata", "--calib_samples", "--num_iter",
                "--flux2"]:
        assert tok not in cmd, f"{tok} leaked into kitchen cmd: {cmd}"


def test_build_ctq_cmd_ctq_backend_keeps_shared_toggles(tmp_path):
    # The gating must NOT strip flags from real CTQ-backend runs (fp8/int8).
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    c = rc.CtqConfig(
        input=str(m), output=str(tmp_path / "m-int8.safetensors"),
        pybin=sys.executable, format="int8", output_mode="sharded",
        quant_tags=rc.ctq_quant_tags("int8", simple=True, low_memory=True),
        comfy_quant=True, save_quant_metadata=True,
        simple=True, low_memory=True, calib_samples="64",
    )
    cmd = rc.build_ctq_cmd(c)
    for tok in ["--int8", "--comfy_quant", "--save_quant_metadata",
                "--simple", "--low_memory", "--calib_samples", "64",
                "--output-mode", "sharded"]:
        assert tok in cmd, tok


def test_auto_suggest_six_combos(tmp_path):
    # single-file model
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    outdir = tmp_path / "out"
    outdir.mkdir()

    # A: single_file + empty -> <parent>/<base>-fp8_e4m3.safetensors
    c = rc.CtqConfig(input=str(m), output="", format="fp8_e4m3", output_mode="sharded",
                     quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) == \
        str(tmp_path / "model-fp8_e4m3.safetensors")

    # B: single_file + explicit .safetensors -> unchanged
    c = rc.CtqConfig(input=str(m), output=str(tmp_path / "custom.safetensors"),
                     format="fp8_e4m3", output_mode="sharded", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) is None

    # C: single_file + dir_path -> <dir>/<base>-fp8_e4m3.safetensors
    c = rc.CtqConfig(input=str(m), output=str(outdir), format="fp8_e4m3",
                     output_mode="sharded", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) == \
        str(outdir / "model-fp8_e4m3.safetensors")

    # sharded folder
    sh = tmp_path / "mymodel"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    (sh / "model-00001-of-00002.safetensors").write_text("x")

    # D: sharded_folder + empty -> <parent>/<base>-fp8_e4m3 (a directory)
    c = rc.CtqConfig(input=str(sh), output="", format="fp8_e4m3",
                     output_mode="sharded", quant_tags=["fp8_e4m3"])
    suggested = rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode)
    assert suggested == str(tmp_path / "mymodel-fp8_e4m3")
    assert not suggested.endswith(".safetensors")

    # E: sharded_folder + explicit .safetensors -> unchanged
    c = rc.CtqConfig(input=str(sh), output=str(tmp_path / "custom.safetensors"),
                     format="fp8_e4m3", output_mode="sharded", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) is None

    # F: sharded_folder + dir_path -> filename inside the chosen folder
    # (Case-C semantics: the dir is a DESTINATION FOLDER, user-report fix)
    c = rc.CtqConfig(input=str(sh), output=str(outdir), format="fp8_e4m3",
                     output_mode="sharded", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) == \
        str(outdir / "mymodel-fp8_e4m3.safetensors")

    # Feature B: sharded_folder + single mode + empty -> .safetensors file
    c = rc.CtqConfig(input=str(sh), output="", format="fp8_e4m3",
                     output_mode="single", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) == \
        str(tmp_path / "mymodel-fp8_e4m3.safetensors")

    # Feature B variant: single mode + dir-shaped destination -> file INSIDE it
    c = rc.CtqConfig(input=str(sh), output=str(outdir), format="fp8_e4m3",
                     output_mode="single", quant_tags=["fp8_e4m3"])
    assert rc.suggest_comfy_output(c.input, c.output, c.quant_tags, c.output_mode) == \
        str(outdir / "mymodel-fp8_e4m3.safetensors")


def test_ctq_quant_tags_unified_int8():
    # Unified INT8: scaling becomes a tag; convrot+gs only when rotation is on.
    assert rc.ctq_quant_tags("int8", scaling="block") == ["int8", "block"]
    assert rc.ctq_quant_tags("int8", scaling="tensor") == ["int8", "tensor"]
    assert rc.ctq_quant_tags("int8", "256", scaling="row", convrot=True) == \
        ["int8", "row", "convrot", "gs256"]
    assert rc.ctq_quant_tags("int8", "256", scaling="row") == ["int8", "row"]
    assert rc.ctq_quant_tags("int8", simple=True) == ["int8", "simple"]
    assert rc.ctq_quant_tags("fp8_e4m3") == ["fp8_e4m3"]


def test_default_constants_single_home():
    """NTH-001: DEFAULT_* / WORKER_CTQ_MODULE live ONLY in run_config.

    app.py and handlers.py re-export the same objects (import, not redefine);
    identity (``is``) proves no duplicate definition exists anywhere.
    """
    from quantui import app as appmod
    from quantui import handlers
    from quantui import run_config as rc

    assert appmod.DEFAULT_CTQ_FORMAT is rc.DEFAULT_CTQ_FORMAT
    assert handlers.DEFAULT_CTQ_FORMAT is rc.DEFAULT_CTQ_FORMAT
    assert appmod.DEFAULT_CTQ_OUTPUT_MODE is rc.DEFAULT_CTQ_OUTPUT_MODE
    assert handlers.DEFAULT_CTQ_OUTPUT_MODE is rc.DEFAULT_CTQ_OUTPUT_MODE
    assert handlers.WORKER_CTQ_MODULE == rc.WORKER_CTQ_MODULE


# --------------------------------------------------------------------------- #
# T4 (plan 2026-08-31-gguf-unsloth-parity): parse_methods + imatrix + method_list
# --------------------------------------------------------------------------- #
def test_parse_methods_single():
    assert rc.parse_methods("q4_k_m") == ["q4_k_m"]


def test_parse_methods_multi_dedupe():
    # Order-preserving de-dup: "q4_k_m, q5_k_m, q4_k_m" runs q4_k_m once.
    assert rc.parse_methods("q4_k_m, q5_k_m, q4_k_m") == ["q4_k_m", "q5_k_m"]


def test_parse_methods_strips_empties():
    # Stray separators/whitespace never produce empty-string "methods".
    assert rc.parse_methods(" , q8_0 , ") == ["q8_0"]


def test_parse_methods_empty():
    assert rc.parse_methods("") == []
    assert rc.parse_methods(" , ") == []


def test_ggufconfig_imatrix_default_empty():
    g = rc.GgufConfig()
    assert g.imatrix == ""


def test_ggufconfig_method_list():
    g = rc.GgufConfig(method="q4_k_m, q5_k_m")
    assert g.method_list == ["q4_k_m", "q5_k_m"]
    # Property is derived from method -- never stored separately.
    g.method = "q8_0"
    assert g.method_list == ["q8_0"]




# --------------------------------------------------------------------------- #
# T5 (plan 2026-08-31-gguf-unsloth-parity): validate_gguf gates
# --------------------------------------------------------------------------- #
def _g(method, imatrix="", **kw):
    return rc.GgufConfig(
        model=sys.executable, output="/tmp/out", pybin=sys.executable,
        method=method, imatrix=imatrix, **kw,
    )


def test_validate_blocks_iq_without_imatrix():
    # unsloth raises RuntimeError (after a FULL model load) unless imatrix_file=
    # is passed for every IQ* id; the TUI must refuse BEFORE spending the load.
    errs = rc.validate_gguf(_g("iq2_xs"))
    assert len(errs) == 1
    assert "iq2_xs" in errs[0]
    assert "imatrix" in errs[0].lower()


def test_validate_allows_iq_with_imatrix_path(tmp_path):
    imatrix = tmp_path / "imatrix.dat"
    imatrix.write_bytes(b"raw")
    assert rc.validate_gguf(_g("iq2_xs", str(imatrix))) == []


def test_validate_allows_iq_with_auto():
    # "auto" = fetch the upstream Unsloth imatrix at run time; existence
    # cannot be checked locally (architect-ratified boundary).
    assert rc.validate_gguf(_g("iq2_xs", "auto")) == []


def test_validate_rejects_missing_imatrix_path(tmp_path):
    missing = tmp_path / "nope.dat"
    errs = rc.validate_gguf(_g("iq2_xs", str(missing)))
    assert len(errs) == 1
    assert "not found" in errs[0]


def test_validate_unknown_id_error():
    # The old custom free-text path let typos through; unsloth would fail
    # after the model load. The whitelist catches them up front.
    errs = rc.validate_gguf(_g("bogus"))
    assert len(errs) == 1
    assert "bogus" in errs[0]


def test_validate_multi_id_ok():
    # Comma-joined ids all in the whitelist, none IQ* -> no error.
    assert rc.validate_gguf(_g("q4_k_m,q5_k_m")) == []


def test_validate_multi_id_partial_failure():
    # One bad id among valid ones: exactly that id is named.
    errs = rc.validate_gguf(_g("q4_k_m, bogus ,iq2_s"))
    assert len(errs) == 2  # bogus (unknown) + iq2_s (needs imatrix)
    assert any("bogus" in e for e in errs)
    assert any("iq2_s" in e for e in errs)


def test_validate_single_id_unchanged(tmp_path):
    # Regression pin: today's passing cases still pass.
    assert rc.validate_gguf(_g("q4_k_m")) == []
    assert rc.validate_gguf(_g("q8_0")) == []
