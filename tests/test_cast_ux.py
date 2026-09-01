"""STEP 3.2 (plan 2026-08-26): validation, auto-naming, and UI flow for the
combine / bf16 / fp16 formats.

Covers:
* validate_ctq: combine/bf16/fp16 require a .safetensors output path OR a
  directory (the builder appends the stem); a sharded input with a
  .safetensors output is ACCEPTED in sharded mode too (these formats ignore
  output mode -- they always merge into ONE file).
* build_ctq_cmd: a sharded input + directory output gets <stem>.safetensors
  appended even in sharded mode (worker writes one merged file).
* suggest_comfy_output: bf16/fp16 tags produce <dir>/<base>-bf16.safetensors
  (Case C regression guard).
* Headless UI flow: selecting bf16/fp16 shows NO dynamic option widgets;
  Run builds a cmd containing --cast_dtype; profile roundtrip keeps the new
  format ids and falls back safely on unknown ids.
"""

import asyncio
import os
import sys

from textual.widgets import Input, Select

from quantui import run_config as rc

# ---- validation -------------------------------------------------------------- #

def test_validate_combine_bf16_fp16_accept_safetensors_or_dir(tmp_path):
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    outdir = tmp_path / "out"
    outdir.mkdir()

    for fmt in ("combine", "bf16", "fp16"):
        # .safetensors file output: fine.
        c_file = rc.CtqConfig(
            input=str(m), output=str(tmp_path / f"o-{fmt}.safetensors"),
            pybin=sys.executable, format=fmt,
        )
        assert rc.validate_ctq(c_file) == [], (fmt, rc.validate_ctq(c_file))
        # Directory output: also fine (builder appends the stem).
        c_dir = rc.CtqConfig(
            input=str(m), output=str(outdir), pybin=sys.executable, format=fmt,
        )
        assert rc.validate_ctq(c_dir) == [], (fmt, rc.validate_ctq(c_dir))


def test_validate_combine_sharded_input_safetensors_output_ok_in_sharded_mode(tmp_path):
    # rev. 2: combine/bf16/fp16 IGNORE output mode -- they always merge to one
    # file, so a .safetensors output must NOT trigger the "sharded output must
    # be a directory" error for these formats.
    sh = tmp_path / "mymodel"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")

    for fmt in ("combine", "bf16", "fp16"):
        c = rc.CtqConfig(
            input=str(sh), output=str(tmp_path / "merged.safetensors"),
            pybin=sys.executable, format=fmt, output_mode="sharded",
        )
        errs = rc.validate_ctq(c)
        assert not any("Sharded output must be a directory" in e for e in errs), \
            (fmt, errs)


def test_build_ctq_cmd_cast_formats_append_stem_for_dir_output(tmp_path):
    # Sharded input + directory output + sharded mode: the builder must append
    # <stem>.safetensors because these formats ALWAYS write one merged file.
    sh = tmp_path / "mymodel"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    outdir = tmp_path / "out"
    outdir.mkdir()

    for fmt, flag_val in (("combine", None), ("bf16", "bfloat16"),
                          ("fp16", "float16")):
        c = rc.CtqConfig(
            input=str(sh), output=str(outdir), pybin=sys.executable,
            format=fmt, output_mode="sharded",
            quant_tags=rc.ctq_quant_tags(fmt),
        )
        cmd = rc.build_ctq_cmd(c)
        o = cmd.index("-o")
        outval = cmd[o + 1]
        assert outval.endswith(".safetensors"), (fmt, outval)
        parent_and_name = os.path.basename(os.path.dirname(outval)) == "out"
        assert parent_and_name, (fmt, outval)
        assert f"-{fmt}.safetensors" in outval, (fmt, outval)
        if fmt != "combine":
            i = cmd.index("--cast_dtype")
            assert cmd[i + 1] == flag_val


# ---- auto-naming regression guards ------------------------------------------- #

def test_suggest_comfy_output_case_c_with_new_format_tags(tmp_path):
    # Case C (single_file + dir_path): the artifact filename inside the chosen
    # folder carries the format tag. Regression guard so future Case-C refactors
    # cannot silently break bf16/fp16/combine naming.
    m = tmp_path / "model.safetensors"
    m.write_text("x")
    outdir = tmp_path / "out"
    outdir.mkdir()

    for fmt in ("combine", "bf16", "fp16"):
        got = rc.suggest_comfy_output(str(m), str(outdir), [fmt], "sharded")
        assert got == str(outdir / f"model-{fmt}.safetensors"), fmt

    # Case F (sharded_folder + dir_path): same tag lands in the filename.
    sh = tmp_path / "mymodel"
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    for fmt in ("bf16", "fp16", "combine"):
        got = rc.suggest_comfy_output(str(sh), str(outdir), [fmt], "sharded")
        assert got == str(outdir / f"mymodel-{fmt}.safetensors"), fmt


def test_ctq_quant_tags_new_formats_are_the_id():
    for fmt in ("combine", "bf16", "fp16"):
        assert rc.ctq_quant_tags(fmt) == [fmt]


# ---- headless UI flow ---------------------------------------------------------- #

def _make_app():
    from quantui.app import QuantApp
    return QuantApp()


def _run(coro):
    return asyncio.run(coro)


def test_ui_bf16_fp16_show_no_dynamic_option_widgets(tmp_path):
    async def main():
        from quantui.quant_methods import COMFY_FORMATS
        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            all_keys = {opt.key for f in COMFY_FORMATS for opt in f.extra_options}
            for fmt in ("bf16", "fp16", "combine"):
                a.query_one("#ctq_format", Select).value = fmt
                a.refresh_ctq_visibility()
                await pilot.pause()
                for key in sorted(all_keys):
                    w = a.query_one(f"#{key}")
                    assert w.display is False, (fmt, key)

    _run(main())


def test_ui_run_builds_cmd_with_cast_dtype(tmp_path):
    # Unit-level mirror of test_comfy_run_builds_cmd_and_streams: the app's
    # build_ctq_cmd must emit --cast_dtype bfloat16 when bf16 is selected.
    async def main():
        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            m = tmp_path / "model.safetensors"
            m.write_text("x")
            a.query_one("#ctq_input", Input).value = str(m)
            a.query_one("#ctq_format", Select).value = "bf16"
            await pilot.pause()
            cmd = a.build_ctq_cmd()
            i = cmd.index("--cast_dtype")
            assert cmd[i + 1] == "bfloat16"
            assert "--int8" not in cmd and "--combine" not in cmd

            a.query_one("#ctq_format", Select).value = "fp16"
            await pilot.pause()
            cmd = a.build_ctq_cmd()
            i = cmd.index("--cast_dtype")
            assert cmd[i + 1] == "float16"

    _run(main())


def test_profile_roundtrip_keeps_new_format_ids(tmp_path):
    async def main():
        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            for fmt in ("bf16", "fp16", "combine"):
                a.query_one("#ctq_format", Select).value = fmt
                await pilot.pause()
                fields = a._profile_fields()
                assert fields["ctq_format"] == fmt
                # Apply back onto a fresh default app state -> survives intact.
                b = _make_app()
                async with b.run_test():
                    b.action_family_comfy()
                    b._apply_profile_fields(fields)
                    assert b.ctq_format() == fmt

    _run(main())


def test_profile_unknown_format_id_falls_back_to_default(tmp_path):
    async def main():
        a = _make_app()
        async with a.run_test() as pilot:
            # Wait for mount: action_family_comfy() -> on_radio_set_changed queries
            # #gguf_panel, which does not exist until compose has run. Without this
            # pause the test is a mount race (flaked under full-gate load).
            await pilot.pause()
            a.action_family_comfy()
            # A saved profile referencing the dead 'onthefly' id must not crash
            # and must leave the select at its current valid value.
            fields = {"family": "comfy", "ctq_format": "onthefly"}
            a._apply_profile_fields(fields)
            assert a.ctq_format() in {
                f.id for f in __import__(
                    "quantui.quant_methods", fromlist=["COMFY_FORMATS"]
                ).COMFY_FORMATS
            }

    _run(main())
