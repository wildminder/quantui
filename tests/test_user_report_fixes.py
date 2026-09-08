"""Tests for the user-reported fixes (session: Raon parity + UI issues).

Covers:
1. exclude_layers / output_dtype options (Raon int8-convrot recipe parity):
   registry declaration, build_ctq_cmd emission, streaming QuantConfig matching,
   worker build_quantize_kwargs passthrough.
2. Header progress stickiness: a mid-run log line clearing the progress store
   must NOT reset #header_progress to 0.
3. Output-mode radio visibility: shown only when #ctq_input is a sharded folder.
4. Log drawer layout: #log fills remaining space; buttons docked bottom.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from textual.widgets import Input

from quantui import app as appmod
from quantui import panels, run_config
from quantui.quant_methods import comfy_format
from quantui.tensor_quant import QuantConfig

# ---- 1. exclude_layers / output_dtype registry + command emission ---------- #

def test_int8_formats_declare_exclude_layers_and_output_dtype():
    # v0.4.0: one unified INT8 format carries all shared INT8 options.
    cf = comfy_format("int8")
    keys = [o.key for o in cf.extra_options]
    assert "exclude_layers" in keys
    assert "output_dtype" in keys
    ex = next(o for o in cf.extra_options if o.key == "exclude_layers")
    assert ex.cli_flag == "--exclude_layers"
    od = next(o for o in cf.extra_options if o.key == "output_dtype")
    assert od.cli_flag == "--output_dtype"
    assert od.default == "bfloat16"


def test_build_ctq_cmd_emits_exclude_and_dtype(tmp_path):
    c = run_config.CtqConfig(
        input=str(tmp_path / "m.safetensors"),
        output=str(tmp_path / "out.safetensors"),
        format="int8",
        option_values={"scaling_mode": "row", "convrot": True,
                       "convrot_group_size": "256",
                       "exclude_layers": "attn_norm|text_embed"},
    )
    cmd = run_config.build_ctq_cmd(c)
    joined = " ".join(cmd)
    assert "--exclude_layers attn_norm|text_embed" in joined
    assert "--int8" in cmd and "--convrot" in cmd
    # output_dtype at its default bfloat16 IS emitted (explicitly pins parity).
    assert "--output_dtype bfloat16" in joined


def test_build_ctq_cmd_omits_empty_exclude():
    c = run_config.CtqConfig(
        input="m.safetensors", output="out.safetensors", format="int8",
        option_values={"scaling_mode": "row"},
    )
    cmd = run_config.build_ctq_cmd(c)
    assert "--exclude_layers" not in cmd


def test_worker_kwargs_passthrough_output_dtype():
    from quantui.worker_ctq import build_quantize_kwargs
    args = types.SimpleNamespace(
        input="in.safetensors", output="out.safetensors",
        int8=True, scaling_mode="row", convrot=True, convrot_group_size=None,
        nvfp4=False, mxfp8=False, comfy_quant=True, save_quant_metadata=False,
        simple=False, low_memory=False, verbose=False, heur=False,
        flux2=False, wan=False, t5xxl=False, hunyuan=False, zimage=False,
        calib_samples=None, exclude_layers="attn_norm|text_embed",
        block_size=None, manual_seed=None, output_dtype="bfloat16",
    )
    kw = build_quantize_kwargs(args)
    assert kw["exclude_layers"] == "attn_norm|text_embed"
    assert kw["output_dtype"] == "bfloat16"


def test_streaming_quantconfig_excluded_regex():
    args = types.SimpleNamespace(
        int8=True, scaling_mode="row", block_size=None, simple=False,
        convrot=True, convrot_group_size=256, heur=False, manual_seed=None,
        exclude_layers=r"attn_norm|text_embed", output_dtype=None,
    )
    cfg = QuantConfig.from_args(args)
    assert cfg.excluded("transformer_blocks.0.attn_norm.weight")
    assert cfg.excluded("text_embed.0.linear.weight")
    assert not cfg.excluded("transformer_blocks.0.attn.to_q.weight")
    assert cfg.orig_dtype == "bfloat16"


# ---- 2. Header progress stickiness ----------------------------------------- #

def _make_app():
    return appmod.QuantApp()


def test_header_progress_holds_value_across_log_line_clears():
    """USER REPORT (S1.7): progress must HOLD across log-line clears — the
    behavior migrated from the removed header strip to #footer_bar (own hold)."""
    async def main():
        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            from textual.widgets import ProgressBar

            a._show_run_footer()  # footer must be visible for the bar to exist
            await pilot.pause()
            bar = a.query_one("#footer_bar", ProgressBar)
            # A structured progress frame sets 40%.
            a._live.update('CTQ_PROGRESS {"phase":"quantize","cur":40,"total":100}')
            a.update_progress(a._live.states())
            await pilot.pause()
            assert bar.progress == 40
            # A real log line clears the live store -> footer must HOLD 40, not reset.
            a._live.clear()
            a.update_progress(a._live.states())
            await pilot.pause()
            assert bar.progress == 40, f"expected hold at 40, got {bar.progress}"
            # Monotonic: a lower aggregate does not move it backwards.
            a._live.update('CTQ_PROGRESS {"phase":"prepare","cur":1,"total":100}')
            a.update_progress(a._live.states())
            await pilot.pause()
            assert bar.progress == 40
            # New run boundary -> reset to 0 for the next run.
            a._show_run_footer()
            await pilot.pause()
            assert bar.progress == 0

    asyncio.run(main())


# ---- 3. Output-mode visibility gated on sharded input ----------------------- #

def test_output_mode_hidden_for_single_file_input(tmp_path):
    async def main():
        inp = tmp_path / "model.safetensors"
        inp.write_bytes(b"\x00" * 64)

        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            # Switch to ComfyUI panel so #ctq_input exists & refresh runs.
            a.action_family_comfy()
            await pilot.pause()
            a.query_one("#ctq_input").value = str(inp)
            a.refresh_ctq_visibility()
            await pilot.pause()
            assert a.query_one("#ctq_output_mode").display is False

    asyncio.run(main())


def test_output_mode_visible_for_sharded_input(tmp_path):
    async def main():
        d = tmp_path / "sharded"
        d.mkdir()
        (d / "model.safetensors.index.json").write_text("{}")

        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            a.query_one("#ctq_input").value = str(d)
            a.refresh_ctq_visibility()
            await pilot.pause()
            assert a.query_one("#ctq_output_mode").display is True

    asyncio.run(main())


def test_output_mode_appears_on_blur_of_sharded_folder_path(tmp_path):
    """User bug report: typing/pasting a sharded folder then just losing focus
    (no Enter) must reveal the Output mode radio."""
    async def main():
        d = tmp_path / "sharded"
        d.mkdir()
        (d / "model.safetensors.index.json").write_text("{}")

        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            inp = a.query_one("#ctq_input", Input)
            # Radio hidden while the path is empty/invalid.
            assert a.query_one("#ctq_output_mode").display is False
            # Real user flow: click into the field (focus), paste the folder
            # path WITHOUT pressing Enter, then move focus away (blur).
            inp.focus()
            await pilot.pause()
            inp.value = str(d)
            await pilot.pause()
            assert a.query_one("#ctq_output_mode").display is False  # not yet
            a.set_focus(None)  # <- field loses focus
            await pilot.pause()
            assert a.query_one("#ctq_output_mode").display is True

    asyncio.run(main())


def test_blur_of_single_file_path_keeps_output_mode_hidden(tmp_path):
    async def main():
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"\x00" * 64)

        a = _make_app()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            inp = a.query_one("#ctq_input", Input)
            inp.focus()
            await pilot.pause()
            inp.value = str(f)
            await pilot.pause()
            a.set_focus(None)
            await pilot.pause()
            assert a.query_one("#ctq_output_mode").display is False

    asyncio.run(main())


# ---- 4. Log drawer layout ---------------------------------------------------- #

def test_log_drawer_css_fills_and_docks():
    """The drawer's #log must be 1fr and .log_buttons docked bottom."""
    css = appmod.QuantApp.CSS
    assert "#log_drawer > #log { height: 1fr;" in css
    assert ".log_buttons { dock: bottom;" in css


def test_drawer_size_presets_resize_the_drawer_not_just_buttons():
    sizes = panels.LOG_DRAWER_SIZES
    assert set(sizes) == {"S", "M", "L"}
    assert sizes["S"] < sizes["M"] < sizes["L"]
