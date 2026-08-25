"""Headless TUI tests for app.py (Textual run_test()).

Covers: panel swap (C1/C3), GGUF regression happy path (E1), ComfyUI arg-build +
streaming (E2/C5), conditional visibility (C6), validate/auto_suggest branching (C4),
and the non-blocking capability badge (D2).

GGUF happy path is asserted byte-for-byte equivalent to the pre-M1 app: the same
``worker.py`` is spawned and the fake worker's echo line reaches the log. ``Popen`` is
mocked so no real torch / convert_to_quant is needed.
"""

import glob
import os
import sys
import tempfile
import threading
import time

import pytest
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, RichLog, Select

from quantui import app as appmod
from quantui import run_config as rc_mod
from quantui import worker_runner as wr
from quantui.quant_methods import Family


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_popen(monkeypatch):
    captured = {}

    class FakeProc:
        def __init__(self, cmd):
            captured["cmd"] = list(cmd)
            # _pump reads the raw fd via os.read, so the fake stdout must expose a real
            # fileno(). Back it with an os.pipe and pre-fill it with the worker's output.
            r, w = os.pipe()
            with os.fdopen(w, "wb") as fw:
                fw.write(b"FAKE_WORKER_LINE\n=== finished ===\n")
            self.stdout = os.fdopen(r, "rb", 0)  # unbuffered so os.read(fd) sees it
            self.returncode = 0

        def wait(self):
            return 0

    def fake_Popen(cmd, **kwargs):
        return FakeProc(cmd)

    # Patch the runner's popen (the app delegates subprocess handling to WorkerRunner,
    # so app.py no longer imports subprocess directly). Resolved at WorkerRunner.__init__
    # call-time, so this takes effect for the QuantApp constructed in the test.
    monkeypatch.setattr(wr.subprocess, "Popen", fake_Popen)
    return captured

def attach_recorder(app):
    """Replace log_msg/set_status with recorders (still call the originals)."""
    log_lines = []
    statuses = []
    orig_log = appmod.QuantApp.log_msg
    orig_status = appmod.QuantApp.set_status

    def rec_log(msg):
        log_lines.append(msg)
        orig_log(app, msg)

    def rec_status(msg):
        statuses.append(msg)
        orig_status(app, msg)

    app.log_msg = rec_log
    app.set_status = rec_status
    return log_lines, statuses

def switch_family(app, fam: Family):
    app.family = fam
    app.query_one("#gguf_panel").display = (fam == Family.GGUF)
    app.query_one("#comfy_panel").display = (fam == Family.COMFY)
    app.refresh_ctq_visibility()

# --------------------------------------------------------------------------- #
# C1 / C3: panel structure + ComfyUI widgets
# --------------------------------------------------------------------------- #
async def test_panels_exist():
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.query_one("#family", RadioSet) is not None
        buttons = list(a.query_one("#family", RadioSet).query(RadioButton))
        assert len(buttons) == 2
        assert a.query_one("#gguf_panel") is not None
        assert a.query_one("#comfy_panel") is not None
        # default visible panel is GGUF
        assert a.query_one("#gguf_panel").display is True
        assert a.query_one("#comfy_panel").display is False

async def test_output_mode_radio_does_not_switch_family():
    # Regression: selecting "Single file (merge)" inside #ctq_output_mode must NOT
    # flip the whole tab back to GGUF. on_radio_set_changed must ignore any RadioSet
    # other than the top-level #family one.
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        assert a.family == Family.COMFY
        rs = a.query_one("#ctq_output_mode", RadioSet)
        single = a.query_one("#om_single", RadioButton)
        a.on_radio_set_changed(RadioSet.Changed(rs, single))
        assert a.family == Family.COMFY, "selecting output mode must not change family"
        assert a.query_one("#comfy_panel").display is True

async def test_family_radio_still_switches_family():
    # Positive check: the #family RadioSet handler still drives family switching.
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.family == Family.GGUF
        fam = a.query_one("#family", RadioSet)
        comfy = a.query_one("#fam_comfy", RadioButton)
        a.on_radio_set_changed(RadioSet.Changed(fam, comfy))
        assert a.family == Family.COMFY
        assert a.query_one("#comfy_panel").display is True
        gguf = a.query_one("#fam_gguf", RadioButton)
        a.on_radio_set_changed(RadioSet.Changed(fam, gguf))
        assert a.family == Family.GGUF
        assert a.query_one("#gguf_panel").display is True

async def test_comfy_widgets_exist_and_convrot_hidden():
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        for wid in ["#ctq_input", "#ctq_output", "#ctq_format", "#ctq_preset",
                    "#pybin_ctq", "#ctq_comfy_quant", "#ctq_save_quant_metadata"]:
            assert a.query_one(wid) is not None, wid
        # convrot_group_size hidden initially (default format fp8_e4m3)
        assert a.query_one("#convrot_group_size").display is False

# --------------------------------------------------------------------------- #
# C6: conditional visibility wiring
# --------------------------------------------------------------------------- #
async def test_format_visibility():
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        a.query_one("#ctq_format", Select).value = "int8_convrot"
        a.refresh_ctq_visibility()
        assert a.query_one("#convrot_group_size").display is True
        a.query_one("#ctq_format", Select).value = "fp8_e4m3"
        a.refresh_ctq_visibility()
        assert a.query_one("#convrot_group_size").display is False
        a.query_one("#ctq_format", Select).value = "int8_block"
        a.refresh_ctq_visibility()
        # scaling mode visible for int8 formats, convrot only for int8_convrot
        assert a.query_one("#ctq_scaling_mode").display is True
        assert a.query_one("#convrot_group_size").display is False


async def test_int8_block_exposes_block_size_heur_manual_seed():
    # UI-exposure (Msg 4): selecting INT8 blockwise must reveal block_size / heur /
    # manual_seed widgets; the default fp8_e4m3 format must keep them hidden.
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        # default format fp8_e4m3 -> all INT8-specific options hidden
        for wid in ["#block_size", "#block_size_label", "#heur", "#heur_label",
                    "#manual_seed", "#manual_seed_label", "#convrot_group_size"]:
            assert a.query_one(wid).display is False, wid
        # switch to int8_block -> block_size + heur + manual_seed become visible
        a.query_one("#ctq_format", Select).value = "int8_block"
        a.refresh_ctq_visibility()
        for wid in ["#block_size", "#block_size_label", "#heur", "#heur_label",
                    "#manual_seed", "#manual_seed_label"]:
            assert a.query_one(wid).display is True, wid
        # convrot only applies to int8_convrot -> hidden for int8_block
        assert a.query_one("#convrot_group_size").display is False


async def test_int8_convrot_exposes_convrot_heur_manual_seed():
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        a.query_one("#ctq_format", Select).value = "int8_convrot"
        a.refresh_ctq_visibility()
        assert a.query_one("#convrot_group_size").display is True
        for wid in ["#heur", "#heur_label", "#manual_seed", "#manual_seed_label"]:
            assert a.query_one(wid).display is True, wid
        # block_size is blockwise-only -> hidden for int8_convrot
        assert a.query_one("#block_size").display is False

async def test_preset_sets_format():
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        a.query_one("#ctq_preset", Select).value = "flux2"
        a.on_select_changed(Select.Changed(a.query_one("#ctq_preset", Select), "flux2"))
        assert a.query_one("#ctq_format", Select).value == "int8_convrot"

# --------------------------------------------------------------------------- #
# C4: validate + auto_suggest_output branching
# --------------------------------------------------------------------------- #
async def test_validate_branches(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        # COMFY: folder input -> must be a .safetensors file
        switch_family(a, Family.COMFY)
        a.query_one("#ctq_input", Input).value = str(tmp_path)  # existing folder (no index/weights)
        a.query_one("#ctq_output", Input).value = ""
        a.query_one("#pybin_ctq", Input).value = sys.executable
        errs = a.validate()
        # plan §8.2: an unusable folder (no index json, not a single .safetensors) errors
        assert any("folder must contain model.safetensors.index.json" in e.lower() for e in errs)

        # GGUF: missing output folder
        switch_family(a, Family.GGUF)
        a.query_one("#model", Input).value = str(tmp_path)
        a.query_one("#output", Input).value = ""
        a.query_one("#pybin", Input).value = sys.executable
        errs = a.validate()
        assert any("output folder is required" in e.lower() for e in errs)

async def test_auto_suggest_comfy(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = ""
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(tmp_path / "model-fp8_e4m3.safetensors")

# --------------------------------------------------------------------------- #
# E1: GGUF regression happy path (worker.py, identical to today)
# --------------------------------------------------------------------------- #
async def test_gguf_regression(fake_popen, tmp_path):
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.query_one("#model", Input).value = str(tmp_path)
        a.query_one("#output", Input).value = str(tmp_path / "out")
        a.query_one("#pybin", Input).value = sys.executable
        logs, statuses = attach_recorder(a)
        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        # Worker stream is routed through the LogRouter to the RichLog (not via log_msg
        # directly), so assert on the RichLog widget content here.
        rich = "\n".join(str(l) for l in a.query_one(RichLog).lines)
        assert "FAKE_WORKER_LINE" in rich
        assert "Done ✓" in statuses
        cmd = " ".join(fake_popen["cmd"])
        assert "quantui.worker" in cmd
        assert "quantui.worker_ctq" not in cmd

# --------------------------------------------------------------------------- #
# E2 / C5: ComfyUI arg-build + streaming (worker_ctq.py)
# --------------------------------------------------------------------------- #
async def test_comfy_run_builds_cmd_and_streams(fake_popen, tmp_path):
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(tmp_path / "model-fp8_e4m3.safetensors")
        a.query_one("#ctq_format", Select).value = "int8_convrot"
        a.query_one("#ctq_preset", Select).value = "flux2"
        a.query_one("#pybin_ctq", Input).value = sys.executable
        logs, statuses = attach_recorder(a)
        a.action_run()
        await a.workers.wait_for_complete()
        await pilot.pause()

        cmd = " ".join(fake_popen["cmd"])
        assert "quantui.worker_ctq" in cmd
        for tok in ["-i", "-o", "--int8", "--scaling_mode", "row",
                    "--convrot", "--convrot_group_size", "256", "--flux2"]:
            assert tok in cmd, tok
        # Worker stream is routed through the LogRouter to the RichLog directly.
        rich = "\n".join(str(l) for l in a.query_one(RichLog).lines)
        assert "FAKE_WORKER_LINE" in rich
        assert "Done ✓" in statuses

# --------------------------------------------------------------------------- #
# D2: capability badge is non-blocking
# --------------------------------------------------------------------------- #
async def test_capability_badge_nonblocking(tmp_path, monkeypatch):
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        switch_family(a, Family.COMFY)
        # Stub the probe so it reports torch 2.8 / CUDA 12.8 / no Triton.
        from quantui.capabilities import CapabilityReport
        fake_report = CapabilityReport(
            python_version="3.13.12", torch_version="2.8.0", cuda_version="12.8",
            triton=False, comfy_kitchen=False, convert_to_quant=True,
        )
        monkeypatch.setattr(appmod.capabilities, "probe_worker_env", lambda *a, **k: fake_report)

        a.query_one("#pybin_ctq", Input).value = sys.executable
        a.query_one("#ctq_format", Select).value = "int8_convrot"
        a.refresh_ctq_visibility()
        a.refresh_capabilities(sys.executable)
        await a.workers.wait_for_complete()
        await pilot.pause()

        warn = a.query_one("#ctq_cap_warn", Label).content
        assert "Triton" in str(warn)

        # Capability mismatch must NEVER block Run.
        m = tmp_path / "m.safetensors"
        m.write_text("x")
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(tmp_path / "m-fp8_e4m3.safetensors")
        errs = a.validate()
        assert not any("triton" in e.lower() or "capability" in e.lower() for e in errs)

# --------------------------------------------------------------------------- #
# A2: auto_suggest_output 6-combo naming (plan §5.2)
# --------------------------------------------------------------------------- #
def _sharded_dir(tmp_path, name="mymodel"):
    sh = tmp_path / name
    sh.mkdir()
    (sh / "model.safetensors.index.json").write_text("{}")
    (sh / "model-00001-of-00002.safetensors").write_text("x")
    return sh

async def test_auto_suggest_single_file_empty(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # A: single_file + empty
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = ""
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(tmp_path / "model-fp8_e4m3.safetensors")

async def test_auto_suggest_single_file_dir(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # C: single_file + dir_path
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(outdir / "model-fp8_e4m3.safetensors")

async def test_auto_suggest_sharded_empty(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # D: sharded_folder + empty -> directory
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = ""
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(tmp_path / "mymodel-fp8_e4m3")
        assert not a.query_one("#ctq_output", Input).value.endswith(".safetensors")

async def test_auto_suggest_sharded_dir(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # F: sharded_folder + dir_path -> unchanged
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(outdir)

async def test_auto_suggest_single_file_explicit_file(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # B: single_file + file_path -> unchanged
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        custom = tmp_path / "custom.safetensors"
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(custom)
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(custom)

async def test_auto_suggest_sharded_explicit_file(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():  # E: sharded_folder + file_path -> unchanged
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        custom = tmp_path / "custom.safetensors"
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = str(custom)
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(custom)

# --------------------------------------------------------------------------- #
# D1: validate() COMFY branch accepts sharded dir + directory output
# --------------------------------------------------------------------------- #
async def test_validate_sharded(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)

        sh = _sharded_dir(tmp_path, "mymodel")
        outdir = tmp_path / "out"
        outdir.mkdir()

        # sharded dir input + dir output -> OK
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.query_one("#pybin_ctq", Input).value = sys.executable
        errs = a.validate()
        assert not errs, errs

        # sharded dir input + .safetensors output -> error
        a.query_one("#ctq_output", Input).value = str(tmp_path / "out.safetensors")
        errs = a.validate()
        assert any("Sharded output must be a directory" in e for e in errs)

        # dir with no index json and two .safetensors -> error
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "a.safetensors").write_text("x")
        (bad / "b.safetensors").write_text("x")
        a.query_one("#ctq_input", Input).value = str(bad)
        a.query_one("#ctq_output", Input).value = str(outdir)
        errs = a.validate()
        assert any("Folder must contain model.safetensors.index.json" in e for e in errs)

        # single-file input + dir output -> OK (builder appends filename)
        f = tmp_path / "model.safetensors"
        f.write_text("x")
        a.query_one("#ctq_input", Input).value = str(f)
        a.query_one("#ctq_output", Input).value = str(outdir)
        errs = a.validate()
        assert not errs, errs

# --------------------------------------------------------------------------- #
# D2: build_ctq_cmd sharded passes dir / single-file dir appends filename
# --------------------------------------------------------------------------- #
async def test_build_ctq_sharded_passes_dir(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.query_one("#ctq_format", Select).value = "fp8_e4m3"
        a.query_one("#pybin_ctq", Input).value = sys.executable
        cmd = a.build_ctq_cmd()
        assert "-i" in cmd and "-o" in cmd
        i = cmd.index("-i")
        o = cmd.index("-o")
        assert cmd[i + 1] == str(sh)
        assert cmd[o + 1] == str(outdir)
        assert not cmd[o + 1].endswith(".safetensors")

async def test_build_ctq_single_file_dir_appends(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.query_one("#ctq_format", Select).value = "fp8_e4m3"
        a.query_one("#pybin_ctq", Input).value = sys.executable
        cmd = a.build_ctq_cmd()
        o = cmd.index("-o")
        outval = cmd[o + 1]
        assert outval == str(outdir / "model-fp8_e4m3.safetensors")
        assert outval.endswith("model-fp8_e4m3.safetensors")

# --------------------------------------------------------------------------- #
# D3: #ctq_output submit triggers auto-naming
# --------------------------------------------------------------------------- #
async def test_output_submit_triggers_naming(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(m)
        a.query_one("#ctq_output", Input).value = str(outdir)
        a.on_input_submitted(Input.Submitted(a.query_one("#ctq_output", Input), str(outdir)))
        assert a.query_one("#ctq_output", Input).value == str(outdir / "model-fp8_e4m3.safetensors")

async def test_browse_out_triggers_naming(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        m = tmp_path / "model.safetensors"
        m.write_text("x")
        outdir = tmp_path / "out"
        outdir.mkdir()
        a.query_one("#ctq_input", Input).value = str(m)
        a.set_ctq_out(str(outdir))
        assert a.query_one("#ctq_output", Input).value == str(outdir / "model-fp8_e4m3.safetensors")

# --------------------------------------------------------------------------- #
# Feature A: temp-file run log + 100-line RichLog cap + Copy/Save read the file
# --------------------------------------------------------------------------- #
async def test_run_log_file_opened(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        assert a._run_log_path.endswith(".log")
        assert os.path.isfile(a._run_log_path)
        assert a._run_log_fh is not None
        assert a._run_log_fh.closed is False
    # after on_unmount the handle is closed
    assert a._run_log_fh.closed is True

async def test_log_msg_writes_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        a.log_msg("x")
        with open(a._run_log_path, encoding="utf-8") as fh:
            assert fh.read() == "x\n"
        a.log_msg("y")
        a.log_msg("z")
        with open(a._run_log_path, encoding="utf-8") as fh:
            assert fh.read() == "x\ny\nz\n"

async def test_richlog_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        for i in range(250):
            a.log_msg(f"line{i}")
        await pilot.pause()
        # display window capped
        assert len(a.query_one(RichLog).lines) <= 100
        # full log intact on disk
        with open(a._run_log_path, encoding="utf-8") as fh:
            assert len(fh.read().splitlines()) == 250

async def test_log_msg_collapses_progress_lines(tmp_path, monkeypatch):
    # Regression: tqdm/optimizer progress lines must NOT flood the RichLog; they are
    # collapsed into the single live-progress widget, while the full stream is kept
    # in the authoritative temp log.
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        for i in range(5):
            a.log_msg(
                f"Optimizing INT8 (Prodigy-plateau):   {i}%|          | {i*10}/4000 "
                f"[00:00<?, ?it/s, loss=0.{i}]"
            )
        await pilot.pause()
        # No progress line reaches the RichLog (no flood).
        assert len(a.query_one(RichLog).lines) == 0
        # They are collapsed into the stacked progress rail, keeping only the latest update.
        assert a.query_one("#progress_rail").display is True
        # The tqdm line is parsed into a REAL determinate state (not frozen raw text):
        # the latest frame (i=4 -> 40/4000) drives the bar.
        states = a._live.states()
        assert len(states) == 1, states
        st = states[0]
        assert st.phase == "quantize"
        assert st.determinate is True
        assert st.cur == 40 and st.total == 4000, (st.cur, st.total)
        value = next(iter(a._live.snapshot().values()))
        assert "Optimizing INT8" in value and "40/4000" in value, value
        # Full stream still captured to disk.
        full = a._read_full_log()
        assert full.count("Optimizing INT8") == 5

async def test_log_msg_progress_cleared_on_real_line(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("Optimizing INT8 (x):   0%|          | 0/4000 [00:00<?, ?it/s]")
        await pilot.pause()
        # The rail is always mounted (display is never toggled) so the layout
        # doesn't jump when progress appears.
        assert a.query_one("#progress_rail").display is True
        # The tqdm line is parsed into a REAL determinate state (cur/total), not raw text.
        states = a._live.states()
        assert len(states) == 1, states
        assert states[0].phase == "quantize" and states[0].determinate is True
        assert states[0].cur == 0 and states[0].total == 4000
        value = next(iter(a._live.snapshot().values()))
        assert "Optimizing INT8 (x)" in value and "0/4000" in value, value
        # A real (non-progress) log line clears the live-progress *content* but keeps the
        # widget mounted, and is written to the RichLog.
        a.log_msg("=== Quantization finished successfully ===")
        await pilot.pause()
        assert a.query_one("#progress_rail").display is True  # still mounted -> no jump
        assert a._live.snapshot() == {}  # content cleared
        assert len(a.query_one(RichLog).lines) == 1

async def test_copy_and_save_full_file(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        a.log_msg("alpha")
        a.log_msg("beta")
        with open(a._run_log_path, encoding="utf-8") as fh:
            assert fh.read() == "alpha\nbeta\n"

async def test_run_proc_streams_real_subprocess_progress_to_live_widget(
    tmp_path, monkeypatch
):
    # Faithful regression for the "real app shows no bar" bug. Unlike the WorkerRunner
    # pump test (which monkeypatches os.read), this drives QuantApp._run_proc through a
    # GENUINE child process (tests/_tqdm_launcher.py) that
    # emits tqdm-style '\r'-rewritten frames on its real stdout. It exercises the actual
    # os.read(fd) pipe path end-to-end and asserts the live-progress widget receives the
    # streamed frames (the real library's bars, not just a simulated one).
    launcher = os.path.join(os.path.dirname(__file__), "_tqdm_launcher.py")
    venv_py = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".venv-test", "Scripts", "python.exe")
    )
    if not os.path.exists(venv_py):
        pytest.skip("venv python not found")
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # call_from_thread synchronous -> deterministic; _run_proc blocks reading the real
        # child's stdout via os.read on the raw pipe fd (no patched os.read here).
        monkeypatch.setattr(
            appmod.QuantApp, "call_from_thread",
            lambda self, fn, *args: fn(*args),
        )
        # Record every live-progress update so we can prove a streamed frame arrived even
        # though trailing log lines clear the widget after the child exits.
        captured: list[dict[str, str]] = []
        orig_update_progress = appmod.QuantApp.update_progress

        def rec_update(self, states):
            orig_update_progress(self, states)
            captured.append(dict(self._live.snapshot()))

        monkeypatch.setattr(appmod.QuantApp, "update_progress", rec_update)

        rc = a._run_proc([venv_py, launcher], os.path.dirname(launcher))
        await pilot.pause()

        assert rc == 0
        # The live-progress widget received streamed frames from the REAL pipe (not just
        # the sim). The last bar frame is 100% / 10-of-10.
        assert captured, "live-progress never received a frame from the real subprocess"
        latest = next(iter(captured[-1].values()))
        assert "10/10" in latest, latest
        # Trailing real log lines cleared the widget and were routed to the RichLog.
        assert a._live.snapshot() == {}
        assert len(a.query_one(RichLog).lines) >= 2


async def test_copy_and_save_clipboard_and_file(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        a.log_msg("alpha")
        a.log_msg("beta")

        # Copy: full log from file -> clipboard (rstrip trailing newline).
        a.copy_log()
        assert a._clipboard == "alpha\nbeta"

        # Save: dump full log to a temp file (appends a "Log dumped to ..." note).
        a.save_log()
        pattern = os.path.join(tempfile.gettempdir(), "unsloth-quant-tui-log-*.txt")
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        assert matches, "save_log did not create an unsloth-quant-tui-log-*.txt file"
        newest = matches[-1]
        try:
            with open(newest, encoding="utf-8") as fh:
                content = fh.read()
            assert "alpha\nbeta" in content
        finally:
            try:
                os.remove(newest)
            except OSError:
                pass

async def test_copy_full_when_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test():
        for i in range(250):
            a.log_msg(f"L{i}")
        a.copy_log()
        # Copy reads the FILE (250 lines), not the capped 100-line window.
        lines = a._clipboard.splitlines()
        assert len(lines) == 250, len(lines)

# --------------------------------------------------------------------------- #
# Feature B: ctq output-mode toggle + validate / auto_suggest / build_ctq_cmd
# --------------------------------------------------------------------------- #
async def test_output_mode_default_sharded():
    a = appmod.QuantApp()
    async with a.run_test():
        assert a.ctq_output_mode() == "sharded"

async def test_auto_suggest_sharded_single_file(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = ""
        a.query_one("#om_single", RadioButton).value = True  # single-file mode
        assert a.ctq_output_mode() == "single"
        a.auto_suggest_output(Family.COMFY)
        assert a.query_one("#ctq_output", Input).value == str(tmp_path / "mymodel-fp8_e4m3.safetensors")

async def test_validate_sharded_single_file_ok(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#pybin_ctq", Input).value = sys.executable

        # sharded + single + .safetensors output -> OK
        # (A real RadioSet click enforces mutual exclusivity; emulate it here.)
        a.query_one("#om_single", RadioButton).value = True
        a.query_one("#om_sharded", RadioButton).value = False
        assert a.ctq_output_mode() == "single"
        a.query_one("#ctq_output", Input).value = str(tmp_path / "out.safetensors")
        errs = a.validate()
        assert not errs, errs

        # sharded + sharded + .safetensors output -> still errors (existing rule)
        a.query_one("#om_sharded", RadioButton).value = True
        a.query_one("#om_single", RadioButton).value = False
        assert a.ctq_output_mode() == "sharded"
        a.query_one("#ctq_output", Input).value = str(tmp_path / "out.safetensors")
        errs = a.validate()
        assert any("Sharded output must be a directory" in e for e in errs)

async def test_build_ctq_single_mode_emits_flag(tmp_path):
    a = appmod.QuantApp()
    async with a.run_test():
        switch_family(a, Family.COMFY)
        sh = _sharded_dir(tmp_path, "mymodel")
        a.query_one("#ctq_input", Input).value = str(sh)
        a.query_one("#ctq_output", Input).value = ""  # empty -> builder appends file
        a.query_one("#om_single", RadioButton).value = True
        a.query_one("#ctq_format", Select).value = "fp8_e4m3"
        a.query_one("#pybin_ctq", Input).value = sys.executable
        cmd = a.build_ctq_cmd()
        o = cmd.index("-o")
        outval = cmd[o + 1]
        assert outval.endswith("mymodel-fp8_e4m3.safetensors"), outval
        assert "--output-mode" in cmd
        i = cmd.index("--output-mode")
        assert cmd[i + 1] == "single"

# --------------------------------------------------------------------------- #
# Bugfix: ctq (convert_to_quant) MAIN-quant progress bar in the ComfyUI tab.
# The third-party library emits "(N/M) Processing (INT8): <layer>" headers with
# indented "    - " detail lines. Previously the classifier missed the headers
# (no "%" token) and let the detail lines clear the live-progress widget -> an
# empty bar. We now (1) recognize the header as progress and (2) keep the bar
# alive through the detail lines.
# --------------------------------------------------------------------------- #
async def test_log_msg_ctq_processing_collapses_and_detail_does_not_clear(tmp_path, monkeypatch):
    # Drive log_msg with a ctq-style header, a detail line, then the next header;
    # assert the live-progress widget collapses to ONE entry (latest header) and the
    # detail line did NOT clear it.
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("(1/211) Processing (INT8): a.weight")
        a.log_msg("    - Tensor shape: [576, 1536], Max rank: 576. Using k=256 components.")
        a.log_msg("(2/211) Processing (INT8): b.weight")
        await pilot.pause()

        # Collapsed: exactly ONE live-progress entry, and it is the LATEST step.
        assert len(a._live.snapshot()) == 1, a._live.snapshot()
        value = next(iter(a._live.snapshot().values()))
        assert "(2/211)" in value, value
        assert "b.weight" in value, value
        assert "(1/211)" not in value, value  # the earlier step was overwritten

        # The detail line was echoed to the RichLog (not swallowed, not clearing).
        rich_lines = a.query_one(RichLog).lines
        joined = "\n".join(str(line) for line in rich_lines)
        assert "Tensor shape" in joined, joined

async def test_log_msg_ctq_separator_keeps_bar_and_log(tmp_path, monkeypatch):
    # Regression for the "no bar + fewer strings" ComfyUI bug: convert_to_quant emits a
    # "----" separator between each (N/M) Processing header and its "    - " details. The
    # separator must echo to the RichLog WITHOUT clearing the live-progress widget.
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        a.log_msg("(1/211) Processing (INT8): a.weight")
        bar_after_header = dict(a._live.snapshot())
        a.log_msg("----")
        bar_after_separator = dict(a._live.snapshot())
        # Core fix: the separator did NOT wipe the bar.
        assert bar_after_separator == bar_after_header, (
            "separator cleared the live-progress bar",
            bar_after_header,
            bar_after_separator,
        )
        a.log_msg("    - Tensor shape: [1,2], Using k=256")
        a.log_msg("(2/211) Processing (INT8): b.weight")
        await pilot.pause()

        # Collapsed to latest header.
        assert len(a._live.snapshot()) == 1, a._live.snapshot()
        value = next(iter(a._live.snapshot().values()))
        assert "(2/211)" in value and "b.weight" in value, value

        # Separator + detail echoed to RichLog (fixed "fewer strings").
        rich = "\n".join(str(l) for l in a.query_one(RichLog).lines)
        assert "----" in rich, rich
        assert "Tensor shape" in rich, rich

        # Authoritative stream intact.
        full = a._read_full_log()
        assert "(1/211) Processing (INT8): a.weight" in full
        assert "----" in full
        assert "    - Tensor shape: [1,2], Using k=256" in full
        assert "(2/211) Processing (INT8): b.weight" in full

async def test_run_proc_streams_ctq_progress_to_live_widget(tmp_path, monkeypatch):
    # Faithful regression: a GENUINE child (tests/_ctq_launcher.py) emits ctq-style
    # "(N/M) Processing (INT8): ..." headers with "    - " detail lines. The live
    # widget must end on the final header and must NOT be cleared by the detail lines.
    launcher = os.path.join(os.path.dirname(__file__), "_ctq_launcher.py")
    venv_py = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".venv-test", "Scripts", "python.exe")
    )
    if not os.path.exists(venv_py):
        pytest.skip("venv python not found")
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        monkeypatch.setattr(
            appmod.QuantApp, "call_from_thread",
            lambda self, fn, *args: fn(*args),
        )
        captured: list[dict[str, str]] = []
        orig_update_progress = appmod.QuantApp.update_progress

        def rec_update(self, states):
            orig_update_progress(self, states)
            captured.append(dict(self._live.snapshot()))

        monkeypatch.setattr(appmod.QuantApp, "update_progress", rec_update)

        rc = a._run_proc([venv_py, launcher], os.path.dirname(launcher))
        await pilot.pause()

        assert rc == 0
        assert captured, "live-progress never received a ctq frame from the real subprocess"
        # The final live-progress entry is the last (N/M) header; detail lines did
        # not clear it mid-way (we saw at least one collapsed ctq entry).
        final = captured[-1]
        assert any("(3/3) Processing" in v for v in final.values()), final
        # Detail lines were echoed to the RichLog, not lost.
        rich_lines = a.query_one(RichLog).lines
        joined = "\n".join(str(line) for line in rich_lines)
        assert "Tensor shape" in joined, joined

async def test_run_proc_streams_ctq_progress_with_separators(tmp_path, monkeypatch):
    # HARD regression for the "no bar + fewer strings" ComfyUI bug. A GENUINE child
    # (tests/_ctq_launcher.py) emits the faithful convert_to_quant MAIN-phase stream:
    #     (1/3) Processing (INT8): layer.a
    #     ----
    #         - Tensor shape: [1, 2]
    #         - Trying svd_lowrank ...
    #     ... (repeated) ... ending on the per-tensor details, with NO trailing clear
    # line. The "----" separators and "    - " details historically cleared the bar.
    #
    # This test proves they do NOT: we record every call to _clear_live_progress
    # together with the bar's snapshot BEFORE the clear. The only clear must be the
    # benign "$ <cmd>" command echo (which fires when the bar is EMPTY, before any
    # ctq header appears). NO clear may happen while the bar holds a (N/M) header.
    launcher = os.path.join(os.path.dirname(__file__), "_ctq_launcher.py")
    venv_py = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".venv-test", "Scripts", "python.exe")
    )
    if not os.path.exists(venv_py):
        pytest.skip("venv python not found")
    monkeypatch.setenv("UNSLOTH_CTQ_LOG_DIR", str(tmp_path))
    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        # call_from_thread synchronous -> deterministic (we drive _run_proc from the
        # app thread, where the real call_from_thread would refuse to run).
        monkeypatch.setattr(
            appmod.QuantApp, "call_from_thread",
            lambda self, fn, *args: fn(*args),
        )

        # Record live-progress updates so we can prove the (3/3) header streamed in.
        captured: list[dict[str, str]] = []
        orig_update_progress = appmod.QuantApp.update_progress

        def rec_update(self, states):
            orig_update_progress(self, states)
            captured.append(dict(self._live.snapshot()))

        monkeypatch.setattr(appmod.QuantApp, "update_progress", rec_update)

        # Record every clear together with the bar's snapshot BEFORE the clear. This is
        # the precise bug-fix proof: a populated ctq bar must never be cleared mid-stream.
        cleared: list[dict[str, str]] = []
        orig_clear_live = appmod.QuantApp.clear_live

        def rec_clear(self):
            cleared.append(dict(self._live.snapshot()))  # snapshot BEFORE clearing
            orig_clear_live(self)

        monkeypatch.setattr(appmod.QuantApp, "clear_live", rec_clear)

        rc = a._run_proc([venv_py, launcher], os.path.dirname(launcher))
        await pilot.pause()

        assert rc == 0
        # The final ctq header streamed into the live widget.
        assert captured, "live-progress never received a ctq frame from the real subprocess"
        all_values = [v for snap in captured for v in snap.values()]
        assert any("(3/3) Processing (INT8)" in v for v in all_values), all_values

        # THE BUG-FIX PROOF: the "----" separators / "    - " details did NOT clear a
        # populated bar. The only allowed clear is the benign "$ <cmd>" echo, which fires
        # when the bar is empty (before any ctq header); such snapshots are {}.
        populated_clears = [snap for snap in cleared if snap]  # clears with a NON-empty bar
        assert populated_clears == [], (
            "live-progress bar was cleared mid-ctq-stream (the original bug): "
            f"{populated_clears}"
        )
        # (Sanity: the only clear we expect is the empty-bar command echo.)
        assert all(snap == {} for snap in cleared), cleared

        # The bar survives the run, ending on the LAST (N/M) header (no trailing clear
        # line in a real ctq stream). All three headers collapse to the shared "(#/3)" key.
        assert a._live.snapshot() == {"(#/3)": "(3/3) Processing (INT8): layer.c"}, a._live.snapshot()

        # Separator + details echoed to the RichLog (fixed "fewer strings").
        rich_lines = a.query_one(RichLog).lines
        joined = "\n".join(str(line) for line in rich_lines)
        assert "----" in joined, joined
        assert "Tensor shape" in joined, joined
        assert "Trying svd_lowrank" in joined, joined

        # Authoritative temp file keeps the full faithful stream.
        full = a._read_full_log()
        assert "(3/3) Processing (INT8): layer.c" in full
        assert "----" in full
        assert "    - Tensor shape: [5, 6]" in full
        assert "    - Trying svd_lowrank" in full

# --------------------------------------------------------------------------- #
# Stop button: app-level headless lifecycle (no real 30s subprocess)
# --------------------------------------------------------------------------- #
async def _wait_until(pred, pilot, timeout: float = 5.0) -> None:
    """Poll a predicate on the main thread, yielding to the app loop, until True.

    Used to bridge the gap between the main-thread test and the daemon worker
    thread (Stop arming / stop-branch status are scheduled via call_from_thread
    and only land on the main loop after a pause).
    """
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for expected UI condition")
        await pilot.pause()


async def test_stop_button_terminates_comfy_run(tmp_path):
    # App-level headless test for the Stop button. A FakeRunner replaces the real
    # WorkerRunner so no real 30s subprocess is spawned: its run() arms a flag and
    # BLOCKS on a threading.Event until terminate() is called (mirroring the real
    # runner's behaviour of blocking in _pump on a live child), then returns non-zero.
    # This exercises the full Stop lifecycle end-to-end through QuantApp.
    safetensors = tmp_path / "model.safetensors"
    safetensors.write_text("x")
    cfg = rc_mod.RunConfig(
        family=Family.COMFY,
        ctq=rc_mod.CtqConfig(
            input=str(safetensors),
            output=str(tmp_path / "out-fp8_e4m3.safetensors"),
            pybin=sys.executable,
            format="fp8_e4m3",
            output_mode="sharded",
            option_values={},
            quant_tags=["fp8_e4m3"],
        ),
    )

    class FakeRunner:
        def __init__(self) -> None:
            self._running = False
            self._ev = threading.Event()

        def run(self, cmd, cwd, observer, progress_debug_fh=None) -> int:
            self._running = True
            self._ev.wait()  # blocked until terminate() fires
            self._running = False
            return 1  # non-zero: the worker was stopped, not a clean success

        def is_running(self) -> bool:
            return self._running

        def terminate(self) -> None:
            self._running = False
            self._ev.set()

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        switch_family(a, Family.COMFY)
        # Feed a valid config directly (single source of truth; skips widget filling).
        a._read_config = lambda: cfg
        # Swap in the blocking fake runner so no real subprocess is launched.
        a.runner = FakeRunner()
        logs, statuses = attach_recorder(a)

        # Trigger the run from the main thread.
        a.action_run()

        # (1) The morphed Run button (#run_ctq) must relabel to "Stop" and stay enabled
        #     once the worker launches (no separate Stop button exists).
        await _wait_until(
            lambda: a.query_one("#run_ctq", Button).label == "Stop"
            and a.query_one("#run_ctq", Button).disabled is False
            and a._run_active is True,
            pilot,
            timeout=5,
        )
        btn = a.query_one("#run_ctq", Button)
        assert btn.label == "Stop", "Run button must morph to Stop on run start"
        assert btn.disabled is False, "Stop button must stay enabled while running"
        assert a._run_active is True

        # (2) The fake worker must actually be running before we press Stop.
        await _wait_until(lambda: a.runner.is_running() is True, pilot, timeout=5)
        assert a.runner.is_running() is True

        # (3) Press the MORPHED Stop button -> verifies on_button_pressed routes it to stop.
        btn.press()
        await pilot.pause()

        # terminate() was invoked -> the worker is no longer reported running.
        assert a.runner.is_running() is False
        # The stop flag is raised so the worker thread takes the stop branch.
        assert a._stop_requested is True

        # (4) The button morphs back to "Run Quantization" (enabled) once the run ends.
        await _wait_until(
            lambda: a.query_one("#run_ctq", Button).label == "Run Quantization"
            and a.query_one("#run_ctq", Button).disabled is False,
            pilot,
            timeout=5,
        )
        assert a.query_one("#run_ctq", Button).label == "Run Quantization"
        assert a.query_one("#run_ctq", Button).disabled is False

        # (5) The worker thread observes the flag and emits the user-stop transition.
        await _wait_until(
            lambda: "Stopped" in statuses
            and any("stopped by user" in line for line in logs),
            pilot,
            timeout=5,
        )
        assert "Stopped" in statuses, statuses
        assert any("stopped by user" in line for line in logs), logs
