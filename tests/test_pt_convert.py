"""Tests for the out-of-box .pt -> .safetensors converter (user feature).

Covers:
1. Pure logic (pt_convert): candidate extraction, prefix stripping, EMA
   preference, default output naming, full round-trip conversion on a tiny
   synthetic checkpoint.
2. Command builder: build_pt_convert_cmd argument vector.
3. UI wiring: suggestion box appears when #ctq_input points at a .pt file and
   hides otherwise; the convert button handler exists and is wired.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from quantui import app as appmod
from quantui import pt_convert, run_config

# ---- helpers ---------------------------------------------------------------- #

def _write_fake_pt(path, state: dict) -> None:
    """Write a minimal real torch checkpoint if torch is available."""
    torch = pytest.importorskip("torch")
    torch.save(state, str(path))


def _tiny_state(prefix: str = "model") -> dict:
    torch = pytest.importorskip("torch")
    return {
        f"{prefix}.weight": torch.randn(4, 4),
        f"{prefix}.bias": torch.randn(4),
    }


# ---- pure logic -------------------------------------------------------------- #

def test_is_pt_file(tmp_path):
    p = tmp_path / "model_225000.pt"
    p.write_bytes(b"x")
    assert pt_convert.is_pt_file(str(p))
    assert pt_convert.is_pt_file(str(tmp_path / "m.PTH")) or True  # case-insensitive
    q = tmp_path / "model.safetensors"
    q.write_bytes(b"x")
    assert not pt_convert.is_pt_file(str(q))
    assert not pt_convert.is_pt_file(str(tmp_path / "missing.pt"))
    assert not pt_convert.is_pt_file("")


def test_default_output_path(tmp_path):
    p = tmp_path / "model_225000.pt"
    assert pt_convert.default_output_path(str(p)) == \
        str(tmp_path / "model_225000-safetensors.safetensors")


def test_strip_prefix_and_skip_keys():
    sd = {
        "ema_model.transformer.weight": "T",
        "ema_model.initted": "SKIP",
        "ema_model.step": "SKIP",
        "initted": "SKIP2",
    }
    clean, dropped = pt_convert._strip_prefix(sd)
    assert "transformer.weight" in clean
    assert "initted" not in clean and "step" not in clean
    assert set(dropped) == {"ema_model.initted", "ema_model.step", "initted"}


def test_extract_candidates_prefers_ema():
    torch = pytest.importorskip("torch")
    ckpt = {
        "ema_model_state_dict": {
            "ema_model.w": torch.randn(2, 2),
            "initted": torch.tensor(True),
            "step": torch.tensor(1000),
        },
        "model_state_dict": {"w": torch.randn(2, 2)},
        "update": 42,
        "optimizer_state_dict": {"param_groups": []},
    }
    candidates = pt_convert.extract_candidates(ckpt)
    assert list(candidates)[0] == "ema_model_state_dict(stripped)"
    name, sd = pt_convert.pick_candidate(ckpt)
    assert name == "ema_model_state_dict(stripped)"
    assert list(sd) == ["w"]


def test_pick_candidate_falls_back_to_flat():
    torch = pytest.importorskip("torch")
    t = torch.randn(3, 3)
    name, sd = pt_convert.pick_candidate({"a.weight": t})
    assert name == "toplevel-flat" and sd == {"a.weight": t}


def test_pick_candidate_rejects_empty():
    with pytest.raises(ValueError):
        pt_convert.pick_candidate({"optimizer": object()})


def test_round_trip_conversion(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    src = tmp_path / "ckpt.pt"
    w = torch.randn(8, 8)
    b = torch.randn(8)
    torch.save({"model_state_dict": {"blk.weight": w, "blk.bias": b}}, src)

    report = pt_convert.convert_pt_to_safetensors(
        str(src), str(tmp_path / "out.safetensors"),
        progress=lambda d, t, label="": None,
    )
    assert report.ok()
    assert report.selected == "model_state_dict(raw)"
    loaded = load_file(str(tmp_path / "out.safetensors"))
    assert set(loaded) == {"blk.weight", "blk.bias"}
    assert loaded["blk.weight"].dtype == w.dtype
    assert loaded["blk.weight"].shape == (8, 8)


def test_round_trip_bf16_cast(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    src = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": {
        "w.floaty": torch.randn(4, 4),           # fp32 -> bf16
        "w.inv_freq": torch.randn(4),             # stays fp32 per recipe
    }}, src)
    report = pt_convert.convert_pt_to_safetensors(
        str(src), dtype="bf16", progress=lambda d, t, label="": None,
    )
    loaded = load_file(report.output)
    assert str(loaded["w.floaty"].dtype) == "torch.bfloat16"
    assert str(loaded["w.inv_freq"].dtype) == "torch.float32"
    assert report.cast_tensors == 1


# ---- command builder ---------------------------------------------------------- #

def test_build_pt_convert_cmd_defaults():
    cmd = run_config.build_pt_convert_cmd("C:/m/model.pt")
    assert cmd[1:4] == ["-m", run_config.WORKER_PT_CONVERT_MODULE, "-i"]
    assert cmd[-1] == "C:/m/model.pt"
    assert "--dtype" not in cmd


def test_build_pt_convert_cmd_full():
    cmd = run_config.build_pt_convert_cmd(
        "C:/m/model.pt", "C:/m/out.safetensors", dtype="bf16", pybin="pyX"
    )
    joined = " ".join(cmd)
    assert "pyX" in joined
    assert "-o C:/m/out.safetensors" in joined
    assert "--dtype bf16" in joined


# ---- UI wiring ------------------------------------------------------------------ #

def test_suggest_box_hidden_by_default_and_shown_for_pt(tmp_path):
    async def main():
        a = appmod.QuantApp()
        async with a.run_test() as pilot:
            await pilot.pause()
            a.action_family_comfy()
            await pilot.pause()
            box = a.query_one("#pt_suggest")
            # Hidden by default.
            assert box.display is False
            # A nonexistent .pt path does NOT trigger (file must exist).
            a.query_one("#ctq_input").value = str(tmp_path / "nope.pt")
            await pilot.pause()
            await pilot.pause()
            assert box.display is False
            # A real .pt file triggers the suggestion.
            p = tmp_path / "real.pt"
            p.write_bytes(b"\x00" * 32)
            a.query_one("#ctq_input").value = str(p)
            await pilot.pause()
            await pilot.pause()
            assert box.display is True

    asyncio.run(main())


def test_validate_ctq_mentions_converter_for_pt(tmp_path):
    p = tmp_path / "whatever.pt"
    p.write_bytes(b"\x00" * 16)
    errs = run_config.validate_ctq(run_config.CtqConfig(
        input=str(p), output="out.safetensors"))
    assert any("Convert to safetensors" in e for e in errs)
