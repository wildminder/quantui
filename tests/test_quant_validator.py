"""Tests for the ComfyUI INT8 quantized-file validator (Msg E).

Covers the model-agnostic structural validation in ``quantui.quant_validator``
plus a headless TUI integration test that exercises the new "Validate" button.
"""

import math
from pathlib import Path

import numpy as np

from quantui import quant_validator as qv
from quantui.comfy_quant_schema import encode_comfy_quant_config, write_safetensors


# --------------------------------------------------------------------------- #
# Fixture builder (pure bytes via numpy; mirrors convert_to_quant's on-disk layout)
# --------------------------------------------------------------------------- #
def _write_quant_file(path: str, layers: list[tuple[str, tuple[int, int], dict]]) -> None:
    """Write a safetensors file with the given quantized layers.

    ``layers`` is ``(name, (out_features, in_features), comfy_quant_cfg)``.
    """
    specs: dict[str, tuple[str, list[int], bytes]] = {}
    for name, shape, cfg in layers:
        out_f, in_f = shape
        weight = np.random.randint(-120, 120, size=shape, dtype=np.int8)
        if cfg.get("format") == "int8_blockwise":
            bs = int(cfg["group_size"])
            s_shape = (math.ceil(out_f / bs), math.ceil(in_f / bs))
        else:
            s_shape = (out_f, 1)
        scale = np.full(s_shape, 0.01, dtype=np.float32)
        specs[f"{name}.weight"] = ("I8", list(shape), weight.tobytes())
        specs[f"{name}.weight_scale"] = ("F32", list(s_shape), scale.tobytes())
        if cfg.get("format") == "int8_blockwise":
            specs[f"{name}.input_scale"] = ("F32", [], np.array(1.0, dtype=np.float32).tobytes())
        blob = encode_comfy_quant_config(cfg)
        specs[f"{name}.comfy_quant"] = ("U8", [len(blob)], blob)
    Path(path).write_bytes(write_safetensors(specs))


CONVROT_CFG = {
    "format": "int8_tensorwise",
    "convrot": True,
    "convrot_groupsize": 4,
    "per_row": True,
    "orig_dtype": "torch.bfloat16",
}
BLOCK_CFG = {"format": "int8_blockwise", "group_size": 4, "orig_dtype": "torch.bfloat16"}


# --------------------------------------------------------------------------- #
# Unit tests: structural validation
# --------------------------------------------------------------------------- #
def test_valid_convrot_passes(tmp_path):
    f = tmp_path / "q.st"
    _write_quant_file(str(f), [("t.block.0.attn.to_q", (16, 8), CONVROT_CFG)])
    r = qv.validate_comfy_quant(str(f))
    assert r.ok
    assert "int8_tensorwise" in r.summary["formats_found"]
    assert r.summary["quantized_matrices"] == 1


def test_valid_blockwise_passes(tmp_path):
    f = tmp_path / "q.st"
    _write_quant_file(str(f), [("t.block.0.ff.0", (16, 8), BLOCK_CFG)])
    r = qv.validate_comfy_quant(str(f))
    assert r.ok
    assert "int8_blockwise" in r.summary["formats_found"]


def test_invalid_wrong_dtype_and_missing_scale_fails(tmp_path):
    # weight stored as F32 (not int8) and the weight_scale is missing entirely.
    specs = {
        "m.weight": ("F32", [16, 8], np.full((16, 8), 0.5, dtype=np.float32).tobytes()),
        "m.comfy_quant": ("U8", [len(encode_comfy_quant_config(CONVROT_CFG))],
                          encode_comfy_quant_config(CONVROT_CFG)),
    }
    f = tmp_path / "bad.st"
    f.write_bytes(write_safetensors(specs))
    r = qv.validate_comfy_quant(str(f))
    assert not r.ok
    assert any("missing weight_scale" in e for e in r.errors)
    assert any("dtype" in e and "I8" in e for e in r.errors)


def test_invalid_convrot_groupsize_not_power_of_four_fails(tmp_path):
    bad = dict(CONVROT_CFG)
    bad["convrot_groupsize"] = 6  # not a power of four, and 8 % 6 != 0
    f = tmp_path / "bad.st"
    _write_quant_file(str(f), [("m", (16, 8), bad)])
    r = qv.validate_comfy_quant(str(f))
    assert not r.ok
    assert any("convrot_groupsize" in e for e in r.errors)


def test_invalid_blockwise_missing_input_scale_fails(tmp_path):
    # blockwise layer with no input_scale scalar.
    specs = {
        "m.weight": ("I8", [16, 8], np.random.randint(-120, 120, size=(16, 8), dtype=np.int8).tobytes()),
        "m.weight_scale": ("F32", [4, 2], np.full((4, 2), 0.01, dtype=np.float32).tobytes()),
        "m.comfy_quant": ("U8", [len(encode_comfy_quant_config(BLOCK_CFG))],
                          encode_comfy_quant_config(BLOCK_CFG)),
    }
    f = tmp_path / "bad.st"
    f.write_bytes(write_safetensors(specs))
    r = qv.validate_comfy_quant(str(f))
    assert not r.ok
    assert any("input_scale" in e for e in r.errors)


def test_orphan_scale_detected(tmp_path):
    # weight_scale with no matching .comfy_quant marker.
    specs = {
        "m.weight": ("I8", [16, 8], np.random.randint(-120, 120, size=(16, 8), dtype=np.int8).tobytes()),
        "m.weight_scale": ("F32", [16, 1], np.full((16, 1), 0.01, dtype=np.float32).tobytes()),
    }
    f = tmp_path / "orphan.st"
    f.write_bytes(write_safetensors(specs))
    r = qv.validate_comfy_quant(str(f))
    # No markers -> reported ok with a warning (plain checkpoint), but an orphan scale
    # only matters when a marker exists; here there are none so it's the "no markers" path.
    assert r.ok
    assert any("no .comfy_quant markers" in w for w in r.warnings)


def test_no_markers_is_ok_with_warning(tmp_path):
    specs = {"w.weight": ("F32", [4, 4], np.full((4, 4), 0.1, dtype=np.float32).tobytes())}
    f = tmp_path / "plain.st"
    f.write_bytes(write_safetensors(specs))
    r = qv.validate_comfy_quant(str(f))
    assert r.ok
    assert r.summary["quantized_matrices"] == 0


def test_missing_file_reports_error():
    r = qv.validate_comfy_quant("does_not_exist.st")
    assert not r.ok
    assert any("file not found" in e for e in r.errors)


def test_format_report_contains_pass_and_stats(tmp_path):
    f = tmp_path / "q.st"
    _write_quant_file(str(f), [("t.block.0.attn.to_q", (16, 8), CONVROT_CFG)])
    r = qv.validate_comfy_quant(str(f))
    rep = qv.format_report(r)
    assert "PASS" in rep
    assert "quantized matrices" in rep
    assert "quantized share" in rep


def test_quantized_share_stat(tmp_path):
    # 1 quantized matrix (16*8=128 params) vs 1 full-precision weight (4*4=16) -> ~88.9%
    f = tmp_path / "mix.st"
    specs = {
        "q.weight": ("I8", [16, 8], np.random.randint(-120, 120, size=(16, 8), dtype=np.int8).tobytes()),
        "q.weight_scale": ("F32", [16, 1], np.full((16, 1), 0.01, dtype=np.float32).tobytes()),
        "q.comfy_quant": ("U8", [len(encode_comfy_quant_config(CONVROT_CFG))],
                          encode_comfy_quant_config(CONVROT_CFG)),
        "fp.weight": ("F32", [4, 4], np.full((4, 4), 0.1, dtype=np.float32).tobytes()),
    }
    f.write_bytes(write_safetensors(specs))
    r = qv.validate_comfy_quant(str(f))
    assert r.summary["quantized_params"] == 128
    assert r.summary["full_precision_params"] == 16
    # quantized_share_pct is rounded to 3 decimals in the summary.
    assert abs(r.summary["quantized_share_pct"] - round(100.0 * 128 / 144, 3)) < 1e-9


# --------------------------------------------------------------------------- #
# Headless TUI integration: the "Validate" button
# --------------------------------------------------------------------------- #
async def _wait_until(pred, pilot, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for expected UI condition")
        await pilot.pause()


def _switch_family(app, fam):
    from quantui.quant_methods import Family

    app.family = fam
    app.query_one("#gguf_panel").display = (fam == Family.GGUF)
    app.query_one("#comfy_panel").display = (fam == Family.COMFY)
    app.refresh_ctq_visibility()


async def test_validate_button_logs_valid(tmp_path):
    from textual.widgets import Button, RichLog

    from quantui import app as appmod
    from quantui.quant_methods import Family

    f = tmp_path / "q.st"
    _write_quant_file(str(f), [("t.block.0.attn.to_q", (16, 8), CONVROT_CFG)])

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        _switch_family(a, Family.COMFY)
        a.query_one("#validate_path").value = str(f)
        a.query_one("#validate", Button).press()
        await _wait_until(
            lambda: "PASS" in "\n".join(str(line) for line in a.query_one(RichLog).lines),
            pilot, timeout=5,
        )
        rich = "\n".join(str(line) for line in a.query_one(RichLog).lines)
        assert "PASS" in rich
        assert "int8_tensorwise" in rich


async def test_validate_button_logs_invalid(tmp_path):
    from textual.widgets import Button, RichLog

    from quantui import app as appmod
    from quantui.quant_methods import Family

    # broken: weight dtype F32 + missing scale
    specs = {
        "m.weight": ("F32", [16, 8], np.full((16, 8), 0.5, dtype=np.float32).tobytes()),
        "m.comfy_quant": ("U8", [len(encode_comfy_quant_config(CONVROT_CFG))],
                          encode_comfy_quant_config(CONVROT_CFG)),
    }
    f = tmp_path / "bad.st"
    f.write_bytes(write_safetensors(specs))

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        _switch_family(a, Family.COMFY)
        a.query_one("#validate_path").value = str(f)
        a.query_one("#validate", Button).press()
        await _wait_until(
            lambda: "FAIL" in "\n".join(str(line) for line in a.query_one(RichLog).lines),
            pilot, timeout=5,
        )
        rich = "\n".join(str(line) for line in a.query_one(RichLog).lines)
        assert "FAIL" in rich


async def test_validate_button_empty_path_errors(tmp_path):
    from textual.widgets import RichLog

    from quantui import app as appmod
    from quantui.quant_methods import Family

    a = appmod.QuantApp()
    async with a.run_test() as pilot:
        _switch_family(a, Family.COMFY)
        a.query_one("#validate_path").value = ""
        a.query_one("#validate").press()
        await _wait_until(
            lambda: any("no file path" in str(line) for line in a.query_one(RichLog).lines),
            pilot, timeout=5,
        )
        rich = "\n".join(str(line) for line in a.query_one(RichLog).lines)
        assert "no file path provided" in rich
