"""T7 (plan 2026-08-31-gguf-unsloth-parity): worker.py GGUF argument surface.

Pure-function tests import the small helpers (parse_methods reuse, imatrix
mapping) with NO unsloth/torch import; subprocess tests prove the pre-import
gate fires BEFORE the heavy `from unsloth import ...` (which would otherwise
burn 10-30s + a full model load before failing).
"""

import os
import subprocess
import sys

import pytest

from quantui import worker as wk
from quantui.quant_methods import IMATRIX_QUANT_IDS


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_imatrix_value_maps_auto_to_true():
    # "auto" -> True: unsloth's own contract for "fetch the upstream imatrix".
    assert wk.imatrix_value("auto") is True


def test_imatrix_value_maps_path():
    # A path passes through verbatim (string), so unsloth loads THAT file.
    assert wk.imatrix_value("C:/imat/imatrix.dat") == "C:/imat/imatrix.dat"


def test_imatrix_value_maps_empty_to_none():
    assert wk.imatrix_value("") is None


def test_worker_method_split_single_str():
    # One id -> plain string: exactly what save_pretrained_gguf expects.
    assert wk.method_or_methods("q4_k_m") == "q4_k_m"


def test_worker_method_split_multi_list():
    # >=2 ids -> list, matching unsloth's str-or-list API surface.
    assert wk.method_or_methods("q4_k_m,q5_k_m") == ["q4_k_m", "q5_k_m"]
    assert wk.method_or_methods("q4_k_m, q5_k_m , q4_k_m") == ["q4_k_m", "q5_k_m"]


def test_worker_method_split_empty():
    with pytest.raises(ValueError):
        wk.method_or_methods(" , ")


# --------------------------------------------------------------------------- #
# Subprocess: pre-import gate + --list-methods
# --------------------------------------------------------------------------- #
def _run_worker(*argv):
    """Run worker.py as a subprocess with the repo on sys.path."""
    env = dict(os.environ)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "quantui.worker", *argv],
        capture_output=True, text=True, env=env, timeout=120,
    )


def test_preimport_gate_blocks_iq_no_imatrix(tmp_path):
    # The gate must fire BEFORE `from unsloth import ...`: exit 1, friendly
    # message, and NO unsloth-import traceback in stderr (the import may take
    # 10-30s in a real env and the failure would come way too late).
    r = _run_worker("--model", str(tmp_path), "--output", str(tmp_path / "o"),
                    "--method", "iq2_xs")
    assert r.returncode == 1
    assert "imatrix" in r.stderr.lower()
    assert "Traceback" not in r.stderr


def test_preimport_gate_missing_imatrix_file(tmp_path):
    missing = tmp_path / "nope.dat"
    r = _run_worker("--model", str(tmp_path), "--output", str(tmp_path / "o"),
                    "--method", "iq2_xs", "--imatrix", str(missing))
    assert r.returncode == 1
    assert str(missing) in r.stderr
    assert "Traceback" not in r.stderr


def test_preimport_gate_allows_auto(tmp_path):
    # "auto" skips the existence check (fetched at run time) -> reaches the
    # model-path resolution stage (fails there with the model-path message,
    # NOT the imatrix message).
    r = _run_worker("--model", str(tmp_path / "no-model"), "--output", str(tmp_path / "o"),
                    "--method", "iq2_xs", "--imatrix", "auto")
    assert "imatrix" not in r.stderr.lower()
    assert "Model path does not exist" in r.stderr


def test_list_methods_marks_imatrix():
    r = _run_worker("--list-methods")
    assert r.returncode == 0
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    for mid in IMATRIX_QUANT_IDS:
        assert f"{mid} [IMATRIX]" in lines, (mid, lines)
    assert "q8_0" in lines  # plain id, NO marker
    for ln in lines:
        if ln.startswith("q8_0"):
            assert "[IMATRIX]" not in ln


# --------------------------------------------------------------------------- #
# _run_with_output_progress: export-thread failure MUST propagate (regression)
# --------------------------------------------------------------------------- #
def test_progress_wrapper_propagates_export_failure(tmp_path):
    # Found by the LFM2.5-VL imatrix run: unsloth raised inside the export
    # thread, but the wrapper printed DONE + 100% and exited 0 -- the TUI
    # showed success while no file existed. The exception must re-raise on
    # the calling thread and the 100%/DONE line must NOT print.
    import contextlib
    import io

    def boom():
        raise RuntimeError("Unsloth: imatrix_file=True but no upstream imatrix was found.")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pytest.raises(RuntimeError, match="no upstream imatrix"):
            wk._run_with_output_progress(boom, str(tmp_path / "progress-empty"), 1000, "Exporting GGUF")
    assert '"pct": 100.0' not in buf.getvalue()


def test_progress_wrapper_prints_100_on_success(tmp_path):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wk._run_with_output_progress(lambda: None, str(tmp_path / "progress-empty"), 1000, "Exporting GGUF")
    assert '"pct": 100.0' in buf.getvalue()
