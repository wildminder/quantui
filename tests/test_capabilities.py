"""Tests for capability probing + soft requirement checks (capabilities.py).

The probe runs ``pybin -c "<script>"``. In the sandbox the real ``convert_to_quant`` /
``torch`` are absent, so we stub ``capabilities.subprocess.run`` to return a JSON env
snapshot -- functionally the plan's "stub python shim" but cross-platform safe (a real
``.cmd``/``.sh`` shim can't be launched by CreateProcess without shell=True on this
Windows Python, and the production probe must stay shell=False).
"""

import json
import subprocess

from quantui import capabilities
from quantui.capabilities import CapabilityReport, check_ctq_requirements, probe_worker_env


def _fake_result(payload, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    out = json.dumps(payload) if returncode == 0 else ""
    return subprocess.CompletedProcess(["pybin", "-c", ""], returncode, stdout=out, stderr=stderr)


def patch_probe(monkeypatch, payload, returncode: int = 0, stderr: str = ""):
    res = _fake_result(payload, returncode, stderr)
    monkeypatch.setattr(capabilities.subprocess, "run", lambda *a, **k: res)


def _raise_no_interp(*a, **k):
    raise FileNotFoundError("no such interpreter")


REPORT_OK = {
    "python_version": "3.13.12",
    "torch_version": "2.8.0",
    "cuda_version": "12.8",
    "triton": False,
    "comfy_kitchen": False,
    "convert_to_quant": True,
}


# ---- D1: probe parsing ----------------------------------------------------- #

def test_probe_parses_payload(monkeypatch):
    patch_probe(monkeypatch, REPORT_OK)
    rep = probe_worker_env("fakepy")
    assert rep.python_version == "3.13.12"
    assert rep.torch_version == "2.8.0"
    assert rep.cuda_version == "12.8"
    assert rep.triton is False
    assert rep.comfy_kitchen is False
    assert rep.convert_to_quant is True


def test_probe_missing_interpreter(monkeypatch):
    monkeypatch.setattr(capabilities.subprocess, "run", _raise_no_interp)
    rep = probe_worker_env("this_interpreter_does_not_exist_xyz")
    assert rep.torch_version is None
    assert rep.warnings  # advisory, not an exception


def test_probe_nonzero_rc(monkeypatch):
    patch_probe(monkeypatch, {}, returncode=1, stderr="boom")
    rep = probe_worker_env("fakepy")
    assert rep.warnings


# ---- D1: requirement checks ------------------------------------------------ #

def test_check_nvfp4_needs_torch_and_cuda():
    rep = CapabilityReport("3.13", "2.8.0", "12.8", False, False, True)
    warns = check_ctq_requirements(rep, "nvfp4")
    assert any("torch" in w for w in warns)  # 2.8 < 2.10
    assert any("CUDA" in w for w in warns)   # 12.8 < 13.0


def test_check_int8_needs_triton():
    rep = CapabilityReport("3.13", "2.8.0", "12.8", False, False, True)
    warns = check_ctq_requirements(rep, "int8_convrot")
    assert any("Triton" in w for w in warns)


def test_fp8_no_warning_when_ok():
    rep = CapabilityReport("3.13", "2.8.0", "12.8", False, False, True)
    assert check_ctq_requirements(rep, "fp8_e4m3") == []


def test_missing_ctq_advisory():
    rep = CapabilityReport("3.13", "2.8.0", "12.8", False, False, False)
    warns = check_ctq_requirements(rep, "int8_block")
    assert any("convert_to_quant" in w for w in warns)


def test_blackwell_ok_with_full_stack():
    rep = CapabilityReport("3.12", "2.10.0", "13.0", True, True, True)
    assert check_ctq_requirements(rep, "nvfp4") == []


# ---- D3: stub-driven integration of probe + checks ------------------------- #

def test_stub_int8_flags_triton_fp8_clean(monkeypatch):
    patch_probe(monkeypatch, REPORT_OK)
    rep = probe_worker_env("fakepy")
    assert any("Triton" in w for w in check_ctq_requirements(rep, "int8_convrot"))
    assert check_ctq_requirements(rep, "fp8_e4m3") == []
