"""Unit tests for quantui.profiles_store (plan S2.4 / S2.5).

Pure module: no Textual, no app boot needed. Uses tmp_path as config_dir.
"""

import os

import pytest

from quantui import profiles_store as ps


def _record(**kw):
    base = dict(
        ts="2026-08-24 10:00:00",
        family="gguf",
        method="q4_k_m",
        output="/tmp/out",
        status="success",
        exit_code=0,
        duration_s=12.3,
    )
    base.update(kw)
    return ps.RunRecord(**base)


def test_store_path_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ps.CONFIG_ENV_VAR, str(tmp_path))
    assert ps.store_path() == os.path.join(str(tmp_path), ps.STORE_FILENAME)


def test_store_path_default_home():
    path = ps.store_path("/unused") if False else ps.store_path(config_dir=None)
    # Without env var (test env may or may not set it): falls back to ~/.quantui
    expected = os.path.join(
        os.path.expanduser("~"), ".quantui", ps.STORE_FILENAME
    )
    assert path == expected


def test_roundtrip(tmp_path):
    d = str(tmp_path)
    cfg = {"family": "gguf", "model": "/m", "output": "/o"}
    ps.save_profile("p1", cfg, config_dir=d)
    assert ps.list_profiles(d) == ["p1"]
    got = ps.get_profile("p1", config_dir=d)
    assert got == cfg
    # Deep copy: mutating the returned dict must not corrupt the store.
    got["model"] = "CHANGED"
    assert ps.get_profile("p1", config_dir=d)["model"] == "/m"
    ps.delete_profile("p1", config_dir=d)
    assert ps.list_profiles(d) == []
    assert ps.get_profile("p1", config_dir=d) is None


def test_corrupt_json_returns_empty(tmp_path):
    d = str(tmp_path)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ps.STORE_FILENAME), "w", encoding="utf-8") as fh:
        fh.write("{not valid json!!")
    st = ps.load_store(config_dir=d)
    assert st == {"profiles": {}, "recents": []}
    # And list functions are safe on a corrupt store.
    assert ps.list_profiles(config_dir=d) == []
    assert ps.list_recents(config_dir=d) == []


def test_recents_capped_at_20(tmp_path):
    d = str(tmp_path)
    for i in range(25):
        ps.add_recent(_record(ts=f"t{i}").to_dict(), config_dir=d)
    recents = ps.list_recents(config_dir=d)
    assert len(recents) == ps.MAX_RECENTS == 20
    # Newest first.
    assert recents[0]["ts"] == "t24"
    assert recents[-1]["ts"] == "t5"


def test_run_record_roundtrip_dataclass():
    r = _record(status="failed", exit_code=2)
    r2 = ps.RunRecord.from_dict(r.to_dict())
    assert r2 == r
    # Unknown keys are dropped, not raised.
    r3 = ps.RunRecord.from_dict({**r.to_dict(), "bogus": 1})
    assert r3.status == "failed"


def test_save_profile_rejects_empty_name(tmp_path):
    with pytest.raises(ValueError):
        ps.save_profile("  ", {"a": 1}, config_dir=str(tmp_path))


def test_save_creates_missing_dir(tmp_path):
    d = str(tmp_path / "nested" / "cfg")
    ps.save_profile("p", {"x": 1}, config_dir=d)
    assert os.path.exists(os.path.join(d, ps.STORE_FILENAME))
