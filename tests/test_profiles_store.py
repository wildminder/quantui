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


# --------------------------------------------------------------------------- #
# T10 (plan 2026-08-31-gguf-unsloth-parity): profiles saved before the
# registry rebuild may reference method ids that no longer exist
# (q4_k_xl / q3_k_xl / q2_k_xl were proprietary UD mixes; q4_nl never existed
# as an official id). Loading must downgrade instead of crashing or silently
# running a dead id.
# --------------------------------------------------------------------------- #
def test_old_profile_q4_k_xl_downgrades(tmp_path):
    d = str(tmp_path)
    ps.save_profile(
        "old", {"family": "gguf", "method": "q4_k_xl"}, config_dir=d
    )
    prof = ps.get_profile("old", config_dir=d)
    assert prof is not None
    assert prof["method"] == "q4_k_m"
    # The downgrade must be visible, not silent.
    assert "downgrade" in prof.get("_notes", "").lower()


def test_old_profile_q3_k_xl_downgrades(tmp_path):
    d = str(tmp_path)
    ps.save_profile(
        "old", {"family": "gguf", "method": "q3_k_xl"}, config_dir=d
    )
    prof = ps.get_profile("old", config_dir=d)
    assert prof is not None
    assert prof["method"] == "q4_k_m"


def test_old_custom_q4_k_xl_downgrades(tmp_path):
    d = str(tmp_path)
    ps.save_profile(
        "old",
        {"family": "gguf", "method": "q4_k_m", "custom": "q4_k_xl"},
        config_dir=d,
    )
    prof = ps.get_profile("old", config_dir=d)
    assert prof is not None
    assert prof["custom"] == ""
    assert prof["method"] == "q4_k_m"


def test_old_profile_q4_nl_downgrades(tmp_path):
    # q4_nl never existed (the real id is the imatrix-gated iq4_nl).
    d = str(tmp_path)
    ps.save_profile(
        "old", {"family": "gguf", "method": "q4_nl"}, config_dir=d
    )
    prof = ps.get_profile("old", config_dir=d)
    assert prof is not None
    assert prof["method"] == "q4_k_m"


def test_valid_profile_untouched(tmp_path):
    d = str(tmp_path)
    ps.save_profile(
        "ok",
        {"family": "gguf", "method": "q5_k_m", "custom": "iq2_xs"},
        config_dir=d,
    )
    prof = ps.get_profile("ok", config_dir=d)
    assert prof is not None
    assert prof["method"] == "q5_k_m"
    assert prof["custom"] == "iq2_xs"
    assert "_notes" not in prof


def test_mixed_comma_list_partially_dead(tmp_path):
    # A comma list where SOME ids are dead: dead ids are dropped, valid ones
    # kept (a fully-dead list degrades to the single default).
    d = str(tmp_path)
    ps.save_profile(
        "mix",
        {"family": "gguf", "method": "q4_k_xl, q5_k_m"},
        config_dir=d,
    )
    prof = ps.get_profile("mix", config_dir=d)
    assert prof is not None
    assert prof["method"] == "q5_k_m"


def test_recent_with_removed_id_loads():
    # Recents are display + re-run only; a dead id in a record must never
    # raise on from_dict (display tolerates any string).
    rec = ps.RunRecord.from_dict(
        {"ts": "t", "family": "gguf", "method": "q4_k_xl", "status": "success"}
    )
    assert rec.method == "q4_k_xl"
    assert rec.status == "success"


def test_no_removed_id_fixtures_remain():
    # Grep-equivalent sweep over tests/: the dead ids may appear ONLY in the
    # downgrade tests themselves (and registry tripwires), never as a valid
    # fixture value elsewhere.
    import glob

    dead = ("q4_k_xl", "q3_k_xl", "q2_k_xl", "q4_nl")
    allowed = (
        "test_profiles_store.py",  # this downgrade suite
        "test_quant_methods.py",   # registry tripwires (absence asserts)
    )
    for path in glob.glob(os.path.join(os.path.dirname(__file__), "test_*.py")):
        base = os.path.basename(path)
        if base in allowed:
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for gone in dead:
            assert gone not in src, f"{base} still references dead id {gone}"
