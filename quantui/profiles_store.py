"""Profiles + recent-jobs persistence (plan S2.4 / S2.5, architecture Q3).

Pure module: NO Textual imports, so it is unit-testable anywhere and safe to
use from worker threads.

Storage layout (Q3): one JSON document at
``$UNSLOTH_QUANT_CONFIG_DIR/store.json`` when the env var is set, else
``~/.quantui/store.json``. Shape::

    {
      "profiles": {"<name>": {"family": "gguf", "fields": {...}}, ...},
      "recents": [ {"ts": ..., "family": ..., "method": ..., "output": ...,
                    "status": ..., "exit_code": 0, "duration_s": 12.3}, ... ]
    }

``recents`` is capped at ``MAX_RECENTS`` (20, newest first). All functions are
best-effort about I/O errors on READ (corrupt json -> empty store) but raise on
SAVE failures only where documented (callers wrap in try/except).
"""

import copy
import json
import os
from dataclasses import asdict, dataclass, field

CONFIG_ENV_VAR = "UNSLOTH_QUANT_CONFIG_DIR"
STORE_FILENAME = "store.json"
MAX_RECENTS = 20
# Filename inside an output dir that marks a partial/interrupted checkpoint
# (S2.5 resume prompt). worker.py writes this when a GGUF save is interrupted.
PARTIAL_MARKER = ".quantui_partial"


@dataclass
class RunRecord:
    """One finished quantization run (plan Q6: data shape now, multi-run later)."""

    ts: str = ""
    family: str = ""
    method: str = ""
    output: str = ""
    status: str = ""  # "success" | "failed" | "stopped"
    exit_code: int = 0
    duration_s: float = 0.0
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


def store_path(config_dir: str | None = None) -> str:
    """Resolve the JSON store path (env override wins; Q3)."""
    if config_dir is None:
        config_dir = os.environ.get(CONFIG_ENV_VAR) or os.path.join(
            os.path.expanduser("~"), ".quantui"
        )
    return os.path.join(config_dir, STORE_FILENAME)


def load_store(config_dir: str | None = None) -> dict:
    """Load the full store document; corrupt/missing file -> empty store."""
    path = store_path(config_dir)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        # Missing file or corrupt content both fall back to a fresh store.
        return {"profiles": {}, "recents": []}
    if not isinstance(data, dict):
        return {"profiles": {}, "recents": []}
    data.setdefault("profiles", {})
    data.setdefault("recents", [])
    if not isinstance(data["profiles"], dict):
        data["profiles"] = {}
    if not isinstance(data["recents"], list):
        data["recents"] = []
    return data


def save_store(store: dict, config_dir: str | None = None) -> None:
    """Persist the whole store document (creates the directory if needed).

    Raises on I/O failure -- callers that must not crash the UI wrap this.
    """
    path = store_path(config_dir)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)


# --------------------------------------------------------------------------- #
# Profiles (S2.4)
# --------------------------------------------------------------------------- #
def save_profile(
    name: str,
    cfg_dict: dict,
    config_dir: str | None = None,
    store: dict | None = None,
) -> dict:
    """Save a named profile; returns the updated store.

    With ``store`` given, mutates and returns it WITHOUT touching disk (the
    caller persists); otherwise loads + saves the on-disk store.
    """
    if not name or not name.strip():
        raise ValueError("profile name must be non-empty")
    profile = copy.deepcopy(cfg_dict)
    if store is not None:
        store.setdefault("profiles", {})[name] = profile
        return store
    st = load_store(config_dir)
    st.setdefault("profiles", {})[name] = profile
    save_store(st, config_dir)
    return st


def get_profile(
    name: str, config_dir: str | None = None, store: dict | None = None
) -> dict | None:
    """Return a deep copy of the named profile, or None when absent."""
    if store is None:
        store = load_store(config_dir)
    prof = store.get("profiles", {}).get(name)
    return copy.deepcopy(prof) if prof is not None else None


def delete_profile(
    name: str, config_dir: str | None = None, store: dict | None = None
) -> dict:
    """Delete a named profile (no-op when absent); returns the updated store."""
    if store is not None:
        store.setdefault("profiles", {}).pop(name, None)
        return store
    st = load_store(config_dir)
    st.setdefault("profiles", {}).pop(name, None)
    save_store(st, config_dir)
    return st


def list_profiles(config_dir: str | None = None, store: dict | None = None) -> list[str]:
    """Sorted profile names."""
    if store is None:
        store = load_store(config_dir)
    return sorted(store.get("profiles", {}).keys())


# --------------------------------------------------------------------------- #
# Recents (S2.5)
# --------------------------------------------------------------------------- #
def add_recent(
    record: dict,
    config_dir: str | None = None,
    store: dict | None = None,
) -> dict:
    """Prepend a RunRecord dict to recents (newest first), cap at MAX_RECENTS.

    Same store-mutation contract as :func:`save_profile`.
    """
    if store is not None:
        recents = store.setdefault("recents", [])
        recents.insert(0, dict(record))
        del recents[MAX_RECENTS:]
        return store
    st = load_store(config_dir)
    recents = st.setdefault("recents", [])
    recents.insert(0, dict(record))
    del recents[MAX_RECENTS:]
    save_store(st, config_dir)
    return st


def list_recents(config_dir: str | None = None, store: dict | None = None) -> list[dict]:
    """Recent RunRecord dicts, newest first."""
    if store is None:
        store = load_store(config_dir)
    return copy.deepcopy(store.get("recents", []))
