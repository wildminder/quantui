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

# T10 (plan 2026-08-31-gguf-unsloth-parity): method ids that were removed
# from the registry and therefore from unsloth's official surface. Profiles
# saved before the rebuild may reference them; load must downgrade instead
# of crashing or silently running a dead id. q4_nl never existed (the real
# id is the imatrix-gated iq4_nl); the *_k_xl trio were proprietary UD mixes.
REMOVED_METHOD_IDS = frozenset({"q4_k_xl", "q3_k_xl", "q2_k_xl", "q4_nl"})
# The fallback for a fully-dead method / custom list.
DOWNGRADE_METHOD = "q4_k_m"


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
def _downgrade_method_list(raw: str, *, keep_empty: bool = False) -> str:
    """Filter dead ids out of a comma-separated method list.

    Valid ids are kept in order. A list left empty degrades to the single
    default UNLESS ``keep_empty`` (the custom field: an all-dead override is
    simply removed, deferring to ``method``). Only ids known-DEAD are removed
    -- unknown-but-not-dead ids (e.g. ids from a newer unsloth release) pass
    through so the store never erases forward-compatible values.
    """
    if not raw:
        return raw
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    kept = [p for p in parts if p not in REMOVED_METHOD_IDS]
    if not kept:
        return "" if keep_empty or not parts else DOWNGRADE_METHOD
    return ", ".join(kept)


def _downgrade_profile_fields(fields: dict) -> dict:
    """T10 downgrade pass over one profile's fields (in place, returns it).

    - ``method`` / ``custom``: dead ids dropped (comma lists filtered).
    - Records the downgrade in ``_notes`` so it is visible, not silent.
    - Never raises on odd shapes: a profile is user data on disk, so a
      non-string method/custom is left untouched (validated downstream).
    """
    notes = []
    for key in ("method", "custom"):
        val = fields.get(key)
        if not isinstance(val, str) or not val:
            continue
        if not any(p.strip() in REMOVED_METHOD_IDS for p in val.split(",")):
            continue
        fields[key] = _downgrade_method_list(val, keep_empty=(key == "custom"))
        if key == "custom" and not fields[key]:
            # The custom override's every id was dead: remove the override so
            # the run defers to the (already-downgraded) method field.
            notes.append(f"downgraded custom: dead ids removed ({val})")
        else:
            notes.append(f"downgraded {key}: '{val}' -> '{fields[key]}'")
    if notes:
        prev = fields.get("_notes", "")
        fields["_notes"] = (prev + "; " if prev else "") + "; ".join(notes)
    return fields


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
    """Return a deep copy of the named profile, or None when absent.

    T10: the returned fields pass through the downgrade pass -- a profile
    saved before the registry rebuild (dead method ids) loads with the dead
    ids replaced and a visible ``_notes`` entry, never a crash. The downgrade
    is applied to the COPY; the on-disk store keeps the original until the
    profile is re-saved (no implicit disk writes on read).
    """
    if store is None:
        store = load_store(config_dir)
    prof = store.get("profiles", {}).get(name)
    if prof is None:
        return None
    return _downgrade_profile_fields(copy.deepcopy(prof))


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
