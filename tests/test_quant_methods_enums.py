"""Contract tests for the Family/Backend enums (NTH-008, UP042 -> StrEnum).

`Family` and `Backend` were ``class X(str, Enum)``. On Python 3.11+ that mixin
formats as ``"Family.GGUF"`` (``__str__``/``__format__`` include the class
name), whereas ``enum.StrEnum`` formats as the value ``"gguf"``. The app only
ever compares these members or reads ``.value``, so the switch is safe -- but
the formatting difference is exactly the kind of thing that silently corrupts
an argv or a log line later, so it is pinned here.

The three ``str``/``format`` tests below are the behavioural change; the rest
are regression pins that must hold before and after.
"""

import json

from quantui.quant_methods import Backend, Family

# --- the behaviour change (StrEnum formats as its value) ------------------- #


def test_family_str_is_its_value():
    assert str(Family.GGUF) == "gguf"
    assert str(Family.COMFY) == "comfy"


def test_family_fstring_is_its_value():
    # f-strings use __format__, which for a (str, Enum) mixin on 3.11+ yields
    # "Family.GGUF"; every log/status line depends on getting "gguf" here.
    assert f"{Family.GGUF}" == "gguf"
    assert f"{Family.COMFY}" == "comfy"


def test_backend_str_is_its_value():
    assert str(Backend.UNSLOTH) == "unsloth"
    assert str(Backend.CTQ) == "convert_to_quant"
    assert str(Backend.COMFY_KITCHEN) == "comfy_kitchen"


# --- regression pins (unchanged by the switch) ----------------------------- #


def test_family_compares_equal_to_plain_string():
    assert Family.GGUF == "gguf"
    assert Family.COMFY == "comfy"
    assert Backend.COMFY_KITCHEN == "comfy_kitchen"


def test_family_lookup_by_value():
    assert Family("gguf") is Family.GGUF
    assert Family("comfy") is Family.COMFY
    assert Backend("comfy_kitchen") is Backend.COMFY_KITCHEN


def test_family_value_attribute():
    assert Family.GGUF.value == "gguf"
    assert Family.COMFY.value == "comfy"


def test_member_order_is_preserved():
    # The data-driven UI iterates these members; order is load-bearing.
    assert [m.value for m in Family] == ["gguf", "comfy"]
    assert [m.value for m in Backend] == [
        "unsloth", "convert_to_quant", "comfy_kitchen", "native",
    ]


def test_json_serializable_as_value():
    # Profiles/recents persist the family as a plain string; json must emit the
    # value, never "Family.GGUF".
    assert json.dumps({"family": Family.GGUF}) == '{"family": "gguf"}'
    assert json.dumps({"backend": Backend.CTQ}) == '{"backend": "convert_to_quant"}'


def test_string_concatenation_uses_value():
    assert Family.GGUF + "-run" == "gguf-run"
