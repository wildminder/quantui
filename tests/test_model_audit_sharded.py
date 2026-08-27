"""Sharded model folder support for the audit engine (tests first).

HuggingFace sharded models are a folder with ``model.safetensors.index.json``
plus one or more shard ``.safetensors`` files. :func:`audit_sharded_folder`
merges every shard's header into one :class:`AuditReport`; :func:`audit`
dispatches between single files and sharded folders.

Pure stdlib fixtures: the shard files are built with
``quantui.comfy_quant_schema.write_safetensors`` (no torch / safetensors lib).
"""

import json
import struct

import pytest

from quantui import comfy_quant_schema
from quantui.comfy_quant_schema import (
    FORMAT_INT8_TENSORWISE,
    default_quant_config,
    serialize_comfy_quant_layer,
    write_safetensors,
)
from quantui.model_audit import (
    AuditError,
    audit,
    audit_file,
    audit_sharded_folder,
)

INDEX_NAME = "model.safetensors.index.json"


def _f32_bytes(n: int) -> bytes:
    return b"".join(struct.pack("<f", 0.0) for _ in range(n))


def _write_shard(folder, name: str, specs: dict, metadata: dict | None = None) -> None:
    (folder / name).write_bytes(write_safetensors(specs, metadata=metadata))


def _write_index(folder, weight_map: dict, metadata: dict | None = None) -> None:
    index: dict = {"weight_map": weight_map}
    if metadata is not None:
        index["metadata"] = metadata
    (folder / INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")


def _plain_specs_a() -> dict:
    return {
        "backbone_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 4], b"\x00" * 32),
        "backbone_model.norm.weight": ("BF16", [4], b"\x00" * 8),
    }


def _plain_specs_b() -> dict:
    return {
        "text_encoder.embed_tokens.weight": ("BF16", [10, 4], b"\x00" * 80),
        "lm_head.weight": ("BF16", [10, 4], b"\x00" * 80),
    }


def _make_plain_sharded(folder) -> None:
    """Two plain shards + an index mapping every tensor to its shard."""
    specs_a, specs_b = _plain_specs_a(), _plain_specs_b()
    _write_shard(folder, "model-00001-of-00002.safetensors", specs_a)
    _write_shard(folder, "model-00002-of-00002.safetensors", specs_b)
    weight_map = {
        name: "model-00001-of-00002.safetensors" for name in specs_a
    }
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in specs_b})
    _write_index(folder, weight_map)


def test_sharded_folder_merges_tensors(tmp_path):
    """The merged report covers every tensor of every shard exactly once."""
    _make_plain_sharded(tmp_path)

    report = audit_sharded_folder(str(tmp_path))

    assert report.path == str(tmp_path)
    assert len(report.tensors) == 4
    names = {info.name for info in report.tensors}
    assert names == set(_plain_specs_a()) | set(_plain_specs_b())
    assert report.category_counts == {
        "linear": 1,  # q_proj.weight
        "vector": 1,  # norm.weight
        "embedding": 1,  # embed_tokens.weight
        "head": 1,  # lm_head.weight
    }
    assert report.total_bytes == 32 + 8 + 80 + 80
    assert report.quantized_layers == []
    assert report.metadata == {}


def test_sharded_folder_quantized_layers(tmp_path):
    """A shard carrying a .comfy_quant blob lands in quantized_layers + histogram."""
    quant_specs = serialize_comfy_quant_layer(
        "model.l0.weight",
        {
            "weight": ("U8", [4], b"\x01" * 4),
            "weight_scale": ("F32", [1], _f32_bytes(1)),
        },
        default_quant_config(FORMAT_INT8_TENSORWISE),
    )
    plain_specs = _plain_specs_b()
    _write_shard(tmp_path, "model-00001-of-00002.safetensors", quant_specs)
    _write_shard(tmp_path, "model-00002-of-00002.safetensors", plain_specs)
    weight_map = {name: "model-00001-of-00002.safetensors" for name in quant_specs}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in plain_specs})
    _write_index(tmp_path, weight_map)

    report = audit_sharded_folder(str(tmp_path))

    assert report.quantized_layers == [
        ("model.l0.weight", {"format": FORMAT_INT8_TENSORWISE})
    ]
    assert report.quant_format_histogram == {FORMAT_INT8_TENSORWISE: 1}
    # The quant descriptor tensors are merged into the tensor list too.
    names = {info.name for info in report.tensors}
    assert "model.l0.weight.comfy_quant" in names
    assert report.category_counts.get("quant_meta") == 1


def test_sharded_folder_metadata_precedence(tmp_path):
    """Index "metadata" wins; otherwise the first shard's __metadata__ is used."""
    specs_a, specs_b = _plain_specs_a(), _plain_specs_b()

    # Case 1: index metadata wins over shard metadata.
    folder1 = tmp_path / "with_index_meta"
    folder1.mkdir()
    _write_shard(folder1, "s1.safetensors", specs_a, metadata={"source": "shard"})
    _write_shard(folder1, "s2.safetensors", specs_b)
    _write_index(
        folder1,
        {**{n: "s1.safetensors" for n in specs_a}, **{n: "s2.safetensors" for n in specs_b}},
        metadata={"source": "index"},
    )
    assert audit_sharded_folder(str(folder1)).metadata == {"source": "index"}

    # Case 2: no index metadata -> first shard (sorted order) with __metadata__.
    folder2 = tmp_path / "shard_meta_only"
    folder2.mkdir()
    _write_shard(folder2, "s1.safetensors", specs_a)
    _write_shard(folder2, "s2.safetensors", specs_b, metadata={"source": "shard"})
    _write_index(
        folder2,
        {**{n: "s1.safetensors" for n in specs_a}, **{n: "s2.safetensors" for n in specs_b}},
    )
    assert audit_sharded_folder(str(folder2)).metadata == {"source": "shard"}


def test_sharded_folder_dedupes_tensor_names(tmp_path):
    """A tensor present in two shards is kept once (first in sorted-shard order)."""
    dup = ("BF16", [2, 2], b"\x00" * 8)
    specs_a = {"shared.q_proj.weight": dup, "only_a.norm.weight": ("BF16", [2], b"\x00" * 4)}
    specs_b = {"shared.q_proj.weight": dup, "only_b.norm.weight": ("BF16", [2], b"\x00" * 4)}
    _write_shard(tmp_path, "s1.safetensors", specs_a)
    _write_shard(tmp_path, "s2.safetensors", specs_b)
    _write_index(
        tmp_path,
        {
            "shared.q_proj.weight": "s1.safetensors",
            "only_a.norm.weight": "s1.safetensors",
            "only_b.norm.weight": "s2.safetensors",
        },
    )

    report = audit_sharded_folder(str(tmp_path))
    names = [info.name for info in report.tensors]
    assert names.count("shared.q_proj.weight") == 1
    assert len(report.tensors) == 3


def test_audit_dispatcher_file(tmp_path):
    """audit(single_file) is field-identical to audit_file(single_file)."""
    fixture = tmp_path / "single.safetensors"
    fixture.write_bytes(write_safetensors(_plain_specs_a()))

    assert audit(str(fixture)) == audit_file(str(fixture))


def test_audit_dispatcher_sharded(tmp_path):
    """audit(sharded_folder) returns the merged sharded report."""
    _make_plain_sharded(tmp_path)

    report = audit(str(tmp_path))
    assert report.path == str(tmp_path)
    assert len(report.tensors) == 4
    assert report == audit_sharded_folder(str(tmp_path))


def test_audit_dispatcher_error(tmp_path):
    """audit() rejects nonexistent paths and folders without an index."""
    with pytest.raises(AuditError):
        audit(str(tmp_path / "nope.safetensors"))

    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    _write_shard(bare_dir, "model.safetensors", _plain_specs_a())
    with pytest.raises(AuditError):
        audit(str(bare_dir))


def test_sharded_missing_shard_file(tmp_path):
    """An index referencing a nonexistent shard raises AuditError naming it."""
    _write_shard(tmp_path, "s1.safetensors", _plain_specs_a())
    _write_index(
        tmp_path,
        {
            **{n: "s1.safetensors" for n in _plain_specs_a()},
            "ghost.weight": "ghost-shard.safetensors",
        },
    )
    with pytest.raises(AuditError, match="ghost-shard.safetensors"):
        audit_sharded_folder(str(tmp_path))


def test_sharded_missing_index(tmp_path):
    """A folder without model.safetensors.index.json raises AuditError."""
    _write_shard(tmp_path, "s1.safetensors", _plain_specs_a())
    with pytest.raises(AuditError, match=INDEX_NAME.replace(".", r"\.")):
        audit_sharded_folder(str(tmp_path))


def test_sharded_malformed_index_json(tmp_path):
    """A corrupt index JSON raises AuditError naming the folder."""
    _write_shard(tmp_path, "s1.safetensors", _plain_specs_a())
    (tmp_path / INDEX_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(AuditError):
        audit_sharded_folder(str(tmp_path))


def test_sharded_malformed_shard_header(tmp_path):
    """A shard with a corrupt safetensors header raises AuditError."""
    _write_shard(tmp_path, "s1.safetensors", _plain_specs_a())
    (tmp_path / "bad.safetensors").write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00xx")
    _write_index(tmp_path, {"a.weight": "s1.safetensors", "b.weight": "bad.safetensors"})
    with pytest.raises(AuditError):
        audit_sharded_folder(str(tmp_path))


def test_sharded_determinism(tmp_path):
    """Two audits of the same folder produce identical reports."""
    _make_plain_sharded(tmp_path)
    first = audit_sharded_folder(str(tmp_path))
    second = audit_sharded_folder(str(tmp_path))
    assert first == second
    # And the comfy_quant reader agrees shard-by-shard (sanity on the merge input).
    shard = str(tmp_path / "model-00001-of-00002.safetensors")
    assert comfy_quant_schema.read_comfy_quant_configs(shard) == []
