"""STEP 1.2: module grouping + byte/param aggregation (synthetic headers, no I/O)."""

from quantui.model_audit import (
    DTYPE_ITEMSIZE,
    TensorInfo,
    collect_tensors,
    module_of,
    summarize_modules,
)


def test_module_of():
    assert module_of("a.b.c") == "a"
    assert module_of("backbone_model.layers.0.self_attn.q_proj.weight") == "backbone_model"
    assert module_of("solo") == "(root)"
    assert module_of("") == "(root)"


def test_collect_tensors_skips_metadata_and_sorts():
    header = {
        "__metadata__": {"format": "pt"},
        # Deliberately unsorted insertion order.
        "b.norm.weight": {"dtype": "BF16", "shape": [2048], "data_offsets": [0, 4096]},
        "a.q_proj.weight": {"dtype": "BF16", "shape": [2048, 2048], "data_offsets": [4096, 8392704]},
        "a.q_proj.bias": {"dtype": "BF16", "shape": [2048], "data_offsets": [8392704, 8396800]},
    }
    infos = collect_tensors(header)
    assert [i.name for i in infos] == [
        "a.q_proj.bias",
        "a.q_proj.weight",
        "b.norm.weight",
    ]
    by_name = {i.name: i for i in infos}
    assert by_name["a.q_proj.weight"].category == "linear"
    assert by_name["a.q_proj.weight"].module == "a"
    assert by_name["a.q_proj.weight"].nbytes == 2048 * 2048 * DTYPE_ITEMSIZE["BF16"]
    assert by_name["a.q_proj.weight"].shape == (2048, 2048)
    assert by_name["a.q_proj.bias"].category == "bias"
    assert by_name["b.norm.weight"].category == "vector"
    assert by_name["b.norm.weight"].nbytes == 2048 * DTYPE_ITEMSIZE["BF16"]


def test_collect_tensors_empty_header():
    assert collect_tensors({}) == []
    assert collect_tensors({"__metadata__": {"format": "pt"}}) == []


def test_summarize_modules_aggregates():
    header = {
        # Module "big": one 2D linear (4096*4096*2 bytes) + one bias.
        "big.q_proj.weight": {"dtype": "BF16", "shape": [4096, 4096], "data_offsets": [0, 0]},
        "big.q_proj.bias": {"dtype": "BF16", "shape": [4096], "data_offsets": [0, 0]},
        # Module "small": one 1D vector.
        "small.norm.weight": {"dtype": "F32", "shape": [512], "data_offsets": [0, 0]},
    }
    infos = collect_tensors(header)
    summaries = summarize_modules(infos)
    assert [s.module for s in summaries] == ["big", "small"]  # bytes desc

    big = summaries[0]
    assert big.tensors == 2
    assert big.params == 4096 * 4096 + 4096
    assert big.nbytes == 4096 * 4096 * 2 + 4096 * 2
    assert big.category_counts == {"linear": 1, "bias": 1}

    small = summaries[1]
    assert small.tensors == 1
    assert small.params == 512
    assert small.nbytes == 512 * 4
    assert small.category_counts == {"vector": 1}


def test_summarize_modules_tiebreak_by_name():
    # Equal byte totals -> alphabetical module order (determinism).
    header = {
        "zeta.w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 0]},
        "alpha.w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 0]},
    }
    summaries = summarize_modules(collect_tensors(header))
    assert [s.module for s in summaries] == ["alpha", "zeta"]


def test_tensor_info_is_frozen_dataclass():
    info = TensorInfo(
        name="a.weight",
        dtype="BF16",
        shape=(2, 2),
        category="linear_review",
        module="a",
        nbytes=8,
    )
    assert info == TensorInfo(
        name="a.weight",
        dtype="BF16",
        shape=(2, 2),
        category="linear_review",
        module="a",
        nbytes=8,
    )
