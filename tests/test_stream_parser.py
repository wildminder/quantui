"""Pure unit tests for quantui.stream_parser (T01 extraction).

These exercise the Textual-free parser/classifier directly -- no UI, no subprocess.
Mirrors the assertions that previously lived inside test_app_headless.py so the
extraction is provably behavior-preserving.
"""
from quantui import stream_parser as sp


def test_split_frames_carriage_return_is_progress():
    # "a\rb\nc": 'a' is a \r progress frame, 'b' is a \n line, 'c' is carried.
    segs, partial = sp.split_frames("a\rb\nc", "")
    assert segs == [("a", "progress"), ("b", "line")], segs
    assert partial == "c"

    # Multiple \r frames collapse in order; only the final \n ends a line.
    segs, partial = sp.split_frames("f1\rf2\rf3\n", "")
    assert segs == [("f1", "progress"), ("f2", "progress"), ("f3", "line")], segs
    assert partial == ""

    # Carried partial across chunk boundaries.
    segs, partial = sp.split_frames("tail", "")
    assert segs == [] and partial == "tail"
    segs2, partial2 = sp.split_frames("more\ndone\n", partial)
    assert segs2 == [("tailmore", "line"), ("done", "line")], segs2
    assert partial2 == ""


def test_is_progress_line_matches_ctq_processing():
    assert sp.ProgressClassifier.is_progress_line(
        "(1/211) Processing (INT8): model.embed_tokens.weight"
    )
    # tqdm / optimizer forms still match.
    assert sp.ProgressClassifier.is_progress_line("Optimizing INT8 (Prodigy): 0%| | 0/4000 [00:00<?, ?it/s]")
    assert sp.ProgressClassifier.is_progress_line("1234/5678 [00:01<?, ?it/s]")


def test_progress_key_collapses_ctq_counter():
    k1 = sp.ProgressClassifier.progress_key("(1/211) Processing (INT8): a.weight")
    k2 = sp.ProgressClassifier.progress_key("(50/211) Processing (INT8): z.weight")
    k3 = sp.ProgressClassifier.progress_key("(1/50) Processing (INT8): other")
    assert k1 == k2, (k1, k2)  # same total -> same slot
    assert k1 != k3
    assert "(#/211)" in k1


def test_is_progress_detail_matches_indented_dash():
    assert sp.ProgressClassifier.is_progress_detail("    - Tensor shape: [1,2]")
    assert not sp.ProgressClassifier.is_progress_detail("Some normal line")


def test_is_progress_line_rejects_log_lines_with_slash():
    # Real log lines that happen to contain N/M must NOT be treated as progress.
    assert not sp.ProgressClassifier.is_progress_line("Epoch 3/10 loss=0.1")
    assert not sp.ProgressClassifier.is_progress_line("[1/3] Quantizing shard x")


def test_classify_separator():
    # Pure rules -> "separator" (the ctq "----" between a header and its details).
    assert sp.ProgressClassifier.classify("----") == "separator"
    assert sp.ProgressClassifier.classify("===") == "separator"
    assert sp.ProgressClassifier.classify("────────") == "separator"
    # A status line with letters is NOT a separator -> it clears the bar (phase boundary).
    assert sp.ProgressClassifier.classify("=== Quantization finished successfully ===") == "line"
    # Everything else routes as before.
    assert sp.ProgressClassifier.classify("(1/211) Processing (INT8): a.weight") == "progress"
    assert sp.ProgressClassifier.classify("    - Tensor shape: [1,2]") == "detail"
    assert sp.ProgressClassifier.classify("Optimizing INT8: 0%| | 0/4000 [...]") == "progress"
    assert sp.ProgressClassifier.classify("Epoch 3/10 loss=0.1") == "line"
    assert sp.ProgressClassifier.classify("[1/3] Quantizing shard x") == "line"


def test_parse_ctq_progress_envelope():
    # A well-formed envelope parses to its JSON payload dict.
    obj = sp.parse_ctq_progress('CTQ_PROGRESS {"phase":"shard","cur":1,"total":3,"label":"x"}')
    assert obj == {"phase": "shard", "cur": 1, "total": 3, "label": "x"}

    # Leading/trailing whitespace tolerated.
    obj = sp.parse_ctq_progress('  CTQ_PROGRESS {"phase":"p"}  ')
    assert obj == {"phase": "p"}

    # Non-envelopes return None so the regex path still applies.
    assert sp.parse_ctq_progress("(1/211) Processing (INT8): a.weight") is None
    assert sp.parse_ctq_progress("[1/3] Quantizing shard x") is None
    assert sp.parse_ctq_progress("CTQ_PROGRESS not-json") is None
    assert sp.parse_ctq_progress("CTQ_PROGRESS ") is None
    assert sp.parse_ctq_progress("") is None


def test_classify_ctq_progress_envelope_is_progress():
    # Our own structured envelope is unambiguously classified as progress (the root-cause
    # fix: previously no line ever hit the "progress" category, so the bar never appeared).
    assert sp.ProgressClassifier.classify(
        'CTQ_PROGRESS {"phase":"shard","cur":1,"total":3}'
    ) == "progress"
    # A malformed envelope is NOT progress (falls through to the line path).
    assert sp.ProgressClassifier.classify("CTQ_PROGRESS {bad json") == "line"


def test_parse_tqdm_progress_extracts_cur_total_pct():
    # convert_to_quant's own calibration bar -> real (cur, total, pct) parsed out.
    line = "Optimizing INT8 (Prodigy-plateau):  50%|#####| 2000/4000 [00:01<?, ?it/s]"
    d = sp.parse_tqdm_progress(line)
    assert d is not None
    assert d["label"] == "Optimizing INT8 (Prodigy-plateau)"
    assert d["cur"] == 2000 and d["total"] == 4000
    assert d["pct"] == 50.0


def test_parse_tqdm_progress_loading_tensors():
    line = "Loading tensors:  12%|###| 30/250 [00:01<?, ?it/s]"
    d = sp.parse_tqdm_progress(line)
    assert d is not None
    assert d["label"] == "Loading tensors"
    assert d["cur"] == 30 and d["total"] == 250


def test_parse_tqdm_progress_pct_only_bar():
    # A bar with a leading % and no explicit counter still yields a pct.
    d = sp.parse_tqdm_progress("Optimizing:  50%|###### |")
    assert d is not None
    assert d["pct"] == 50.0


def test_parse_tqdm_progress_derives_pct_from_counter():
    # No leading "%" token but a real "cur/total" -> pct derived.
    d = sp.parse_tqdm_progress("Loading:  | 40/4000 [00:00<?, ?it/s]")
    assert d is not None
    assert d["cur"] == 40 and d["total"] == 4000
    assert d["pct"] == 1.0


def test_parse_tqdm_progress_rejects_ctq_header_and_epoch():
    # ctq "(N/M) Processing" headers and "Epoch 3/10" log lines are NOT tqdm bars.
    assert sp.parse_tqdm_progress("(1/211) Processing (INT8): a.weight") is None
    assert sp.parse_tqdm_progress("Epoch 3/10 loss=0.1") is None


def test_parse_tqdm_progress_rejects_unknown_total():
    # Unknown-total bars ("?/?") carry no usable signal -> None (legacy text path).
    assert sp.parse_tqdm_progress("Optimizing:  ?%|###| ?/? [?it/s]") is None


def test_ctq_progress_prefix_is_pinned():
    # Load-bearing: worker_ctq.py / worker_ctq_kitchen.py emit exactly this
    # prefix and stream_parser classifies lines by it, so changing the value
    # (or dropping the trailing space) silently disables structured progress.
    # Pinned so the NTH-008 import reordering cannot regress it.
    assert sp.CTQ_PROGRESS_PREFIX == "CTQ_PROGRESS "
