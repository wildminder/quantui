"""Unit tests for ``quantui.live_progress.LiveProgressStore`` (T04).

Pure (no UI): the store collapses consecutive progress frames into one slot keyed by a
stable bar key, clears on demand, and renders a constant-height string.
"""

from quantui.live_progress import LiveProgressStore


def test_collapse_and_clear():
    store = LiveProgressStore()
    store.update("(1/211) Processing (INT8): a.weight")
    store.update("(2/211) Processing (INT8): b.weight")
    # Collapsed to a single slot (latest step overwrites the prior one).
    assert len(store) == 1, store.snapshot()
    value = next(iter(store.snapshot().values()))
    assert "(2/211)" in value, value
    assert "b.weight" in value, value
    assert "(1/211)" not in value, value  # the earlier step was overwritten

    # clear() empties the store and render() returns the constant-height placeholder.
    store.clear()
    assert store.snapshot() == {}
    assert store.render() == " "

    # snapshot() returns a COPY (mutating it must not affect the store).
    store.update("(9/9) Processing (INT8): z.weight")
    snap = store.snapshot()
    snap["injected"] = "x"
    assert "injected" not in store.snapshot()


def test_progress_key_total_preserved():
    store = LiveProgressStore()
    store.update("(1/211) Processing (INT8): a.weight")
    store.update("(1/50) Processing (INT8): a.weight")
    # Distinct totals keep DISTINCT keys -> both slots survive (not collapsed together).
    assert len(store) == 2, store.snapshot()
    keys = set(store.snapshot().keys())
    assert "(#/211)" in keys, keys
    assert "(#/50)" in keys, keys


def test_render_joins_multiple_slots():
    store = LiveProgressStore()
    store.update("(1/2) Job A", key="A")
    store.update("(1/2) Job B", key="B")
    rendered = store.render()
    assert "(1/2) Job A" in rendered and "(1/2) Job B" in rendered
    assert "\n" in rendered


def test_ctq_progress_envelope_structured_state():
    # A CTQ_PROGRESS envelope is stored as a DETERMINATE ProgressState keyed by phase.
    store = LiveProgressStore()
    store.update('CTQ_PROGRESS {"phase":"shard","cur":1,"total":3,"label":"Quantizing x"}')
    store.update('CTQ_PROGRESS {"phase":"shard","cur":2,"total":3,"label":"Quantizing y"}')

    # Collapsed to ONE slot (same phase), progress advanced to cur=2/3.
    assert len(store) == 1, store.snapshot()
    states = store.states()
    assert len(states) == 1
    st = states[0]
    assert st.phase == "shard"
    assert st.cur == 2 and st.total == 3
    assert st.determinate is True
    # snapshot() still returns the human-readable text (back-compat with existing tests).
    assert next(iter(store.snapshot().values())) == "Quantizing y [2/3]"

    # A percentage-only envelope is also determinate.
    store.update('CTQ_PROGRESS {"phase":"q","pct":42.5,"label":"Calibrating"}')
    q = store.states()[-1]
    assert q.determinate is True
    assert q.text == "Calibrating [42%]"

    # render() of an empty store is still the constant-height placeholder.
    empty = LiveProgressStore()
    assert empty.render() == " "


def test_tqdm_line_becomes_determinate_calibrate_state():
    # A third-party tqdm bar is parsed into a DETERMINATE state (real cur/total)
    # instead of frozen raw text -- this is what makes the bar advance (Task #25).
    # It gets its own "calibrate" slot; see the regression test below for why.
    store = LiveProgressStore()
    store.update("Optimizing INT8 (Prodigy-plateau):  50%|#####| 2000/4000 [00:01<?, ?it/s]")
    states = store.states()
    assert len(states) == 1
    st = states[0]
    assert st.phase == "calibrate"
    assert st.determinate is True
    assert st.cur == 2000 and st.total == 4000
    assert st.text == "Optimizing INT8 (Prodigy-plateau) [2000/4000]"

    # A ctq "(N/M) Processing" header stays on its own legacy-text slot (distinct key),
    # so the two never collide.
    store.update("(1/211) Processing (INT8): a.weight")
    assert len(store) == 2, store.snapshot()


def test_calibration_bar_does_not_overwrite_overall_quantize_progress():
    # Regression: convert_to_quant's calibration tqdm is a SUB-STEP. While it shared
    # the "quantize" slot with the worker's output-file-size poll, finishing
    # calibration pinned the bar at 100% and the next real update knocked it back
    # to a low percentage -- fp8 showed 100% at the start while quantization ran on.
    store = LiveProgressStore()
    store.update("Optimizing INT8 (Prodigy-plateau): 100%|##########| 4000/4000 [00:02<?, ?it/s]")
    store.update('CTQ_PROGRESS {"phase": "quantize", "pct": 8.0, "label": "Writing quantized model"}')

    by_phase = {s.phase: s for s in store.states()}
    assert by_phase["calibrate"].pct == 100.0, "calibration keeps its own reading"
    assert by_phase["quantize"].pct == 8.0, "overall progress must not be reset by calibration"
