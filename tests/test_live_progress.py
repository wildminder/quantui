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


def test_tqdm_line_becomes_determinate_quantize_state():
    # A third-party tqdm bar is parsed into a DETERMINATE "quantize" state (real cur/total)
    # instead of frozen raw text -- this is what makes the bar advance (Task #25).
    store = LiveProgressStore()
    store.update("Optimizing INT8 (Prodigy-plateau):  50%|#####| 2000/4000 [00:01<?, ?it/s]")
    states = store.states()
    assert len(states) == 1
    st = states[0]
    assert st.phase == "quantize"
    assert st.determinate is True
    assert st.cur == 2000 and st.total == 4000
    assert st.text == "Optimizing INT8 (Prodigy-plateau) [2000/4000]"

    # A ctq "(N/M) Processing" header stays on its own legacy-text slot (distinct key),
    # so the two never collide.
    store.update("(1/211) Processing (INT8): a.weight")
    assert len(store) == 2, store.snapshot()
