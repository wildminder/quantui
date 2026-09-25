## What changed

<!-- What does this PR change, and why? Keep unrelated refactors out. -->

## How it was tested

<!--
Commands run locally, in this order (see CONTRIBUTING.md for the full gate
descriptions). The same three gates run on every pull request in CI.
-->

- [ ] `python -m ruff check quantui/ tests/` — passes with zero findings
- [ ] `python -m mypy quantui/quant_methods.py quantui/stream_parser.py quantui/live_progress.py quantui/run_config.py quantui/profiles_store.py` — passes
- [ ] `python -m pytest tests/ -q --ignore=tests/test_incremental_safetensors.py --ignore=tests/test_stream_quant.py` — passes (headless suite; the two ignored suites need torch and are run separately with `scripts/gate_torch.sh`)
- [ ] Manual/functional verification (describe what you ran in the TUI and what you observed)

## Notes for the reviewer

<!--
- Which files carry the real logic, and what is deliberately left out.
- Anything that only shows up in a separate worker environment
  (Unsloth GGUF or ComfyUI `convert_to_quant`), which CI cannot exercise.
- For user-visible changes: is there a `CHANGELOG.md` entry?
-->

## Hygiene confirmation

- [ ] No model weights, generated files, local absolute paths, or credentials are included.
- [ ] No local absolute paths (for example development or model directories) appear in the diff, logs, or new files.
- [ ] Tests that depend on the gitignored `docs/` tree are unchanged or remain skippable on a fresh clone.
