# Repository Guidelines

## Project Structure & Module Organization
Primary code lives in `src/translator/`. Keep new runtime modules there, grouped by responsibility: CLI in `cli.py`, orchestration in `pipeline.py`, and focused helpers such as `chunking.py`, `preservation.py`, and `markdown_*`. The repo-root `translator/` package is only a thin compatibility layer for `python -m translator` and imports from the repo root; do not add runtime logic outside `src/translator/`. Tests live in `tests/`, with reusable inputs under `tests/fixtures/`. Sample Markdown inputs for manual runs live in `documents/`.

## Build, Test, and Development Commands
Create an environment and install dependencies with `pip install -r requirements.txt`. Use `python -m translator --help` to inspect the CLI. Common local runs:

```bash
python -m translator translate --in documents/6.031_note1.md --out output/note.zh.md
python -m translator translate --url https://example.com --out output/page.zh.md
python -m translator translate --url https://a.com --url https://b.com --out-dir output/batch
python -m translator lint-md --in output/page.zh.md
pytest -q
```

`translate` is the preferred one-command entrypoint for single files and URL batches, `lint-md` checks Markdown guardrails, and `pytest -q` runs the automated test suite.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and explicit type hints where the code already uses them. Prefer small, single-purpose modules over large utility files. Match the current import style and keep CLI flag names descriptive, for example `--prompt-glossary-mode`. No formatter or linter is enforced in the repo today, so keep changes PEP 8 aligned and consistent with neighboring files.

## Testing Guidelines
Add or update `pytest` tests for every behavior change. Name test files `test_<feature>.py` and test functions `test_<scenario>()`. Place sample inputs in `tests/fixtures/` instead of inline blobs when they are reused. Integration-style tests that depend on `DEEPSEEK_API_KEY` should skip cleanly when the key is absent; keep default test runs offline-safe.

## Commit & Pull Request Guidelines
Recent history mostly uses short imperative subjects, often in Conventional Commit form such as `feat:` or `feat(diagrams):`. Prefer that format and keep the first line under roughly 72 characters. Pull requests should describe the user-visible change, list validation steps such as `pytest -q`, and include before/after output snippets or screenshots when CLI output or rendered Markdown behavior changes.

## Security & Configuration Tips
Store secrets in `.env`, not in code or fixtures. Expected variables include `DEEPSEEK_API_KEY`, optional `JINA_API_KEY`, `DEEPSEEK_MODEL`, and `DEEPSEEK_BASE_URL`; legacy `MOONSHOT_*` variables remain fallback-only. Do not commit generated caches such as `.pytest_cache/`, `.ruff_cache/`, or API-derived output files unless they are intentional fixtures.
