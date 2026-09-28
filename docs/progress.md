# Progress log

## Phase 0: Setup

- Initialized git repo (empty directory, no prior repo).
- Environment note: system default `python3` is 3.14.6 (Homebrew), too new for
  reliable PyTorch wheel availability at the time of writing. Used
  `python3.11` (also available via Homebrew, `python@3.11` 3.11.15) to create
  `.venv` instead.
- Created venv with `python3.11 -m venv .venv`, upgraded pip.
- Installing: mteb, sentence-transformers, pandas, numpy, tqdm, pytest (macOS,
  so no CPU-only torch index needed — skipped per instructions).
- Created folder structure: configs/, src/retrieval/, experiments/, results/,
  demo/, scripts/, tests/, docs/.
- Added `.gitignore` (ignores .venv/, caches, indexes, embeddings; keeps
  results/*.json and metrics_log.csv tracked).
- Wrote `scripts/check_env.py`: prints Python/platform/CPU info, installed
  package versions, checks the three required mteb imports, and loads the
  `AppsRetrieval` task metadata without downloading the full dataset.
- Wrote `src/retrieval/metrics_log.py` with `log_run(...)`, appending rows to
  `results/metrics_log.csv` (creating the header if missing).
- Wrote `tests/test_metrics_log.py` to verify header + two appended rows.
- Added root-level empty `conftest.py` so `tests/` can import the `src`
  package without an `__init__.py` in `tests/`.

Next: run `check_env.py` and `pytest -q`, fix any issues, commit Phase 0.
