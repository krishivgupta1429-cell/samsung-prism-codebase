# Code Retrieval Hackathon

This project implements a code retrieval and ranking system for the Samsung PRISM
"Agentic Code Intelligence" problem statement. Given a natural-language query
describing a programming problem, the system retrieves and ranks code snippets
from a corpus by relevance. Generating explanations or answers is out of scope;
only retrieval and ranking matter. The system is scored on the CoIR APPS dataset
via the MTEB `AppsRetrieval` task (NDCG@10, MRR) and is designed to run on CPU
with minimal GPU use.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install mteb sentence-transformers pandas numpy tqdm pytest
.venv/bin/pip freeze > requirements.txt
```

On Linux or a GPU-less Windows machine, install the CPU-only PyTorch build
first to keep the download smaller (skip this on macOS):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running checks

```bash
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest -q
```

## Folder map

- `configs/` — JSON configs for encoder models (model name, prefixes, batch size, etc.)
- `src/retrieval/` — encoder wrapper, metrics logger, and other retrieval pipeline code
- `experiments/` — scripts that run an evaluation end-to-end and log results
- `results/` — MTEB output JSON files per model, plus `metrics_log.csv` tracking every run
- `demo/` — demo application (later phase)
- `scripts/` — one-off utility scripts (env check, timing, etc.)
- `tests/` — pytest test suite
- `docs/` — running progress log and other notes
