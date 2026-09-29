# Code Retrieval Hackathon

This project implements a code retrieval and ranking system for the Samsung PRISM
"Agentic Code Intelligence" problem statement. Given a natural-language query
describing a programming problem, the system retrieves and ranks code snippets
from a corpus by relevance. Generating explanations or answers is out of scope;
only retrieval and ranking matter. The system is scored on the CoIR APPS dataset
via the MTEB `AppsRetrieval` task (NDCG@10, MRR) and is designed to run on CPU
with minimal GPU use.

## Current model (Phase 1 winner)

**`jinaai/jina-code-embeddings-0.5b`** — NDCG@10 **0.84083**, MRR@10 **0.81055**
on the full APPS test split (`results/jina_code_embeddings_0_5b/`), vs. 0.13993
for the next-best candidate and 0.06596 for the `all-MiniLM-L6-v2` baseline.
Selected for its code-specific pretraining, which produces a dramatically
better score than any general-purpose text encoder tried in Phase 1
(see `docs/progress.md` and `results/model_comparison.md` for the full
comparison).

This model is slow to encode on CPU alone — full-corpus timing checks
projected roughly 13.5 hours on CPU, and it was actually evaluated on a CUDA
GPU. This does not conflict with the problem statement's "CPU-friendly,
minimal GPU" requirement, because retrieval systems separate two distinct
costs, and MTEB's own retrieval evaluator (`RetrievalEvaluator.__call__`) is
built around exactly this split:

- **Corpus encoding is a one-time (or occasional, on corpus change) offline
  indexing step.** Once built, the corpus index can be reused for every
  future query until the underlying code changes — this is also this
  project's own P1 goal ("fast index rebuilds when code changes"), and the
  reason `src/retrieval/encoder.py` has an on-disk embedding cache in the
  first place.
- **Query encoding happens per request and is what must stay fast on CPU**
  for the system to be usable.

Caveat: we have verified this distinction architecturally (MTEB times
"Encoding corpus" and "Encoding queries" as separate phases) and via the
dramatic NDCG@10 gain, but we have **not yet directly measured
jina-code-embeddings-0.5b's single-query CPU encoding latency in isolation**
— every CPU timing check run so far (`scripts/time_encode.py`) sampled
documents from the corpus, not queries. That's a recommended follow-up
before treating full CPU-side compliance as proven.

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
