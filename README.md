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

We closed the loop on this: measured single-query CPU latency directly (20
real queries, one at a time) and it was **not** fast — avg 4,382 ms/query.
Corpus-encoding being a one-time cost doesn't help if a single live query
also takes 4+ seconds on CPU. See "Serving: quantized model" below for how
this is actually handled.

## Serving: quantized model vs. official benchmark score

This project uses **two versions of the same model** for two different
purposes:

- **Official benchmark score — full precision:** NDCG@10 **0.84083**, MRR@10
  **0.81055**, measured with `experiments/run_eval.py` on the full
  8,765-document APPS test corpus (`results/jina_code_embeddings_0_5b/`).
  This is the number reported for the hackathon submission.
- **Live query-time serving — INT8 ONNX quantized:** single-query CPU
  latency dropped from **avg 4,382 ms** (full precision) to **avg 450 ms**
  (~9.7x faster; min 113 ms, max 1,069 ms) via dynamic INT8 quantization
  (`experiments/export_quantized_onnx.py`, ONNX Runtime, `arm64` preset). A
  scoped sanity check (300 sample queries against a 500-document pool, not
  the full corpus) showed NDCG@10 0.87636 with no sign of accuracy collapse
  — not directly comparable to the 0.84083 full-corpus score since a 500-doc
  pool is an easier task, but sufficient to confirm quantization didn't break
  retrieval quality.

The quantized model binary (497MB) is not committed to git (GitHub rejects
files over 100MB without Git LFS); `experiments/export_quantized_onnx.py`
regenerates it exactly, in a separate isolated venv (`sentence-transformers[onnx]`
+ `optimum`/`onnxruntime`, kept out of the main `.venv`). See
`docs/progress.md` for the full writeup, including a real ONNX Runtime
warm-up-cost characteristic worth knowing about before deploying it (a
production service needs one warm-up inference at startup).

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
