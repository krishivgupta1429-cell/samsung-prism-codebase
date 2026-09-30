# Code Retrieval Hackathon

This repository evaluates natural-language-to-code retrieval for Samsung PRISM's
Agentic Code Intelligence task, using MTEB `AppsRetrieval`. It contains a shared
PyTorch/ONNX encoder, experiment runners, exact cosine ranking diagnostics, and
profiling tools. A deployed service, repository ingestion, and reranking are future
work. Answer generation is out of scope.

The recorded Phase 1 winner is **jinaai/jina-code-embeddings-0.5b**, with **NDCG@10
0.84083 / MRR@10 0.81055** on 3,765 test queries against 8,765 documents. SFR Code
400M is second at 0.49627 / 0.44957. Historical JSON files and `metrics_log.csv`
remain unchanged. Model architecture, training, prompts, tokenization, and context
limits differ; the comparison does not isolate the effect of code pretraining.

Historical notes report about 4,382 ms per CPU query for full precision and 450 ms
for ARM64 INT8 ONNX. Raw artifacts for those latency runs and the 500-document
quantization sanity check were not committed. Those measurements do not establish
current hardware performance, full-corpus INT8 accuracy, or end-to-end serving
latency. The new commands below produce reproducible artifacts for those checks.
The earlier 34.5-minute first-batch delay has not been causally diagnosed; one
warm-up query cannot guarantee that all future input shapes are warm.

## Install and check

Use Python 3.11. On Linux CPU machines, install CPU PyTorch **before** the remaining
requirements. On macOS omit that CPU-index command. For CUDA, install the matching
PyTorch build from the appropriate PyTorch index instead.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest -q
RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -q -m integration
```

The default tests use local model doubles and never download weights. To run only
those tests without the ML stack, install `requirements-dev.txt` in a separate
venv. The explicitly enabled integration test downloads MiniLM if uncached.

SFR requires a **separate** environment using `requirements-sfr.txt`. ONNX export
and inference require another environment using `requirements-onnx.txt`. The
ONNX dependency currently requires Transformers <4.58, which conflicts with the
main environment's Sentence Transformers 6.1 requirement of Transformers >=5.
The separate ONNX requirements select compatible versions. These auxiliary files
pin direct dependencies; each new run also records every installed package version.
Do not overwrite the main `requirements.txt` during setup.

## Run a fresh evaluation

```bash
.venv/bin/python experiments/run_eval.py \
  --config configs/jina_code_embeddings_0_5b.json \
  --experiment-id jina-baseline
```

Each invocation creates a unique directory under `results/runs/`, even if the
experiment ID is reused. It contains `run.json` (requested/resolved config,
revision, source fingerprint, installed versions, device and status), MTEB results,
per-query predictions, a summary, and logs. Failed runs retain their traceback.
A convenience CSV is written to `results/runs/metrics.csv`. The per-run artifacts
are authoritative if updating that CSV fails. Large new artifacts are ignored by
git; copy a complete run directory when sharing an experiment.

Evaluations disable both MTEB result caching and the optional embedding cache.
`evaluation_seconds` measures the whole evaluation; `query_encode_seconds` and
`corpus_encode_seconds` measure the encoder wrapper separately, including its
pre/postprocessing. `seconds_per_1k_docs` uses corpus encoding only. MTEB's phase
called `Scoring` is metric calculation, not vector search. Historical
`encode_seconds` includes the whole evaluation and is labeled accordingly in
regenerated comparisons. Compare speed only on matched hardware and settings.
Undefined auxiliary nAUC metrics are saved as JSON null and listed in `run.json`;
nonfinite primary metrics fail the run instead of producing a misleading score.

Config paths and configured cache/model paths are relative to the repository,
regardless of the shell's current directory. Absolute paths are supported. Use
`revision` for an immutable model commit, and `code_revision` for remote model
code. Runs record the resolved model revision. Remote-code execution also requires
`trust_remote_code`; SFR uses code from a separate upstream repository.

The encoder rejects unknown config and encoding options. Query/document roles
must be explicit. Float embedding normalization and dimension truncation are
supported; integer embedding quantization is intentionally rejected (ONNX INT8
quantizes model weights instead). The optional vector cache validates arrays,
recomputes corrupt entries, and atomically writes vectors under a pipeline-specific
namespace. PyTorch caching requires a resolved immutable revision; remote-code
caching additionally requires an immutable `code_revision`.

## Sequential candidates

```bash
.venv/bin/python experiments/run_all_candidates.py --budget-seconds 7200
.venv/bin/python experiments/run_all_candidates.py --comparison-only
```

The default candidate list uses the main environment. Run SFR separately in its
own environment. Candidate subprocesses run sequentially with a shared elapsed-time
budget and full logs under `results/batches/`. Failed or skipped candidates produce
a nonzero exit status. Deterministic failures are not automatically retried. Timing
samples are selected with a recorded random seed; their corpus-only projection is
an estimate, not a guarantee about total evaluation time. Regenerating the table
retains the historical SFR result and hardware metadata, as well as new complete runs.

## Export and measure ONNX

Create `.venv-onnx` with Python 3.11, install the CPU torch build as above, then:

```bash
.venv-onnx/bin/python -m pip install -r requirements-onnx.txt
.venv-onnx/bin/python experiments/export_quantized_onnx.py
.venv-onnx/bin/python scripts/measure_query_latency.py \
  --config .cache/onnx_models/jina-code-int8/serving_config.json \
  --num-queries 100 --seed 42 --output results/profiles/jina-int8.json
```

The exporter resolves a model commit, selects ARM64 or conservative AVX2 based on
host architecture (override with `--preset`), and copies matching companion files.
It validates single-query and mixed-length batched inference before publishing the
local artifact directory. Existing directories are never overwritten; choose a new
`--output-dir`. Use the generated `serving_config.json` to preserve the artifact
path, revision and preset. The shared loader applies Jina's position-ID patch and
hashes artifact contents, including external tensors and tokenizer files.

For a comparable full-precision profile, run `measure_query_latency.py` with the
full-precision config, the same seed/query count, and a different output path.
The script selects the device before model loading, disables vector caching, and
records query IDs, individual latencies, cold inference, per-input warm-up, p50/p95,
model-load time and environment. `--threads` controls PyTorch CPU threads;
`onnx_intra_op_threads` in the ONNX config controls ONNX Runtime threads (the Jina
config uses four). These measurements exclude search,
reranking, queueing and concurrent requests.

```bash
.venv-onnx/bin/python experiments/compare_encoders.py \
  --reference-config configs/jina_code_embeddings_0_5b.json \
  --candidate-config .cache/onnx_models/jina-code-int8/serving_config.json \
  --num-queries 300 --seed 42
```

This diagnostic uses the **same sampled queries and full corpus** for reference
queries/reference documents, candidate queries/reference documents, and candidate
queries/candidate documents. Models encode in separate subprocesses. Artifacts
include raw vectors, IDs, configs, logs, scores and top-100 rankings under
`results/comparisons/`. The metric helper checks APPS's one-binary-positive-per-query
assumption. It is a scoped diagnostic; use `run_eval.py` for full-query official
MTEB scores. CPU corpus encoding can still be slow; optional device flags and
per-encoder timeouts are available via `--help`.

Use `--num-queries 3765` to include every APPS test query. Separate environments
are supported with `--reference-python` and `--candidate-python`; each encoder
records its own installed versions. To isolate quantization, use matching
Sentence Transformers/Transformers versions for both encoders. The pinned
`configs/jina_code_embeddings_0_5b_validation.json` uses explicit FP32 and batch
size four, suitable for the available 8 GB GPU, with the same revision as the
ONNX export config. For CPU latency, use the same sampled query IDs and four
threads in both backends and run the profiles sequentially.

## Research follow-up

Before tuning, inspect per-query predictions, especially answers already present
in the top 20. Use training queries for development and reserve test queries for
final comparison. A tokenizer-only audit of the pinned APPS data found roughly
89% of test queries exceed MiniLM's configured limit and 41% exceed E5's; none
exceed Jina's. Slice errors by input length before attributing gains to training
alone. Identical snippets under distinct document IDs also create label ambiguity;
flag them in analysis without changing official benchmark labels.

```bash
.venv/bin/python experiments/analyze_run.py \
  --run-dir results/runs/<run-directory> --output results/profiles/errors.json
```

This reads saved predictions without model inference and records query lengths,
gold-document rank intervals (including score ties), top candidates, and duplicate
code labels. It rejects a dataset revision different from the evaluated run.

`docs/progress.md` preserves the historical experiment narrative. Statements in
older entries describe the code and assumptions at that time; this README describes
the current runners. Further reranking, persistent serving/index management and
full-corpus INT8 validation remain separate experiments.
