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

- Verified: `check_env.py` exits 0 (mteb 2.21.8, sentence-transformers 6.1.0,
  torch 2.14.0, numpy 2.4.6, pandas 3.0.6; all three required imports OK;
  `AppsRetrieval` task metadata loads without downloading the full dataset).
- Verified: `pytest -q` passes (1 test).
- Committed as `d509264` — "Phase 0: project skeleton, env check, metrics
  logger".

**Phase 0: verified and committed.**

## Phase 1: Baseline and first valid submission

- Inspected installed `mteb` 2.21.8 source (`abs_encoder.py`,
  `sentence_transformer_wrapper.py`, `types/_encoder_io.py`) to confirm the
  exact `encode()` signature, that `PromptType.query`/`PromptType.document`
  distinguish queries from documents, that each batch is a dict with a
  `"text"` key, and that `ModelMeta.create_empty(overwrites=...)` is the right
  way to populate `mteb_model_meta`.
- Wrote `src/retrieval/encoder.py`: `PrePostPipelineEncoder(AbsEncoder)`
  wrapping a CPU-forced `SentenceTransformer`, driven by a JSON config (model
  name, query/document prefixes, max sequence length, batch size,
  `trust_remote_code` flag, optional on-disk embedding cache under
  `.cache/embeddings/`, off by default). Every query/document is routed
  through `preprocess_query`/`preprocess_document`/`postprocess` hooks
  (no-ops for now).
- Wrote `configs/gte_modernbert_base.json` for `Alibaba-NLP/gte-modernbert-base`
  (no prefixes, `max_seq_length=2048`). Confirmed it does **not** need
  `trust_remote_code`: its `config.json` has no `auto_map` and uses the
  natively-supported `ModernBertModel` architecture.
- Wrote `tests/test_encoder_smoke.py` (uses
  `configs/minilm_smoke_test.json`, `sentence-transformers/all-MiniLM-L6-v2`)
  and `scripts/time_encode.py`. Both pass/run.
- **Problem — `gte-modernbert-base` is impractically slow on this CPU.**
  Timing the real APPS corpus (8,765 docs test split) with this model showed
  a single batch of 64 real documents taking 5+ minutes even at
  `max_seq_length=512`, which would put the full-corpus encode at many hours
  — far past the brief's 2-hour cap. Spent significant time ruling out
  causes: not a long-document outlier (checked char/token lengths directly,
  all short), not an HF network stall (reproduced identically in full
  `HF_HUB_OFFLINE=1` mode), not a threading pathology
  (`torch.set_num_threads(1)` was equally slow), not denormal floats
  (`torch.set_flush_denormal(True)` made no difference), not
  `max_seq_length` itself (512 was as slow as 2048). Confirmed the machine
  and pipeline are otherwise fine: `sentence-transformers/all-MiniLM-L6-v2`
  encoded the identical 64 real documents in 0.69s. Root cause is most
  likely an inefficient CPU code path for ModernBERT's hybrid local/global
  attention in this transformers/torch build on this machine, not a bug in
  our code.
- **Decision (user-approved): switched the Phase 1 baseline to
  `sentence-transformers/all-MiniLM-L6-v2`** rather than continuing to
  debug `gte-modernbert-base` on CPU. Added `configs/minilm_l6_v2.json`
  (`max_seq_length=256`, `batch_size=64`, no prefixes).
- Formal timing projection (`scripts/time_encode.py`, 200-doc sample,
  `HF_HUB_OFFLINE=1` to avoid HF network flakiness): 11.86s per 1,000 docs →
  **104.0s projected for the full 8,765-doc corpus.** Well within budget.
- Wrote `experiments/run_eval.py`: builds the encoder from a config, runs
  `mteb.evaluate(...)` on `AppsRetrieval` with `encode_kwargs={"batch_size":
  64}`, writes `results/<model_slug>/appsretrieval_results.json`
  (`task_result.to_dict()` needed `default=str` in `json.dump` — it contains
  a `datetime` that isn't JSON-serializable by default), and calls
  `log_run()`. Confirmed exact metric key names from
  `mteb/_evaluators/retrieval_metrics.py`: `ndcg_at_10`, `mrr_at_10` (as
  the brief guessed).
- Ran `experiments/run_eval.py --config configs/minilm_l6_v2.json
  --experiment-id phase1_minilm_l6_v2` (with `HF_HUB_OFFLINE=1`, and cleared
  `~/.cache/mteb`'s result cache first to get an honest, non-cached timing
  measurement). Result:
  - **NDCG@10: 0.06596**
  - **MRR@10: 0.05581**
  - **encode_seconds: 80.74s** (close to the 104s projection)
  - **seconds_per_1k_docs: 9.21**
  - num_queries: 3765, corpus_size: 8765
- `results/minilm_l6_v2/appsretrieval_results.json` written and verified to
  contain real (non-null/non-zero) scores. One row logged in
  `results/metrics_log.csv`.
- `pytest -q`: 2 passed (metrics log test + encoder smoke test).

**Phase 1: baseline established with `all-MiniLM-L6-v2`.** This NDCG@10 of
0.066 is the number every later idea (query cleaning, hybrid search,
reranking, a properly-working code-specific embedding model) has to beat.
`gte-modernbert-base` remains a candidate worth revisiting later — e.g. on a
different machine/GPU, or after investigating the specific CPU slowdown
further — since a code-pretrained model should meaningfully outperform a
general-purpose MiniLM on this task.

Next: await direction on whether to (a) try the other candidate models
(`granite-embedding-english-r2`, `e5-base-v2`, `jina-code-embeddings-0.5b`)
against this same baseline, (b) revisit `gte-modernbert-base`'s CPU
slowness, or (c) move to Phase 2 error analysis on the MiniLM baseline.
