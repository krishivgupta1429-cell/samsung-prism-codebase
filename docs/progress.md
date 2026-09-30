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

## Phase 1 (continued): candidate model comparison

- Installed `hf_transfer` and initially set `HF_HUB_ENABLE_HF_TRANSFER=1` as
  requested, but discovered it's a no-op in the installed `huggingface_hub`
  version — downloads always go through its Xet backend now, and that env
  var only triggers a deprecation warning. Xet stalled indefinitely on this
  machine (as it did for `gte-modernbert-base` earlier), so
  `experiments/run_all_candidates.py` forces `HF_HUB_DISABLE_XET=1` instead,
  which was confirmed reliable in the earlier Phase 1 baseline work.
- Confirmed all three candidate model IDs exist on the Hub and none require
  `trust_remote_code` (checked each `config.json` for `auto_map`):
  `jinaai/jina-code-embeddings-0.5b`, `ibm-granite/granite-embedding-english-r2`,
  `intfloat/e5-base-v2`. All three ship `modules.json`, so they load directly
  through our existing `SentenceTransformer`-based encoder.
  - `jina-code-embeddings-0.5b` uses named prompts rather than simple
    prefixes; used its `nl2code_query`/`nl2code_document` prompt strings
    (the closest match to natural-language-query → code-snippet retrieval)
    as our `query_prefix`/`document_prefix`.
  - `granite-embedding-english-r2` is **also `ModernBertModel`-architected**
    (`configs/granite_embedding_english_r2.json`), same family as the
    already-broken `gte-modernbert-base`.
  - `e5-base-v2` uses the standard `BertModel` architecture and needs
    `"query: "`/`"passage: "` prefixes (set in its config, per the brief).
- Wrote `experiments/run_all_candidates.py`: runs each candidate in its own
  subprocess (real OS-level isolation, not just non-concurrency — needed
  after seeing one CPU-bound model's memory climb past 1.5GB earlier),
  strictly one at a time. Reuses `scripts/time_encode.py` for a 200-doc
  timing projection and `experiments/run_eval.py` for the full evaluation.
  Skips a model automatically if the projected full-corpus time exceeds 20
  minutes, retries once on any failure before giving up, and always
  continues to the next candidate rather than stopping the whole run.
  Updated `experiments/run_eval.py` to read `batch_size` from each model's
  own config instead of a hardcoded 64, since candidates have very
  different memory footprints.
- Results:
  - **`ibm-granite/granite-embedding-english-r2` → FAILED.** Timed out
    (1200s) on both the initial timing-check attempt and its retry. This is
    almost certainly the same ModernBERT CPU-inference slowdown diagnosed
    for `gte-modernbert-base` — same architecture family.
  - **`jinaai/jina-code-embeddings-0.5b` → SKIPPED.** First timing-check
    attempt timed out (slow download); the retry succeeded and measured a
    projected full-corpus time of **48,726s (~13.5 hours)** — this 0.5B
    decoder-style (Qwen2) model is also impractically slow for CPU
    inference on this machine.
  - **`intfloat/e5-base-v2` → OK.** NDCG@10 **0.11523**, MRR@10 **0.09878**,
    encode_seconds 787.74 (actual full run — docs + queries — ran longer
    than the docs-only 531.2s projection, expected since queries average
    longer than documents on APPS and the projection only samples the
    corpus). Sanity-checked `results/e5_base_v2/appsretrieval_results.json`
    directly: `ndcg_at_1` (0.073) < `ndcg_at_10` (0.115) < `ndcg_at_100`
    (0.150), `recall_at_10` (0.169) < `recall_at_100` (0.343) — monotonic
    and non-degenerate, not a silent all-zero/identical-score bug.
- `results/model_comparison.md` written, sorted by NDCG@10 descending:
  `e5-base-v2` (0.11523) > `all-MiniLM-L6-v2` (0.06596, baseline) >
  `jina-code-embeddings-0.5b` (skipped) > `granite-embedding-english-r2`
  (failed).
- `pytest -q`: 2 passed.

**Current leader: `intfloat/e5-base-v2`, NDCG@10=0.11523 — roughly 1.75x the
`all-MiniLM-L6-v2` baseline's 0.06596.** Every candidate with a ModernBERT or
large-decoder architecture has now failed or been skipped for CPU-speed
reasons on this machine; the two working, reasonably fast models so far are
both standard BERT-family encoders (MiniLM, e5-base-v2).

## Phase 1 (continued): GPU support added, SFR-Embedding-Code attempted

- Added MPS/CUDA device support to `src/retrieval/encoder.py`: device
  selection now tries CUDA, then Apple MPS, then CPU (`select_device()`),
  and a config `"dtype"` field is resolved through `resolve_dtype()`, which
  downgrades a requested `bfloat16` to `float16` on MPS specifically (MPS's
  bf16 support is incomplete/unstable). `run_eval.py`/`time_encode.py` now
  print `device=..., dtype=...` for every run and include it in the
  `metrics_log.csv` notes field going forward. On this machine (MacBook Air
  M4, no CUDA), device auto-selects to `mps`.
- New candidate `Salesforce/SFR-Embedding-Code-400M_R` needs
  `trust_remote_code=True` (approved by user) — its `NewModel` architecture
  is loaded from a separate `Alibaba-NLP/new-impl` repo's custom code, not
  from Salesforce's own repo. First attempt crashed on MPS: the remote
  code's `self.get_extended_attention_mask(...)` call doesn't exist in our
  installed `transformers==5.17.0` (the code was written for `4.45.1`,
  per the model's `config.json`) — a real version incompatibility, not
  MPS-specific (would raise the same `AttributeError` on CPU too, just
  without the fatal Metal-kernel-level crash MPS produced).
- Per user's direction, built an isolated `.venv-sfr-test/` (fully separate
  from the main `.venv`, never touched it) with `transformers==4.45.1`,
  `sentence-transformers==3.4.1`, `mteb==2.21.8`, `torch==2.14.0` — verified
  no dependency conflicts (mteb only needs `sentence_transformers>=3.0.0`,
  and our own encoder never touches mteb's built-in
  `SentenceTransformerEncoderWrapper`, so the older sentence-transformers is
  safe). In that isolated env, the model loaded and ran successfully:
  `device=mps, dtype=float16`, 64-doc timing sample → **projected 2,586.6s
  (~43.1 min) for the full corpus** — over the 30-minute threshold but
  approved to proceed anyway.
- **Stopped mid-run per user instruction** (moving further GPU testing to a
  separate RTX 4090 machine for speed/reliability) before the full eval
  completed. No result was ever logged — the process was killed before
  reaching `log_run()`/writing `appsretrieval_results.json`, so
  `results/metrics_log.csv` and `results/model_comparison.md` needed no
  cleanup; there was nothing partial to remove.
- Cleanup performed: `.venv-sfr-test/` deleted entirely. Confirmed the main
  `.venv` is untouched — `transformers==5.17.0` (unchanged), `check_env.py`
  exits 0, `pytest -q` passes (2 passed).
- **Note:** `results/metrics_log.csv`'s two completed rows
  (`all-MiniLM-L6-v2`, `e5-base-v2`) predate the device-tracking change
  above and do not record which device they ran on (both ran on CPU, since
  that was hardcoded before this session). There is no separate "device"
  column in either `metrics_log.csv` or `model_comparison.md` — device info
  is only appended into the free-text `notes` field, and only for runs
  after this change (none logged yet).

**Current leader remains `intfloat/e5-base-v2`, NDCG@10=0.11523.** GPU
testing for `SFR-Embedding-Code-400M_R`, `granite-embedding-english-r2`,
and `jina-code-embeddings-0.5b` is continuing on a separate RTX 4090
machine.

## Phase 1 (continued): CUDA runs on RTX 4050 laptop, new leader found

- Picked up on a different machine (Windows laptop). Note: the machine
  actually has an **RTX 4050 Laptop GPU (6GB VRAM)**, not the RTX 4090
  mentioned at the start of this session — confirmed via `nvidia-smi`
  (driver 592.82, CUDA 13.1) and flagged to the user, who confirmed to
  proceed on the 4050.
- Created a fresh `.venv` (Python 3.14.6, the only interpreter on this
  machine) and installed `requirements.txt`, which brings in CPU-only
  `torch==2.14.0` by default. Replaced it with the matching CUDA build,
  `torch==2.14.0+cu130` (from `download.pytorch.org/whl/cu130` — the
  `cu126`/`cu130` indexes were the only ones with a `2.14.0` wheel for
  `cp314`; `cu128`/`cu124`/`cu121` did not have both), via
  `pip install --force-reinstall --no-deps` (plain `pip install` no-ops
  since pip considers a same-version CPU wheel as already satisfying a
  CUDA-tagged requirement). `torch.cuda.is_available()` confirmed `True`,
  device name "NVIDIA GeForce RTX 4050 Laptop GPU". `check_env.py` and
  `pytest -q` both pass (2 passed) before touching any model, as required.
- Added a `device` column to `results/metrics_log.csv` / `log_run()` /
  `experiments/run_eval.py` (previously device was only recorded in the
  free-text `notes` field, per the prior session's note that there was "no
  separate device column"). Backfilled the two pre-existing CPU rows
  (`all-MiniLM-L6-v2`, `e5-base-v2`) with `device=cpu`.
- **`ibm-granite/granite-embedding-english-r2` → OK on CUDA.** Same
  ModernBERT architecture that timed out after 1200s on CPU on a prior
  machine. Timing check (200-doc sample): projected 905s (~15 min), well
  under the 30-minute threshold. Full run: **NDCG@10 0.13993, MRR@10
  0.11960**, encode_seconds 751.33 (device=cuda). GPU memory during the run
  peaked around 5.9/6.1GB — tight on the 4050's 6GB but did not OOM.
- **`jinaai/jina-code-embeddings-0.5b` → OK on CUDA.** Projected ~13.5
  hours on CPU on a prior machine; on CUDA the timing check projected
  1111.5s (~18.5 min), also under threshold. Full run: **NDCG@10 0.84083,
  MRR@10 0.81055**, encode_seconds 1431.18 (device=cuda) — the actual run
  took about 1.3x the doc-only projection, consistent with queries
  averaging longer than documents (same pattern seen with `e5-base-v2`
  earlier). Sanity-checked `results/jina_code_embeddings_0_5b/
  appsretrieval_results.json` directly: `ndcg_at_1` (0.74104) <
  `ndcg_at_10` (0.84083) < `ndcg_at_100` (0.85213), `recall_at_10` (0.9344)
  < `recall_at_100` (0.98566) — monotonic, non-degenerate. This is a large
  jump over every other candidate (next best 0.13993), which is plausible
  here: it's a code-specific model using its own `nl2code_query`/
  `nl2code_document` prompt templates (already wired up in
  `configs/jina_code_embeddings_0_5b.json` from the earlier candidate
  comparison), unlike the generic BERT-family encoders. **New leader.**
- **`Salesforce/SFR-Embedding-Code-400M_R` → OK on CUDA**, resuming the
  test that was stopped mid-run on a different machine before completing.
  Its remote code (from `Alibaba-NLP/new-impl`, loaded via
  `trust_remote_code=True`) previously required `transformers==4.45.1` (an
  older-than-`requirements.txt` version) — but `transformers==4.45.1`'s
  pinned `tokenizers<0.21` has no prebuilt wheel for Python 3.14, so used
  `transformers==4.49.0` instead (top of the user-approved ~4.45–4.49
  range; dry-run confirmed it resolves cleanly to `tokenizers==0.21.4`,
  which does have a `cp314`-compatible wheel). Built an isolated
  `.venv-sfr-test/` (`transformers==4.49.0`, `sentence-transformers==3.4.1`,
  `mteb==2.21.8`, then the same CUDA `torch==2.14.0+cu130` swap as the main
  venv) — main `.venv` untouched throughout. Timing check: projected
  1204.2s (~20 min), under threshold. Full run: **NDCG@10 0.49627, MRR@10
  0.44957**, encode_seconds 1721.35 (device=cuda, dtype=bfloat16).
  Sanity-checked the results file: `ndcg_at_1` (0.36042) < `ndcg_at_10`
  (0.49627) < `ndcg_at_100` (0.54178) — monotonic, non-degenerate. Second
  place overall.
- All three models ran one at a time (never concurrently) to avoid
  contending for the 4050's 6GB VRAM.
- `results/metrics_log.csv` and `results/model_comparison.md` updated with
  all three results (device=cuda for all three). `pytest -q`: 2 passed
  (re-verified after the `device` column change).

**New leader: `jinaai/jina-code-embeddings-0.5b`, NDCG@10=0.84083** — about
6x the previous leader (`e5-base-v2`, 0.11523) and 7.3x further above the
`all-MiniLM-L6-v2` baseline (0.06596. `Salesforce/SFR-Embedding-Code-400M_R`
is a clear second (0.49627). Every candidate attempted on CPU that failed or
was skipped for speed reasons (`granite-embedding-english-r2`,
`jina-code-embeddings-0.5b`, and previously the CPU-crashing
`SFR-Embedding-Code-400M_R`) ran successfully once moved to CUDA, so GPU
availability — not architecture — was the real blocker for those models on
this benchmark.

## Phase 1: formal winner decision — `jina-code-embeddings-0.5b`

Before formally selecting a winner, confirmed exactly how corpus vs. query
encoding is handled, since that determines whether jina's slow CPU
corpus-encoding time conflicts with the problem statement's "CPU-friendly,
minimal GPU" requirement.

**Confirmed via MTEB's own source** (`mteb/_evaluators/retrieval_evaluator.py`,
`RetrievalEvaluator.__call__`): the evaluator is structured as two distinct
phases — `search_model.index(corpus=...)`, timed as `"Encoding corpus"`, then
`search_model.search(queries=...)`, timed as `"Encoding queries"`. The code
explicitly branches on whether the search backend has a persistent
`index_backend`: with one, corpus encoding happens once during indexing and
only queries are re-encoded per search; without one (our current setup —
a plain encoder, no vector index in front of it), the two get fused into a
single measurement, which is exactly what our logged `encode_seconds` numbers
reflect (corpus + queries combined, in one `mteb.evaluate()` call).

This matches this project's own stated design intent from the original Phase 1
brief for `src/retrieval/encoder.py`'s on-disk embedding cache
(`.cache/embeddings/`): "reuse cached embeddings **to avoid re-encoding the
corpus**." The whole point of that cache is that corpus encoding is meant to
be a one-time (or occasional, on corpus change) offline step — also this
project's stated P1 goal ("fast index rebuilds when code changes") — while
query encoding is the per-request, online cost that actually needs to be fast.

**Conclusion: confirmed.** Corpus encoding is architecturally a one-time
indexing cost that can reasonably use a GPU without violating the spirit of
"CPU-friendly, minimal GPU" — that requirement is about the serving/query
path, not the one-time index build. jina-code-embeddings-0.5b's slow CPU
corpus-encoding time (~13.5h projected) is therefore not by itself
disqualifying.

**Open caveat, not yet closed out:** we have not directly measured
jina-code-embeddings-0.5b's single-query CPU encoding latency in isolation —
every CPU timing check so far (`scripts/time_encode.py`) sampled documents
from the corpus, never queries. The architectural argument for compliance is
sound, but a concrete query-only CPU latency number is still a recommended
follow-up before calling CPU-side compliance fully proven end-to-end.

**Formal decision: `jinaai/jina-code-embeddings-0.5b` is the Phase 1 winner.**
NDCG@10 0.84083 / MRR@10 0.81055 — a dramatic, non-degenerate improvement
(6x the previous leader, 12.7x the baseline) driven by code-specific
pretraining, which none of the general-purpose text encoders tried in Phase 1
(MiniLM, e5-base-v2, granite) can match regardless of device. `README.md`
updated to reflect this as the current model and to state the corpus/query
reasoning and its open caveat.

## Phase 1 (continued): closed out the single-query CPU latency follow-up

Closed out the open caveat above: measured `jina-code-embeddings-0.5b`'s
actual single-query CPU latency (20 real queries, one at a time, `batch_size=1`,
one untimed warm-up call excluded from the stats):

- **min 1,121.3 ms / avg 4,382.1 ms / max 9,396.0 ms**

Not comfortably fast — 4+ seconds average is a real problem for live
query-time serving, not just a theoretical corpus-encoding cost. This
disproves the earlier "corpus is one-time, queries are fine" assumption for
*this specific model as-is*: the model is too large (0.5B, decoder
architecture) for its own per-query forward pass to be CPU-fast, independent
of corpus size.

**Response: ONNX INT8 dynamic quantization**, via
`experiments/export_quantized_onnx.py` (requires a separate, isolated venv —
`sentence-transformers[onnx]` pulls in `optimum`/`onnxruntime`, deliberately
kept out of the main `.venv`; the main `.venv` was never touched and was
re-verified after — `check_env.py` exits 0, `pytest -q` passes). Exported via
`sentence_transformers.backend.export_dynamic_quantized_onnx_model(model,
"arm64", ...)` (arm64 preset, this machine's CPU architecture), producing a
497MB `model_qint8_arm64.onnx` (roughly half the original bf16 size — dynamic
quantization only quantizes weight matrices, not the full model).

One real snag: the exported graph requires `position_ids` as an explicit
input (Qwen2 uses RoPE), which sentence-transformers' tokenizer doesn't
produce and the original PyTorch model computed internally when absent — that
fallback isn't preserved by tracing. Fixed by monkeypatching the ONNX model's
forward to compute `position_ids` from `attention_mask` before delegating to
the real forward (see `patch_position_ids()` in the export script).

**Quantized single-query CPU latency** (identical methodology, same 20
queries):

- **min 113.0 ms / avg 449.7 ms / max 1,068.6 ms — a ~9.7x speedup**, now
  mostly under a second. This is the number that matters for live serving.

**Scoped accuracy sanity check** (300 randomly-sampled queries — not the full
3,765 — against a 500-document pool: their 300 gold-answer documents plus 200
random distractors, not the full 8,765-document corpus):

- **NDCG@10: 0.87636, MRR@10: 0.84623**

No accuracy collapse from quantization — that was the actual purpose of this
check. **Important caveat: this number is not directly comparable to the
official benchmark score (0.84083).** A 500-candidate pool that's guaranteed
to contain every correct answer is a much easier ranking task than the full
8,765-document corpus, so 0.876 here reflects an easier task, not a claim
that quantization *improved* accuracy. A true apples-to-apples comparison
would need the same query sample run against the full corpus with both
model versions.

Along the way, discovered a real ONNX Runtime characteristic worth recording:
the very first inference batch at a new/largest sequence shape pays a one-time
graph/memory-arena initialization cost that can dominate a short run's total
time (a 16-batch corpus-pool encode showed batch 1 taking 34.5 minutes and
the remaining 15 batches taking under 2 minutes combined). The single-query
latency measurements above are unaffected because they include an explicit
untimed warm-up call before measuring — exactly the practice that absorbs
this cost. **A production deployment needs one warm-up inference at
startup**; without it, the very first real query would pay this one-time
cost.

The quantized model binary (497MB) is **not committed to git** — GitHub
rejects pushes of files over 100MB without Git LFS, which this repo doesn't
use, and a 500MB binary isn't appropriate to carry in ordinary git history
for a hackathon submission. Instead, `experiments/export_quantized_onnx.py`
reproduces it exactly, and `configs/jina_code_embeddings_0_5b_int8_onnx.json`
documents the serving-time config (file name, provider, prefixes,
`max_seq_length`, the position_ids caveat). Regenerating it takes roughly the
time reported above (base export + quantization, a few minutes) plus the
corpus/query encode times if re-validating.

**Two-tier model setup, going forward:**

- **Official benchmark score: the full-precision `jina-code-embeddings-0.5b`,
  NDCG@10 0.84083, MRR@10 0.81055** — measured on the full 8,765-document
  APPS test corpus (`results/jina_code_embeddings_0_5b/`). This is the number
  reported for the hackathon submission.
- **Live query-time serving: the INT8 ONNX quantized version** — ~450ms
  average single-query CPU latency, ~9.7x faster than full precision, with a
  scoped sanity check showing no sign of accuracy collapse. This is what an
  actual deployed/demo system should use for encoding incoming queries.
  Corpus embeddings (computed once, offline, with either model version) are
  reused across queries either way, per the corpus/query architecture
  discussion above.

`README.md` updated to reflect this two-tier setup.

**Phase 1 is now complete**, including the CPU-serving-latency follow-up.

Next: awaiting direction on Phase 2 (error analysis) or Phase 3 (hybrid
search + reranking) on top of `jina-code-embeddings-0.5b`.
