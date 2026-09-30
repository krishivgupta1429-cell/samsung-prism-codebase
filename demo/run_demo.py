import random
import time

import mteb
import numpy as np
from sentence_transformers import SentenceTransformer

random.seed(42)
N_CANDIDATES = 60
N_SHOW = 5
N_DISTRACTOR_DOCS = 200
QUERY_PREVIEW_CHARS = 180

print("Indexing code corpus...")
task = mteb.get_task("AppsRetrieval")
task.load_data()
split = task.dataset["default"]["test"]
corpus = {r["id"]: r["text"] for r in split["corpus"]}
queries = {r["id"]: r["text"] for r in split["queries"]}
qrels = split["relevant_docs"]

candidate_qids = random.sample(list(qrels.keys()), N_CANDIDATES)
relevant_doc_ids = {qid: list(qrels[qid].keys())[0] for qid in candidate_qids}

distractor_pool = [d for d in corpus if d not in relevant_doc_ids.values()]
demo_doc_ids = list(relevant_doc_ids.values()) + random.sample(distractor_pool, N_DISTRACTOR_DOCS)
demo_doc_ids = list(dict.fromkeys(demo_doc_ids))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

doc_texts = [corpus[d] for d in demo_doc_ids]
doc_embs = model.encode(doc_texts, convert_to_numpy=True)

cand_texts = [queries[q] for q in candidate_qids]
cand_embs = model.encode(cand_texts, convert_to_numpy=True)

results = []
for qid, q_emb in zip(candidate_qids, cand_embs):
    sims = doc_embs @ q_emb / (np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(q_emb) + 1e-9)
    ranking = np.argsort(-sims)
    rel_id = relevant_doc_ids[qid]
    rank = next((i for i, idx in enumerate(ranking) if demo_doc_ids[idx] == rel_id), None)
    results.append((qid, sims, ranking, rank))

hits = [r for r in results if r[3] is not None and r[3] < 5]
misses = [r for r in results if r not in hits]
results.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 999))
chosen = (hits + misses)[:N_SHOW]

print("Ready.\n")

for qid, sims, ranking, rank in chosen:
    qt = queries[qid].strip().replace("\n", " ")
    preview = qt[:QUERY_PREVIEW_CHARS] + ("..." if len(qt) > QUERY_PREVIEW_CHARS else "")
    t0 = time.time()
    top5 = ranking[:5]
    elapsed = time.time() - t0

    print(f"{'='*80}\nQuery: {preview}")
    for i, idx in enumerate(top5, 1):
        did = demo_doc_ids[idx]
        marker = "  <-- correct match" if did == relevant_doc_ids[qid] else ""
        print(f"  {i}. {did}  (score: {sims[idx]:.4f}){marker}")
    print(f"Ranked in {elapsed*1000:.1f} ms\n")
