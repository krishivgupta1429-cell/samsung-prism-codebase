"""Exact cosine ranking and binary single-answer diagnostics for APPS."""

import math
import numpy as np


def cosine_top_k(queries, documents, document_ids, k=100):
    queries, documents = np.asarray(queries), np.asarray(documents)
    if (
        queries.ndim != 2
        or documents.ndim != 2
        or queries.shape[1] != documents.shape[1]
    ):
        raise ValueError("Query and document embeddings must have matching dimensions")
    if len(documents) != len(document_ids) or len(set(document_ids)) != len(
        document_ids
    ):
        raise ValueError("Document IDs must be unique and match the vector rows")
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0 or not len(documents):
        raise ValueError("Search requires positive k and a nonempty corpus")
    if not np.isfinite(queries).all() or not np.isfinite(documents).all():
        raise ValueError("Embeddings must be finite")
    qnorm, dnorm = np.linalg.norm(queries, axis=1), np.linalg.norm(documents, axis=1)
    if (qnorm == 0).any() or (dnorm == 0).any():
        raise ValueError("Cosine search requires nonzero vectors")
    documents = documents / dnorm[:, None]
    ranked = []
    # Bound the similarity matrix; tie order follows the recorded corpus order.
    for start in range(0, len(queries), 32):
        block = queries[start : start + 32] / qnorm[start : start + 32, None]
        scores = block @ documents.T
        order = np.argsort(-scores, axis=1, kind="stable")[:, : min(k, len(documents))]
        ranked.extend([[document_ids[i] for i in row] for row in order])
    return ranked


def single_answer_metrics(ranked, query_ids, relevant_docs):
    if len(ranked) != len(query_ids) or not query_ids:
        raise ValueError("Rankings must match a nonempty set of queries")
    records = []
    for qid, ranking in zip(query_ids, ranked, strict=True):
        positives = [doc for doc, score in relevant_docs[qid].items() if score > 0]
        if len(positives) != 1 or relevant_docs[qid][positives[0]] != 1:
            raise ValueError(
                "This APPS diagnostic requires one binary positive per query"
            )
        gold = positives[0]
        rank = ranking.index(gold) + 1 if gold in ranking else None
        records.append(
            {
                "query_id": qid,
                "gold_document_id": gold,
                "rank": rank,
                "ndcg_at_10": 1 / math.log2(rank + 1) if rank and rank <= 10 else 0.0,
                "mrr_at_10": 1 / rank if rank and rank <= 10 else 0.0,
                "top_document_ids": ranking,
            }
        )
    return {
        "ndcg_at_10": sum(r["ndcg_at_10"] for r in records) / len(records),
        "mrr_at_10": sum(r["mrr_at_10"] for r in records) / len(records),
        "queries": records,
    }
