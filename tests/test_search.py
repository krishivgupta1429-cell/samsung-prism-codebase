import math
import numpy as np
import pytest
from src.retrieval.search import cosine_top_k, single_answer_metrics


def test_exact_cosine_retrieval_and_known_metrics():
    docs = np.array([[9.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    queries = np.array([[2.0, 0.0], [0.0, 4.0]])
    ranked = cosine_top_k(queries, docs, ["east", "north", "west"], k=3)
    assert ranked == [["east", "north", "west"], ["north", "east", "west"]]
    result = single_answer_metrics(
        ranked, ["q1", "q2"], {"q1": {"east": 1}, "q2": {"east": 1}}
    )
    assert result["mrr_at_10"] == 0.75
    assert result["ndcg_at_10"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert result["queries"][1]["rank"] == 2


@pytest.mark.parametrize("vectors", [np.zeros((1, 2)), np.array([[np.nan, 1.0]])])
def test_invalid_vectors_rejected(vectors):
    with pytest.raises(ValueError):
        cosine_top_k(vectors, np.ones((1, 2)), ["doc"])


def test_binary_metric_rejects_incompatible_labels():
    with pytest.raises(ValueError, match="one binary"):
        single_answer_metrics([["a"]], ["q"], {"q": {"a": 1, "b": 1}})


def test_error_analysis_flags_duplicate_code_and_tie_intervals():
    from experiments.analyze_run import analyze

    report = analyze(
        {"q": {"a": 0.8, "b": 0.8, "c": 0.9}},
        [{"id": "q", "text": "add numbers"}],
        [
            {"id": "a", "text": "same code"},
            {"id": "b", "text": "same code"},
            {"id": "c", "text": "different"},
        ],
        {"q": {"a": 1}},
    )
    record = report["queries"][0]
    assert (record["rank_min"], record["rank_max"]) == (2, 3)
    assert record["identical_document_ids"] == ["a", "b"]
    assert report["queries_with_duplicate_gold"] == 1
