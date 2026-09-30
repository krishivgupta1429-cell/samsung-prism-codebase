from unittest.mock import patch

import os
import pytest

CONFIG_PATH = "configs/minilm_smoke_test.json"


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 for real model inference",
)
def test_encoder_smoke():
    from mteb.types import PromptType
    from src.retrieval.encoder import PrePostPipelineEncoder

    encoder = PrePostPipelineEncoder(CONFIG_PATH)

    texts = [
        "def add(a, b): return a + b",
        "Write a function that adds two numbers.",
        "class Foo: pass",
    ]
    inputs = [{"text": texts}]

    with (
        patch.object(
            encoder, "preprocess_query", wraps=encoder.preprocess_query
        ) as mock_preprocess_query,
        patch.object(
            encoder, "preprocess_document", wraps=encoder.preprocess_document
        ) as mock_preprocess_document,
        patch.object(
            encoder, "postprocess", wraps=encoder.postprocess
        ) as mock_postprocess,
    ):
        query_embeddings = encoder.encode(
            inputs,
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.query,
        )
        document_embeddings = encoder.encode(
            inputs,
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            prompt_type=PromptType.document,
        )

    assert query_embeddings.shape == (3, document_embeddings.shape[1])
    assert document_embeddings.shape == (3, document_embeddings.shape[1])

    assert mock_preprocess_query.call_count == 3
    assert mock_preprocess_document.call_count == 3
    assert mock_postprocess.call_count == 2


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 for real model inference",
)
def test_real_mteb_evaluation_writes_artifacts(monkeypatch, tmp_path):
    import json
    import mteb
    from datasets import Dataset
    from experiments.run_eval import run_evaluation

    task = mteb.get_task("AppsRetrieval")
    texts = [
        "sort a list",
        "read a file",
        "add two integers",
        "reverse a string",
        "parse JSON",
        "find the maximum",
    ]
    task.dataset = {
        "default": {
            "test": {
                "corpus": Dataset.from_dict(
                    {"id": [f"d{i}" for i in range(6)], "text": texts}
                ),
                "queries": Dataset.from_dict(
                    {"id": [f"q{i}" for i in range(6)], "text": texts}
                ),
                "relevant_docs": {
                    f"q{i}": {f"d{i if i < 4 else 0}": 1} for i in range(6)
                },
                "top_ranked": None,
            }
        }
    }
    task.data_loaded = True
    monkeypatch.setattr(mteb, "get_task", lambda name: task)
    path = run_evaluation(
        CONFIG_PATH, "synthetic-integration-only", output_root=tmp_path, device="cpu"
    )
    assert json.loads((path / "run.json").read_text())["status"] == "complete"
    summary = json.loads((path / "summary.json").read_text())
    assert summary["num_queries"] == 6 and summary["corpus_size"] == 6
    assert summary["corpus_encode_seconds"] > 0 and summary["query_encode_seconds"] > 0
    assert list((path / "predictions").rglob("*.json"))


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ONNX_TEST_CONFIG"),
    reason="Set ONNX_TEST_CONFIG to a local exported model config",
)
def test_real_onnx_single_and_padded_batch():
    import numpy as np
    from src.retrieval.encoder import PrePostPipelineEncoder

    encoder = PrePostPipelineEncoder(
        os.environ["ONNX_TEST_CONFIG"], cache_embeddings=False
    )
    assert encoder.config["backend"] == "onnx"
    single = encoder.encode_texts(["Sort a list."], kind="query")
    batch = encoder.encode_texts(
        [
            "Sort a list.",
            "Find the longest increasing subsequence in a list of integers.",
        ],
        kind="query",
    )
    assert batch.shape == (2, single.shape[1])
    assert np.isfinite(batch).all()
    cosine = float(
        single[0] @ batch[0] / (np.linalg.norm(single[0]) * np.linalg.norm(batch[0]))
    )
    assert cosine > 0.99
