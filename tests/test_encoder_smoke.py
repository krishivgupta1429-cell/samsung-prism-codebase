from unittest.mock import patch

from mteb.types import PromptType

from src.retrieval.encoder import PrePostPipelineEncoder

CONFIG_PATH = "configs/minilm_smoke_test.json"


def test_encoder_smoke():
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
