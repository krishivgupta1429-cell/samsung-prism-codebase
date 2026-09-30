import json
from unittest.mock import patch

import numpy as np
import pytest

from src.retrieval.config import load_config


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"max_seq_length": -1},
        {"dtype": "fp8"},
        {"unknown": 1},
        {"backend": "onnx"},
        {"provider": "CPUExecutionProvider"},
    ],
)
def test_bad_configuration_rejected_before_model_load(tmp_path, overrides):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model_name": "test/model", **overrides}))
    with pytest.raises(ValueError):
        load_config(path)


def test_cache_invalidation_and_mixed_hit_order(encoder_factory):
    first = encoder_factory(cache_embeddings=True)
    old = first.encode_texts(["alpha", "beta"], kind="query")
    fresh = encoder_factory(cache_embeddings=True)
    result = fresh.encode_texts(["beta", "gamma", "alpha", "gamma"], kind="query")
    np.testing.assert_array_equal(result[0], old[1])
    np.testing.assert_array_equal(result[2], old[0])
    assert fresh.model.calls[0][0] == ["gamma"]
    changed = encoder_factory(cache_embeddings=True, max_seq_length=32)
    new = changed.encode_texts(["alpha"], kind="query")
    assert changed.cache_dir != first.cache_dir
    assert new[0, 1] == 32
    assert len(changed.model.calls) == 1


def test_corrupt_wrong_shape_and_nonfinite_cache_recomputed(encoder_factory):
    encoder = encoder_factory(cache_embeddings=True)
    expected = encoder.encode_texts(["alpha"], kind="query")
    path = next(encoder.cache_dir.glob("*.npy"))
    for contents in (b"partial file", None, "nan"):
        if contents == b"partial file":
            path.write_bytes(contents)
        else:
            np.save(path, np.array([np.nan] * 3) if contents == "nan" else np.ones(2))
        np.testing.assert_array_equal(
            encoder.encode_texts(["alpha"], kind="query"), expected
        )
    assert len(encoder.model.calls) == 4
    assert not list(encoder.cache_dir.glob(".tmp-*"))


def test_interrupted_write_preserves_existing_cache(encoder_factory):
    encoder = encoder_factory(cache_embeddings=True)
    encoder.encode_texts(["saved"], kind="query")
    path = next(encoder.cache_dir.glob("*.npy"))
    original = path.read_bytes()
    with patch.object(
        encoder_factory.module.os, "replace", side_effect=OSError("disk error")
    ):
        with pytest.raises(OSError):
            encoder.encode_texts(["new"], kind="query")
    assert path.read_bytes() == original
    assert not list(encoder.cache_dir.glob(".tmp-*"))


def test_options_prefixes_empty_and_batches(encoder_factory):
    encoder = encoder_factory(
        cache_embeddings=True, query_prefix="Q:", document_prefix="D:"
    )
    p = encoder_factory.prompt_type
    inputs = [{"text": ["alpha"]}, {"text": ["beta", "gamma"]}]
    output = encoder.encode(
        inputs,
        task_metadata=None,
        hf_split="test",
        hf_subset="default",
        prompt_type=p.query,
        normalize_embeddings=True,
        truncate_dim=2,
    )
    assert output.shape == (3, 2)
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1)
    assert encoder.model.calls[0][0] == ["Q:alpha"]
    assert encoder.model.calls[0][1]["prompt"] == ""
    encoder.encode_texts(["alpha"], kind="document")
    assert encoder.model.calls[-1][0] == ["D:alpha"]
    assert encoder.encode_texts([], kind="query").shape == (0, 3)
    with pytest.raises(ValueError, match="explicit"):
        encoder.encode([], task_metadata=None, hf_split="test", hf_subset="default")
    with pytest.raises(ValueError, match="Unsupported"):
        encoder.encode(
            [],
            task_metadata=None,
            hf_split="test",
            hf_subset="default",
            prompt_type=p.query,
            imaginary=True,
        )


def test_onnx_config_loads_local_graph_and_patch(
    encoder_factory, tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(SessionOptions=SimpleNamespace)
    )
    root = tmp_path / "onnx-model"
    (root / "onnx").mkdir(parents=True)
    graph = root / "onnx/model.onnx"
    graph.write_bytes(b"graph-one")
    options = dict(
        backend="onnx",
        onnx_model_dir=str(root),
        onnx_file_name="onnx/model.onnx",
        onnx_position_ids=True,
        onnx_intra_op_threads=4,
    )
    encoder = encoder_factory(**options)
    assert encoder.model.name == str(root)
    assert encoder.model.kwargs["backend"] == "onnx"
    assert encoder.model.kwargs["model_kwargs"]["export"] is False
    assert (
        encoder.model.kwargs["model_kwargs"]["session_options"].intra_op_num_threads
        == 4
    )
    assert encoder.model.auto_model._prism_position_ids_patched
    graph.write_bytes(b"graph-two")
    assert (
        encoder_factory(**options).pipeline_fingerprint != encoder.pipeline_fingerprint
    )


def test_bad_model_outputs_fail(encoder_factory):
    encoder = encoder_factory()
    for result in (
        np.ones((2, 3)),
        np.full((1, 3), np.nan),
        np.ones((1, 3), dtype=np.int32),
    ):
        with (
            patch.object(encoder.model, "encode", return_value=result),
            pytest.raises(ValueError, match="finite floating"),
        ):
            encoder.encode_texts(["alpha"], kind="query")


def test_device_override_applied_during_construction(encoder_factory, tmp_path):
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps({"model_name": "test/model", "device": "cuda"}))
    encoder = encoder_factory.module.PrePostPipelineEncoder(path, device="cpu")
    assert encoder.model.kwargs["device"] == "cpu"
