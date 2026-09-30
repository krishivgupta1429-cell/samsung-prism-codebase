import json
import sys
from types import SimpleNamespace

import pytest

from experiments import run_all_candidates, run_eval
from src.retrieval.artifacts import atomic_json
from src.retrieval.metrics_log import append_row, HEADER, log_run


def test_fresh_evaluations_have_unique_artifacts_and_predictions(
    encoder_factory, monkeypatch, tmp_path
):
    config = tmp_path / "model.json"
    config.write_text(
        json.dumps(
            {"model_name": "test/model", "device": "cpu", "cache_embeddings": True}
        )
    )
    scores = {"ndcg_at_10": 0.5, "mrr_at_10": 0.4}
    task = SimpleNamespace(
        dataset={"default": {"test": {"corpus": [1, 2], "queries": [1]}}},
        load_data=lambda: None,
    )
    calls = []

    def evaluate(encoder, tasks, **kwargs):
        calls.append(kwargs)
        assert not encoder.cache_embeddings
        encoder.encode_seconds.update(document=2.0, query=1.0)
        result = SimpleNamespace(
            scores={"test": [dict(scores)]},
            to_dict=lambda: {
                "scores": {"test": [dict(scores)]},
                "dataset_revision": "dataset-sha",
            },
        )
        return SimpleNamespace(task_results=[result])

    monkeypatch.setattr(
        sys.modules["mteb"], "get_task", lambda name: task, raising=False
    )
    monkeypatch.setattr(sys.modules["mteb"], "evaluate", evaluate, raising=False)
    first = run_eval.run_evaluation(config, "same-id", output_root=tmp_path / "runs")
    first_bytes = (first / "appsretrieval_results.json").read_bytes()
    scores["ndcg_at_10"] = 0.75
    second = run_eval.run_evaluation(config, "same-id", output_root=tmp_path / "runs")
    assert first != second
    assert (first / "appsretrieval_results.json").read_bytes() == first_bytes
    assert json.loads((second / "summary.json").read_text())["ndcg_at_10"] == 0.75
    for call in calls:
        assert call["cache"] is None and call["overwrite_strategy"] == "always"
        assert call["prediction_folder"].parent in (first, second)
    summary = json.loads((first / "summary.json").read_text())
    assert summary["corpus_encode_seconds"] == 2
    assert summary["seconds_per_1k_docs"] == 1000
    assert "encode_seconds" not in summary


def test_failed_eval_preserves_traceback(encoder_factory, monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_name": "test/model", "device": "cpu"}))

    def fail(_):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr(sys.modules["mteb"], "get_task", fail, raising=False)
    with pytest.raises(RuntimeError, match="dataset unavailable"):
        run_eval.run_evaluation(config, "failure", output_root=tmp_path / "runs")
    run = next((tmp_path / "runs").iterdir())
    assert json.loads((run / "run.json").read_text())["status"] == "failed"
    assert "dataset unavailable" in (run / "failure.log").read_text()


def test_all_candidates_fail_returns_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(run_all_candidates, "run_subprocess", lambda *args: 1)
    assert (
        run_all_candidates.main(
            ["--configs", "configs/minilm_l6_v2.json", "--results-root", str(tmp_path)]
        )
        == 1
    )
    attempts = json.loads(
        next((tmp_path / "batches").glob("*/attempts.json")).read_text()
    )
    assert attempts[0]["status"] == "failed"


def test_comparison_preserves_historical_sfr_and_device(tmp_path):
    from src.retrieval.config import REPO_ROOT

    rows = run_all_candidates.completed_results(REPO_ROOT / "results")
    path = tmp_path / "comparison.md"
    run_all_candidates.write_comparison_table(rows, path)
    text = path.read_text()
    assert "Salesforce/SFR-Embedding-Code-400M_R" in text
    assert "| device |" in text and "| cuda |" in text


def test_empty_csv_gets_header_and_wrong_schema_is_untouched(tmp_path):
    path = tmp_path / "empty.csv"
    path.touch()
    log_run("exp", "model", "config", 0.5, 0.4, 1, 1, 1, path=str(path))
    assert path.read_text().splitlines()[0] == ",".join(HEADER)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        append_row(path, ["wrong"], {"wrong": "value"})
    assert path.read_bytes() == original


def test_atomic_json_rejects_nonfinite_without_replacing_file(tmp_path):
    path = tmp_path / "artifact.json"
    atomic_json(path, {"valid": 1})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        atomic_json(path, {"bad": float("nan")})
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".tmp-*"))


def test_comparison_uses_separate_interpreters(encoder_factory, monkeypatch, tmp_path):
    from pathlib import Path
    import numpy as np
    from experiments import compare_encoders

    class Rows(list):
        def __getitem__(self, key):
            return (
                [row[key] for row in self]
                if isinstance(key, str)
                else super().__getitem__(key)
            )

    split = {
        "queries": Rows([{"id": "q", "text": "query"}]),
        "corpus": Rows([{"id": "d", "text": "code"}]),
        "relevant_docs": {"q": {"d": 1}},
    }
    task = SimpleNamespace(
        load_data=lambda: None,
        dataset={"default": {"test": split}},
        metadata=SimpleNamespace(dataset={"revision": "test-revision"}),
    )
    monkeypatch.setattr(sys.modules["mteb"], "get_task", lambda _: task, raising=False)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        job = json.loads(Path(command[-1]).read_text())
        assert job["python"] == command[0]
        output = Path(job["output"])
        output.mkdir()
        for name in ("queries", "documents"):
            np.save(output / f"{name}.npy", np.ones((1, 3)))

    monkeypatch.setattr(compare_encoders.subprocess, "run", run)
    monkeypatch.setattr(compare_encoders, "environment", lambda: {})
    compare_encoders.main(
        [
            "--reference-config",
            "configs/jina_code_embeddings_0_5b.json",
            "--candidate-config",
            "configs/jina_code_embeddings_0_5b_int8_onnx.json",
            "--reference-python",
            "/reference/python",
            "--candidate-python",
            "/candidate/python",
            "--output-root",
            str(tmp_path),
        ]
    )
    assert [call[0] for call in calls] == ["/reference/python", "/candidate/python"]
    manifest = json.loads(next(tmp_path.glob("*/comparison.json")).read_text())
    assert manifest["status"] == "complete"
    assert manifest["results"]["candidate"]["ndcg_at_10"] == 1


def test_undefined_auxiliary_metric_is_null_but_invalid_primary_fails():
    data = {
        "scores": {"test": [{"ndcg_at_10": 0.5, "nauc_ndcg_at_10_max": float("nan")}]}
    }
    missing = run_eval.sanitize_auxiliary_metrics(data)
    assert data["scores"]["test"][0]["nauc_ndcg_at_10_max"] is None
    assert missing[0]["metric"] == "nauc_ndcg_at_10_max"
    with pytest.raises(ValueError, match="Nonfinite benchmark"):
        run_eval.sanitize_auxiliary_metrics(
            {"scores": {"test": [{"ndcg_at_10": float("nan")}]}}
        )
