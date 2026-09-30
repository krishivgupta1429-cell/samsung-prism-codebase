"""Fresh full-corpus evaluations with immutable run artifacts."""

import argparse
import logging
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, create_run, environment
from src.retrieval.config import load_config, repo_path
from src.retrieval.metrics_log import log_evaluation


def sanitize_auxiliary_metrics(result_data):
    """MTEB nAUC is undefined for constant scores; keep that distinct from zero."""
    undefined = []
    for split, records in result_data.get("scores", {}).items():
        for record in records:
            for name, value in list(record.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    if not name.startswith("nauc_"):
                        raise ValueError(f"Nonfinite benchmark metric: {name}")
                    record[name] = None
                    undefined.append(
                        {
                            "split": split,
                            "subset": record.get("hf_subset"),
                            "metric": name,
                        }
                    )
    return undefined


def run_evaluation(config_path, experiment_id, *, output_root=None, device=None):
    config = load_config(config_path, device=device)
    root = repo_path(output_root or "results/runs")
    run_dir = create_run(root, experiment_id)
    manifest = {
        "status": "running",
        "experiment_id": experiment_id,
        "config_path": str(repo_path(config_path)),
        "requested_config": config,
        "environment": environment(),
        "run_dir": str(run_dir),
    }
    atomic_json(run_dir / "run.json", manifest)
    handler = logging.FileHandler(run_dir / "evaluation.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    root_logger.setLevel(min(previous_level, logging.INFO))
    root_logger.addHandler(handler)
    try:
        import mteb
        from src.retrieval.encoder import PrePostPipelineEncoder

        start = time.perf_counter()
        # Benchmark times must measure inference, even if a serving config enables caching.
        encoder = PrePostPipelineEncoder(
            config_path, device=device, cache_embeddings=False
        )
        manifest["model_load_seconds"] = time.perf_counter() - start
        manifest["resolved_config"] = encoder.resolved_config
        atomic_json(run_dir / "run.json", manifest)
        task = mteb.get_task("AppsRetrieval")
        start = time.perf_counter()
        task.load_data()
        manifest["dataset_load_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        result = mteb.evaluate(
            encoder,
            [task],
            encode_kwargs={"batch_size": encoder.default_batch_size},
            cache=None,
            overwrite_strategy="always",
            prediction_folder=run_dir / "predictions",
        )
        evaluation_seconds = time.perf_counter() - start
        results = list(result.task_results)
        if len(results) != 1:
            raise ValueError(f"Expected one task result, got {len(results)}")
        task_result = results[0]
        result_data = task_result.to_dict()
        manifest["undefined_auxiliary_metrics"] = sanitize_auxiliary_metrics(
            result_data
        )
        scores = task_result.scores["test"][0]
        for metric in ("ndcg_at_10", "mrr_at_10"):
            if not math.isfinite(scores[metric]) or not 0 <= scores[metric] <= 1:
                raise ValueError(f"Invalid benchmark metric: {metric}")
        corpus_size = len(task.dataset["default"]["test"]["corpus"])
        num_queries = len(task.dataset["default"]["test"]["queries"])
        corpus_seconds = encoder.encode_seconds["document"]
        summary = {
            "run_id": run_dir.name,
            "experiment_id": experiment_id,
            "model": config["model_name"],
            "backend": config["backend"],
            "device": encoder.device,
            "dtype": encoder.dtype_used,
            "pipeline_fingerprint": encoder.pipeline_fingerprint,
            "ndcg_at_10": scores["ndcg_at_10"],
            "mrr_at_10": scores["mrr_at_10"],
            "num_queries": num_queries,
            "corpus_size": corpus_size,
            "evaluation_seconds": evaluation_seconds,
            "corpus_encode_seconds": corpus_seconds,
            "query_encode_seconds": encoder.encode_seconds["query"],
            "seconds_per_1k_docs": corpus_seconds / corpus_size * 1000
            if corpus_size
            else None,
            "run_dir": str(run_dir),
        }
        atomic_json(run_dir / "appsretrieval_results.json", result_data)
        atomic_json(run_dir / "summary.json", summary)
        manifest.update(
            status="complete", dataset_revision=result_data.get("dataset_revision")
        )
        atomic_json(run_dir / "run.json", manifest)
        # The CSV is a convenience view; complete run artifacts are authoritative.
        try:
            log_evaluation(summary, root / "metrics.csv")
        except (OSError, ValueError) as exc:
            manifest["metrics_log_error"] = str(exc)
            atomic_json(run_dir / "run.json", manifest)
            print(f"Run saved; could not update metrics.csv: {exc}", file=sys.stderr)
        return run_dir
    except Exception:
        failure = traceback.format_exc()
        (run_dir / "failure.log").write_text(failure, encoding="utf-8")
        manifest.update(status="failed", error=failure)
        atomic_json(run_dir / "run.json", manifest)
        raise
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(previous_level)
        handler.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    path = run_evaluation(
        args.config,
        args.experiment_id,
        output_root=args.output_root,
        device=args.device,
    )
    print(f"RUN_DIR={path}")
    print((path / "summary.json").read_text())


if __name__ == "__main__":
    main()
