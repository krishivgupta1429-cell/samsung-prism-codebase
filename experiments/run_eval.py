import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mteb

from src.retrieval.encoder import PrePostPipelineEncoder
from src.retrieval.metrics_log import log_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    model_slug = Path(args.config).stem

    encoder = PrePostPipelineEncoder(args.config)
    device_note = f"device={encoder.device}"
    if encoder.dtype_used:
        device_note += f", dtype={encoder.dtype_used}"
    print(device_note)

    task = mteb.get_task("AppsRetrieval")
    task.load_data()

    batch_size = config.get("batch_size", 64)

    start = time.monotonic()
    result = mteb.evaluate(encoder, [task], encode_kwargs={"batch_size": batch_size})
    encode_seconds = time.monotonic() - start

    task_result = list(result.task_results)[0]

    output_dir = Path("results") / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "appsretrieval_results.json"
    with open(output_path, "w") as f:
        json.dump(task_result.to_dict(), f, indent=2, default=str)

    scores = task_result.scores["test"][0]
    ndcg_at_10 = scores["ndcg_at_10"]
    mrr = scores["mrr_at_10"]

    corpus_size = len(task.dataset["default"]["test"]["corpus"])
    num_queries = len(task.dataset["default"]["test"]["queries"])

    log_run(
        experiment_id=args.experiment_id,
        model=config["model_name"],
        pipeline_config=args.config,
        ndcg_at_10=ndcg_at_10,
        mrr=mrr,
        num_queries=num_queries,
        corpus_size=corpus_size,
        encode_seconds=encode_seconds,
        notes=f"task={task_result.task_name}, main_score={scores['main_score']}, {device_note}",
    )

    print(f"Wrote {output_path}")
    print(f"ndcg_at_10: {ndcg_at_10}")
    print(f"mrr_at_10: {mrr}")
    print(f"encode_seconds: {encode_seconds:.2f}")
    print(f"num_queries: {num_queries}, corpus_size: {corpus_size}")


if __name__ == "__main__":
    main()
