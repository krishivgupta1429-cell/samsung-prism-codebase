import csv
import os
import subprocess
from datetime import datetime, timezone

HEADER = [
    "timestamp",
    "experiment_id",
    "model",
    "pipeline_config",
    "ndcg_at_10",
    "mrr",
    "num_queries",
    "corpus_size",
    "encode_seconds",
    "seconds_per_1k_docs",
    "git_commit",
    "notes",
]


def _git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def log_run(
    experiment_id,
    model,
    pipeline_config,
    ndcg_at_10,
    mrr,
    num_queries,
    corpus_size,
    encode_seconds,
    notes="",
    path="results/metrics_log.csv",
):
    file_exists = os.path.isfile(path)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    seconds_per_1k_docs = (
        (encode_seconds / corpus_size) * 1000 if corpus_size else 0
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "model": model,
        "pipeline_config": pipeline_config,
        "ndcg_at_10": ndcg_at_10,
        "mrr": mrr,
        "num_queries": num_queries,
        "corpus_size": corpus_size,
        "encode_seconds": encode_seconds,
        "seconds_per_1k_docs": seconds_per_1k_docs,
        "git_commit": _git_commit(),
        "notes": notes,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
