import csv
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from .config import REPO_ROOT

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
    "device",
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
            cwd=REPO_ROOT,
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
    device="",
    notes="",
    path="results/metrics_log.csv",
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    seconds_per_1k_docs = (encode_seconds / corpus_size) * 1000 if corpus_size else 0

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
        "device": device,
        "git_commit": _git_commit(),
        "notes": notes,
    }

    append_row(path, HEADER, row)


def append_row(path, header, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=30):
        nonempty = path.exists() and path.stat().st_size > 0
        if nonempty:
            with path.open(newline="", encoding="utf-8") as stream:
                if next(csv.reader(stream)) != header:
                    raise ValueError(f"Unexpected CSV schema: {path}")
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            if not nonempty:
                writer.writeheader()
            writer.writerow(row)


EVALUATION_HEADER = [
    "run_id",
    "experiment_id",
    "model",
    "backend",
    "device",
    "dtype",
    "pipeline_fingerprint",
    "ndcg_at_10",
    "mrr_at_10",
    "num_queries",
    "corpus_size",
    "evaluation_seconds",
    "corpus_encode_seconds",
    "query_encode_seconds",
    "seconds_per_1k_docs",
    "run_dir",
]


def log_evaluation(summary, path):
    append_row(
        path, EVALUATION_HEADER, {key: summary[key] for key in EVALUATION_HEADER}
    )
