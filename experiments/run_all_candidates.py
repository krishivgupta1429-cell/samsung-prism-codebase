"""Sequential model trials with a wall-clock budget and durable subprocess logs."""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, create_run
from src.retrieval.config import REPO_ROOT, load_config, positive_int, repo_path

CANDIDATES = [
    "configs/jina_code_embeddings_0_5b.json",
    "configs/granite_embedding_english_r2.json",
    "configs/e5_base_v2.json",
]


def completed_results(root):
    """Retain historical scores and hardware metadata without rerunning old models."""
    rows = []
    legacy = root / "metrics_log.csv"
    if legacy.exists():
        with legacy.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "model": row["model"],
                        "status": "complete",
                        "device": row["device"],
                        "ndcg_at_10": float(row["ndcg_at_10"]),
                        "mrr_at_10": float(row["mrr"]),
                        "evaluation_seconds": float(row["encode_seconds"]),
                        "corpus_encode_seconds": None,
                        "source": f"legacy:{row['experiment_id']}",
                        "notes": row["notes"],
                    }
                )
    for path in sorted((root / "runs").glob("*/run.json")):
        run = json.loads(path.read_text())
        if run["status"] == "complete":
            summary = json.loads((path.parent / "summary.json").read_text())
            rows.append(
                {
                    **summary,
                    "status": "complete",
                    "source": str(path.parent.relative_to(root)),
                    "notes": "",
                }
            )
    return rows


def write_comparison_table(rows, destination):
    def number(value):
        return "—" if value is None else f"{value:.5f}"

    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Model comparison — AppsRetrieval",
        "",
        "Historical timings include the entire evaluation. Corpus-only timings are available for new runs. Compare speed only on matched hardware.",
        "",
        "| model | status | device | NDCG@10 | MRR@10 | evaluation_seconds | corpus_encode_seconds | source | notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda r: r.get("ndcg_at_10") if r.get("ndcg_at_10") is not None else -1,
        reverse=True,
    ):
        values = [
            row["model"],
            row["status"],
            row.get("device", "unknown"),
            number(row.get("ndcg_at_10")),
            number(row.get("mrr_at_10")),
            number(row.get("evaluation_seconds")),
            number(row.get("corpus_encode_seconds")),
            row.get("source", ""),
            row.get("notes", ""),
        ]
        lines.append("| " + " | ".join(map(escape, values)) + " |")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".comparison-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_subprocess(command, log_path, timeout):
    # Stream to disk: progress and full tracebacks survive failures and timeouts.
    with Path(log_path).open("w", encoding="utf-8") as stream:
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=CANDIDATES)
    parser.add_argument("--budget-seconds", type=positive_int, default=7200)
    parser.add_argument("--max-projected-seconds", type=positive_int, default=1200)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--comparison-only", action="store_true")
    args = parser.parse_args(argv)
    root = repo_path(args.results_root)
    if args.comparison_only:
        write_comparison_table(completed_results(root), root / "model_comparison.md")
        return 0
    batch = create_run(root / "batches", "candidates")
    deadline = time.monotonic() + args.budget_seconds
    attempts = []
    failed = False
    for index, config_path in enumerate(args.configs):
        attempt = {
            "config": config_path,
            "status": "failed",
            "model": config_path,
            "source": str(batch),
        }
        print(f"Evaluating {config_path}; logs: {batch}", flush=True)
        try:
            config = load_config(config_path)
            attempt["model"] = config["model_name"]
            if time.monotonic() >= deadline:
                raise TimeoutError("Total batch budget exhausted")
            timing_path = batch / f"{index}-timing.json"
            code = run_subprocess(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/time_encode.py"),
                    "--config",
                    config_path,
                    "--output",
                    str(timing_path),
                ],
                batch / f"{index}-timing.log",
                min(1200, deadline - time.monotonic()),
            )
            if code:
                raise RuntimeError(
                    f"Timing process exited {code}; see {index}-timing.log"
                )
            timing = json.loads(timing_path.read_text())
            attempt["device"] = timing["resolved_config"]["device"]
            if timing["projected_corpus_seconds"] > args.max_projected_seconds:
                attempt.update(
                    status="skipped",
                    notes="Projected corpus encoding exceeds configured threshold",
                )
                failed = True
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Total batch budget exhausted")
                code = run_subprocess(
                    [
                        sys.executable,
                        str(REPO_ROOT / "experiments/run_eval.py"),
                        "--config",
                        config_path,
                        "--experiment-id",
                        Path(config_path).stem,
                        "--output-root",
                        str(root / "runs"),
                    ],
                    batch / f"{index}-evaluation.log",
                    min(3600, remaining),
                )
                if code:
                    raise RuntimeError(
                        f"Evaluation process exited {code}; see {index}-evaluation.log"
                    )
                attempt["status"] = "complete"
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.TimeoutExpired,
            TimeoutError,
        ) as exc:
            failed = True
            attempt["notes"] = str(exc)
        attempts.append(attempt)
        atomic_json(batch / "attempts.json", attempts)
    rows = completed_results(root) + [r for r in attempts if r["status"] != "complete"]
    write_comparison_table(rows, root / "model_comparison.md")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
