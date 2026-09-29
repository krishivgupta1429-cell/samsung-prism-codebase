import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TIME_ENCODE = REPO_ROOT / "scripts" / "time_encode.py"
RUN_EVAL = REPO_ROOT / "experiments" / "run_eval.py"
METRICS_LOG = REPO_ROOT / "results" / "metrics_log.csv"
COMPARISON_MD = REPO_ROOT / "results" / "model_comparison.md"

MAX_PROJECTED_SECONDS = 20 * 60

# Each candidate is run in its own subprocess, one at a time, never
# concurrently. Running multiple transformer models at once on shared CPU
# cores makes timing numbers meaningless (they'd all be competing for the
# same cores) and risks a crash from memory pressure once two or three
# multi-hundred-MB models are loaded simultaneously. A subprocess per model
# also guarantees its memory is fully released before the next one loads,
# which matters after seeing one CPU-bound model's memory climb past 1.5GB
# during Phase 1. Running sequentially with no manual step between models
# gets the "don't babysit each one" outcome without that risk.
CANDIDATES = [
    ("configs/jina_code_embeddings_0_5b.json", "phase1_jina_code_embeddings_0_5b"),
    ("configs/granite_embedding_english_r2.json", "phase1_granite_embedding_english_r2"),
    ("configs/e5_base_v2.json", "phase1_e5_base_v2"),
]

# Already computed in an earlier run (see results/minilm_l6_v2/ and
# metrics_log.csv) — included here for the comparison table, not re-run.
EXISTING_RESULTS = [
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "status": "ok",
        "ndcg_at_10": 0.06596,
        "mrr": 0.05581,
        "encode_seconds": 80.74445525021292,
        "seconds_per_1k_docs": 9.21214549346411,
        "notes": "baseline, from earlier run",
    }
]


def build_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    # HF_HUB_ENABLE_HF_TRANSFER is a no-op in this huggingface_hub version
    # (downloads always go through its Xet backend now). That Xet backend
    # stalled indefinitely on this machine even with HF_TOKEN set, so
    # downloads are forced onto the plain HTTPS path instead, which was
    # confirmed reliable earlier in this project.
    env["HF_HUB_DISABLE_XET"] = "1"
    return env


def run_subprocess(cmd, env, timeout):
    return subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout
    )


def short_error(result, exc=None):
    if exc is not None:
        return f"{type(exc).__name__}: {exc}"
    tail = [line for line in result.stderr.strip().splitlines() if line.strip()]
    return tail[-1] if tail else f"exit code {result.returncode} (no stderr)"


def with_retry(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  first attempt failed ({e}); retrying once...", flush=True)
        return fn(*args, **kwargs)


def check_timing(config_path, env):
    result = run_subprocess(
        [sys.executable, str(TIME_ENCODE), "--config", config_path],
        env=env,
        timeout=1200,
    )
    if result.returncode != 0:
        raise RuntimeError(short_error(result))
    match = re.search(r"Projected full-corpus encode time: ([\d.]+) s", result.stdout)
    if not match:
        raise RuntimeError("could not parse projected time from time_encode.py output")
    return float(match.group(1))


def read_last_metrics_row(experiment_id):
    with open(METRICS_LOG, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if row["experiment_id"] == experiment_id:
            return row
    raise RuntimeError(f"no metrics_log.csv row found for experiment_id={experiment_id}")


def run_full_eval(config_path, experiment_id, model_slug, env):
    result = run_subprocess(
        [sys.executable, str(RUN_EVAL), "--config", config_path, "--experiment-id", experiment_id],
        env=env,
        timeout=3600,
    )
    if result.returncode != 0:
        raise RuntimeError(short_error(result))

    results_path = REPO_ROOT / "results" / model_slug / "appsretrieval_results.json"
    with open(results_path) as f:
        data = json.load(f)
    scores = data["scores"]["test"][0]

    metrics_row = read_last_metrics_row(experiment_id)

    return {
        "ndcg_at_10": scores["ndcg_at_10"],
        "mrr": scores["mrr_at_10"],
        "encode_seconds": float(metrics_row["encode_seconds"]),
        "seconds_per_1k_docs": float(metrics_row["seconds_per_1k_docs"]),
    }


def fmt(value, digits=5):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def write_comparison_table(rows):
    def sort_key(row):
        return row["ndcg_at_10"] if isinstance(row["ndcg_at_10"], (int, float)) else -1

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    lines = [
        "# Model comparison — AppsRetrieval",
        "",
        "| model | status | NDCG@10 | MRR | encode_seconds | seconds_per_1k_docs | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows_sorted:
        lines.append(
            f"| {row['model']} | {row['status']} | {fmt(row['ndcg_at_10'])} | "
            f"{fmt(row['mrr'])} | {fmt(row['encode_seconds'], 2)} | "
            f"{fmt(row['seconds_per_1k_docs'], 2)} | {row['notes']} |"
        )

    COMPARISON_MD.write_text("\n".join(lines) + "\n")
    print(f"Wrote {COMPARISON_MD}")


def main():
    print(f"HF_TOKEN is set: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")

    env = build_env()
    rows = list(EXISTING_RESULTS)

    for config_path, experiment_id in CANDIDATES:
        with open(REPO_ROOT / config_path) as f:
            config = json.load(f)
        model_name = config["model_name"]
        model_slug = Path(config_path).stem
        print(f"\n=== {model_name} ===", flush=True)

        try:
            projected_seconds = with_retry(check_timing, config_path, env)
        except Exception as e:
            print(f"  FAILED (timing check): {e}", flush=True)
            rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "ndcg_at_10": None,
                    "mrr": None,
                    "encode_seconds": None,
                    "seconds_per_1k_docs": None,
                    "notes": f"failed — {e}",
                }
            )
            continue

        print(f"  projected full-corpus encode time: {projected_seconds:.1f}s", flush=True)

        if projected_seconds > MAX_PROJECTED_SECONDS:
            print("  SKIPPED — too slow", flush=True)
            rows.append(
                {
                    "model": model_name,
                    "status": "skipped",
                    "ndcg_at_10": None,
                    "mrr": None,
                    "encode_seconds": None,
                    "seconds_per_1k_docs": None,
                    "notes": f"skipped — too slow (projected {projected_seconds:.0f}s)",
                }
            )
            continue

        try:
            result = with_retry(run_full_eval, config_path, experiment_id, model_slug, env)
        except Exception as e:
            print(f"  FAILED (full eval): {e}", flush=True)
            rows.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "ndcg_at_10": None,
                    "mrr": None,
                    "encode_seconds": None,
                    "seconds_per_1k_docs": None,
                    "notes": f"failed — {e}",
                }
            )
            continue

        print(
            f"  OK — NDCG@10={result['ndcg_at_10']}, MRR@10={result['mrr']}",
            flush=True,
        )
        rows.append(
            {
                "model": model_name,
                "status": "ok",
                "ndcg_at_10": result["ndcg_at_10"],
                "mrr": result["mrr"],
                "encode_seconds": result["encode_seconds"],
                "seconds_per_1k_docs": result["seconds_per_1k_docs"],
                "notes": "",
            }
        )

    write_comparison_table(rows)


if __name__ == "__main__":
    main()
