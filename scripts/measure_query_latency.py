"""Measure both torch and ONNX encoders with recorded query IDs and cold/warm timings."""

import argparse
import random
import statistics
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, environment
from src.retrieval.config import positive_int, repo_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-queries", type=positive_int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=positive_int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    import mteb
    import torch
    from src.retrieval.encoder import PrePostPipelineEncoder

    if args.threads:
        torch.set_num_threads(args.threads)
    start = time.perf_counter()
    encoder = PrePostPipelineEncoder(
        args.config, device=args.device, cache_embeddings=False
    )
    load_seconds = time.perf_counter() - start
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    queries = task.dataset["default"]["test"]["queries"]
    if not len(queries):
        raise ValueError("Cannot profile empty queries")
    indices = random.Random(args.seed).sample(
        range(len(queries)), min(args.num_queries, len(queries))
    )
    samples = [queries[i] for i in indices]

    def measure(text):
        start = time.perf_counter()
        encoder.encode_texts([text], kind="query", batch_size=1)
        return (time.perf_counter() - start) * 1000

    cold_ms = measure(samples[0]["text"])
    # Every sampled input is warmed once; this does not promise unseen shapes are warm.
    warmup_ms = [measure(q["text"]) for q in samples]
    measurements = [
        {"query_id": q["id"], "characters": len(q["text"]), "ms": measure(q["text"])}
        for q in samples
    ]
    times = [q["ms"] for q in measurements]
    result = {
        "resolved_config": encoder.resolved_config,
        "environment": environment(),
        "dataset_revision": task.metadata.dataset["revision"],
        "seed": args.seed,
        "torch_threads": torch.get_num_threads(),
        "model_load_seconds": load_seconds,
        "cold_first_query_ms": cold_ms,
        "warmup_ms": warmup_ms,
        "queries": measurements,
        "min_ms": min(times),
        "mean_ms": statistics.mean(times),
        "max_ms": max(times),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "scope": "query encoding only; each sampled query warmed; excludes search, queueing and reranking",
    }
    atomic_json(repo_path(args.output), result)
    print(
        f"Mean {result['mean_ms']:.1f} ms; p50 {result['p50_ms']:.1f}; p95 {result['p95_ms']:.1f}; cold {cold_ms:.1f}"
    )


if __name__ == "__main__":
    main()
