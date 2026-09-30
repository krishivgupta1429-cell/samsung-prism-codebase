"""Seeded corpus-only timing sample; projection is an estimate, not a run budget."""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, environment
from src.retrieval.config import positive_int, repo_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gte_modernbert_base.json")
    parser.add_argument("--sample-size", type=positive_int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    import mteb
    from src.retrieval.encoder import PrePostPipelineEncoder

    encoder = PrePostPipelineEncoder(
        args.config, device=args.device, cache_embeddings=False
    )
    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    corpus = task.dataset["default"]["test"]["corpus"]
    if not len(corpus):
        raise ValueError("Cannot profile an empty corpus")
    indices = random.Random(args.seed).sample(
        range(len(corpus)), min(args.sample_size, len(corpus))
    )
    texts = [corpus[i]["text"] for i in indices]
    start = time.perf_counter()
    encoder.encode_texts(texts[:1], kind="document", batch_size=1)
    warmup = time.perf_counter() - start
    start = time.perf_counter()
    encoder.encode_texts(texts, kind="document")
    elapsed = time.perf_counter() - start
    result = {
        "resolved_config": encoder.resolved_config,
        "environment": environment(),
        "dataset_revision": task.metadata.dataset["revision"],
        "seed": args.seed,
        "sample_indices": indices,
        "document_ids": [corpus[i]["id"] for i in indices],
        "sample_size": len(texts),
        "corpus_size": len(corpus),
        "warmup_seconds": warmup,
        "sample_encode_seconds": elapsed,
        "seconds_per_1k_docs": elapsed / len(texts) * 1000,
        "projected_corpus_seconds": elapsed / len(texts) * len(corpus),
    }
    if args.output:
        atomic_json(repo_path(args.output), result)
    print(
        f"Projected full-corpus encode time: {result['projected_corpus_seconds']:.1f} s"
    )
    print(
        f"Sample: {len(texts)} randomly selected documents, seed={args.seed}; excludes query encoding and search"
    )


if __name__ == "__main__":
    main()
