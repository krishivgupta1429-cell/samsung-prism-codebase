"""Compare float/quantized retrieval on identical seeded queries and the full corpus.

Each encoder runs in a separate process. The comparison retains IDs, raw vectors,
resolved configs and rankings, including INT8 queries against reference vectors.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json, create_run, environment
from src.retrieval.config import positive_int, repo_path
from src.retrieval.search import cosine_top_k, single_answer_metrics


def encode_job(path):
    import numpy as np
    from src.retrieval.encoder import PrePostPipelineEncoder

    job = json.loads(Path(path).read_text())
    data = json.loads(Path(job["inputs"]).read_text())
    encoder = PrePostPipelineEncoder(
        job["config"], device=job.get("device"), cache_embeddings=False
    )
    print(f"Loaded {encoder.model_name} on {encoder.device}", flush=True)
    output = Path(job["output"])
    output.mkdir()
    np.save(output / "queries.npy", encoder.encode_texts(data["queries"], kind="query"))
    print(f"Encoded {len(data['queries'])} queries", flush=True)
    np.save(
        output / "documents.npy",
        encoder.encode_texts(data["documents"], kind="document"),
    )
    print(f"Encoded {len(data['documents'])} documents", flush=True)
    atomic_json(
        output / "encoder.json",
        {
            "resolved_config": encoder.resolved_config,
            "environment": environment(),
            "encode_seconds": dict(encoder.encode_seconds),
        },
    )


def main(argv=None):
    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encode-job")
    parser.add_argument("--reference-config")
    parser.add_argument("--candidate-config")
    parser.add_argument("--num-queries", type=positive_int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="results/comparisons")
    parser.add_argument("--reference-device")
    parser.add_argument("--candidate-device")
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=positive_int, default=7200)
    args = parser.parse_args(argv)
    if args.encode_job:
        encode_job(args.encode_job)
        return
    if not args.reference_config or not args.candidate_config:
        parser.error("--reference-config and --candidate-config are required")
    import mteb

    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    data = task.dataset["default"]["test"]
    queries, corpus = data["queries"], data["corpus"]
    indices = random.Random(args.seed).sample(
        range(len(queries)), min(args.num_queries, len(queries))
    )
    query_ids = [queries[i]["id"] for i in indices]
    document_ids = list(corpus["id"])
    output = create_run(repo_path(args.output_root), "encoder-comparison")
    manifest = {
        "status": "running",
        "seed": args.seed,
        "query_ids": query_ids,
        "document_ids": document_ids,
        "dataset_revision": task.metadata.dataset["revision"],
        "num_queries": len(query_ids),
        "corpus_size": len(document_ids),
        "environment": environment(),
        "scope": (
            "all test queries against full corpus; exact cosine diagnostic"
            if len(query_ids) == len(queries)
            else "sampled test queries against full corpus; exact cosine diagnostic"
        ),
    }
    atomic_json(output / "comparison.json", manifest)
    atomic_json(
        output / "inputs.json",
        {
            "queries": [queries[i]["text"] for i in indices],
            "documents": list(corpus["text"]),
        },
    )
    try:
        for role, config, device, interpreter in [
            (
                "reference",
                args.reference_config,
                args.reference_device,
                args.reference_python,
            ),
            (
                "candidate",
                args.candidate_config,
                args.candidate_device,
                args.candidate_python,
            ),
        ]:
            job = output / f"{role}-job.json"
            atomic_json(
                job,
                {
                    "inputs": str(output / "inputs.json"),
                    "config": str(repo_path(config)),
                    "device": device,
                    "output": str(output / role),
                    "python": interpreter,
                },
            )
            with (output / f"{role}.log").open("w") as stream:
                subprocess.run(
                    [
                        interpreter,
                        str(Path(__file__).resolve()),
                        "--encode-job",
                        str(job),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=args.timeout_seconds,
                )
        results = {}
        for name, query_role, document_role in [
            ("reference", "reference", "reference"),
            ("candidate_queries_reference_corpus", "candidate", "reference"),
            ("candidate", "candidate", "candidate"),
        ]:
            ranked = cosine_top_k(
                np.load(output / query_role / "queries.npy"),
                np.load(output / document_role / "documents.npy"),
                document_ids,
                k=100,
            )
            result = single_answer_metrics(ranked, query_ids, data["relevant_docs"])
            atomic_json(output / f"{name}-rankings.json", result)
            results[name] = {k: result[k] for k in ("ndcg_at_10", "mrr_at_10")}
        manifest.update(status="complete", results=results)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        atomic_json(output / "comparison.json", manifest)
    print(json.dumps(results, indent=2))
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
