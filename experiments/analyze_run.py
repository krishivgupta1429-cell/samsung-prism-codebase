"""Inspect saved MTEB predictions without rerunning a model."""

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retrieval.artifacts import atomic_json
from src.retrieval.config import repo_path


def analyze(predictions, queries, corpus, relevant_docs):
    by_text = defaultdict(list)
    for doc in corpus:
        by_text[doc["text"]].append(doc["id"])
    duplicates = {
        doc_id: ids for ids in by_text.values() if len(ids) > 1 for doc_id in ids
    }
    records = []
    for query in queries:
        qid = query["id"]
        positives = [doc for doc, score in relevant_docs[qid].items() if score > 0]
        if len(positives) != 1:
            raise ValueError("APPS analysis requires one positive document per query")
        gold = positives[0]
        scores = predictions[qid]
        rank_min = rank_max = None
        if gold in scores:
            rank_min = 1 + sum(score > scores[gold] for score in scores.values())
            rank_max = sum(score >= scores[gold] for score in scores.values())
        records.append(
            {
                "query_id": qid,
                "query": query["text"],
                "query_characters": len(query["text"]),
                "gold_document_id": gold,
                "rank_min": rank_min,
                "rank_max": rank_max,
                "identical_document_ids": duplicates.get(gold, []),
                "top_document_ids": sorted(scores, key=lambda doc: (-scores[doc], doc))[
                    :20
                ],
            }
        )
    return {
        "num_queries": len(records),
        "queries_with_duplicate_gold": sum(
            bool(r["identical_document_ids"]) for r in records
        ),
        "rank_note": "Rank intervals account for tied scores; missing gold is outside the saved candidate pool. Top-document display breaks ties by ID.",
        "queries": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_dir = repo_path(args.run_dir)
    manifest = json.loads((run_dir / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Analyze a completed run")
    predictions = json.loads(
        (run_dir / "predictions/AppsRetrieval_predictions.json").read_text()
    )["default"]["test"]
    import mteb

    task = mteb.get_task("AppsRetrieval")
    if task.metadata.dataset["revision"] != manifest["dataset_revision"]:
        raise ValueError(
            "Installed task dataset revision differs from the recorded run"
        )
    task.load_data()
    split = task.dataset["default"]["test"]
    report = analyze(
        predictions, split["queries"], split["corpus"], split["relevant_docs"]
    )
    report.update(
        run_dir=str(run_dir),
        pipeline_fingerprint=manifest["resolved_config"]["pipeline_fingerprint"],
        dataset_revision=manifest["dataset_revision"],
    )
    atomic_json(repo_path(args.output), report)
    print(
        f"Analyzed {report['num_queries']} queries; {report['queries_with_duplicate_gold']} have duplicate gold code"
    )


if __name__ == "__main__":
    main()
