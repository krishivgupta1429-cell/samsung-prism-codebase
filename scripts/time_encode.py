import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mteb

from src.retrieval.encoder import PrePostPipelineEncoder

SAMPLE_SIZE = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gte_modernbert_base.json")
    args = parser.parse_args()

    encoder = PrePostPipelineEncoder(args.config)

    task = mteb.get_task("AppsRetrieval")
    task.load_data()

    corpus = task.dataset["default"]["test"]["corpus"]
    corpus_size = len(corpus)

    sample_size = min(SAMPLE_SIZE, corpus_size)
    sample_texts = corpus["text"][:sample_size]

    processed = [
        encoder.document_prefix + encoder.preprocess_document(text)
        for text in sample_texts
    ]

    start = time.monotonic()
    encoder.model.encode(
        processed,
        batch_size=encoder.default_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.monotonic() - start

    seconds_per_1k = (elapsed / sample_size) * 1000
    projected_full_seconds = seconds_per_1k * corpus_size / 1000
    projected_full_hours = projected_full_seconds / 3600

    print(f"Model: {encoder.model_name}")
    print(f"Real corpus size (test split): {corpus_size}")
    print(f"Sample size encoded: {sample_size}")
    print(f"Elapsed for sample: {elapsed:.2f} s")
    print(f"Seconds per 1,000 documents: {seconds_per_1k:.2f}")
    print(
        f"Projected full-corpus encode time: {projected_full_seconds:.1f} s "
        f"({projected_full_hours:.3f} hours)"
    )


if __name__ == "__main__":
    main()
