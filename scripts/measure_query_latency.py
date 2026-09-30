import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mteb

from src.retrieval.encoder import PrePostPipelineEncoder

NUM_QUERIES = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    encoder = PrePostPipelineEncoder(args.config)
    encoder.model.to(args.device)
    encoder.device = args.device
    print(f"Model: {encoder.model_name}")
    print(f"Device (forced): {args.device}")

    task = mteb.get_task("AppsRetrieval")
    task.load_data()
    queries = task.dataset["default"]["test"]["queries"]["text"][: args.num_queries]
    texts = [encoder.query_prefix + encoder.preprocess_query(q) for q in queries]

    # One untimed warm-up call: a live service pays this once at startup, not
    # per query, so excluding it reflects steady-state serving latency.
    encoder.model.encode(
        [texts[0]], batch_size=1, show_progress_bar=False, convert_to_numpy=True
    )

    times_ms = []
    for text in texts:
        start = time.monotonic()
        encoder.model.encode(
            [text], batch_size=1, show_progress_bar=False, convert_to_numpy=True
        )
        times_ms.append((time.monotonic() - start) * 1000)

    print(f"Queries measured: {len(times_ms)}")
    for i, t in enumerate(times_ms):
        print(f"  query {i}: {t:.1f} ms")
    print(f"min: {min(times_ms):.1f} ms")
    print(f"avg: {statistics.mean(times_ms):.1f} ms")
    print(f"max: {max(times_ms):.1f} ms")


if __name__ == "__main__":
    main()
