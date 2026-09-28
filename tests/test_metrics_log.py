import csv

from src.retrieval.metrics_log import HEADER, log_run


def test_log_run_creates_header_and_appends_rows(tmp_path):
    path = tmp_path / "metrics_log.csv"

    log_run(
        experiment_id="exp1",
        model="test-model",
        pipeline_config="baseline",
        ndcg_at_10=0.5,
        mrr=0.4,
        num_queries=100,
        corpus_size=1000,
        encode_seconds=10.0,
        path=str(path),
    )
    log_run(
        experiment_id="exp2",
        model="test-model",
        pipeline_config="baseline",
        ndcg_at_10=0.6,
        mrr=0.45,
        num_queries=100,
        corpus_size=1000,
        encode_seconds=12.0,
        path=str(path),
    )

    with open(path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert rows[0] == HEADER
    assert len(rows) == 3
