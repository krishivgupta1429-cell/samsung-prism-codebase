# Model comparison — AppsRetrieval

| model | status | NDCG@10 | MRR | encode_seconds | seconds_per_1k_docs | notes |
|---|---|---|---|---|---|---|
| intfloat/e5-base-v2 | ok | 0.11523 | 0.09878 | 787.74 | 89.87 |  |
| sentence-transformers/all-MiniLM-L6-v2 | ok | 0.06596 | 0.05581 | 80.74 | 9.21 | baseline, from earlier run |
| jinaai/jina-code-embeddings-0.5b | skipped | — | — | — | — | skipped — too slow (projected 48726s) |
| ibm-granite/granite-embedding-english-r2 | failed | — | — | — | — | failed — Command '['/Users/krishivgupta/Desktop/Samsung_Prism/.venv/bin/python', '/Users/krishivgupta/Desktop/Samsung_Prism/scripts/time_encode.py', '--config', 'configs/granite_embedding_english_r2.json']' timed out after 1200 seconds |
