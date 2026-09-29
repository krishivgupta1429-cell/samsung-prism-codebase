# Model comparison — AppsRetrieval

| model | status | device | NDCG@10 | MRR | encode_seconds | seconds_per_1k_docs | notes |
|---|---|---|---|---|---|---|---|
| jinaai/jina-code-embeddings-0.5b | ok | cuda | 0.84083 | 0.81055 | 1431.18 | 163.28 | RTX 4050 Laptop GPU (6GB); CPU timing-check projected ~13.5h, infeasible on CPU |
| Salesforce/SFR-Embedding-Code-400M_R | ok | cuda | 0.49627 | 0.44957 | 1721.35 | 196.39 | RTX 4050 Laptop GPU (6GB); trust_remote_code=True, bfloat16; run from isolated `.venv-sfr-test` (transformers==4.49.0) — main `.venv` untouched |
| ibm-granite/granite-embedding-english-r2 | ok | cuda | 0.13993 | 0.11960 | 751.33 | 85.72 | RTX 4050 Laptop GPU (6GB); timed out (1200s) on CPU on a prior machine, same ModernBERT architecture as gte-modernbert-base |
| intfloat/e5-base-v2 | ok | cpu | 0.11523 | 0.09878 | 787.74 | 89.87 |  |
| sentence-transformers/all-MiniLM-L6-v2 | ok | cpu | 0.06596 | 0.05581 | 80.74 | 9.21 | baseline |
