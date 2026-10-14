# Research results status ledger

| Component | Status | Evidence |
|---|---|---|
| Historical six-model evaluation including LightGBM | **Manuscript-reported** | Public results/model_comparison.csv |
| Historical LightGBM confusion matrix | **Manuscript-reported** | Public results/lightgbm_confusion_matrix.csv |
| Synthetic fixture, mock direct/RAG/hybrid | **Implemented smoke test** | Tests and CLI; NOT scientific evidence |
| Actual LLM-only model on WUSTL | **Not run** | Requires authorized CSV, API key and consent |
| Actual RAG+LLM on WUSTL | **Not run** | Same |
| Actual hybrid on WUSTL | **Not run** | Same |
| Paired matched-split LightGBM comparison | **Not run** | Requires retraining on same feature set and split |
| v2 explicit WUSTL/CIC manifests and chunked ingestion | **Implemented; synthetic tests passing** | medguard/datasets.py; tests/test_multidataset.py |
| Within-dataset group/time-aware evaluation | **Implemented; synthetic tests passing** | medguard/generalization.py |
| Zero-target-training cross-dataset transfer | **Implemented; synthetic tests passing** | medguard/generalization.py; test fixtures |
| Bounded-memory CSV replay | **Implemented; synthetic tests passing** | medguard/replay.py |
| v2 synthetic scale stress harness | **Implemented; synthetic tests passing** | medguard/scale.py |
| Actual WUSTL, CICIoMT2024, CICIoT2023 transfer results | **Not run** | Requires publisher datasets, original labels and verified manifests |
| Robustness, expert explanation audit, clinical deployment validation | **Not run** | Future independent research phase |

**Do not treat source code, a synthetic smoke test or historical manuscript figures as a new LLM experimental result.**

Previously reported LightGBM performance (original manuscript): accuracy 98.96%, attack recall 93.64%, F1 95.75%, ROC-AUC 0.9984. These values have **not** been freshly reproduced by this framework.

The optional fresh LightGBM implementation and read-only aggregate dashboard are available in this public repository. A successful **synthetic** integration test does not establish WUSTL performance.

**Scope discipline:** Cross-dataset generalization software does not establish clinical transfer. Dataset templates contain intentional placeholder column mappings. All synthetic CI outcomes must remain labeled synthetic.
