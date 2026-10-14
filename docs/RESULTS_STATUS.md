# Research results status ledger

| Component | Status | Evidence |
|---|---|---|
| Historical six-model evaluation including LightGBM | **Manuscript-reported** | Private results/model_comparison.csv |
| Historical LightGBM confusion matrix | **Manuscript-reported** | Private results/lightgbm_confusion_matrix.csv |
| Synthetic fixture, mock direct/RAG/hybrid | **Implemented smoke test** | Tests and CLI; NOT scientific evidence |
| Actual LLM-only model on WUSTL | **Not run** | Requires authorized CSV, API key and consent |
| Actual RAG+LLM on WUSTL | **Not run** | Same |
| Actual hybrid on WUSTL | **Not run** | Same |
| Paired matched-split LightGBM comparison | **Not run** | Requires retraining on same feature set and split |
| Robustness, expert evidence audit, deployment validation | **Not run** | Later research phase |

**Do not treat source code, a synthetic smoke test or historical manuscript figures as a new LLM experimental result.**

Previously reported LightGBM performance (original manuscript): accuracy 98.96%, attack recall 93.64%, F1 95.75%, ROC-AUC 0.9984. These values have **not** been freshly reproduced by this framework.

The optional fresh LightGBM implementation and read-only aggregate dashboard are available in this public repository. A successful **synthetic** integration test does not establish WUSTL performance.
