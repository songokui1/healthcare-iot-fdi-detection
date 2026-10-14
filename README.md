# MedGuard-AI — LLM-Powered Healthcare IoT Threat Analysis

**Public Python research prototype + original six-model WUSTL-EHMS-2020 evaluation**

## Overview

This repository documents an experimental study comparing six machine learning classifiers for binary false data injection (FDI) attack detection in healthcare IoT traffic. It emphasizes predictive performance, inference latency, training time, and SHAP-based interpretability.

**Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Multi-Layer Perceptron (MLP).

## MedGuard-AI research implementation

**The complete MedGuard-AI source implementation is publicly available in this repository.** It compares three approaches to numerical IoMT telemetry: (1) LLM-only assessment, (2) retrieval-augmented LLM analysis (RAG), and (3) a hybrid anomaly gate followed by RAG/LLM investigation.

- [Python modules](medguard/) — data sanitization, summarization, LLM/RAG analysis, and matched evaluation
- [Runnable tests](tests/test_medguard.py) and [GitHub Actions](.github/workflows/test.yml)
- [Streamlit research dashboard](dashboard.py) — aggregate results only
- [Full implementation guide](docs/MEDGUARD_IMPLEMENTATION.md) and [architecture overview](docs/MEDGUARD_AI.md)
- [Research protocol](docs/PROTOCOL.md), [safety boundaries](docs/SECURITY.md), [results-status ledger](docs/RESULTS_STATUS.md), and [new paper blueprint](docs/NEW_PAPER_BLUEPRINT.md)

### Run the public research prototype locally

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m medguard.cli --max-test-rows 20 --output outputs/smoke.json
```

The default uses a **synthetic fixture and deterministic mock**, with no paid API calls. Its scores are **not research findings**. A real OpenAI-powered run requires an authorized local dataset, an API key, and two explicit CLI flags authorizing external telemetry transfer and API spending; see the [implementation guide](docs/MEDGUARD_IMPLEMENTATION.md).

The original dataset, real credentials, and unpublished manuscripts are **not distributed**. There are **no verified real-LLM WUSTL benchmark findings** in this repository.

## Reported results

The following values are transcribed from the original research manuscript; they have **not been independently reproduced** in this repository.

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | ROC-AUC | MCC | Training (s) | Inference (ms/sample) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 92.22 | 72.21 | 61.49 | 66.87 | 0.8719 | 0.624 | 1.722 | 0.0018 |
| Random Forest | 95.25 | 99.22 | 62.76 | 76.62 | 0.9670 | 0.767 | 2.664 | 0.0208 |
| XGBoost | 96.57 | 97.75 | 74.33 | 84.44 | 0.9900 | 0.835 | 0.250 | 0.0024 |
| LightGBM | 98.96 | 97.95 | 93.64 | 95.75 | 0.9984 | 0.952 | 0.429 | 0.0066 |
| CatBoost | 97.82 | 98.84 | 83.60 | 90.61 | 0.9980 | 0.898 | 0.792 | 0.0077 |
| MLP | 95.22 | 96.68 | 64.06 | 77.50 | 0.9550 | 0.765 | 3.877 | 0.0030 |

The manuscript reports the LightGBM confusion matrix as **TN = 2,847, FP = 8, FN = 26, TP = 383** (3,264 observations). These counts are consistent with approximately 98.96% accuracy, 93.64% recall, and 97.95% precision.

## Research workflow

1. Load and preprocess WUSTL-EHMS-2020 traffic data.
2. Train six classifiers using a shared train/test evaluation framework.
3. Evaluate accuracy, precision, recall, F1, ROC-AUC, MCC, training time, and inference latency.
4. Inspect LightGBM's confusion matrix and SHAP feature attributions.
5. Compare deployment trade-offs for resource-constrained healthcare IoT systems.

The manuscript highlights Source Port, TCP Flag, Destination Jitter, Packet Count, Inter-Packet Time, and physiological features such as Temperature, Pulse Rate, Respiratory Rate, and ST Measurements among the influential predictors.

## Contents

- [Reported results (CSV)](results/model_comparison.csv)
- [LightGBM confusion matrix (CSV)](results/lightgbm_confusion_matrix.csv)
- [Methodology and reproducibility limitations](docs/METHODOLOGY.md)

## Research source and distribution

The source code, tests, research design, dashboard, and comparison tables are publicly viewable. The authorized WUSTL dataset, API secrets, original experimental notebooks and weights, and any unpublished manuscript remain outside this repository. Those missing historical artifacts have not been recreated or represented as recovered.

## Reproducibility status

**MedGuard-AI is an executable prototype with an offline synthetic smoke test.** The historical six-model figures above come from the original manuscript, not a fresh experiment. A fair real-LLM vs. LightGBM comparison still requires authorized WUSTL data and a matched evaluation run. The original six-model notebooks, train/test indices, model weights, and SHAP artifacts have not been recovered; no reconstructed code is represented as original.

## Rights and data

**All rights reserved.** Public visibility is for portfolio review only; no general permission to reuse, modify, or redistribute original content is granted. See [LICENSE](LICENSE) for the current notice. Prior licensing grants, if any, may remain effective.

Third-party datasets, publications, and jointly authored material remain subject to their own rights and permissions.

## Research context

Original work titled *Deployment-Oriented Evaluation of Lightweight Machine Learning Models for False Data Injection Detection in Healthcare IoT*. This repository is a technical project summary and does not claim peer-reviewed acceptance.
