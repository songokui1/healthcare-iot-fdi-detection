# Healthcare IoT False Data Injection Detection

**Deployment-oriented comparison of lightweight machine learning models using WUSTL-EHMS-2020**

## Overview

This repository documents an experimental study comparing six machine learning classifiers for binary false data injection (FDI) attack detection in healthcare IoT traffic. It emphasizes predictive performance, inference latency, training time, and SHAP-based interpretability.

**Models:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Multi-Layer Perceptron (MLP).

## New research extension: MedGuard-AI

**An experimental LLM + retrieval + hybrid IoMT threat-investigation framework is now implemented in a separate PRIVATE research repository.** The [public research overview](docs/MEDGUARD_AI.md) describes the three approaches, system architecture, evaluation protocol, and current validation status. No real LLM benchmark performance is claimed yet, and the private code/data are not published.

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

## Public versus private materials

This public repository is a recruiter-facing showcase of reported model performance and deployment trade-offs. The reported metrics are intentionally public; the original manuscript, experimental notebooks, and trained artifacts are not published here. A separate private repository holds a copy of the comparison tables for ongoing work.

## Reproducibility status

**This is a manuscript-grounded research results archive, not yet an executable reproduction.** The original experimental notebooks, dataset copy, preprocessing configuration, train/test indices, trained models, and SHAP artifacts have not been recovered and verified for publication here. No fabricated training code or reconstructed results are presented as original.

## Rights and data

**All rights reserved.** Public visibility is for portfolio review only; no general permission to reuse, modify, or redistribute original content is granted. See [LICENSE](LICENSE) for the current notice. Prior licensing grants, if any, may remain effective.

Third-party datasets, publications, and jointly authored material remain subject to their own rights and permissions.

## Research context

Original work titled *Deployment-Oriented Evaluation of Lightweight Machine Learning Models for False Data Injection Detection in Healthcare IoT*. This repository is a technical project summary and does not claim peer-reviewed acceptance.
