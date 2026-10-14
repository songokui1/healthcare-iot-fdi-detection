# MedGuard-AI
### Multi-Dataset Healthcare IoT Threat Detection and AI-Assisted Investigation

[![Research CI](https://github.com/songokui1/healthcare-iot-fdi-detection/actions/workflows/test.yml/badge.svg)](https://github.com/songokui1/healthcare-iot-fdi-detection/actions/workflows/test.yml)

**A public research software project exploring how conventional detectors, large language models, retrieval-augmented reasoning, and selective anomaly screening behave under heterogeneous IoT/IoMT traffic.**

> **Research maturity:** Working engineering prototype and synthetic CI—not a validated intrusion detection product. Real cross-dataset and real-LLM benchmarks are **not yet claimed**. This repository contains neither third-party dataset copies nor private API credentials. **All rights reserved; see [LICENSE](LICENSE).**

## Research questions

1. **Generalization:** Do detectors trained on one device/protocol/dataset remain reliable on unseen IoT/medical IoT distributions?
2. **AI assistance:** Can an LLM produce useful evidence-grounded investigation notes without sacrificing attack recall or hallucinating support?
3. **Operational trade-offs:** When is a fast anomaly gate worthwhile compared with calling an LLM on every event?
4. **Reproducibility:** Can we evaluate unseen sessions/devices and report missing attacks, feature drift, throughput, uncertainty and compute costs without leaking test labels?

## System architecture

~~~mermaid
flowchart TD
    A[Locally authorized IoMT/IoT CSVs] --> B[Explicit schema and label manifests]
    B --> C[Leakage guard + bounded-memory ingestion]
    C --> D[Train-only preprocessing]
    D --> E[Within-domain tests]
    D --> F[Cross-dataset transfer]
    D --> G[CSV replay and throughput]
    D --> H[Anomaly gate]
    D --> I[LLM-only / RAG + LLM]
    H --> J[LLM investigation of flagged events]
    E --> K[Aggregate research reports]
    F --> K
    G --> K
    I --> K
    J --> K
    K --> L[Local Streamlit dashboard]
~~~

## What's implemented

| Research subsystem | Capability |
| --- | --- |
| **LLM / RAG / hybrid evaluation** | Structured JSON classification, bounded numeric summaries, curated local TF-IDF retrieval, evidence/citation validation, and hybrid anomaly gating |
| **Multi-dataset adapters** | Explicit CSV schema, attack label and physical-unit manifests; uniform bounded-memory sampling; audit counts |
| **Leakage-conscious experiments** | Train-only imputation; stratified, device/session-disjoint or chronological splits where actual metadata supports them |
| **Generalization** | Source-only model fitting and unseen-target evaluation on explicitly matched canonical features and units |
| **Reliability statistics** | Confusion matrix, attack recall/precision/F1, MCC, ROC-AUC, PR-AUC, false negatives, cluster-aware/bootstrap intervals |
| **Large-scale engineering** | Bounded-memory CSV replay measuring events per second and batch-inference latency |
| **Dashboard and CI** | Aggregate-only Streamlit results display, synthetic fixtures, unit tests, security tests, automated smoke runs |

The original LightGBM six-model comparison is preserved as **historical manuscript-reported research**, not passed off as the result of the new model pipeline.

## Try it in two commands — no data download or API key

Python 3.10+:

~~~bash
python -m pip install -e '.[dev,baseline]'
python -m medguard.research_cli demo --output outputs/v2-demo.json
~~~

This generates two synthetic IoT-style datasets with differing column names, then runs a within-dataset group split, a cross-dataset transfer experiment, and 300-event local batch replay. It is an **engineering demonstration**, not real attack-detection evidence.

To visualize:

~~~bash
python -m pip install -e '.[dashboard]'
streamlit run dashboard.py
~~~

To run all tests:

~~~bash
python -m pytest -q
~~~

### Local large-file stress harness (synthetic only)

The scale harness generates CSVs incrementally and measures bounded-memory sampling plus source-trained batch replay. These numbers are machine-specific **engineering smoke measurements**, not clinical throughput or measured LLM serving capacity.

~~~bash
python -m medguard.scale --rows 100000 --source-cap 5000 --replay-cap 25000 --output outputs/synthetic-scale.json
~~~

The dashboard can open both v2 demonstration and synthetic scale reports. Source datasets are generated in a temporary directory and automatically discarded.

**Original LLM pilot interface** (synthetic, deterministic mock by default):

~~~bash
python -m medguard.cli --max-test-rows 20 --output outputs/mock-llm.json
~~~

Real provider calls require an authorized dataset, locally configured API credentials, and **two explicit consent flags** authorizing numerical telemetry transfer and billable API calls. No CI workflow makes billable API calls.

## Expanding beyond WUSTL

| Dataset | Intended study role | Official source |
| --- | --- | --- |
| **WUSTL-EHMS-2020** | Medical sensor + flow telemetry; original research foundation | [Washington University](https://www.cs.wustl.edu/~jain/ehms/index.html) |
| **CICIoMT2024** | New medical devices, attack families, Wi-Fi/MQTT/Bluetooth contexts | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) |
| **CICIoT2023** | Larger heterogeneous IoT traffic and scale / cross-domain tests | [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iotdataset-2023.html) |

The repository includes [example manifest templates](dataset_profiles/) for these datasets. **They deliberately contain placeholder headers, labels and/or units that must be verified against the downloaded source.** Dataset schemas differ across protocols and source distributions; no guessed feature mapping or fabricated label conversion is shipped as validated.

To run real research after verifying the profiles and acquiring authorized data:

~~~bash
python -m medguard.research_cli inspect --source dataset_profiles/wustl_ehms_2020.json --sha256
python -m medguard.research_cli evaluate --source dataset_profiles/wustl_ehms_2020.json --split group --output outputs/wustl.json
python -m medguard.research_cli transfer --source dataset_profiles/wustl_ehms_2020.json --target dataset_profiles/ciciomt2024_wifi_mqtt.json --output outputs/transfer.json
python -m medguard.research_cli replay --source dataset_profiles/ciciomt2024_wifi_mqtt.json --target dataset_profiles/ciciot2023.json --max-events 100000 --output outputs/replay.json
~~~

These are **interface examples**, not executable commands with the unmodified \`.example.json\` templates. A group split specifically requires a real group identifier. See [v2 methodology](docs/V2_RESEARCH_PROTOCOL.md) for responsible data preparation.

## Original research: conventional model comparison

The values below are transcribed from the earlier *Deployment-Oriented Evaluation of Lightweight Machine Learning Models for False Data Injection Detection in Healthcare IoT* manuscript. They have **not** been independently reproduced by MedGuard-AI.

| Model | Accuracy (%) | Recall (%) | F1 (%) | ROC-AUC | Training (s) | Inference (ms/sample) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 92.22 | 61.49 | 66.87 | 0.8719 | 1.722 | 0.0018 |
| Random Forest | 95.25 | 62.76 | 76.62 | 0.9670 | 2.664 | 0.0208 |
| XGBoost | 96.57 | 74.33 | 84.44 | 0.9900 | 0.250 | 0.0024 |
| **LightGBM** | **98.96** | **93.64** | **95.75** | **0.9984** | 0.429 | 0.0066 |
| CatBoost | 97.82 | 83.60 | 90.61 | 0.9980 | 0.792 | 0.0077 |
| MLP | 95.22 | 64.06 | 77.50 | 0.9550 | 3.877 | 0.0030 |

The original manuscript reports a LightGBM confusion matrix of TN 2,847, FP 8, FN 26, TP 383. **This is not a new, paired LLM-versus-LightGBM result.**

## Project map

- [medguard/](medguard/) — LLM/RAG/hybrid analysis, dataset registry, splits, transfer, replay and reporting modules
- [tests/](tests/) — leakage, label mapping, separation, provenance and integration tests
- [dataset_profiles/](dataset_profiles/) — schemas that must be configured from actual downloaded datasets
- [dashboard.py](dashboard.py) — read-only, aggregate-only Streamlit UI
- [docs/V2_RESEARCH_PROTOCOL.md](docs/V2_RESEARCH_PROTOCOL.md) — scientifically grounded expansion protocol
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — initial LLM research design
- [docs/RESULTS_STATUS.md](docs/RESULTS_STATUS.md) — completed vs unrun evidence ledger
- [docs/NEW_PAPER_BLUEPRINT.md](docs/NEW_PAPER_BLUEPRINT.md) — prospective manuscript structure without invented findings

## Important limitations

A conventional model can outperform an LLM on tabular network features; the evidence must decide. WUSTL's binary attack classification is **not automatically an FDI-only endpoint**, and CICIoMT/CICIoT attack families do not share equivalent ground truths by default. Network features may encode capture/session artifacts or MAC-derived label proxies. Real results need dataset-specific label audits, capture/device isolation, external validation and expert checks of LLM explanations. Synthetic scores, bootstrap intervals over correlated flows, and local batch replay cannot establish hospital deployment readiness.

**Rights:** Public visibility permits portfolio inspection, not unrestricted reuse. See [LICENSE](LICENSE). Previously granted rights, if any, may remain effective, and third-party datasets retain their own licenses and citation requirements.
