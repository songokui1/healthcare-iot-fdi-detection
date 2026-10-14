# MedGuard-AI — LLM-Powered Healthcare IoT Threat Analysis

**PUBLIC RESEARCH REPOSITORY · EXPERIMENTAL PROTOTYPE · ALL RIGHTS RESERVED**

An executable research framework comparing (1) **LLM-only threat assessment**, (2) **RAG-grounded LLM assessment**, and (3) **hybrid anomaly screening + RAG/LLM investigation** on an authorized local copy of WUSTL-EHMS-2020.

This is an experimental research prototype—not a clinical device, autonomous intrusion-prevention product, or validated healthcare decision tool.

## Architecture

~~~mermaid
flowchart TD
    A[Local WUSTL-EHMS CSV] --> B[Leakage-aware numeric features]
    B --> C[Stratified train/test split]
    C --> D[Training-benign statistics]
    C --> E[IsolationForest anomaly gate]
    D --> F[Bounded numeric event summaries]
    F --> G[Approach 1: LLM-only]
    F --> H[Approach 2: local RAG + LLM]
    E --> I{Suspicious?}
    I -->|Yes| J[Approach 3: RAG + LLM]
    I -->|No| K[Benign prediction; missed attacks measured]
    G --> L[Same held-out test events]
    H --> L
    J --> L
    K --> L
    L --> M[Aggregate metrics, latency, API calls, tokens]
~~~

## Implemented components

- CSV loader filters labels, attack indicators, identifiers, and nonnumeric free text.
- Train-only imputation and benign-feature reference ranges.
- LLM assessments are constrained to JSON; evidence and citations must reference supplied data.
- Local TF-IDF retrieval over a curated knowledge corpus.
- IsolationForest gate calibrated using training-benign data and evaluated for missed attacks.
- Consistent comparison of three approaches over the *same held-out subset*.
- Confusion counts, accuracy, balanced accuracy, attack precision/recall/F1, FP/FN rates, API tokens/calls, and measured wall-clock time.
- A deterministic **mock for smoke tests only**, optional OpenAI provider, and offline unit tests.

## Current research status

**No live LLM experiments have been executed or validated on WUSTL-EHMS-2020 in this repository.** The historical LightGBM results stored under results/ are from the *earlier research manuscript* and are NOT a same-split baseline for this prototype.

## Project availability

This source implementation, its tests and demonstration dashboard are now publicly viewable. External datasets, private credentials, original unpublished manuscripts, and real experimental artifacts are **not** included.

## Quick start (no network/API key needed)

Requirements: Python 3.10+.

~~~bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m medguard.cli --max-test-rows 20 --output outputs/smoke.json
~~~

The mock test runs on artificially generated non-clinical data. Its classifications are **not scientific results**.

## Run on authorized local WUSTL data (still offline mock)

Acquire the dataset from the source: https://www.cs.wustl.edu/~jain/ehms/index.html

Keep the CSV in your local gitignored data/ folder; verify the target label meaning and exclude any other proxy columns.

~~~bash
python -m medguard.cli --csv data/wustl.csv --label Label \
  --exclude 'Attack Category' --max-test-rows 30 \
  --output outputs/wustl_mock_smoke.json
~~~

This only checks data processing. Mock metrics are never valid research findings.

## Real LLM comparison: explicit telemetry and billing opt-in

Only after authorization to send testbed numeric summaries to an external API:

~~~bash
python -m pip install -e '.[llm,dev]'
export OPENAI_API_KEY='YOUR_SECRET_KEY'
python -m medguard.cli --csv data/wustl.csv --label Label \
  --provider openai --model gpt-4.1-mini --seed 42 \
  --max-test-rows 30 --allow-external-telemetry --confirm-api-spend \
  --output outputs/llm_pilot.json
~~~

The three approaches can make up to three times max-test-rows API calls; the gate reduces the third approach's calls. Provider cost varies; no dollar values are invented. Keep real patient records and identifiers out of all experiments.

## Optional local results dashboard

The read-only dashboard uses **aggregate metrics only**; it never loads individual patient/network records.

~~~bash
python -m pip install -e '.[dashboard]'
python -m medguard.cli --max-test-rows 20 --output outputs/aggregate.json
streamlit run dashboard.py
~~~

A red warning labels the default synthetic mock results as **NOT actual LLM evidence**.

## Fairer conventional baseline (optional)

After installing the optional LightGBM dependency, the experiment can *train a new baseline on the exact same train/test split*, instead of comparing against historical manuscript numbers:

~~~bash
python -m pip install -e '.[baseline]'
python -m medguard.cli --csv data/wustl.csv --label Label \
  --include-lightgbm --output outputs/baseline_and_mock.json
~~~

The example still uses a mock, so its LLM columns are smoke tests only. For a valid comparison, run the same option with the actual provider and explicit consent flags. Do not claim FDI-only detection unless the target labels have been validated.

## Critical scientific caveats

- WUSTL-EHMS attack labels cover more than one type of attack; binary intrusion classification is **not** automatically FDI-only evaluation.
- Ground truth was generated using attacker MAC addresses. Excluding identifiers is necessary but may not remove all correlated leakage proxies.
- Random row split may overestimate generalization with shared collection sessions and duplicates.
- Small pilot subsets and LLM confidence ratings cannot support publication-grade accuracy or calibration claims.
- Explanations are model-generated interpretations, not proof of causal attack mechanisms.
- A fair baseline comparison requires rerunning LightGBM on the exact same split, hardware, features, and evaluation.
- The hybrid can miss attacks at its gate; these are scored as false negatives.

Read [protocol](docs/PROTOCOL.md), [safety and rights](docs/SECURITY.md), and [results ledger](docs/RESULTS_STATUS.md).

## Layout

~~~text
medguard/data.py         numeric feature sanitization, split, event summary
medguard/analyst.py      real and mock providers, structured reasoning
medguard/knowledge.py    local curated retrieval
medguard/experiment.py   matched direct/RAG/hybrid evaluation
medguard/cli.py          CLI + explicit external-transfer/billing gates
tests/test_medguard.py
dashboard.py             local aggregate-only research dashboard
docs/PROTOCOL.md
docs/SECURITY.md
docs/RESULTS_STATUS.md
results/                 historical manuscript comparisons
.github/workflows/test.yml
~~~

**Rights:** All rights reserved. Public access permits viewing the source, but no general reuse or redistribution permission is granted. The dataset, external research references, and any jointly authored manuscripts remain subject to their own rights. No personal names are included here.
