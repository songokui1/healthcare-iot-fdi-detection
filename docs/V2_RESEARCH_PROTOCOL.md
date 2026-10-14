# MedGuard-AI v2 — multi-dataset research protocol

**Status:** Implemented software protocol with synthetic CI, NOT a completed cross-dataset empirical study.

## Why these datasets?

| Official dataset | Purpose | Interpretation |
|---|---|---|
| [WUSTL-EHMS-2020](https://www.cs.wustl.edu/~jain/ehms/index.html) | Initial healthcare telemetry / network study | 16,318 examples, 44 columns including label; spoofing and injected traffic are both described. MAC-derived labels are a major leakage threat. |
| [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) | Domain expansion across medical devices and protocols | 40 medical IoT devices, 18 attacks, Wi-Fi, MQTT, Bluetooth. The original distribution contains different file formats and pre-provided splits; do not assume identical schemas across protocols. |
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | Large IoT robustness, generalized-network and performance test | 105 IoT devices, 33 attacks in several categories. NOT inherently clinical data. |

None of these datasets are automatically downloaded, mirrored or published here. Follow originating license/access terms and cite the dataset authors.

## Dataset acquisition and mapping

1. Acquire original files yourself from their official publishers and document any preprocessing or feature extraction.
2. Review the **actual CSV headers** and unit definitions. Filenames, labeling, and schemas vary.
3. Copy one of the manifests in [dataset_profiles](../dataset_profiles/). Replace ALL \`VERIFY_...\` values. Give every attack label explicitly; never infer 'everything non-benign is an attack' from unknown classes.
4. Configure \`features\` as **canonical feature name -> raw dataset column**. The v2 engine refuses any unknown canonical name, duplicate mapping, obvious label/identifier mapping, or missing units.
5. Set optional group/time metadata separately from model features. Do not assert device-separated or chronological validation without genuine source metadata.
6. Keep real data in locally gitignored \`data/\`; inspect sample/labels before training.

### Canonical features and semantics

Each shared feature must have the same **meaning and unit** in source and target. For example, 'duration' can only transfer when both refer to equivalent flow durations measured in the same units. A name match or unit match is **not proof** of equivalence; review how each dataset generated its flows. Device-specific biometrics and protocol-specific attributes can remain dataset-specific and should not be silently imputed as shared network features.

Be especially careful with IPs, MACs, IDs, timestamps, ports, attack classes, capture labels, and flow/session metadata. The registry blocks obvious direct identifiers and label terms, but even harmless-looking numerical fields may be proxies for the attack source. Source/destination ports, for example, can encode deployment or attack setup rather than true behavior.

## Working commands

~~~bash
python -m pip install -e '.[dev,baseline]'
python -m medguard.research_cli demo --output outputs/v2-demo.json
python -m medguard.research_cli inspect --source dataset_profiles/wustl_ehms_2020.json --source-limit 20000 --sha256
python -m medguard.research_cli evaluate --source dataset_profiles/wustl_ehms_2020.json --split group --output outputs/wustl_group.json
python -m medguard.research_cli transfer --source dataset_profiles/wustl_ehms_2020.json --target dataset_profiles/ciciomt2024_wifi_mqtt.json --output outputs/wustl_to_ciciomt.json
python -m medguard.research_cli replay --source dataset_profiles/ciciomt2024_wifi_mqtt.json --target dataset_profiles/ciciot2023.json --max-events 100000 --output outputs/replay.json
~~~

**Important:** The three real-dataset commands are examples of the interface, **not turnkey commands**: they require local CSVs and completed/non-example manifests. Group split requires a real \`group_column\`. If not available, use \`--split stratified\`, report its limitations, and do not call it group-validated.

### What the new framework actually implements

- CSV chunk ingestion with bounded-memory uniform random-priority reservoir sampling and explicit class prevalence counts.
- Stratified, group-disjoint, and chronological split options, with train-only imputation.
- Within-domain Random Forest baseline with ROC-AUC, PR-AUC, MCC, attack-class detection metrics, and bootstrapped intervals.
- Cross-domain transfer from one declared dataset to another without fitting on target rows or target labels.
- Fail-closed shared feature-unit checks, descriptive median shift relative to source IQR, and task compatibility warnings.
- Local bounded-memory chunk replay, confusion counts, batch inference timing, and throughput including CSV ingestion.
- Deterministic synthetic demonstration and tests without API calls or copied external data.

### Scientific restrictions

- **Never compare historical 98.96% LightGBM accuracy from the old manuscript against a new model's unpaired split.** Re-run the baseline with equal task, examples, feature space, and split.
- Cross-dataset comparison of **binary intrusion** across different attack types answers a domain-transfer question. It does not establish FDI-specific sensitivity unless equivalent attack definitions are verified.
- Bootstrap intervals assume independent examples. Repeated flows from one device or capture violate this; use capture-/device-level bootstrap for publication.
- The reservoir sample is uniform, but finite samples might miss rare attack variants. The full scanned class proportions are recorded; do not claim sampled prevalence is deployed prevalence.
- A fixed group-heldout test from one dataset does not prove generalization to entirely new hospitals or devices. External prospective data and reviews are needed.
- Throughput is **local CSV ingestion + batched classical ML inference**, not live network packet ingestion or remote LLM serving performance. Do not call it production-ready.
- No model's natural-language rationale is validated merely because its cited IDs exist. Human review of correctness is a separate task.

## LLM research direction

The v1 LLM, RAG and gated-hybrid modules remain available, with opt-in remote calls and synthetic smoke coverage. v2 focuses first on **defensible dataset generalization**. The next empirical phase should run actual LLM evaluations on the same heldout examples with per-sample cost and human citation audits, not call an LLM on millions of rows.

A publication-worthy research contribution requires results on authorized datasets, preregistered evaluation questions, appropriate group/time separation, external shift analyses, failure audits, and reproducible model/environment specifications.
