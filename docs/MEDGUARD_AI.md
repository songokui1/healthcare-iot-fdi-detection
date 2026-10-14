# MedGuard-AI: LLM-Powered Healthcare IoT Threat Investigation

**Public experimental research prototype — real-model results pending**

## Research direction

MedGuard-AI explores whether large language models can help interpret suspicious Internet of Medical Things (IoMT) activity, including network-flow behavior and physiological sensor measurements. The research extends an earlier six-model comparison on WUSTL-EHMS-2020, where LightGBM was the strongest **reported** conventional baseline.

Unlike an ordinary tabular classifier, the proposed system investigates **explanations and evidence retrieval** alongside attack-detection performance, latency, false negatives, and inference cost.

## Three experimental designs

| Approach | Function | Research trade-off |
| --- | --- | --- |
| LLM-only | Convert numerical telemetry into bounded, training-referenced evidence and request structured threat assessment | Direct inference; uncertain numerical reasoning |
| Retrieval-augmented LLM | Retrieve relevant security knowledge before generating an assessment and cited rationale | Adds context; incurs retrieval overhead and potential hallucinations |
| Hybrid anomaly gate + RAG/LLM | Screen events with a lightweight anomaly detector; investigate flagged events with RAG and an LLM | Fewer LLM calls; risk of missing attacks at the gate |

## Conceptual architecture

~~~mermaid
flowchart TD
 A[Authorized IoMT traffic dataset] --> B[Privacy-aware feature processing]
 B --> C[Numeric event summaries]
 C --> D[LLM-only]
 C --> E[Retrieval and LLM]
 B --> F[Lightweight anomaly gate]
 F --> G{Flagged?}
 G -->|Yes| H[Retrieval and LLM]
 G -->|No| I[Benign prediction]
 D --> J[Matched held-out evaluation]
 E --> J
 H --> J
 I --> J
 J --> K[Attack recall, F1, false negatives, latency, calls, tokens]
~~~

## Engineering completed (public implementation)

- Modular Python prototype for event preparation and bounded prompt generation.
- Curated local knowledge retrieval and machine-readable LLM response validation.
- Same-subset experimental evaluation for the three approaches.
- A hybrid gate with measurable missed-attack risk.
- Offline synthetic smoke-testing and a consent-gated API implementation.
- Test suite and research protocol documenting leakage, privacy and generalization limits.
- Read-only local results dashboard that displays aggregate recall, F1, false negatives, API calls and latency without exposing row-level telemetry.
- Optional freshly trained LightGBM reference evaluated on the same held-out events; it remains separate from historical manuscript metrics.

**All current MedGuard-AI source modules, automated tests, the optional conventional baseline, the synthetic smoke test, and the dashboard are available publicly.** See [the implementation guide](MEDGUARD_IMPLEMENTATION.md), [Python package](../medguard/), and [dashboard](../dashboard.py). Original WUSTL data, API secrets, model assets, and unpublished manuscripts are not provided.

## Research integrity and next phase

Prior private CI verified **10 unit tests, Python source compilation, and a synthetic mock smoke test**. These are engineering checks, **not research performance measurements**.

No real LLM-vs-LightGBM benchmark results have been validated yet. The existing LightGBM metrics in this public repository come from an earlier manuscript and are not a paired evaluation of MedGuard-AI.

Before performance claims, the researchers must complete authorized dataset acquisition, FDI-specific class validation, leakage audits, same-split baseline testing, live LLM experiments, uncertainty estimates, and human evaluation of explanations.

The source dataset includes more than one attack context, so broad binary intrusion-detection results must not be mislabeled as exclusively false-data-injection detection. In addition, the original dataset labels are associated with attacker MAC addresses; identifier and proxy leakage require special care.

**No API keys, raw telemetry, or unpublished manuscripts are published here. The research implementation is now public.**

Research source: https://www.cs.wustl.edu/~jain/ehms/index.html

**All rights reserved.** Portfolio viewing does not grant permission to reuse this original research.
