# Draft research-paper blueprint — MedGuard-AI

**Working title:** *MedGuard-AI: A Controlled Evaluation of LLM-Only, Retrieval-Augmented, and Hybrid Threat Investigation for Healthcare IoT*

**Status:** Study design and software prototype, **NOT an empirical paper with verified LLM results**.

## Proposed abstract (protocol stage)

Healthcare IoT intrusion detection research commonly evaluates numerical machine learning classifiers on network-flow and physiological data. Large language models offer a different capability: translating numerical anomalies into contextual threat assessments and evidence-linked narratives. This study proposes a controlled comparison of three approaches on the WUSTL-EHMS-2020 benchmark: direct LLM classification of sanitized numerical evidence, local retrieval-augmented LLM assessment, and an anomaly-screened hybrid that invokes an LLM only on flagged events. A matched train/test protocol will measure attack-class recall, F1, confusion errors, API usage, and end-to-end latency while separately auditing the correctness of model-generated explanations. Dataset labeling and leakage risks, particularly identifiers related to attacker MAC addresses, will be examined before drawing conclusions. The objective is to characterize the accuracy-cost-explainability trade-off, not to assume an LLM outperforms conventional models.

## Paper sections

1. **Introduction.** IoMT integrity threats; traditional tabular-detection strengths; language-based interpretation opportunity; explicit hypotheses.
2. **Related work.** Healthcare intrusion detection, WUSTL benchmark, anomaly gating, RAG for security operations, explanation evaluation, tabular reasoning by LLMs.
3. **Problem formulation and data audit.** Formal binary target definition; spoofing vs data injection; features, leakage proxies, grouping and duplicate handling.
4. **System architecture.** Numeric summarizer, zero-shot assessor, curated retrieval, hybrid gate, JSON evidence/citation validation.
5. **Experimental design.** Frozen prompts, split, hyperparameters, matched test rows, budgets and model selection; LightGBM same-split comparator.
6. **Results (pending).** Accuracy and recall table; confusion matrices; API calls/latency/cost; gate sensitivity; uncertainty intervals. **DO NOT FILL BEFORE RUNNING EXPERIMENTS.**
7. **Error and explanation study (pending).** Expert review, unsupported rationales, hallucinated claims, model failure handling, biometric privacy.
8. **Discussion.** Trade-offs, clinical/security safety limits, interpretability versus verifiability, attack generalization.
9. **Limitations and threats to validity.** Single benchmark, label construction, proxy leakage, missing device deployment, API model updates and nondeterminism.
10. **Conclusion.** Only claims supported by final measurements.

## Required tables (placeholders, not reported data)

### Detection
Rows: LightGBM (fresh), LLM-only, RAG+LLM, hybrid gate+RAG/LLM.
Columns: n, TP, TN, FP, FN, precision, attack recall, attack F1, balanced accuracy, 95% CI.

### Operations
Columns: provider/model, API calls, input/output tokens, cost per 100 events (verified price), total elapsed s, median/p95 end-to-end latency, gate false-negative count.

### Explanation reliability
Columns: grounded feature names, supported retrieval IDs, correct factual explanation %, analyst rating, hallucination/error %. Requires independent human expert evaluation.

## Contribution claim discipline

A novel *comparison framework* and reproducible trade-off study may be a defensible contribution even if the LLM underperforms LightGBM. Do not claim algorithmic novelty, clinical deployment, superior accuracy, real-world hospital readiness or peer-reviewed acceptance without support.

## Practical next experiments

1. Acquire authorized dataset and record provenance, schema, SHA-256, label map.
2. Run full feature-leakage audit and session-split assessment.
3. Execute offline mock smoke suite and fresh conventional baseline.
4. Authorize a paid, capped real LLM pilot on matched held-out events.
5. Review unexpected classifications and code reliability.
6. Expand sample, repeat seeds, assess external model/corpus and expert ratings.
7. Write genuine empirical results and revise manuscript.
