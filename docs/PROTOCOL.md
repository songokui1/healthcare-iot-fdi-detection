# MedGuard-AI research protocol v0.1

## Research questions

RQ1: How does LLM-only classification compare with retrieval-assisted LLM classification on held-out WUSTL-EHMS-2020 telemetry?

RQ2: Does an anomaly-gated hybrid reduce API calls and elapsed time while preserving acceptable attack recall?

RQ3: Can the LLM provide grounded evidence citations rather than plausible but unsupported explanations?

These are **open experimental questions**. Strong tabular classifiers may outperform LLMs. No superiority claim is justified until evaluated.

## Dataset and task boundaries

Official dataset: https://www.cs.wustl.edu/~jain/ehms/index.html

The source describes network flow and physiological readings with spoofing and data-injection activity; a broad attack-vs-normal classification is not necessarily the same as **FDI-only** detection. Verify individual attack category provenance before an FDI-specific claim, and exclude category fields from model inputs.

Because attack labels were constructed from attacker MAC addresses, inspect every input for identifiers and correlation proxies. The loader blocks known names automatically but cannot certify all leakage sources.

## Protocol for a publishable experiment

1. Obtain the dataset from its publisher; record source, file hash, license terms, class counts and schema privately.
2. Audit near-duplicates, device/session identifiers, class mapping, label proxies and collection chronology.
3. Reserve an untuned final test set. Prefer session/capture/device-group splits when supported, rather than random row splitting; implement group strategy before making real generalization claims.
4. Fit imputation statistics, summarizer and anomaly threshold on the training partition only.
5. Freeze model, prompt, retrieval corpus and screening threshold before final evaluation. Do not feed test labels to prompts or use test results for tuning.
6. Evaluate LLM-only, RAG, and hybrid on the SAME test events.
7. Report confusion matrices, attack-class recall/F1, false positive and missed-attack rates, throughput, full wall latency, API calls, tokens and verified cost.
8. Analyze gate-only recall and failure cases; a gated-out attack cannot be recovered by the LLM.
9. Assess explanations by blinded expert review; validated citations do not establish factual or causal correctness.
10. Repeat with multiple seeds and calculate paired uncertainty; evaluate out-of-distribution performance on an external site/device dataset if accessible.
11. Rerun a conventional LightGBM baseline on the SAME split and features for meaningful head-to-head comparison. Historical manuscript numbers are not interchangeable with new runs.

## Security and privacy

- Numeric summaries only are sent externally in real mode; no patient names, IP/MAC identifiers, row text, labels or raw dataset records.
- Additional potential proxies must be excluded by the researcher, including time/capture identifiers and source/destination session fields.
- Knowledge retrieval is local and static; documents supply context, not labeled examples.
- Prompt data is untrusted; generated JSON evidence references must match provided features/knowledge IDs.
- Invalid outputs fail loudly; count and audit invalid outputs in future batch experiments.
- No automatic blocking, isolation, or healthcare diagnosis is allowed.

## Status and limitations

Current tools support exploratory pilot evaluation, not a verified clinical or publication-grade result. The default 30-row sample has insufficient statistical power. A fixed random stratified row split can have correlated train/test events. Neither the mock provider nor a good-looking generated explanation is evidence of LLM effectiveness.
