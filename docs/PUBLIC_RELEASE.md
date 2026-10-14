# MedGuard-AI public-source release

This repository now contains the MedGuard-AI source package in its publicly viewable working tree. The release integrates the implementation with the earlier healthcare IoT research summary.

## Included implementation

| Path | Scope |
| --- | --- |
| `medguard/data.py` | Leakage-aware CSV loading, train/test separation, numerical event summarization |
| `medguard/analyst.py` | Structured LLM classification and evidence validation |
| `medguard/knowledge.py` | Curated local knowledge retrieval |
| `medguard/experiment.py` | LLM-only, RAG, hybrid gate, and optional freshly trained LightGBM comparison |
| `medguard/cli.py` | Local reproducible CLI with explicit API permission flags |
| `dashboard.py` | Streamlit visualization of aggregate output |
| `tests/test_medguard.py` | Offline test suite |
| `.github/workflows/test.yml` | Python compilation, unit tests, synthetic pipeline smoke test |
| `docs/PROTOCOL.md` | Dataset, target validity, and experimental evaluation protocol |
| `docs/SECURITY.md` | Risk, privacy and scientific limitations |
| `docs/RESULTS_STATUS.md` | Historical results versus unrun LLM experiment ledger |
| `docs/NEW_PAPER_BLUEPRINT.md` | Proposed research-paper design (not empirical results) |
| `docs/MEDGUARD_IMPLEMENTATION.md` | Full technical setup and CLI usage |

## Verification and research boundaries

The Python modules, dashboard, test script, build configuration, CI workflow, and original comparison CSVs have been checked against the previous private repository copies. Documentation containing older private-repository wording was updated for this public release.

A previous private CI run passed ten unit tests, Python source compilation, and a synthetic end-to-end smoke test. The public release's CI run is the separate validation of its own checkout.

**No real LLM-on-WUSTL metrics are claimed.** No raw WUSTL dataset, source credentials, per-event data, medical records, trained models, or unpublished full manuscript has been published.

## Rights

Copyright and reuse restrictions are documented in [../LICENSE](../LICENSE). Public visibility does not grant a permissive software license. Existing rights from earlier licenses, if any, may not be retroactively revoked.

## Running tests

```bash
python -m pip install -e '.[dev,baseline]'
python -m pytest -q
python -m medguard.cli --max-test-rows 12 --output outputs/ci-smoke.json
```

The default synthetic/mocked output is for verifying plumbing only and **must not** be cited as measured LLM threat-detection performance.
