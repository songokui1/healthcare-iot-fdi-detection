# Security, rights, and privacy boundaries

**ALL RIGHTS RESERVED.** No permission is granted to redistribute original source, experimental records, or unpublished papers.

## Keep private
- Raw WUSTL and derived per-event data, model files, draft manuscripts and reviewer feedback.
- API keys, individual prompts containing telemetry, model provider responses, patient information, or device identifiers.
- Original notebooks and output artifacts until rights and disclosure have been reviewed.

## Risks and controls
- **Data leakage:** labels based on attacker MAC addresses can be trivially reconstructed from direct identifiers or indirectly from correlated fields. Automated filtering is only a first defense.
- **External privacy:** an API request contains a subset of sanitized numeric testbed fields. Ensure the source permits external transfer. Never send identifiable clinical records.
- **Prompt injection:** untrusted free text is dropped; retrieval is curated locally. The model is forbidden from treating provided event data as instructions.
- **Hallucination:** evidence IDs are mechanically validated; factual validity of a narrative remains unproven.
- **Billing:** mock is the default. Real API mode requires explicit double consent; requests have bounded timeouts and retry limits.
- **Reliability:** no autonomous safety-critical interventions or medical diagnosis; human oversight is mandatory.

The public healthcare repository contains the source implementation and documentation. Public GitHub history can retain earlier commits even after deletion.
