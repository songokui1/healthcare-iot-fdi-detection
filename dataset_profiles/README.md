# Local dataset manifest templates

These are **examples requiring manual verification**, not validated column mappings for the original datasets. Do not run them without completing every \`VERIFY_...\` field and ensuring the CSV files, labels, units and provenance match the publisher's actual release.

The original datasets are **not bundled or downloaded automatically**.

## Steps

1. Download the appropriate CSV extracts from the official dataset publisher under its terms.
2. Read the publisher's field descriptions; record the exact original attack category/label semantics.
3. Copy \`*.example.json\` to a corresponding \`*.json\` local profile. Replace placeholders with actual raw CSV column names and units.
4. List **every** allowed benign and attack label under \`benign_values\` and \`attack_values\`. The pipeline rejects unexpected classes rather than silently converting them.
5. Define only true, reproducibly comparable numerical features in \`features\`, mapping canonical names (left) to actual raw columns (right). Add a physical unit for each in \`units\`.
6. For verified capture/device-heldout evaluation, supply a real non-target metadata field as \`group_column\`. For time splits, use \`time_column\`. These identifiers are withheld from model inputs.
7. Run \`python -m medguard.research_cli inspect --source dataset_profiles/YOUR_PROFILE.json\` and review sample counts and missingness before any score calculation.
8. Record the input files' SHA-256 with \`--sha256\` when provenance matters; note this reads each file again and can be expensive on large input.

Do not assume that different CICIoMT2024 protocols use the same CSV feature extraction schema. Do not equate WUSTL FDI, IoMT general intrusion and IoT denial-of-service labels by renaming everything to "FDI."

The v2 source code supports: duration, source/destination and total packets/bytes, packet/byte rates, mean packet size, inter-arrival time, jitter, TCP flags, source/destination ports, temperature, pulse rate, respiratory rate, oxygen saturation, ECG ST, and aggregate flow packets. The registry validates canonical names and matching declared units, but cannot prove that two independently extracted fields mean the same thing. Inspect both data dictionaries and flag incompatibilities.

All real CSV data, derived individual records, original private manuscripts and credentials belong in locally ignored directories and must not be committed.
