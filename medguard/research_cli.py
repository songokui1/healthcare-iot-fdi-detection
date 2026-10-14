"""MedGuard v2: local, no-API multi-dataset research commands.

Profiles must be checked against ORIGINAL dataset headers and source licenses.
Data files stay local and gitignored. No LLM calls or remote downloads.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DatasetSpec, file_sha256, uniform_reservoir
from .experiment import write_aggregate
from .generalization import evaluate_transfer, evaluate_within
from .replay import replay_transfer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medguard-research",
        description="Leakage-conscious multi-dataset IoMT evaluation and replay",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "evaluate", "transfer", "replay"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--source", required=True, help="Local JSON dataset manifest")
        cmd.add_argument("--source-limit", type=int, default=20_000)
        cmd.add_argument("--seed", type=int, default=42)
        cmd.add_argument("--chunk-rows", type=int, default=10_000)
        cmd.add_argument("--sha256", action="store_true",
                         help="Hash CSV inputs for provenance (extra disk pass)")
        if name in ("transfer", "replay"):
            cmd.add_argument("--target", required=True, help="Local target manifest")
        if name in ("evaluate", "transfer", "replay"):
            cmd.add_argument("--output", default=f"outputs/{name}.json")
        if name == "evaluate":
            cmd.add_argument("--split", choices=["stratified", "group", "chronological"],
                             default="stratified")
            cmd.add_argument("--bootstrap", type=int, default=300)
        if name == "transfer":
            cmd.add_argument("--target-limit", type=int, default=20_000)
            cmd.add_argument("--bootstrap", type=int, default=300)
        if name == "replay":
            cmd.add_argument("--max-events", type=int, default=100_000)
    demo = sub.add_parser("demo", help="Entirely synthetic, no external data/API")
    demo.add_argument("--output", default="outputs/v2-demo.json")
    demo.add_argument("--seed", type=int, default=42)
    return parser


def _source_audit(spec: DatasetSpec, with_hash: bool) -> dict:
    out = {"dataset_id": spec.dataset_id, "task": spec.task,
           "source_url": spec.source_url}
    if with_hash:
        out["source_csv_sha256"] = [file_sha256(p) for p in spec.files]
    return out


def _demo_files(folder: Path, seed: int) -> tuple[Path, Path]:
    """Generate two independently shifted testbed-like *synthetic* domains."""
    rng = np.random.default_rng(seed)
    paths = []
    for label, shift, field_names in [
        ("synthetic_training", 0.0, ("Packet Count", "Jitter MS", "Flow Seconds")),
        ("synthetic_external", 1.5, ("packets_received", "jitter_val", "duration_s")),
    ]:
        n = 360
        y = rng.choice([0, 1], n, p=[.7, .3])
        df = pd.DataFrame({
            field_names[0]: rng.poisson(12, n) + y * 9 + shift,
            field_names[1]: rng.exponential(3, n) + y * 5 + shift,
            field_names[2]: rng.exponential(1, n) + y * 1.2,
            "Truth": y,
            "session": [f"device_{i%12}" for i in range(n)],
            "tstamp": np.arange(n),
            "attacker_mac": ["SYNTHETIC_IDENTIFIER_NEVER_PREDICT"] * n,
        })
        path = folder / f"{label}.csv"
        df.to_csv(path, index=False)
        manifest = {
            "dataset_id": label, "files": [path.name],
            "label_column": "Truth", "benign_values": ["0"],
            "attack_values": ["1"], "task": "binary_intrusion",
            "source_url": "synthetic://local-ci-only",
            "group_column": "session", "time_column": "tstamp",
            "features": {
                "total_packets": field_names[0],
                "jitter": field_names[1],
                "duration": field_names[2],
            },
            "units": {
                "total_packets": "count",
                "jitter": "ms",
                "duration": "seconds",
            },
        }
        config = folder / f"{label}.json"
        config.write_text(json.dumps(manifest), encoding="utf-8")
        paths.append(config)
    return paths[0], paths[1]


def run_demo(*, seed: int = 42) -> dict:
    with tempfile.TemporaryDirectory(prefix="medguard-synthetic-") as directory:
        src, dst = _demo_files(Path(directory), seed)
        source = uniform_reservoir(DatasetSpec.load(src), max_rows=500, seed=seed)
        target = uniform_reservoir(DatasetSpec.load(dst), max_rows=500, seed=seed)
        within = evaluate_within(source, split_mode="group", seed=seed,
                                 repetitions=60)
        transfer = evaluate_transfer(source, target, seed=seed, repetitions=60)
        replay = replay_transfer(source, DatasetSpec.load(dst),
                                 chunk_rows=45, max_events=300, seed=seed)
        return {
            "status": "synthetic_fixture_not_scientific_evidence",
            "within": within, "transfer": transfer, "replay": replay,
            "warning": "Synthetic data and labels, not WUSTL/CIC or real detection.",
        }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        report = run_demo(seed=args.seed)
        write_aggregate(report, args.output)
        print(json.dumps({
            "status": report["status"], "output": args.output,
            "within_f1_synthetic": report["within"]["metrics"]["attack_f1"],
            "transfer_f1_synthetic": report["transfer"]["metrics"]["attack_f1"],
            "replayed_synthetic_events": report["replay"]["target_events_processed"],
            "warning": report["warning"],
        }, indent=2))
        return 0
    source_spec = DatasetSpec.load(args.source)
    source = uniform_reservoir(
        source_spec, max_rows=args.source_limit, seed=args.seed,
        chunk_rows=args.chunk_rows,
    )
    source_record = _source_audit(source_spec, args.sha256)
    if args.command == "inspect":
        # Never print raw rows, names, identifiers, or raw device metadata.
        print(json.dumps({
            "status": "local_manifest_audit_not_a_performance_result",
            "source": source_record, "sample": source.audit,
            "canonical_features": list(source.x.columns),
            "units": source.units,
            "missing_fraction_in_sample": {
                col: float(source.x[col].isna().mean()) for col in source.x
            },
        }, indent=2))
        return 0
    if args.command == "evaluate":
        report = evaluate_within(
            source, split_mode=args.split, seed=args.seed,
            repetitions=args.bootstrap,
        )
    elif args.command == "transfer":
        target_spec = DatasetSpec.load(args.target)
        target = uniform_reservoir(
            target_spec, max_rows=args.target_limit, seed=args.seed,
            chunk_rows=args.chunk_rows,
        )
        report = evaluate_transfer(source, target, seed=args.seed,
                                   repetitions=args.bootstrap)
        report["target_provenance"] = _source_audit(target_spec, args.sha256)
    elif args.command == "replay":
        report = replay_transfer(
            source, DatasetSpec.load(args.target), seed=args.seed,
            max_events=args.max_events, chunk_rows=args.chunk_rows,
        )
    else:
        raise AssertionError("Unknown research command")
    report["source_provenance"] = source_record
    write_aggregate(report, args.output)
    print(json.dumps({
        "status": report["status"], "method": report["method"],
        "output": args.output, "warning":
        "Experimental research only. Audit provenance, labels and leakage.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
