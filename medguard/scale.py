"""Generate reproducible synthetic CSV streams for scalability engineering.

A throughput harness, NOT a real network simulator or real attack benchmark.
Temporary synthetic CSVs are discarded after execution.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import DatasetSpec, uniform_reservoir
from .experiment import write_aggregate
from .replay import replay_transfer


def _write_stream(path: Path, *, rows: int, chunk_rows: int,
                  rng: np.random.Generator, domain_shift: float,
                  fields: tuple[str, str, str]) -> None:
    first = True
    for start in range(0, rows, chunk_rows):
        n = min(chunk_rows, rows - start)
        y = rng.choice([0, 1], n, p=[.87, .13])
        frame = pd.DataFrame({
            fields[0]: rng.poisson(15, n) + y * 9 + domain_shift,
            fields[1]: rng.exponential(4, n) + y * 3 + domain_shift,
            fields[2]: rng.exponential(1, n) + y * 1.5,
            "BinaryLabel": y,
        })
        frame.to_csv(path, mode="w" if first else "a",
                     index=False, header=first)
        first = False


def _profile(path: Path, dataset_id: str,
             fields: tuple[str, str, str]) -> Path:
    doc = {
        "dataset_id": dataset_id, "files": [path.name],
        "label_column": "BinaryLabel", "benign_values": ["0"],
        "attack_values": ["1"], "task": "binary_intrusion",
        "source_url": "synthetic://generated-at-runtime",
        "features": {
            "total_packets": fields[0],
            "jitter": fields[1],
            "duration": fields[2],
        },
        "units": {
            "total_packets": "count",
            "jitter": "ms", "duration": "seconds",
        },
    }
    output = path.with_suffix(".json")
    output.write_text(json.dumps(doc), encoding="utf-8")
    return output


def run_scale(*, rows: int = 100_000, source_cap: int = 5_000,
              replay_cap: int = 25_000, chunk_rows: int = 10_000,
              seed: int = 42) -> dict:
    if not (rows >= 100 and 10 <= source_cap <= rows and replay_cap >= 10
            and chunk_rows >= 1):
        raise ValueError("Need rows>=100, 10<=source_cap<=rows, "
                         "replay_cap>=10, chunk_rows>=1")
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    tracemalloc.start()
    try:
        with tempfile.TemporaryDirectory(prefix="medguard-scale-") as directory:
            folder = Path(directory)
            source_fields = ("Source Packets", "Net Jitter", "Duration Sec")
            target_fields = ("PacketCount", "DelayVariance", "FlowDuration")
            source_csv = folder / "source.csv"
            target_csv = folder / "target.csv"
            _write_stream(source_csv, rows=rows, chunk_rows=chunk_rows,
                          rng=rng, domain_shift=0, fields=source_fields)
            _write_stream(target_csv, rows=rows, chunk_rows=chunk_rows,
                          rng=rng, domain_shift=2, fields=target_fields)
            source_manifest = DatasetSpec.load(
                _profile(source_csv, "synthetic_scale_source", source_fields)
            )
            target_manifest = DatasetSpec.load(
                _profile(target_csv, "synthetic_scale_target", target_fields)
            )
            training = uniform_reservoir(
                source_manifest, max_rows=source_cap,
                chunk_rows=chunk_rows, seed=seed,
            )
            replay = replay_transfer(
                training, target_manifest, chunk_rows=chunk_rows,
                max_events=replay_cap, seed=seed,
            )
            peak_bytes = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    total_seconds = time.perf_counter() - start
    return {
        "status": "synthetic_scale_smoke_only",
        "provenance": "GENERATED SYNTHETIC RANDOM TRAFFIC; NOT A DATASET RESULT",
        "generated_rows_per_domain": rows,
        "source_rows_retained": source_cap,
        "target_replay_event_cap": replay_cap,
        "chunk_rows": chunk_rows,
        "seed": seed,
        "total_execution_seconds_including_csv_generation": round(
            total_seconds, 5
        ),
        "peak_python_tracemalloc_mb": round(peak_bytes / 1e6, 3),
        "peak_memory_caveat": "Python tracemalloc excludes many native "
                             "NumPy/pandas/scikit-learn allocations; not RSS.",
        "replay": replay,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--source-cap", type=int, default=5_000)
    parser.add_argument("--replay-cap", type=int, default=25_000)
    parser.add_argument("--chunk-rows", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/synthetic-scale.json")
    a = parser.parse_args(argv)
    report = run_scale(rows=a.rows, source_cap=a.source_cap,
                       replay_cap=a.replay_cap, chunk_rows=a.chunk_rows,
                       seed=a.seed)
    write_aggregate(report, a.output)
    print(json.dumps({
        "status": report["status"], "output": a.output,
        "generated_rows_per_domain": a.rows,
        "events_replayed": report["replay"]["target_events_processed"],
        "events_per_second": report["replay"][
            "events_per_second_including_csv_ingestion"],
        "warning": report["provenance"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
