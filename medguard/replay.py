"""Bounded-memory replay of held-out IoT traffic through a source-trained model.

Reports throughput of local CSV processing + batch inference. It is NOT
a real hospital deployment, packet-capture test, or LLM serving benchmark.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .datasets import DatasetSpec, SampledDataset, iter_chunks
from .generalization import fitted_baseline


def replay_transfer(
    trained_on: SampledDataset,
    replay_spec: DatasetSpec, *,
    chunk_rows: int = 10_000,
    max_events: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    if trained_on.dataset_id == replay_spec.dataset_id:
        raise ValueError("Replay benchmark requires a separate target dataset")
    if max_events < 10 or chunk_rows < 1:
        raise ValueError("max_events >= 10 and chunk_rows >= 1 required")
    shared = sorted(set(trained_on.x) & set(replay_spec.features))
    if len(shared) < 2:
        raise ValueError("Insufficient shared canonical features for replay")
    wrong = [col for col in shared
             if trained_on.units[col] != replay_spec.units[col]]
    if wrong:
        raise ValueError("Conflicting feature units: " + ", ".join(wrong))
    clf, columns = fitted_baseline(trained_on.x[shared],
                                   trained_on.y, seed=seed)
    n = tn = fp = fn = tp = chunks = 0
    batch_latencies_ms: list[float] = []
    overall_start = time.perf_counter()
    for frame in iter_chunks(replay_spec, chunk_rows=chunk_rows):
        if n >= max_events:
            break
        batch = frame.iloc[:max_events - n]
        if batch.empty:
            break
        start = time.perf_counter()
        predictions = clf.predict(batch[columns]).astype(int)
        elapsed = time.perf_counter() - start
        truth = batch["__label__"].to_numpy(dtype=int)
        tn += int(((truth == 0) & (predictions == 0)).sum())
        fp += int(((truth == 0) & (predictions == 1)).sum())
        fn += int(((truth == 1) & (predictions == 0)).sum())
        tp += int(((truth == 1) & (predictions == 1)).sum())
        batch_latencies_ms.append(1000 * elapsed)
        n += len(batch)
        chunks += 1
    wall = time.perf_counter() - overall_start
    if not n:
        raise ValueError("No replay rows")
    return {
        "status": "synthetic_fixture" if (
            trained_on.dataset_id.startswith("synthetic") or
            replay_spec.dataset_id.startswith("synthetic")
        ) else "local_replay_unreviewed",
        "method": "source_trained_batch_replay",
        "source_dataset": trained_on.dataset_id,
        "target_dataset": replay_spec.dataset_id,
        "source_sample_size": len(trained_on.x),
        "target_events_processed": n,
        "max_events": max_events,
        "chunk_rows": chunk_rows,
        "chunks": chunks,
        "features": columns,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "attack_recall": tp / (tp + fn) if tp + fn else None,
        "attack_precision": tp / (tp + fp) if tp + fp else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "wall_seconds_total": round(wall, 5),
        "events_per_second_including_csv_ingestion": round(n / max(wall, 1e-9), 2),
        "median_batch_inference_ms": round(float(np.median(batch_latencies_ms)), 5),
        "p95_batch_inference_ms": round(float(np.percentile(batch_latencies_ms, 95)), 5),
        "bounded_memory_note": "Input reads at most chunk_rows records per iteration "
                               "and source fitting retains at most source_sample_size.",
        "limitations": [
            "Local CPU batch replay only: not a live intrusion prevention system.",
            "No LLM provider calls included in throughput values.",
            "Batch percentile is NOT per-event p95 latency.",
            "No provenance or semantic feature equivalence implied by matching units.",
            "Training labels may define different attack families than replay labels.",
            "Do not benchmark on training records or treat synthetic scores as research.",
        ],
    }
