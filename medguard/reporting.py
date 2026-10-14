"""Safe projection of aggregate evaluation reports into a local dashboard.

The presenter does not pass arbitrary JSON fields through to the UI.
This prevents accidental rendering of unknown raw-record/secret fields.
"""
from __future__ import annotations

from typing import Any

V1_STATES = {"synthetic_mock_smoke_test", "experimental_llm_evaluation"}
V2_STATES = {"synthetic_fixture", "local_dataset_experiment_unreviewed",
             "local_cross_dataset_experiment_unreviewed", "local_replay_unreviewed"}


def _metrics(summary: dict) -> dict:
    keys = (
        "n", "tn", "fp", "fn", "tp", "accuracy",
        "balanced_accuracy", "attack_precision", "attack_recall",
        "attack_f1", "false_positive_rate", "false_negative_rate",
        "roc_auc", "pr_auc", "mcc",
    )
    return {k: summary[k] for k in keys if k in summary and
            (summary[k] is None or isinstance(summary[k], (int, float)))}


def _is_synthetic(status: str) -> bool:
    return "synthetic" in status or "mock" in status


def dashboard_projection(document: dict[str, Any]) -> dict:
    if not isinstance(document, dict):
        raise ValueError("Dashboard accepts only aggregate JSON objects")
    status = document.get("status")
    if status == "synthetic_scale_smoke_only":
        if not isinstance(document.get("replay"), dict):
            raise ValueError("Incomplete synthetic scale replay report")
        replay = dashboard_projection(document["replay"])
        if not replay["synthetic"]:
            raise ValueError("Scale report must contain synthetic replay")
        return {
            "kind": "synthetic_scale", "title": "Synthetic CSV scale smoke",
            "status": status, "synthetic": True,
            "sections": [replay],
            "generated_rows_per_domain": int(document.get("generated_rows_per_domain", 0)),
            "peak_python_tracemalloc_mb": document.get("peak_python_tracemalloc_mb"),
            "duration_seconds": document.get("total_execution_seconds_including_csv_generation"),
        }
    if status == "synthetic_fixture_not_scientific_evidence":
        if not all(isinstance(document.get(k), dict)
                   for k in ("within", "transfer", "replay")):
            raise ValueError("Incomplete synthetic demo report")
        sections = [dashboard_projection(document[k]) for k in
                    ("within", "transfer", "replay")]
        if not all(section["synthetic"] for section in sections):
            raise ValueError("Mixed provenance in synthetic demonstration")
        return {"kind": "synthetic_demo", "status": status, "synthetic": True,
                "sections": sections, "title": "Synthetic multi-dataset demonstration"}

    if status in V1_STATES:
        if not isinstance(document.get("approaches"), dict):
            raise ValueError("Missing LLM comparison results")
        rows = []
        display = {
            "llm_direct": "LLM-only",
            "llm_rag": "RAG + LLM",
            "hybrid_gate_rag": "Hybrid gate + RAG/LLM",
        }
        for key, stats in document["approaches"].items():
            if key not in display or not isinstance(stats, dict):
                continue
            rows.append({
                "approach": display[key],
                **_metrics(stats),
                "llm_calls": int(stats.get("llm_calls", 0)),
                "input_tokens": int(stats.get("input_tokens", 0)),
                "output_tokens": int(stats.get("output_tokens", 0)),
                "wall_seconds_total": stats.get("wall_seconds_total"),
            })
        if not rows:
            raise ValueError("Unrecognized LLM comparison approaches")
        return {"kind": "llm", "title": "LLM + RAG + hybrid comparison",
                "status": status, "synthetic": _is_synthetic(status),
                "rows": rows,
                "sample_size": document.get("data", {}).get("test_examples_evaluated"),
                "gate_flagged": document.get("gate", {}).get("flagged_events"),
                "baseline": _metrics(document["conventional_baseline"])
                            if isinstance(document.get("conventional_baseline"), dict)
                            else None}
    if status in V2_STATES:
        method = document.get("method")
        if method in {"within_dataset_random_forest",
                      "zero_target_training_cross_dataset_transfer"}:
            metrics = document.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError("No evaluation metrics")
            return {
                "kind": "detection", "title": method.replace("_", " ").title(),
                "status": status, "synthetic": _is_synthetic(status),
                "metrics": _metrics(metrics),
                "features": list(document.get("features", [])),
                "confidence_intervals": {
                    k: {"low_95": v.get("low_95"), "high_95": v.get("high_95")}
                    for k, v in document.get("confidence_intervals", {}).items()
                    if k in {"attack_recall", "attack_f1", "false_positive_rate"}
                    and isinstance(v, dict)
                },
                "dataset": document.get("dataset"),
                "source": document.get("source"),
                "target": document.get("target"),
                "task_compatibility": document.get("task_compatibility"),
            }
        if method == "source_trained_batch_replay":
            return {
                "kind": "replay", "title": "Local Batch Replay",
                "status": status, "synthetic": _is_synthetic(status),
                "source": document.get("source_dataset"),
                "target": document.get("target_dataset"),
                "events": int(document.get("target_events_processed", 0)),
                "throughput_eps": document.get(
                    "events_per_second_including_csv_ingestion",
                ),
                "median_batch_ms": document.get("median_batch_inference_ms"),
                "p95_batch_ms": document.get("p95_batch_inference_ms"),
                "attack_recall": document.get("attack_recall"),
                "false_negative_rate": document.get("false_negative_rate"),
                "confusion": {
                    k: int(v) for k, v in document.get("confusion", {}).items()
                    if k in {"tn", "fp", "fn", "tp"}
                },
            }
    raise ValueError("Unknown or unapproved aggregate report schema/status")
