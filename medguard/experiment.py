"""Comparison protocol: zero-shot LLM, LLM+RAG, hybrid anomaly gate+RAG.

Evaluation outputs are aggregate only. No row-level biometric data written.
"""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from .analyst import Provider, analyze
from .data import DataSplit, Summarizer, balanced_test_indices
from .knowledge import Retriever

@dataclass
class AnomalyGate:
    model: IsolationForest
    threshold: float
    train_normal_quantile: float

    @classmethod
    def fit(cls, split: DataSplit, *, seed: int = 42,
            normal_quantile: float = 0.85) -> "AnomalyGate":
        if not 0.5 <= normal_quantile < 1.0:
            raise ValueError("normal_quantile must be in [0.5,1)")
        normal = split.x_train.iloc[np.flatnonzero(split.y_train == 0)]
        model = IsolationForest(
            n_estimators=100, random_state=seed, n_jobs=1, contamination="auto"
        )
        model.fit(normal)
        # Gate threshold is selected using training NORMAL examples only.
        scores = -model.score_samples(normal)
        threshold = float(np.quantile(scores, normal_quantile))
        return cls(model, threshold, normal_quantile)

    def flag(self, x) -> np.ndarray:
        return (-self.model.score_samples(x)) >= self.threshold

def scores_for(y_true, y_pred) -> dict:
    tn, fp, fn, tp = [int(n) for n in confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()]
    return {
        "n": int(len(y_true)), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "attack_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "attack_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else None,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else None,
    }

def run_comparison(split: DataSplit, provider: Provider, *,
                   max_test_rows: int = 30, seed: int = 42,
                   normal_quantile: float = 0.85,
                   include_lightgbm: bool = False) -> dict:
    selection = balanced_test_indices(split.y_test, max_test_rows, seed)
    x_eval = split.x_test.iloc[selection].reset_index(drop=True)
    y_eval = np.asarray(split.y_test)[selection]
    if len(set(y_eval)) != 2:
        raise ValueError("Evaluation subset needs both classes")
    summary = Summarizer.fit(split.x_train, split.y_train)
    retrieval = Retriever.create()
    gate = AnomalyGate.fit(split, seed=seed, normal_quantile=normal_quantile)
    flags = gate.flag(x_eval)
    approaches = ("llm_direct", "llm_rag", "hybrid_gate_rag")
    results = {}
    for approach in approaches:
        predicted: list[int] = []
        calls = 0
        input_tokens = output_tokens = 0
        start = time.perf_counter()
        for row_no in range(len(x_eval)):
            if approach == "hybrid_gate_rag" and not flags[row_no]:
                predicted.append(0)  # Explicit skip; may miss real attacks.
                continue
            event = summary.summarize(x_eval.iloc[row_no])
            refs = retrieval.search(json.dumps(event), k=3) \
                if approach != "llm_direct" else []
            assessment, usage = analyze(provider, event, refs)
            predicted.append(assessment.attack)
            calls += 1
            input_tokens += usage.input_tokens
            output_tokens += usage.output_tokens
        wall_seconds = time.perf_counter() - start
        report = scores_for(y_eval, np.asarray(predicted))
        report.update({
            "llm_calls": calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_seconds_total": round(wall_seconds, 5),
            "wall_ms_per_evaluated_event": round(1000 * wall_seconds / len(y_eval), 5),
            "not_calibrated": True,
        })
        results[approach] = report
    # Gate-alone sensitivity quantifies errors that the LLM cannot repair.
    gate_metrics = scores_for(y_eval, flags.astype(int))
    conventional_baseline = None
    if include_lightgbm:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("Install optional baseline: pip install -e '.[baseline]'") from exc
        clf = LGBMClassifier(n_estimators=100, num_leaves=15, random_state=seed,
                             verbosity=-1, n_jobs=1)
        begin_train = time.perf_counter()
        clf.fit(split.x_train, split.y_train)
        train_s = time.perf_counter() - begin_train
        begin_predict = time.perf_counter()
        baseline_pred = clf.predict(x_eval)
        inference_s = time.perf_counter() - begin_predict
        conventional_baseline = scores_for(y_eval, baseline_pred)
        conventional_baseline.update({
            "algorithm": "LightGBM", "comparison": "fresh_same_split",
            "training_seconds": round(train_s, 5),
            "wall_ms_per_evaluated_event": round(1000 * inference_s / len(y_eval), 5),
            "note": "Measured locally on same features/test events; not the historical manuscript model.",
        })
    return {
        "schema_version": "1.0",
        "status": "synthetic_mock_smoke_test"
                  if provider.kind == "synthetic_mock"
                  else "experimental_llm_evaluation",
        "provider": provider.kind,
        "model": provider.model,
        "seed": seed,
        "data": {
            "test_examples_evaluated": int(len(y_eval)),
            "train_examples": int(len(split.x_train)),
            "full_test_examples": int(len(split.x_test)),
            "excluded_columns": split.excluded_columns,
            "label_column": split.label_column,
            "test_class_counts": {
                "normal": int(sum(y_eval == 0)),
                "attack": int(sum(y_eval == 1)),
            },
        },
        "gate": {
            "type": "IsolationForest",
            "calibration": "training_normal_only",
            "quantile": normal_quantile,
            "flagged_events": int(flags.sum()),
            "alone_metrics_on_evaluation_subset": gate_metrics,
        },
        "approaches": results,
        "conventional_baseline": conventional_baseline,
        "environment": {"python": platform.python_version()},
        "limitations": [
            "Small evaluation samples are exploratory, not publication-grade.",
            "LLM self-reported confidence is not calibrated probability.",
            "Only an isolated test split is assessed; no external-site generalization.",
            "Mock-mode metrics are synthetic plumbing checks, not research evidence.",
            "Wall time includes local processing and optional network/API delays.",
            "No model-token pricing is assumed; cost must be estimated from provider bills.",
        ],
    }

def write_aggregate(report: dict, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # No prompts, individual readings, patient telemetry, or model explanations.
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
