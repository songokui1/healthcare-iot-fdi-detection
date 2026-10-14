"""Rigorous within-dataset and cross-dataset experiment design.

Real validation requires unseen devices/sessions and a validated label mapping.
All reported measurements come from actual local arrays, never fixed demo scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline

from .data import DataSplit
from .datasets import SampledDataset
from .experiment import scores_for


@dataclass
class SplitAudit:
    split: DataSplit
    audit: dict
    test_groups: np.ndarray | None = None


def _class_presence(y: np.ndarray) -> bool:
    return set(np.unique(y)) == {0, 1}


def within_domain_split(
    dataset: SampledDataset, *, mode: str = "stratified",
    test_size: float = 0.2, seed: int = 42,
) -> SplitAudit:
    """Only group/chronological splits support respective separation claims."""
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size must be between 0.05 and 0.5")
    if len(dataset.x) < 20:
        raise ValueError("Need at least 20 examples for a defensible split")
    y = np.asarray(dataset.y, dtype=int)
    ids = np.arange(len(y))
    if mode == "stratified":
        train, test = train_test_split(
            ids, test_size=test_size, random_state=seed, stratify=y,
        )
    elif mode == "group":
        if dataset.groups is None:
            raise ValueError("group split requires manifest group_column")
        if len(np.unique(dataset.groups)) < 3:
            raise ValueError("Group split requires at least three groups")
        # Seek a split with both classes represented, without optimizing metrics.
        splitter = GroupShuffleSplit(
            n_splits=20, test_size=test_size, random_state=seed,
        )
        valid = (
            (tr, te) for tr, te in splitter.split(
                ids, y, groups=dataset.groups
            ) if _class_presence(y[tr]) and _class_presence(y[te])
        )
        pair = next(valid, None)
        if pair is None:
            raise ValueError("No group-heldout split contains both classes")
        train, test = pair
        if set(dataset.groups[train]) & set(dataset.groups[test]):
            raise AssertionError("Group leakage between train and test")
    elif mode == "chronological":
        if dataset.times is None:
            raise ValueError("chronological split requires manifest time_column")
        # All observations at the cut timestamp go together.
        ordered = np.sort(np.unique(dataset.times))
        if len(ordered) < 3:
            raise ValueError("Not enough distinct timestamps")
        split_at = max(1, min(len(ordered) - 1,
                              int(round(len(ordered) * (1 - test_size)))))
        threshold = ordered[split_at]
        train = ids[dataset.times < threshold]
        test = ids[dataset.times >= threshold]
        if not len(train) or not len(test):
            raise ValueError("Chronological split produced empty partition")
        if np.max(dataset.times[train]) >= np.min(dataset.times[test]):
            raise AssertionError("Chronological ordering violated")
    else:
        raise ValueError("split must be stratified, group or chronological")

    if not _class_presence(y[train]) or not _class_presence(y[test]):
        raise ValueError("Both partitions must contain both labels")
    # Imputation is NOT performed here; baseline preprocessing is fitted
    # on the training portion only. NaN values remain in the split.
    split = DataSplit(
        x_train=dataset.x.iloc[train].reset_index(drop=True),
        x_test=dataset.x.iloc[test].reset_index(drop=True),
        y_train=y[train], y_test=y[test], excluded_columns=[],
        label_column="explicit_binary_manifest",
    )
    return SplitAudit(split=split, audit={
        "mode": mode, "seed": seed, "train": len(train), "test": len(test),
        "group_overlap": 0 if mode == "group" else None,
        "time_separated": mode == "chronological",
        "train_class_counts": {"normal": int(sum(y[train] == 0)),
                               "attack": int(sum(y[train] == 1))},
        "test_class_counts": {"normal": int(sum(y[test] == 0)),
                              "attack": int(sum(y[test] == 1))},
        "warning": ("Random stratification does not prevent device/session leakage"
                    if mode == "stratified"
                    else "Separation is based on supplied metadata; verify provenance"),
    }, test_groups=dataset.groups[test] if mode == "group" else None)


def _numeric_columns(train: pd.DataFrame) -> list[str]:
    eligible = [c for c in train if train[c].notna().any()
                and train[c].nunique(dropna=True) > 1]
    if len(eligible) < 2:
        raise ValueError("At least two usable varying numeric predictors required")
    return eligible


def fitted_baseline(x_train: pd.DataFrame, y_train: np.ndarray,
                    *, seed: int = 42):
    """Train-only imputer + random forest with bounded baseline complexity."""
    cols = _numeric_columns(x_train)
    clf = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=False),
        RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2,
            random_state=seed, n_jobs=1, class_weight="balanced_subsample",
        ),
    )
    clf.fit(x_train[cols], y_train)
    return clf, cols


def metric_summary(y_true: np.ndarray, predicted: np.ndarray,
                   probabilities: np.ndarray) -> dict:
    report = scores_for(y_true, predicted)
    report["mcc"] = float(matthews_corrcoef(y_true, predicted))
    report["roc_auc"] = (float(roc_auc_score(y_true, probabilities))
                         if _class_presence(y_true) else None)
    report["pr_auc"] = (float(average_precision_score(y_true, probabilities))
                        if _class_presence(y_true) else None)
    return report


def bootstrap_intervals(y_true: np.ndarray, predicted: np.ndarray,
                        *, repetitions: int = 300, seed: int = 42,
                        groups: np.ndarray | None = None) -> dict:
    """Percentile CIs; cluster-resampling when group metadata is available.

    If no groups are supplied, samples are stratified by class. That approach
    assumes independently sampled events and can understate uncertainty.
    """
    if repetitions < 20:
        raise ValueError("repetitions must be >= 20")
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    if not _class_presence(y):
        raise ValueError("Both classes required for bootstrap")
    rng = np.random.default_rng(seed)
    normal, attacks = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    by_group = None
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("groups must align with evaluation labels")
        unique = np.unique(groups)
        if len(unique) < 3:
            raise ValueError("Cluster bootstrap requires >= 3 evaluation groups")
        by_group = {g: np.flatnonzero(groups == g) for g in unique}
    collected = {"attack_recall": [], "attack_f1": [], "false_positive_rate": []}
    for _ in range(repetitions):
        if by_group is None:
            inds = np.r_[rng.choice(normal, len(normal), replace=True),
                         rng.choice(attacks, len(attacks), replace=True)]
        else:
            # A group is the sampling unit, preserving within-group correlation.
            for attempt in range(50):
                cluster_draw = rng.choice(unique, len(unique), replace=True)
                inds = np.concatenate([by_group[g] for g in cluster_draw])
                if _class_presence(y[inds]):
                    break
            else:
                raise ValueError("Insufficient independent class-containing groups")
        scores = scores_for(y[inds], pred[inds])
        for field in collected:
            collected[field].append(scores[field])
    return {key: {
        "low_95": float(np.quantile(values, 0.025)),
        "high_95": float(np.quantile(values, 0.975)),
    } for key, values in collected.items()}

def evaluate_within(
    dataset: SampledDataset, *, split_mode: str = "stratified",
    seed: int = 42, repetitions: int = 300,
) -> dict:
    plan = within_domain_split(dataset, mode=split_mode, seed=seed)
    split = plan.split
    model, cols = fitted_baseline(split.x_train, split.y_train, seed=seed)
    predictions = model.predict(split.x_test[cols]).astype(int)
    probs = model.predict_proba(split.x_test[cols])[:, 1]
    return {
        "status": "synthetic_fixture" if dataset.dataset_id.startswith("synthetic")
                  else "local_dataset_experiment_unreviewed",
        "method": "within_dataset_random_forest",
        "dataset": dataset.dataset_id, "sampling": dataset.audit,
        "split_audit": plan.audit, "features": cols, "units": dataset.units,
        "metrics": metric_summary(split.y_test, predictions, probs),
        "confidence_intervals": bootstrap_intervals(
            split.y_test, predictions, seed=seed, repetitions=repetitions,
            groups=plan.test_groups,
        ),
        "interval_sampling_unit": ("group" if plan.test_groups is not None
                                   else "individual_event"),
        "limitations": [
            "Bootstrap assumes independent evaluation observations.",
            "Dataset sampling can change observed attack prevalence.",
            "Feature leakage and class mappings require human review.",
            "A single benchmark cannot demonstrate real-world readiness.",
        ],
    }


def evaluate_transfer(source: SampledDataset, target: SampledDataset,
                      *, seed: int = 42, repetitions: int = 300) -> dict:
    """Train on source only; no target labels or statistics influence fitting."""
    if source.dataset_id == target.dataset_id:
        raise ValueError("Transfer requires distinct source and target datasets")
    if source.audit["task"] != target.audit["task"]:
        raise ValueError("Incompatible task definitions: cannot directly compare "
                         "binary intrusion with an FDI-specific endpoint")
    shared = sorted(set(source.x) & set(target.x))
    incompatible = [c for c in shared if source.units[c] != target.units[c]]
    if incompatible:
        raise ValueError("Incompatible units across datasets: " + ", ".join(incompatible))
    if len(shared) < 2:
        raise ValueError("At least two canonical features shared across domains")
    if not _class_presence(source.y) or not _class_presence(target.y):
        raise ValueError("Both domains must contain benign and attack examples")
    model, cols = fitted_baseline(source.x[shared], source.y, seed=seed)
    pred = model.predict(target.x[cols]).astype(int)
    prob = model.predict_proba(target.x[cols])[:, 1]
    # Source-only scale for descriptive shift; this never tunes model decisions.
    ref = source.x[cols].median()
    scale = (source.x[cols].quantile(.75)
             - source.x[cols].quantile(.25)).replace(0, 1)
    drift = ((target.x[cols].median() - ref).abs() / scale).replace(
        [np.inf, -np.inf], np.nan,
    ).fillna(0)
    return {
        "status": "synthetic_fixture" if (
            source.dataset_id.startswith("synthetic") or
            target.dataset_id.startswith("synthetic")
        ) else "local_cross_dataset_experiment_unreviewed",
        "method": "zero_target_training_cross_dataset_transfer",
        "source": source.dataset_id, "target": target.dataset_id,
        "source_audit": source.audit, "target_audit": target.audit,
        "task_compatibility": ("same_declared_task" if
                               source.audit["task"] == target.audit["task"]
                               else "different_attack_task_not_directly_comparable"),
        "features": cols, "feature_count": len(cols),
        "median_shift_scaled_by_source_iqr": {
            col: float(drift[col]) for col in cols
        },
        "metrics": metric_summary(target.y, pred, prob),
        "confidence_intervals": bootstrap_intervals(
            target.y, pred, seed=seed, repetitions=repetitions,
        ),
        "limitations": [
            "Different attack families may define different targets.",
            "Raw feature units can match while collection semantics differ.",
            "No target observations or labels were used in model fitting.",
            "Bootstrap assumes independent target observations.",
            "Source/target data provenance, splits and leakage require review.",
        ],
    }
