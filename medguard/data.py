"""Leakage-conscious loading, held-out splitting, and telemetry summarization."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# These columns are excluded even when numeric. MAC-derived labels make these
# fields particularly risky. Users must also review dataset-specific proxies.
BLOCKED = re.compile(
    r"(label|target|attack|category|class|mac|src.?addr|dst.?addr|"
    r"source.?addr|destination.?addr|ip.?addr|src.?ip|dst.?ip|"
    r"source.?ip|destination.?ip|^id$|^index$|patient.?id|"
    r"time.?stamp|^date$|session.?id|flow.?id)",
    re.IGNORECASE,
)

@dataclass
class DataSplit:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    excluded_columns: list[str]
    label_column: str

def encode_binary_labels(series: pd.Series) -> np.ndarray:
    mapping = {"0": 0, "normal": 0, "benign": 0, "legitimate": 0,
               "1": 1, "attack": 1, "malicious": 1, "anomaly": 1}
    values = series.astype(str).str.strip().str.lower()
    invalid = sorted(set(values) - set(mapping))
    if invalid:
        raise ValueError("Binary labels must be 0/1 or normal/attack. "
                         "Unexpected label examples: " + repr(invalid[:5]))
    result = values.map(mapping).to_numpy(dtype=int)
    if len(set(result)) != 2:
        raise ValueError("Dataset must contain both normal and attack classes")
    return result

def load_split(
    path: str, *, label_column: str = "Label", test_size: float = 0.2,
    seed: int = 42, exclude: tuple[str, ...] = (),
) -> DataSplit:
    """Split before learning imputations. Never use labels as model inputs."""
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size must be between 0.05 and 0.5")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError("Empty CSV")
    by_lower = {str(c).lower(): c for c in df.columns}
    key = by_lower.get(label_column.lower())
    if key is None:
        raise ValueError(f"Label column {label_column!r} not found")
    labels = encode_binary_labels(df[key])
    forbidden = [c for c in df.columns if c == key or BLOCKED.search(str(c))
                 or str(c).lower() in {e.lower() for e in exclude}]
    candidates = df.drop(columns=forbidden)
    # Split before fitting *any* feature selection or numerical imputations.
    train_rows, test_rows, y_train, y_test = train_test_split(
        candidates, labels, test_size=test_size, random_state=seed,
        stratify=labels
    )
    # Never send raw IPs, identifiers, strings, or free text into prompts.
    train_numeric = train_rows.apply(pd.to_numeric, errors="coerce")
    test_numeric = test_rows.apply(pd.to_numeric, errors="coerce")
    train_numeric = train_numeric.replace([np.inf, -np.inf], np.nan)
    test_numeric = test_numeric.replace([np.inf, -np.inf], np.nan)
    retained = [c for c in train_numeric
                if train_numeric[c].notna().mean() >= 0.95
                and train_numeric[c].nunique(dropna=True) > 1]
    dropped = forbidden + [c for c in candidates if c not in retained]
    if not retained:
        raise ValueError("No safe varying numeric predictors remain")
    x_train, x_test = train_numeric[retained], test_numeric[retained]
    # Training-only imputation: test distribution never influences parameters.
    medians = x_train.median().fillna(0.0)
    x_train = x_train.fillna(medians).reset_index(drop=True)
    x_test = x_test.fillna(medians).reset_index(drop=True)
    return DataSplit(x_train, x_test, y_train, y_test,
                     [str(c) for c in dropped], str(key))

def balanced_test_indices(y: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """A reproducible stratified evaluation subset, never used in training."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    if limit >= len(y):
        return np.arange(len(y))
    # Stratification is explicit so positives are represented at small budgets.
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    pos_n = max(1, min(len(positives), round(limit * len(positives) / len(y))))
    neg_n = limit - pos_n
    if neg_n > len(negatives):
        neg_n = len(negatives)
        pos_n = limit - neg_n
    chosen = np.r_[rng.choice(positives, pos_n, replace=False),
                   rng.choice(negatives, neg_n, replace=False)]
    rng.shuffle(chosen)
    return chosen

@dataclass
class Summarizer:
    """Training-benign reference ranges; no test labels or test stats used."""
    median: pd.Series
    scale: pd.Series
    columns: list[str]

    @classmethod
    def fit(cls, x_train: pd.DataFrame, y_train: np.ndarray) -> "Summarizer":
        benign = x_train.iloc[np.flatnonzero(y_train == 0)]
        if benign.empty:
            raise ValueError("No benign training samples")
        median = benign.median()
        # Robust scale avoids inflating scores due to outliers in training.
        iqr = (benign.quantile(0.75) - benign.quantile(0.25))
        scale = iqr.where(iqr > 1e-9, benign.std(ddof=0))
        scale = scale.where(scale > 1e-9, 1.0)
        return cls(median, scale, list(x_train.columns))

    def summarize(self, row: pd.Series, *, max_features: int = 10) -> dict[str, Any]:
        if max_features < 1:
            raise ValueError("max_features must be positive")
        z = ((row[self.columns] - self.median) / self.scale).abs()
        ranking = z.nlargest(min(max_features, len(z))).index
        features = []
        for name in ranking:
            value = float(row[name])
            if not np.isfinite(value):
                continue
            features.append({
                "feature": str(name), "value": round(value, 6),
                "benign_median": round(float(self.median[name]), 6),
                "relative_deviation": round(float(z[name]), 3),
            })
        return {"type": "numeric_iomt_telemetry", "features": features,
                "note": "Deviations are relative to TRAINING benign data; "
                        "large deviation alone does not prove an attack."}

def synthetic_demo(seed: int = 7, rows: int = 200) -> DataSplit:
    """Clearly synthetic, non-clinical fixture for demos and CI only."""
    if rows < 30:
        raise ValueError("rows must be >= 30")
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1], rows, p=[0.8, 0.2])
    if len(set(y)) != 2:
        y[:2] = [0, 1]
    x = pd.DataFrame({
        "Source Port": rng.normal(4500, 600, rows) + y * 2200,
        "Destination Jitter": rng.exponential(3, rows) + y * 6,
        "Packet Count": rng.poisson(12, rows) + y * 12,
        "Inter Packet Time": rng.exponential(1, rows) + y * 2,
        "Temperature": rng.normal(37, 0.35, rows) + y * 0.3,
        "Pulse Rate": rng.normal(77, 12, rows) + y * 6,
    })
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=seed
    )
    return DataSplit(xtr.reset_index(drop=True), xte.reset_index(drop=True),
                     ytr, yte, [], "synthetic_label")
