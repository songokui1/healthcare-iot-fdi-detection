"""Manifest-driven, bounded-memory loading for heterogeneous IoMT/IoT datasets.

Nothing is downloaded automatically. Feature mappings, units, and labels are
explicit: accidental schema guesses can invalidate cross-dataset experiments.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .data import BLOCKED

CANONICAL = frozenset({
    "duration", "src_packets", "dst_packets", "total_packets",
    "src_bytes", "dst_bytes", "total_bytes", "packet_rate",
    "byte_rate", "mean_packet_size", "inter_arrival", "jitter",
    "tcp_flags", "src_port", "dst_port", "temperature", "pulse_rate",
    "respiratory_rate", "spo2", "ecg_st", "flow_packets",
})

_ID = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{1,63}$")


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    files: tuple[Path, ...]
    label_column: str
    benign_values: tuple[str, ...]
    attack_values: tuple[str, ...]
    features: dict[str, str]
    units: dict[str, str]
    task: str
    source_url: str
    group_column: str | None = None
    time_column: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSpec":
        manifest = Path(path).resolve()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        required = {
            "dataset_id", "files", "label_column", "benign_values",
            "attack_values", "features", "units", "task", "source_url",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Manifest lacks explicit fields: {sorted(missing)}")
        if not _ID.fullmatch(data["dataset_id"]):
            raise ValueError("Unsafe dataset identifier")
        files = data["files"]
        if not isinstance(files, list) or not files or not all(
            isinstance(f, str) and f.strip() for f in files
        ):
            raise ValueError("files must be a nonempty list of CSV paths")
        resolved = tuple((manifest.parent / f).resolve() for f in files)
        if any(p.suffix.lower() != ".csv" for p in resolved):
            raise ValueError("Only explicit CSV inputs supported; convert PCAP separately")
        features = data["features"]
        if not isinstance(features, dict) or not features:
            raise ValueError("Explicit canonical-to-dataset feature mapping required")
        if any(k not in CANONICAL for k in features):
            raise ValueError("Unrecognized canonical feature; audit mapping and units")
        if any(not isinstance(v, str) or not v.strip() or BLOCKED.search(v)
               for v in features.values()):
            raise ValueError("Feature mapping contains missing or label/identifier-like column")
        if len(set(features.values())) != len(features):
            raise ValueError("Each raw column can map to one canonical feature only")
        units = data["units"]
        if not isinstance(units, dict) or set(units) != set(features):
            raise ValueError("A physical unit is required for every mapped feature")
        if any(not isinstance(u, str) or not u.strip() for u in units.values()):
            raise ValueError("Unit names must be explicit")
        benign = tuple(_norm(v) for v in data["benign_values"])
        attack = tuple(_norm(v) for v in data["attack_values"])
        if not benign or not attack or set(benign) & set(attack):
            raise ValueError("Specify nonoverlapping benign and attack label values")
        if not data["label_column"] or data["label_column"] in features.values():
            raise ValueError("Label column cannot be a predictor")
        for key in ("group_column", "time_column"):
            value = data.get(key)
            if value and value in features.values():
                raise ValueError(f"{key} cannot be a predictor")
        if data.get("group_column") == data.get("label_column"):
            raise ValueError("Group cannot be label")
        if data.get("time_column") == data.get("label_column"):
            raise ValueError("Time cannot be label")
        if data["task"] not in {"binary_intrusion", "fdi_specific"}:
            raise ValueError("Task must be binary_intrusion or fdi_specific")
        if data["task"] == "fdi_specific" and "fdi" not in data.get("target_note", "").lower():
            raise ValueError("FDI-specific task requires explicit target_note explaining labels")
        return cls(
            dataset_id=data["dataset_id"], files=resolved,
            label_column=data["label_column"],
            benign_values=benign, attack_values=attack,
            features=dict(features), units=dict(units),
            task=data["task"], source_url=data["source_url"],
            group_column=data.get("group_column"),
            time_column=data.get("time_column"),
        )

    @property
    def columns(self) -> list[str]:
        fields = list(self.features.values()) + [self.label_column]
        for extra in (self.group_column, self.time_column):
            if extra and extra not in fields:
                fields.append(extra)
        return fields


def _norm(value: object) -> str:
    if value is None:
        raise ValueError("Missing labels are not permitted")
    return str(value).strip().casefold()


def _convert_chunk(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    absent = set(spec.columns) - set(frame.columns)
    if absent:
        raise ValueError(f"{spec.dataset_id}: missing CSV fields: {sorted(absent)}")
    label_raw = frame[spec.label_column]
    if label_raw.isna().any():
        raise ValueError("Missing ground-truth labels")
    labels = label_raw.map(_norm)
    known = set(spec.benign_values) | set(spec.attack_values)
    unexpected = set(labels.unique()) - known
    if unexpected:
        raise ValueError(f"Unknown labels {sorted(unexpected)[:5]}; "
                         "map every observed class explicitly in manifest")
    out = pd.DataFrame({
        canonical: pd.to_numeric(frame[raw], errors="coerce")
        for canonical, raw in spec.features.items()
    })
    out = out.replace([np.inf, -np.inf], np.nan)
    out["__label__"] = labels.isin(spec.attack_values).astype("int8").to_numpy()
    if spec.group_column:
        if frame[spec.group_column].isna().any():
            raise ValueError("Missing group identifiers")
        out["__group__"] = frame[spec.group_column].astype(str).to_numpy()
    if spec.time_column:
        raw_time = frame[spec.time_column]
        parsed = pd.to_numeric(raw_time, errors="coerce")
        if parsed.notna().all():
            out["__time__"] = parsed.astype("float64").to_numpy()
        else:
            dates = pd.to_datetime(raw_time, utc=True, errors="coerce")
            if dates.isna().any():
                raise ValueError("Time column must be consistently numeric or timestamp")
            out["__time__"] = dates.astype("int64").to_numpy()
    return out


def iter_chunks(spec: DatasetSpec, *, chunk_rows: int = 50_000
                ) -> Iterator[pd.DataFrame]:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    for path in spec.files:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset CSV unavailable locally: {path}")
        reader = pd.read_csv(path, usecols=spec.columns, chunksize=chunk_rows)
        for chunk in reader:
            if not chunk.empty:
                yield _convert_chunk(chunk, spec)


@dataclass
class SampledDataset:
    dataset_id: str
    x: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray | None
    times: np.ndarray | None
    units: dict[str, str]
    audit: dict


def uniform_reservoir(spec: DatasetSpec, *, max_rows: int = 20_000,
                      seed: int = 42, chunk_rows: int = 50_000
                      ) -> SampledDataset:
    """Uniform without-replacement reservoir using smallest random priorities.

    Runtime O(total rows), retained memory O(max_rows + chunk_rows).
    A sampled pilot is not a population-accurate prevalence estimate.
    """
    if max_rows < 10:
        raise ValueError("max_rows must be >= 10")
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    total = normal = attacks = 0
    for chunk in iter_chunks(spec, chunk_rows=chunk_rows):
        total += len(chunk)
        attacks += int(chunk["__label__"].sum())
        normal += int(len(chunk) - chunk["__label__"].sum())
        chunk["__priority__"] = rng.random(len(chunk))
        merged = chunk if reservoir is None else pd.concat(
            [reservoir, chunk], ignore_index=True,
        )
        reservoir = merged.nsmallest(max_rows, "__priority__").copy()
    if reservoir is None or total == 0:
        raise ValueError("Dataset has no rows")
    reservoir = reservoir.sort_values("__priority__").reset_index(drop=True)
    y = reservoir["__label__"].to_numpy(dtype=int)
    if set(y) != {0, 1}:
        raise ValueError("Retained pilot lacks one class; expand max_rows or "
                         "obtain more data before evaluation")
    cols = list(spec.features)
    return SampledDataset(
        dataset_id=spec.dataset_id,
        x=reservoir[cols].copy(), y=y,
        groups=reservoir["__group__"].to_numpy()
               if "__group__" in reservoir else None,
        times=reservoir["__time__"].to_numpy()
               if "__time__" in reservoir else None,
        units=dict(spec.units),
        audit={
            "dataset_id": spec.dataset_id, "task": spec.task,
            "source_url": spec.source_url,
            "rows_seen": total, "rows_retained": len(reservoir),
            "original_class_counts": {"normal": normal, "attack": attacks},
            "selected_class_counts": {"normal": int(sum(y == 0)),
                                      "attack": int(sum(y == 1))},
            "sampling": "uniform_without_replacement" if len(reservoir) < total
                        else "entire_dataset",
            "seed": seed,
            "caution": "Sampled pilot; do not infer deployed class prevalence "
                       "or generalization from a random-row split.",
        },
    )


def file_sha256(path: str | Path, *, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()
