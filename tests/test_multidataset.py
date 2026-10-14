"""Tests for multi-dataset manifests, leakage controls, splits, transfer and replay."""
import json

import numpy as np
import pandas as pd
import pytest

from medguard.datasets import DatasetSpec, uniform_reservoir
from medguard.generalization import (
    bootstrap_intervals, evaluate_transfer, evaluate_within,
    within_domain_split,
)
from medguard.replay import replay_transfer
from medguard.research_cli import _demo_files, main, run_demo


def _fixture(tmp_path, *, name="synthetic_alpha", seed=8, unit="ms"):
    rng = np.random.default_rng(seed)
    y = np.array([0] * 120 + [1] * 40)
    rng.shuffle(y)
    frame = pd.DataFrame({
        "label": y,
        "Packet Count": rng.poisson(14, len(y)) + 5 * y,
        "Jitter": rng.normal(3, 0.5, len(y)) + 3 * y,
        "Source MAC": ["aa:bb:cc:dd"] * len(y),
        "device_id": [f"device{i % 16}" for i in range(len(y))],
        "event_time": np.arange(len(y)),
        "Attack Category": ["benign" if not v else "attack" for v in y],
    })
    csv = tmp_path / f"{name}.csv"
    frame.to_csv(csv, index=False)
    config = {
        "dataset_id": name, "files": [csv.name],
        "label_column": "label", "benign_values": ["0"],
        "attack_values": ["1"], "source_url": "synthetic://ci",
        "task": "binary_intrusion",
        "features": {"total_packets": "Packet Count", "jitter": "Jitter"},
        "units": {"total_packets": "count", "jitter": unit},
        "group_column": "device_id", "time_column": "event_time",
    }
    profile = tmp_path / f"{name}.json"
    profile.write_text(json.dumps(config))
    return profile


def test_manifest_filters_attack_labels_identifiers_and_metadata(tmp_path):
    profile = _fixture(tmp_path)
    spec = DatasetSpec.load(profile)
    sample = uniform_reservoir(spec, max_rows=70, seed=3, chunk_rows=19)
    assert list(sample.x) == ["total_packets", "jitter"]
    assert "Source MAC" not in sample.x
    assert "Attack Category" not in sample.x
    assert len(sample.x) == 70
    assert sample.audit["rows_seen"] == 160
    assert sample.audit["rows_retained"] == 70
    assert sample.audit["sampling"] == "uniform_without_replacement"
    assert sample.groups is not None
    assert sample.times is not None
    assert sum(sample.audit["original_class_counts"].values()) == 160


def test_reservoir_independent_of_csv_chunks(tmp_path):
    spec = DatasetSpec.load(_fixture(tmp_path))
    a = uniform_reservoir(spec, max_rows=55, seed=17, chunk_rows=13)
    b = uniform_reservoir(spec, max_rows=55, seed=17, chunk_rows=41)
    pd.testing.assert_frame_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_invalid_label_and_leakage_mapping_are_rejected(tmp_path):
    file = _fixture(tmp_path)
    cfg = json.loads(file.read_text())
    cfg["features"]["total_packets"] = "Attack Category"
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="identifier-like"):
        DatasetSpec.load(file)
    cfg["features"]["total_packets"] = "Packet Count"
    cfg["attack_values"] = ["99"]
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Unknown labels"):
        uniform_reservoir(DatasetSpec.load(file))


def test_group_and_chronological_holdouts(tmp_path):
    spec = DatasetSpec.load(_fixture(tmp_path))
    sample = uniform_reservoir(spec, max_rows=160, seed=1)
    grouped = within_domain_split(sample, mode="group", seed=4)
    assert grouped.audit["group_overlap"] == 0
    chrono = within_domain_split(sample, mode="chronological", seed=4)
    assert chrono.audit["time_separated"]
    assert set(chrono.split.y_test) == {0, 1}
    for method in ("group", "chronological"):
        report = evaluate_within(sample, split_mode=method,
                                 repetitions=20, seed=4)
        assert 0 <= report["metrics"]["attack_recall"] <= 1


def test_cross_dataset_transfer_is_zero_target_fit(tmp_path):
    first = uniform_reservoir(
        DatasetSpec.load(_fixture(tmp_path, name="synthetic_training", seed=4)),
        max_rows=120,
    )
    second = uniform_reservoir(
        DatasetSpec.load(_fixture(tmp_path, name="synthetic_external", seed=7)),
        max_rows=120,
    )
    transfer = evaluate_transfer(first, second, seed=4, repetitions=20)
    assert transfer["source"] != transfer["target"]
    assert transfer["feature_count"] == 2
    assert transfer["metrics"]["n"] == 120
    assert transfer["confidence_intervals"]["attack_recall"]["low_95"] <= 1.0


def test_mismatched_units_fail_closed(tmp_path):
    first = uniform_reservoir(DatasetSpec.load(_fixture(tmp_path, name="synthetic_a")))
    second = uniform_reservoir(
        DatasetSpec.load(_fixture(tmp_path, name="synthetic_b", unit="seconds"))
    )
    with pytest.raises(ValueError, match="Incompatible units"):
        evaluate_transfer(first, second, repetitions=20)
    with pytest.raises(ValueError, match="Conflicting feature units"):
        replay_transfer(first, DatasetSpec.load(
            tmp_path / "synthetic_b.json"), max_events=30)


def test_bounded_replay_and_no_llm_calls(tmp_path):
    source = uniform_reservoir(
        DatasetSpec.load(_fixture(tmp_path, name="synthetic_src")),
        max_rows=120,
    )
    target_spec = DatasetSpec.load(
        _fixture(tmp_path, name="synthetic_dst", seed=5)
    )
    report = replay_transfer(source, target_spec, max_events=49, chunk_rows=11)
    assert report["target_events_processed"] == 49
    assert sum(report["confusion"].values()) == 49
    assert report["chunks"] == 5
    assert report["status"] == "synthetic_fixture"
    assert report["events_per_second_including_csv_ingestion"] > 0


def test_bootstrap_not_point_estimate_claim():
    actual = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 0, 1])
    ci = bootstrap_intervals(actual, pred, repetitions=30, seed=11)
    assert 0 <= ci["attack_f1"]["low_95"] <= ci["attack_f1"]["high_95"] <= 1


def test_complete_multidataset_demo_is_synthetic(tmp_path):
    result = run_demo(seed=9)
    assert result["status"] == "synthetic_fixture_not_scientific_evidence"
    assert result["within"]["status"] == "synthetic_fixture"
    assert result["transfer"]["status"] == "synthetic_fixture"
    assert result["replay"]["status"] == "synthetic_fixture"
    out = tmp_path / "aggregate.json"
    assert main(["demo", "--seed", "9", "--output", str(out)]) == 0
    obj = json.loads(out.read_text())
    assert obj["status"] == "synthetic_fixture_not_scientific_evidence"
    assert not any(key in obj for key in ("raw_posts", "raw_packets", "api_key"))


def test_cli_inspect_only_outputs_aggregate(tmp_path, capsys):
    profile = _fixture(tmp_path)
    assert main(["inspect", "--source", str(profile),
                 "--source-limit", "90"]) == 0
    stdout = capsys.readouterr().out
    audit = json.loads(stdout)
    assert audit["sample"]["rows_seen"] == 160
    assert audit["sample"]["rows_retained"] == 90
    assert "aa:bb:cc:dd" not in stdout
