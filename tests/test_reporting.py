"""Dashboard projection tests: status labels, explicit schemas, no raw-row display."""
import pytest

from medguard.analyst import MockProvider
from medguard.data import synthetic_demo
from medguard.experiment import run_comparison
from medguard.reporting import dashboard_projection
from medguard.research_cli import run_demo


def test_v1_mock_is_never_described_as_real_llm():
    doc = run_comparison(synthetic_demo(rows=120), MockProvider(),
                         max_test_rows=12)
    doc["raw_sensitive_id"] = "NEVER_RENDER_ME"
    rendered = dashboard_projection(doc)
    assert rendered["kind"] == "llm"
    assert rendered["synthetic"] is True
    assert len(rendered["rows"]) == 3
    assert "NEVER_RENDER_ME" not in str(rendered)


def test_multidataset_demo_is_bounded_to_safe_outputs():
    report = run_demo(seed=2)
    report["patient_data"] = [{"name": "SENSITIVE_IDENTIFIER"}]
    safe = dashboard_projection(report)
    assert safe["kind"] == "synthetic_demo"
    assert safe["synthetic"] is True
    assert [section["kind"] for section in safe["sections"]] == [
        "detection", "detection", "replay"
    ]
    assert "SENSITIVE_IDENTIFIER" not in str(safe)
    assert all(section["synthetic"] for section in safe["sections"])


def test_invalid_or_forged_report_is_rejected():
    with pytest.raises(ValueError, match="Unknown"):
        dashboard_projection({"status": "peer_reviewed_production_safe"})
    with pytest.raises(ValueError, match="Incomplete"):
        dashboard_projection({
            "status": "synthetic_fixture_not_scientific_evidence",
            "within": {},
        })
    with pytest.raises(ValueError, match="No evaluation metrics"):
        dashboard_projection({
            "status": "local_dataset_experiment_unreviewed",
            "method": "within_dataset_random_forest",
        })


def test_reproducible_scale_harness_is_synthetic_and_capped():
    from medguard.scale import run_scale
    result = run_scale(rows=450, source_cap=120, replay_cap=175,
                       chunk_rows=37, seed=8)
    assert result["status"] == "synthetic_scale_smoke_only"
    assert result["generated_rows_per_domain"] == 450
    assert result["replay"]["target_events_processed"] == 175
    assert result["replay"]["status"] == "synthetic_fixture"
    assert result["peak_python_tracemalloc_mb"] > 0


def test_synthetic_scale_report_requires_synthetic_replay():
    from medguard.scale import run_scale
    doc = run_scale(rows=200, source_cap=50, replay_cap=70,
                    chunk_rows=19, seed=11)
    projection = dashboard_projection(doc)
    assert projection["kind"] == "synthetic_scale"
    assert projection["synthetic"] is True
    assert projection["generated_rows_per_domain"] == 200
    assert projection["sections"][0]["kind"] == "replay"
    doc["replay"]["status"] = "local_replay_unreviewed"
    with pytest.raises(ValueError, match="must contain synthetic"):
        dashboard_projection(doc)
