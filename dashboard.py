"""Streamlit dashboard for aggregate research outputs. No raw telemetry loaded."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from medguard.reporting import dashboard_projection

st.set_page_config(page_title="MedGuard-AI Research Lab", page_icon="🛡️",
                   layout="wide")

ROOT = Path(__file__).resolve().parent
OUTPUTS = (ROOT / "outputs").resolve()
st.title("MedGuard-AI")
st.caption("Cross-dataset IoMT research • LLM/RAG/hybrid investigation • local replay")
st.sidebar.header("Read a saved aggregate report")
entry = st.sidebar.text_input(
    "JSON file under outputs/",
    value="outputs/v2-demo.json",
)
st.sidebar.caption(
    "Supported: v1 LLM comparisons, v2 within-domain, transfer, replay, "
    "the fully synthetic demonstration, and synthetic scale tests. No individual events are loaded."
)
st.sidebar.markdown("**Rights:** All rights reserved. Research demonstration only.")


def _render(data: dict) -> None:
    kind = data["kind"]
    st.subheader(data["title"])
    if kind == "llm":
        rows = data["rows"]
        display = pd.DataFrame(rows).set_index("approach")
        n = data.get("sample_size")
        if n is not None:
            st.metric("Test events", n)
        cols = [c for c in ("attack_recall", "attack_f1") if c in display]
        if cols:
            st.bar_chart(display[cols])
        st.dataframe(display, use_container_width=True)
        if data.get("baseline"):
            st.write("Fresh same-split LightGBM reference",
                     pd.DataFrame([data["baseline"]]))
        st.caption("LLM self-reported confidence is not calibrated.")
        return
    if kind == "detection":
        m = data["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Examples", m.get("n", "?"))
        c2.metric("Attack recall", _pct(m.get("attack_recall")))
        c3.metric("Attack F1", _pct(m.get("attack_f1")))
        c4.metric("False negative rate", _pct(m.get("false_negative_rate")))
        st.dataframe(pd.DataFrame([m]), hide_index=True,
                     use_container_width=True)
        if data.get("source") or data.get("target"):
            st.caption(
                "Transfer source: " + str(data.get("source")) +
                " · Target: " + str(data.get("target")) +
                " · Task compatibility: " + str(data.get("task_compatibility"))
            )
        else:
            st.caption("Dataset: " + str(data.get("dataset")))
        st.caption("Canonical features: " + ", ".join(data.get("features", [])))
        cis = data.get("confidence_intervals", {})
        if cis:
            st.markdown("**Class-stratified bootstrap intervals (pilot only)**")
            st.dataframe(pd.DataFrame(cis).T, use_container_width=True)
        return
    if kind == "replay":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events replayed", data.get("events", 0))
        c2.metric("Events/s incl. CSV", data.get("throughput_eps"))
        c3.metric("Median batch infer. ms", data.get("median_batch_ms"))
        c4.metric("p95 batch infer. ms", data.get("p95_batch_ms"))
        st.caption("Source: " + str(data.get("source")) +
                   " · Destination: " + str(data.get("target")))
        st.dataframe(pd.DataFrame([data.get("confusion", {})]),
                     hide_index=True, use_container_width=True)
        st.warning("Batch latency is not per-event p95. These are local "
                   "CPU and CSV figures, not live IoMT or LLM throughput.")


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:.1f}%"


try:
    selected = Path(entry).expanduser()
    selected = (ROOT / selected).resolve() if not selected.is_absolute() \
        else selected.resolve()
    # A local analyst cannot accidentally display a private file outside
    # the project's predesignated generated-output directory.
    if not selected.is_relative_to(OUTPUTS):
        raise ValueError("Select an aggregate JSON report within outputs/")
    if not selected.is_file():
        st.info(
            "No aggregate report at that location. Start with:\n\n"
            "    python -m medguard.research_cli demo "
            "--output outputs/v2-demo.json"
        )
        st.stop()
    if selected.stat().st_size > 2_000_000:
        raise ValueError("Aggregate report exceeds 2 MB; no raw event logs allowed")
    report = dashboard_projection(
        json.loads(selected.read_text(encoding="utf-8"))
    )
except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
    st.error(f"Cannot display aggregate report: {exc}")
    st.stop()

if report["synthetic"]:
    st.error(
        "SYNTHETIC OR MOCK RESULTS — engineering smoke tests only. "
        "These are not WUSTL/CIC accuracy results or clinical evidence."
    )
else:
    st.warning(
        "LOCAL EXPERIMENTAL OUTPUT — dataset provenance, label mapping, "
        "feature leakage, and statistical validity require independent review."
    )

if report["kind"] == "synthetic_demo":
    labels = ["Within-domain", "Cross-domain", "Replay"]
    tabs = st.tabs(labels)
    for tab, section in zip(tabs, report["sections"]):
        with tab:
            _render(section)
elif report["kind"] == "synthetic_scale":
    st.metric("Synthetic rows generated per domain",
              report["generated_rows_per_domain"])
    st.metric("Python traced peak allocations (MB)",
              report["peak_python_tracemalloc_mb"])
    st.caption("This is not process RSS; native array allocations are excluded.")
    _render(report["sections"][0])
else:
    _render(report)

st.divider()
st.caption("No remote calls. No clinical diagnoses or autonomous interventions. "
           "Source code is public; raw datasets, keys and event-level records remain local.")
