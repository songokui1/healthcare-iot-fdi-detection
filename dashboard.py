"""Aggregate-only Streamlit research dashboard; no individual telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MedGuard-AI Lab", page_icon="🛡️", layout="wide")
st.title("MedGuard-AI | Healthcare IoT Security Research")
st.caption("Experimental comparison of LLM-only, RAG+LLM, and anomaly-gated RAG+LLM")

path = st.sidebar.text_input("Local aggregate results JSON",
                             value="outputs/aggregate.json")
st.sidebar.caption("Only read aggregate experiment summaries; never select raw datasets.")
st.sidebar.markdown("**Privacy:** Full source and original research remain private.")

try:
    location = Path(path).expanduser()
    if not location.is_file():
        st.info("Run the private pipeline first: python -m medguard.cli "
                "--output outputs/aggregate.json")
        st.stop()
    payload = json.loads(location.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or not isinstance(
        payload.get("approaches"), dict
    ):
        raise ValueError("Not a MedGuard-AI aggregate report")
except (OSError, ValueError, json.JSONDecodeError) as exc:
    st.error(f"Cannot load aggregate report: {exc}")
    st.stop()

status = payload.get("status")
if status == "synthetic_mock_smoke_test":
    st.error("SYNTHETIC MOCK TEST — NOT ACTUAL LLM PERFORMANCE, "
             "NOT CLINICAL EVIDENCE, NOT A PUBLICATION RESULT.")
elif status == "experimental_llm_evaluation":
    st.warning("EXPLORATORY LLM RESULTS — require independent validation, "
               "dataset/proxy audit, larger samples, and expert review.")
else:
    st.error("Unknown report provenance. Do not interpret these metrics.")
    st.stop()

approaches = payload["approaches"]
if not approaches:
    st.error("No approach-level results in this report.")
    st.stop()

names = {
    "llm_direct": "LLM only",
    "llm_rag": "RAG + LLM",
    "hybrid_gate_rag": "Hybrid gate + RAG/LLM",
}
data = []
for key, metrics in approaches.items():
    data.append({
        "Approach": names.get(key, key),
        "Attack recall": metrics.get("attack_recall"),
        "Attack F1": metrics.get("attack_f1"),
        "False positive rate": metrics.get("false_positive_rate"),
        "False negative rate": metrics.get("false_negative_rate"),
        "LLM calls": metrics.get("llm_calls"),
        "Total wall seconds": metrics.get("wall_seconds_total"),
        "ms / event (end to end)": metrics.get("wall_ms_per_evaluated_event"),
        "TN": metrics.get("tn"), "FP": metrics.get("fp"),
        "FN": metrics.get("fn"), "TP": metrics.get("tp"),
    })
frame = pd.DataFrame(data).set_index("Approach")
c1, c2, c3 = st.columns(3)
c1.metric("Held-out events", payload["data"]["test_examples_evaluated"])
c2.metric("Provider", payload.get("provider", "unknown"))
c3.metric("Total LLM calls", int(frame["LLM calls"].sum()))
st.subheader("Predictive performance")
st.bar_chart(frame[["Attack recall", "Attack F1"]], y_label="0–1 score")
st.subheader("Operational costs and false negatives")
st.dataframe(frame, use_container_width=True)
st.markdown("**Gate screening:** " +
            str(payload.get("gate", {}).get("flagged_events", "?")) +
            " selected events for the hybrid LLM.")
baseline = payload.get("conventional_baseline")
if baseline:
    st.subheader("Fresh, same-split LightGBM comparison")
    st.write({k: baseline.get(k) for k in (
        "attack_recall", "attack_f1", "accuracy", "training_seconds",
        "wall_ms_per_evaluated_event")})
st.subheader("Limitations")
for item in payload.get("limitations", []):
    st.markdown("- " + str(item))
st.caption("Read-only dashboard. It does not make medical or security enforcement decisions.")
