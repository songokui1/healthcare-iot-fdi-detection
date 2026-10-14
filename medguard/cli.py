"""Command-line interface for private MedGuard-AI research experiments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .analyst import MockProvider, OpenAIProvider
from .data import load_split, synthetic_demo
from .experiment import run_comparison, write_aggregate

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medguard", description="Explore LLM, RAG, and gated hybrid IoMT detection"
    )
    parser.add_argument("--csv", help="Local authorized WUSTL-EHMS CSV; omitted = synthetic demo")
    parser.add_argument("--label", default="Label", help="Binary label column")
    parser.add_argument("--exclude", action="append", default=[],
                        help="Additional columns to exclude; repeat as needed")
    parser.add_argument("--provider", choices=["mock", "openai"], default="mock")
    parser.add_argument("--include-lightgbm", action="store_true",
                        help="Train a fresh conventional baseline on same split")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI chat-completions JSON-capable model")
    parser.add_argument("--max-test-rows", type=int, default=30,
                        help="Budget cap for each approach's held-out evaluations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--gate-normal-quantile", type=float, default=0.85)
    parser.add_argument("--output", default="outputs/aggregate.json")
    parser.add_argument("--allow-external-telemetry", action="store_true",
                        help="Approve transmitting numerical telemetry to an API")
    parser.add_argument("--confirm-api-spend", action="store_true",
                        help="Approve billable LLM calls; required for real provider")
    return parser

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "openai":
        if not args.csv:
            raise SystemExit("Real LLM evaluation requires --csv; synthetic demo uses mock only.")
        if not (args.allow_external_telemetry and args.confirm_api_spend):
            raise SystemExit("Real LLM requires BOTH --allow-external-telemetry "
                             "and --confirm-api-spend.")
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("Set OPENAI_API_KEY in your local environment; never commit it.")
        provider = OpenAIProvider(args.model)
    else:
        provider = MockProvider()
    split = load_split(
        args.csv, label_column=args.label, test_size=args.test_size,
        seed=args.seed, exclude=tuple(args.exclude),
    ) if args.csv else synthetic_demo(seed=args.seed)
    report = run_comparison(
        split, provider, max_test_rows=args.max_test_rows,
        seed=args.seed, normal_quantile=args.gate_normal_quantile,
        include_lightgbm=args.include_lightgbm,
    )
    write_aggregate(report, args.output)
    print(json.dumps({
        "status": report["status"],
        "approaches": list(report["approaches"]),
        "evaluated_rows": report["data"]["test_examples_evaluated"],
        "total_llm_calls": sum(a["llm_calls"] for a in report["approaches"].values()),
        "aggregate_file": str(Path(args.output)),
        "warning": "MOCK / SYNTHETIC: NOT SCIENTIFIC EVIDENCE"
                   if report["status"] == "synthetic_mock_smoke_test"
                   else "EXPLORATORY REAL-LLM RESULTS: REVIEW BEFORE PUBLICATION",
    }, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
