from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_research_experiments as experiment
from app.core.config import settings
from app.llm.ollama_client import ollama_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated reports with an LLM-only baseline.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=2048)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings.ollama_num_ctx = args.num_ctx
    settings.ollama_num_predict = args.num_predict
    status = await ollama_client.status()
    if not status.model_available:
        raise RuntimeError(f"Configured Ollama model unavailable: {status.detail}; model={status.model}; url={ollama_client.base_url}")

    output = args.output or ROOT / "output" / "experiments" / f"report_faithfulness_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    datasets = experiment.load_datasets(args.cache_dir)
    reports = []
    claims = []
    for name, (frame, target, source) in datasets.items():
        print(f"running grounded and LLM-only reports for {name}", flush=True)
        report, dataset_claims = await experiment.run_workflow_and_llm_baseline(name, frame, target)
        report["dataset_source"] = source
        reports.append(report)
        claims.extend(dataset_claims)
        print(
            f"completed {name}: report={report['workflow_statuses'].get('report')}, "
            f"direct_llm={'generated' if report['llm_only_report'] else report['llm_only_diagnostics'].get('failure_reason', 'empty')}",
            flush=True,
        )

    rows = []
    for report in reports:
        rows.append({
            "dataset": report["dataset"],
            "report_source": "grounded_multi_agent",
            "report_status": report["workflow_statuses"].get("report", "unknown"),
            **report["workflow_numeric_faithfulness"],
            "metric_definition": "Numeric value match only; not semantic verification.",
        })
        diagnostic = report["llm_only_diagnostics"]
        rows.append({
            "dataset": report["dataset"],
            "report_source": "llm_only",
            "report_status": "generated" if report["llm_only_report"] else diagnostic.get("failure_reason", "empty"),
            **report["llm_only_numeric_faithfulness"],
            "metric_definition": "Numeric value match only; not semantic verification.",
        })

    pd.DataFrame(reports).to_json(output / "report_runs.json", orient="records", indent=2, force_ascii=False)
    pd.DataFrame(rows).to_csv(output / "faithfulness_summary.csv", index=False)
    pd.DataFrame(claims).to_csv(output / "faithfulness_claims.csv", index=False)
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "ollama_base_url": ollama_client.base_url,
                "ollama_model": ollama_client.model,
                "ollama_num_ctx": settings.ollama_num_ctx,
                "ollama_num_predict": settings.ollama_num_predict,
                "inference_overrides": "Experiment-only runtime overrides; backend/.env was not modified.",
                "workflow_report_rule": "Only Report Agent status=complete is scored; deterministic fallback reports have undefined faithfulness.",
                "direct_llm_input": "12-row CSV excerpt without backend model metrics.",
                "score_definition": "Extracted numeric value match rate only; manual semantic audit remains required.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved reports, claims, and summary to {output}", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
