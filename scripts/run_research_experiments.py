from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import pandas as pd
import psutil
from flaml import AutoML
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_diabetes, load_iris, load_wine
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.agents.graph import run_analytics
from app.core.config import settings
from app.data_pipeline import clean_dataframe
from app.dataset_store import profile_csv_file
from app.llm.ollama_client import ollama_client
from app.ml_pipeline import _can_stratify, _infer_task_type, _non_predictive_features, _recommended_test_size, train_and_evaluate


SEEDS = [17, 42, 2026]
AUTOML_TIME_BUDGET_SECONDS = 8
REPORT_QUESTION = (
    "Summarize the dataset and modeling results. State the dataset row and column counts, "
    "the selected model, and its reported evaluation metrics. Separate computed findings "
    "from recommendations, and do not invent numbers."
)
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?\s*%?")
METRIC_VALUE_PATTERN = re.compile(
    r"(?P<label>accuracy|precision|recall|f1(?:[- ]score)?|r(?:\^?2|²)|mae|rmse|mse)"
    r"\s*[:=]?\s*(?P<value>[-+]?\d[\d,]*(?:\.\d+)?\s*%?)",
    re.IGNORECASE,
)
LABEL_METRICS = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "r2": "r2",
    "r^2": "r2",
    "mae": "mae",
    "rmse": "rmse",
    "mse": "mse",
}


class PeakMemory:
    def __init__(self, interval: float = 0.05):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                return

    def __enter__(self) -> "PeakMemory":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop_event.set()
        self.thread.join(timeout=1)

    def as_mb(self) -> dict[str, float]:
        return {
            "peak_rss_mb": round(self.peak / (1024 * 1024), 2),
            "rss_delta_mb": round(max(0, self.peak - self.baseline) / (1024 * 1024), 2),
        }


def measure(call: Callable[[], Any]) -> tuple[Any, float, dict[str, float]]:
    with PeakMemory() as memory:
        started = time.perf_counter()
        result = call()
        elapsed = time.perf_counter() - started
    return result, round(elapsed, 3), memory.as_mb()


def frame_from_sklearn(bunch: Any, target_name: str) -> pd.DataFrame:
    frame = bunch.data.copy()
    frame[target_name] = bunch.target
    return frame


def load_datasets(cache_dir: Path) -> dict[str, tuple[pd.DataFrame, str, str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, tuple[pd.DataFrame, str, str]] = {}

    wdbc_path = ROOT / "backend/app/datasets/0dac58be4c0d4f3fb3c6cfb63068eec3/data.csv"
    datasets["WDBC"] = (pd.read_csv(wdbc_path), "diagnosis", "Local project CSV; Wisconsin Diagnostic Breast Cancer data")

    wine = load_wine(as_frame=True)
    datasets["Wine"] = (frame_from_sklearn(wine, "target"), "target", "scikit-learn bundled Wine dataset")
    iris = load_iris(as_frame=True)
    datasets["Iris"] = (frame_from_sklearn(iris, "target"), "target", "scikit-learn bundled Iris dataset")
    diabetes = load_diabetes(as_frame=True)
    datasets["Diabetes"] = (frame_from_sklearn(diabetes, "target"), "target", "scikit-learn bundled Diabetes dataset")
    housing = fetch_california_housing(as_frame=True, data_home=str(cache_dir))
    datasets["California Housing"] = (
        frame_from_sklearn(housing, "target"),
        "target",
        "California Housing dataset fetched by scikit-learn",
    )
    return datasets


def metric_dict(task: str, actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_weighted": float(precision_score(actual, predicted, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(actual, predicted, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(actual, predicted, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        }
    mse = float(mean_squared_error(actual, predicted))
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": math.sqrt(mse),
        "r2": float(r2_score(actual, predicted)),
    }


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = [col for col in features if pd.api.types.is_numeric_dtype(features[col])]
    categorical = [col for col in features if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def flaml_weighted_f1(X_val, y_val, estimator, labels, X_train, y_train, *args, **kwargs):
    prediction = estimator.predict(X_val)
    score = f1_score(y_val, prediction, average="weighted", zero_division=0)
    return 1.0 - float(score), {"weighted_f1": float(score)}


def compare_models(name: str, frame: pd.DataFrame, target_name: str, seed: int) -> list[dict[str, Any]]:
    prepared, _ = clean_dataframe(frame)
    prepared = prepared.dropna(subset=[target_name]).reset_index(drop=True)
    target = prepared[target_name]
    features = prepared.drop(columns=[target_name])
    dropped = _non_predictive_features(features, target)
    features = features.drop(columns=dropped)
    task = _infer_task_type(target)
    test_size = _recommended_test_size(task, target)
    stratify = target if task == "classification" and _can_stratify(target, test_size) else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=seed, stratify=stratify
    )
    output: list[dict[str, Any]] = []

    system_outcome, elapsed, memory = measure(lambda: train_and_evaluate(prepared, target_name, random_state=seed))
    system_row = next(row for row in system_outcome.metrics_table if row["model"] == system_outcome.best_model)
    output.append({
        "dataset": name, "seed": seed, "method": "Project model selector", "task": task,
        "model": system_outcome.best_model, "metrics": system_row, "runtime_seconds": elapsed,
        **memory, "excluded_features": system_outcome.dropped_features, "status": "complete",
    })

    prep = build_preprocessor(X_train)
    X_train_ready, X_test_ready = prep.fit_transform(X_train), prep.transform(X_test)
    dummy = DummyClassifier(strategy="prior") if task == "classification" else DummyRegressor(strategy="mean")

    def fit_dummy():
        dummy.fit(X_train_ready, y_train)
        return metric_dict(task, y_test, dummy.predict(X_test_ready))

    dummy_metrics, elapsed, memory = measure(fit_dummy)
    output.append({
        "dataset": name, "seed": seed, "method": "Dummy baseline", "task": task,
        "model": "most_frequent" if task == "classification" else "mean", "metrics": dummy_metrics,
        "runtime_seconds": elapsed, **memory, "excluded_features": dropped, "status": "complete",
    })

    automl = AutoML()
    metric = flaml_weighted_f1 if task == "classification" else "r2"

    def fit_automl():
        automl.fit(
            X_train=X_train_ready,
            y_train=y_train,
            task=task,
            metric=metric,
            time_budget=AUTOML_TIME_BUDGET_SECONDS,
            eval_method="holdout",
            split_ratio=0.2,
            estimator_list=["lgbm", "rf", "extra_tree", "xgboost"],
            n_jobs=1,
            seed=seed,
            max_iter=30,
            ensemble=False,
            verbose=0,
            log_file_name="",
        )
        return metric_dict(task, y_test, automl.predict(X_test_ready))

    try:
        automl_metrics, elapsed, memory = measure(fit_automl)
        output.append({
            "dataset": name, "seed": seed, "method": "FLAML AutoML", "task": task,
            "model": str(getattr(automl, "best_estimator", "FLAML selected model")), "metrics": automl_metrics,
            "runtime_seconds": elapsed, **memory, "excluded_features": dropped, "status": "complete",
            "automl_time_budget_seconds": AUTOML_TIME_BUDGET_SECONDS,
        })
    except Exception as exc:
        output.append({
            "dataset": name, "seed": seed, "method": "FLAML AutoML", "task": task,
            "model": "", "metrics": {}, "runtime_seconds": None, "peak_rss_mb": None,
            "rss_delta_mb": None, "excluded_features": dropped, "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        })
    return output


def numeric_claims(text: str) -> list[tuple[str, float, bool]]:
    claims = []
    for raw_line in text.splitlines():
        line = re.sub(r"\bUnnamed\s*:\s*\d+\b", "", raw_line, flags=re.IGNORECASE)
        if re.match(r"\s*(?:[-*]\s*)?\d+[.)]\s", line):
            continue

        matched_spans: list[tuple[int, int]] = []
        for match in METRIC_VALUE_PATTERN.finditer(line):
            label = match.group("label").lower().replace("²", "2")
            if label.startswith("f1"):
                label = "f1"
            value_match = re.search(NUMBER_PATTERN, match.group("value"))
            if value_match is None:
                continue
            raw = value_match.group(0).replace(",", "").replace(" ", "")
            percentage = raw.endswith("%")
            value = float(raw[:-1] if percentage else raw)
            if percentage:
                value /= 100.0
            claims.append((f"{label}: {raw_line.strip()}", value, percentage))
            matched_spans.append((match.start("value"), match.end("value")))

        for label, pattern in (
            ("row_count", re.compile(r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+rows?\b", re.IGNORECASE)),
            ("column_count", re.compile(r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+columns?\b", re.IGNORECASE)),
            ("columns_with_missing", re.compile(r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+missing\s+columns?\b", re.IGNORECASE)),
            ("missing_cells", re.compile(r"(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s+missing\s+(?:cells?|values?)\b", re.IGNORECASE)),
        ):
            for match in pattern.finditer(line):
                if any(start <= match.start("value") < end for start, end in matched_spans):
                    continue
                value = float(match.group("value").replace(",", ""))
                claims.append((f"{label}: {raw_line.strip()}", value, False))
                matched_spans.append((match.start("value"), match.end("value")))

        for match in NUMBER_PATTERN.finditer(line):
            if any(start <= match.start() < end for start, end in matched_spans):
                continue
            raw = match.group(0).replace(",", "").replace(" ", "")
            percentage = raw.endswith("%")
            value = float(raw[:-1] if percentage else raw)
            if percentage:
                value /= 100.0
            claims.append((raw_line.strip(), value, percentage))
    return claims


def supported_facts(state: dict[str, Any]) -> dict[str, float]:
    metadata = state.get("metadata", {})
    summary = state.get("model_summary", {})
    facts: dict[str, float] = {
        "row_count": float(metadata.get("row_count", 0)),
        "column_count": float(metadata.get("column_count", 0)),
        "duplicate_rows": float(metadata.get("duplicate_rows", 0)),
        "missing_cells": float(sum(metadata.get("missing_by_column", {}).values())),
        "columns_with_missing": float(len(metadata.get("missing_by_column", {}))),
    }
    for key, value in summary.items():
        if key in {"metrics_table", "candidates"}:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            facts[key] = float(value)
    table = summary.get("metrics_table", [])
    best = next((row for row in table if row.get("model") == summary.get("best_model")), {})
    for key, value in best.items():
        if key != "model" and isinstance(value, (int, float)) and math.isfinite(float(value)):
            facts[key.lower()] = float(value)
    return facts


def score_report(text: str, facts: dict[str, float], dataset: str, source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    claims = numeric_claims(text)
    checked = []
    for context, value, percentage in claims:
        context_lower = context.lower()
        expected_key = context.split(":", 1)[0] if context.split(":", 1)[0] in facts else ""
        if not expected_key:
            expected_key = next((fact for label, fact in LABEL_METRICS.items() if context_lower.startswith(f"{label}: ") and fact in facts), "")
        expected = facts.get(expected_key) if expected_key else None
        matched_key = expected_key if expected is not None and abs(value - expected) <= max(0.00051, abs(expected) * 0.001) else ""
        checked.append({
            "dataset": dataset,
            "report_source": source,
            "claim": context,
            "reported_value": value,
            "percent_form": percentage,
            "matched_fact": matched_key,
            "supported": bool(matched_key),
        })
    total = len(checked)
    score = sum(row["supported"] for row in checked) / total if total else None
    return {"dataset": dataset, "report_source": source, "numeric_claims": total, "matched_claims": sum(row["supported"] for row in checked), "faithfulness": score}, checked


async def run_workflow_and_llm_baseline(name: str, frame: pd.DataFrame, target_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    question = f"{REPORT_QUESTION} The requested prediction target is '{target_name}'."
    csv_text = frame.to_csv(index=False)
    with PeakMemory() as monitor:
        started = time.perf_counter()
        state = await run_analytics(question=question, dataset_preview=csv_text, target_column=target_name)
        elapsed = time.perf_counter() - started
    memory = monitor.as_mb()
    report_result = next((item for item in reversed(state.get("results", [])) if item.get("agent") == "report"), None)
    report_text = report_result.get("output", "") if report_result else ""
    facts = supported_facts(state)
    if report_result and report_result.get("status") == "complete":
        ground_score, claims = score_report(report_text, facts, name, "grounded_multi_agent")
    else:
        ground_score = {
            "dataset": name,
            "report_source": "grounded_multi_agent",
            "numeric_claims": 0,
            "matched_claims": 0,
            "faithfulness": None,
        }
        claims = []

    sample = frame.head(12).to_csv(index=False)
    direct_prompt = (
        f"Question: {question}\n\n"
        "Analyze only the raw CSV excerpt below. The complete dataset and model outputs are not supplied. "
        "Do not claim exact whole-dataset or model metrics if you cannot derive them from this excerpt. "
        "Write a brief report and next steps. Start the visible response with 'Final:'.\n\n"
        f"CSV excerpt:\n{sample}"
    )
    llm_text, llm_diagnostics = await ollama_client.generate_with_diagnostics(
        prompt=direct_prompt,
        system="You are an analytics assistant. State numbers only when they can be computed from the supplied data.",
        require_final_marker=True,
    )
    direct_score, direct_claims = score_report(llm_text, facts, name, "llm_only")
    claims.extend(direct_claims)
    return {
        "dataset": name,
        "workflow_statuses": {item.get("agent"): item.get("status") for item in state.get("results", [])},
        "workflow_model_summary": state.get("model_summary", {}),
        "workflow_report": report_text,
        "workflow_insight": next((item.get("output", "") for item in state.get("results", []) if item.get("agent") == "insight"), ""),
        "workflow_runtime_seconds": round(elapsed, 3),
        **{f"workflow_{key}": value for key, value in memory.items()},
        "workflow_numeric_faithfulness": ground_score,
        "llm_only_report": llm_text,
        "llm_only_diagnostics": llm_diagnostics,
        "llm_only_numeric_faithfulness": direct_score,
        "faithfulness_facts": facts,
    }, claims


async def run_large_file_stress(output_dir: Path) -> dict[str, Any]:
    large_path = ROOT / "backend/app/datasets/b7d02382adf34748959849d78c182b64/data.csv"
    profile, elapsed, memory = measure(lambda: profile_csv_file(large_path))
    sample_rows = min(50_000, int(profile["row_count"]))
    with PeakMemory() as monitor:
        started = time.perf_counter()
        state = await run_analytics(
            question="Analyze the large crop-yield dataset and report the model evaluation.",
            dataset_id=large_path.parent.name,
            target_column="Yield_tons_per_hectare",
        )
        workflow_elapsed = time.perf_counter() - started
    workflow_memory = monitor.as_mb()
    return {
        "file": str(large_path.relative_to(ROOT)),
        "file_size_bytes": large_path.stat().st_size,
        "rows_profiled": int(profile["row_count"]),
        "columns": int(profile["column_count"]),
        "profile_chunk_size_rows": 100_000,
        "model_training_sample_limit_rows": sample_rows,
        "profile_runtime_seconds": elapsed,
        **memory,
        "end_to_end_runtime_seconds": round(workflow_elapsed, 3),
        "end_to_end_peak_rss_mb": workflow_memory["peak_rss_mb"],
        "end_to_end_rss_delta_mb": workflow_memory["rss_delta_mb"],
        "workflow_agent_statuses": {item.get("agent"): item.get("status") for item in state.get("results", [])},
        "model_summary": state.get("model_summary", {}),
        "note": "Local project stress-test CSV; public-dataset provenance is not recorded, so this is not included as an accuracy benchmark.",
    }


def write_outputs(output_dir: Path, records: list[dict[str, Any]], faithfulness: list[dict[str, Any]], reports: list[dict[str, Any]], stress: dict[str, Any], manifest: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "trial_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(faithfulness).to_csv(output_dir / "faithfulness_claims.csv", index=False)
    pd.DataFrame(reports).to_json(output_dir / "agent_reports.json", orient="records", indent=2, force_ascii=False)
    pd.DataFrame(manifest).to_csv(output_dir / "dataset_manifest.csv", index=False)
    faithfulness_rows = []
    for report in reports:
        for key, method in (("workflow_numeric_faithfulness", "grounded_multi_agent"), ("llm_only_numeric_faithfulness", "llm_only")):
            score = report[key]
            faithfulness_rows.append({
                "dataset": report["dataset"],
                "report_source": method,
                "report_status": (
                    report.get("workflow_statuses", {}).get("report", "unknown")
                    if key == "workflow_numeric_faithfulness"
                    else ("generated" if report.get("llm_only_report") else "empty_or_suppressed")
                ),
                "numeric_claims": score["numeric_claims"],
                "matched_claims": score["matched_claims"],
                "faithfulness": score["faithfulness"],
                "metric_definition": "Automatic numeric match only; fallback text is labeled by report_status, and semantic correctness requires manual audit.",
            })
    pd.DataFrame(faithfulness_rows).to_csv(output_dir / "faithfulness_summary.csv", index=False)
    (output_dir / "large_file_stress.json").write_text(json.dumps(stress, indent=2), encoding="utf-8")

    aggregate_rows = []
    for (dataset, method), group in pd.DataFrame(records).query("status == 'complete'").groupby(["dataset", "method"]):
        metric_names = sorted({
            key
            for value in group["metrics"]
            for key, metric_value in value.items()
            if isinstance(metric_value, (int, float, np.integer, np.floating))
            and not isinstance(metric_value, bool)
        })
        row = {"dataset": dataset, "method": method, "runs": int(len(group))}
        for metric_name in metric_names:
            vals = [float(item[metric_name]) for item in group["metrics"] if metric_name in item]
            row[f"{metric_name}_mean"] = float(np.mean(vals))
            row[f"{metric_name}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        row["runtime_seconds_mean"] = float(group["runtime_seconds"].mean())
        row["peak_rss_mb_max"] = float(group["peak_rss_mb"].max())
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "benchmark_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": SEEDS,
        "automl": "FLAML 2.7.0",
        "automl_time_budget_seconds_per_seed_dataset": AUTOML_TIME_BUDGET_SECONDS,
        "ollama_model": ollama_client.model,
        "ollama_num_ctx": settings.ollama_num_ctx,
        "ollama_num_predict": settings.ollama_num_predict,
        "python": sys.version,
        "platform": sys.platform,
        "narrative": "One split per seed; mean and sample standard deviation are across three seeded holdouts, not k-fold cross-validation.",
    }, indent=2), encoding="utf-8")


async def main() -> None:
    ollama_status = await ollama_client.status()
    if not ollama_status.model_available:
        raise RuntimeError(f"Configured Ollama model unavailable: {ollama_status.detail} (model={ollama_status.model}, base_url={ollama_client.base_url})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ROOT / "output/experiments" / f"run_{timestamp}"
    data = load_datasets(output_dir / "sklearn_cache")
    records: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    faithfulness: list[dict[str, Any]] = []
    manifest = []

    for name, (frame, target, source) in data.items():
        manifest.append({
            "dataset": name, "source": source, "rows": len(frame), "columns_including_target": len(frame.columns),
            "target": target, "task": _infer_task_type(frame[target]),
        })
        print(f"[{name}] {frame.shape}; target={target}; running heldout trials")
        for seed in SEEDS:
            records.extend(compare_models(name, frame, target, seed))
        print(f"[{name}] running six-agent workflow and LLM-only report")
        report, claims = await run_workflow_and_llm_baseline(name, frame, target)
        reports.append(report)
        faithfulness.extend(claims)

    del data
    gc.collect()
    stress = await run_large_file_stress(output_dir)
    write_outputs(output_dir, records, faithfulness, reports, stress, manifest)
    print(f"Saved experiment outputs to {output_dir}")
    print(pd.read_csv(output_dir / "benchmark_summary.csv").to_string(index=False))
    print(json.dumps(stress, indent=2))
    faith_df = pd.DataFrame([
        {key: value for key, value in report.items() if key in {"dataset", "workflow_numeric_faithfulness", "llm_only_numeric_faithfulness"}}
        for report in reports
    ])
    print(faith_df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
