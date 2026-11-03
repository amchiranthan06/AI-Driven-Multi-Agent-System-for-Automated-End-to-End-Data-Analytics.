from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_research_experiments as experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeatable selector, dummy, and FLAML comparisons.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 2026, 123, 777])
    parser.add_argument("--budgets", nargs="+", type=int, default=[8, 60])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baselines-only", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(records: list[dict], output: Path) -> None:
    rows = []
    for (dataset, method, budget), group in pd.DataFrame(records).groupby(
        ["dataset", "method", "budget_seconds"], dropna=False
    ):
        metric_names = sorted(
            {
                key
                for metrics in group["metrics"]
                if isinstance(metrics, dict)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
            }
        )
        row = {
            "dataset": dataset,
            "method": method,
            "budget_seconds": None if pd.isna(budget) else int(budget),
            "runs": len(group),
        }
        for metric in metric_names:
            values = [
                float(metrics[metric])
                for metrics in group["metrics"]
                if isinstance(metrics, dict)
                and metric in metrics
                and isinstance(metrics[metric], (int, float, np.number))
            ]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "model_summary.csv", index=False)


def run_dummy(dataset: str, frame: pd.DataFrame, target_name: str, seed: int) -> dict:
    prepared, _ = experiment.clean_dataframe(frame)
    prepared = prepared.dropna(subset=[target_name]).reset_index(drop=True)
    target = prepared[target_name]
    features = prepared.drop(columns=[target_name])
    dropped = experiment._non_predictive_features(features, target)
    features = features.drop(columns=dropped)
    task = experiment._infer_task_type(target)
    test_size = experiment._recommended_test_size(task, target)
    stratify = target if task == "classification" and experiment._can_stratify(target, test_size) else None
    x_train, x_test, y_train, y_test = experiment.train_test_split(
        features, target, test_size=test_size, random_state=seed, stratify=stratify
    )
    preprocessor = experiment.build_preprocessor(x_train)
    x_train_ready = preprocessor.fit_transform(x_train)
    x_test_ready = preprocessor.transform(x_test)
    estimator = (
        experiment.DummyClassifier(strategy="prior")
        if task == "classification"
        else experiment.DummyRegressor(strategy="mean")
    )

    def fit_and_score():
        estimator.fit(x_train_ready, y_train)
        return experiment.metric_dict(task, y_test, estimator.predict(x_test_ready))

    metrics, elapsed, memory = experiment.measure(fit_and_score)
    return {
        "dataset": dataset,
        "seed": seed,
        "method": "Dummy baseline",
        "task": task,
        "model": "most_frequent" if task == "classification" else "mean",
        "metrics": metrics,
        "runtime_seconds": elapsed,
        **memory,
        "excluded_features": dropped,
        "status": "complete",
        "budget_seconds": None,
    }


def main() -> None:
    args = parse_args()
    output = args.output or ROOT / "output" / "experiments" / f"model_refresh_{datetime.now():%Y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=bool(args.output))
    trial_path = output / "model_trials.jsonl"
    cache = output / "sklearn_cache"
    datasets = experiment.load_datasets(cache)
    manifest = {
        name: {"rows": len(frame), "columns": len(frame.columns), "target": target, "source": source}
        for name, (frame, target, source) in datasets.items()
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records: list[dict] = []
    baseline_written: set[tuple[str, int, str]] = set()
    if trial_path.exists():
        records = [json.loads(line) for line in trial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        baseline_written = {
            (row["dataset"], int(row["seed"]), row["method"])
            for row in records
            if row["method"] != "FLAML AutoML"
        }
    for dataset, (frame, target, _) in datasets.items():
        for seed in args.seeds:
            key = (dataset, seed, "Dummy baseline")
            if key not in baseline_written:
                record = run_dummy(dataset, frame, target, seed)
                records.append(record)
                append_jsonl(trial_path, record)
                baseline_written.add(key)
                print(f"completed dummy baseline {dataset}, seed={seed}", flush=True)

    if args.baselines_only:
        summarize(records, output)
        print(f"Updated baseline summary in {output}", flush=True)
        return

    for budget in args.budgets:
        experiment.AUTOML_TIME_BUDGET_SECONDS = budget
        for dataset, (frame, target, _) in datasets.items():
            for seed in args.seeds:
                results = experiment.compare_models(dataset, frame, target, seed)
                for record in results:
                    if record["method"] == "FLAML AutoML":
                        record["budget_seconds"] = budget
                    else:
                        key = (dataset, seed, record["method"])
                        if key in baseline_written:
                            continue
                        record["budget_seconds"] = None
                        baseline_written.add(key)
                    record["seed"] = seed
                    records.append(record)
                    append_jsonl(trial_path, record)
                print(f"completed {dataset}, seed={seed}, FLAML budget={budget}s", flush=True)

    summarize(records, output)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "flaml_budgets_seconds": args.budgets,
                "holdouts": "One 80/20 seeded holdout per seed; repeated holdouts, not k-fold cross-validation.",
                "selection": "The project selector chooses on a validation split within the training partition and reports metrics on the untouched outer test split.",
                "flaml_budget_note": "Search budget; observed wall time may be longer.",
                "ollama_arm": "Not run by this script; requires the configured local model.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved incremental trials and summary to {output}", flush=True)


if __name__ == "__main__":
    main()
