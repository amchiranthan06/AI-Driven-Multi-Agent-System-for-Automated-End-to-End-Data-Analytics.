from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.dataset_store import profile_csv_file
from app.ml_pipeline import _non_predictive_features
from run_research_experiments import PeakMemory
import app.dataset_store as dataset_store
from app.agents.graph import run_analytics


def generate_csv(path: Path, rows: int, seed: int, chunk_rows: int) -> None:
    rng = np.random.default_rng(seed)
    for start in range(0, rows, chunk_rows):
        size = min(chunk_rows, rows - start)
        feature_1 = rng.normal(50, 10, size)
        feature_2 = rng.normal(0, 1, size)
        category = rng.choice(["A", "B", "C"], size)
        target = feature_1 * 2 + feature_2 * 5 + rng.normal(0, 5, size)
        frame = pd.DataFrame(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "category": category,
                "id": np.arange(start, start + size),
                "target": target,
            }
        )
        frame.to_csv(path, index=False, mode="w" if start == 0 else "a", header=start == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and profile a reproducible synthetic CSV stress fixture.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = f"synthetic_{args.rows}_seed_{args.seed}"
    dataset_dir = args.output_dir / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / "data.csv"
    generate_csv(csv_path, args.rows, args.seed, args.chunk_rows)

    with PeakMemory() as memory:
        started = time.perf_counter()
        profile = profile_csv_file(csv_path)
        profile_elapsed = time.perf_counter() - started
    memory_result = memory.as_mb()

    sample = pd.read_csv(csv_path, nrows=min(50_000, args.rows))
    features = sample.drop(columns=["target"])
    excluded = _non_predictive_features(features, sample["target"])
    dataset_store.DATASETS_DIR = args.output_dir
    with PeakMemory() as workflow_memory:
        workflow_started = time.perf_counter()
        state = asyncio.run(
            run_analytics(
                question="Analyze the synthetic dataset and summarize model evaluation.",
                target_column="target",
                dataset_id=dataset_id,
            )
        )
        workflow_elapsed = time.perf_counter() - workflow_started
    report = {
        "file": f"{dataset_id}/data.csv",
        "seed": args.seed,
        "rows": int(profile["row_count"]),
        "columns": int(profile["column_count"]),
        "file_size_bytes": csv_path.stat().st_size,
        "profile_chunk_rows": 100_000,
        "profile_runtime_seconds": round(profile_elapsed, 3),
        **memory_result,
        "workflow_runtime_seconds": round(workflow_elapsed, 3),
        "workflow_memory": workflow_memory.as_mb(),
        "workflow_agent_statuses": {item.get("agent"): item.get("status") for item in state.get("results", [])},
        "model_summary": state.get("model_summary", {}),
        "chart_count": len(state.get("charts", [])),
        "identifier_filter_test": {"identifier": "id", "excluded_features": excluded, "id_excluded": "id" in excluded},
        "data_generation": "NumPy default_rng with fixed seed; target = 2*feature_1 + 5*feature_2 + Gaussian noise; category and sequential ID included.",
        "note": "Synthetic processing fixture; not an accuracy benchmark. This local run used deterministic LLM fallbacks because the configured Ollama model was unavailable. RSS measures the Python backend process, not Ollama or GPU memory.",
    }
    (args.output_dir / "synthetic_stress_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
