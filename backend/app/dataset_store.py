from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import UploadFile

from app.core.paths import DATASETS_DIR
from app.data_pipeline import suggest_target_columns

CHUNK_SIZE = 100_000
TRAINING_SAMPLE_ROWS = 50_000


def dataset_path(dataset_id: str) -> Path:
    safe_id = "".join(ch for ch in dataset_id if ch.isalnum() or ch in ("-", "_"))
    path = DATASETS_DIR / safe_id / "data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset_id}' was not found.")
    return path


async def save_uploaded_dataset(file: UploadFile) -> dict[str, Any]:
    dataset_id = uuid4().hex
    dataset_dir = DATASETS_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / "data.csv"

    with path.open("wb") as output:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    profile = profile_csv_file(path)
    profile["dataset_id"] = dataset_id
    profile["filename"] = file.filename or "uploaded.csv"
    profile["file_size_bytes"] = path.stat().st_size
    return profile


def profile_csv_file(path: Path) -> dict[str, Any]:
    row_count = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    missing_counts: dict[str, int] = {}
    numeric_stats: dict[str, dict[str, float]] = {}
    top_values: dict[str, dict[str, int]] = {}
    sample_rows: list[dict[str, str]] = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):
        chunk.columns = [str(column).strip() for column in chunk.columns]
        if not columns:
            columns = list(chunk.columns)
            missing_counts = {column: 0 for column in columns}

        row_count += int(len(chunk))
        for column in columns:
            series = chunk[column]
            missing_counts[column] += int(series.isna().sum())
            if column not in dtypes:
                dtypes[column] = str(series.dtype)

            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_ratio = float(numeric_series.notna().mean()) if len(series) else 0.0
            if numeric_ratio >= 0.8:
                stats = numeric_stats.setdefault(
                    column,
                    {"count": 0.0, "sum": 0.0, "min": float("inf"), "max": float("-inf")},
                )
                values = numeric_series.dropna()
                if not values.empty:
                    stats["count"] += float(values.count())
                    stats["sum"] += float(values.sum())
                    stats["min"] = min(stats["min"], float(values.min()))
                    stats["max"] = max(stats["max"], float(values.max()))
            elif len(top_values) < 8:
                counts = top_values.setdefault(column, {})
                for value, count in series.dropna().astype(str).value_counts().head(20).items():
                    counts[value] = counts.get(value, 0) + int(count)

        if len(sample_rows) < 10:
            needed = 10 - len(sample_rows)
            sample_rows.extend(chunk.head(needed).fillna("").astype(str).to_dict(orient="records"))

    for stats in numeric_stats.values():
        stats["mean"] = stats["sum"] / stats["count"] if stats["count"] else 0.0
        if stats["min"] == float("inf"):
            stats["min"] = 0.0
        if stats["max"] == float("-inf"):
            stats["max"] = 0.0

    top_values = {
        column: dict(sorted(values.items(), key=lambda item: item[1], reverse=True)[:10])
        for column, values in top_values.items()
    }
    missing_by_column = {column: count for column, count in missing_counts.items() if count}
    sample_df = load_training_sample(path, max_rows=min(TRAINING_SAMPLE_ROWS, max(row_count, 1)))

    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "numeric_columns": list(numeric_stats.keys()),
        "categorical_columns": [column for column in columns if column not in numeric_stats],
        "missing_by_column": missing_by_column,
        "numeric_stats": numeric_stats,
        "top_values": top_values,
        "target_suggestions": suggest_target_columns(sample_df) if not sample_df.empty else [],
        "sample_rows": sample_rows,
    }


def load_training_sample(path: Path, max_rows: int = TRAINING_SAMPLE_ROWS) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_rows)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def compact_profile_text(profile: dict[str, Any]) -> str:
    lines = [
        f"Dataset ID: {profile.get('dataset_id', 'not stored')}",
        f"Filename: {profile.get('filename', 'unknown')}",
        f"Rows: {profile.get('row_count', 0)}",
        f"Columns: {profile.get('column_count', 0)}",
        f"Column names: {', '.join(profile.get('columns', [])[:30])}",
        f"Numeric columns: {', '.join(profile.get('numeric_columns', [])[:20]) or 'none'}",
        f"Categorical columns: {', '.join(profile.get('categorical_columns', [])[:20]) or 'none'}",
    ]
    missing = profile.get("missing_by_column", {})
    if missing:
        worst = sorted(missing.items(), key=lambda item: item[1], reverse=True)[:10]
        lines.append("Missing values: " + ", ".join(f"{col}={count}" for col, count in worst))
    stats = profile.get("numeric_stats", {})
    for column, values in list(stats.items())[:10]:
        lines.append(
            f"{column}: min={values.get('min', 0):.3f}, max={values.get('max', 0):.3f}, mean={values.get('mean', 0):.3f}"
        )
    suggestions = profile.get("target_suggestions", [])
    lines.append(f"Likely target columns: {', '.join(suggestions) or 'none'}")
    return "\n".join(lines)
