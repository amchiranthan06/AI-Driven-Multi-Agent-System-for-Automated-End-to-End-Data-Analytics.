from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd


def load_dataframe(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    df.columns = [str(column).strip() for column in df.columns]
    return df


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    missing_by_column = {column: int(value) for column, value in df.isna().sum().items() if int(value) > 0}
    duplicates = int(df.duplicated().sum())

    pattern_notes: list[str] = []
    if missing_by_column:
        worst_missing = max(missing_by_column.items(), key=lambda item: item[1])
        pattern_notes.append(f"Highest missingness appears in '{worst_missing[0]}' with {worst_missing[1]} null values.")
    if duplicates:
        pattern_notes.append(f"Detected {duplicates} duplicate rows that may need review.")
    if numeric_columns:
        corr = df[numeric_columns].corr(numeric_only=True)
        strongest_pair = _strongest_correlation(corr)
        if strongest_pair:
            a, b, value = strongest_pair
            pattern_notes.append(f"Strongest numeric relationship: {a} vs {b} with correlation {value:.2f}.")
    if categorical_columns:
        low_cardinality = [column for column in categorical_columns if 1 < df[column].nunique(dropna=True) <= 12]
        if low_cardinality:
            pattern_notes.append(f"Useful grouping columns include: {', '.join(low_cardinality[:4])}.")

    target_suggestions = suggest_target_columns(df)

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_by_column": missing_by_column,
        "duplicate_rows": duplicates,
        "pattern_notes": pattern_notes,
        "target_suggestions": target_suggestions,
        "sample_rows": df.head(5).fillna("").astype(str).to_dict(orient="records"),
    }


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    cleaned = df.copy()
    dropped_duplicates = int(cleaned.duplicated().sum())
    if dropped_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned, {
        "dropped_duplicates": dropped_duplicates,
        "fill_actions": [],
        "remaining_missing": int(cleaned.isna().sum().sum()),
        "imputation": "deferred to training-fold pipelines",
    }


def suggest_target_columns(df: pd.DataFrame) -> list[str]:
    ranked: list[tuple[int, str]] = []
    row_count = max(len(df), 1)
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if unique_count <= 1:
            continue
        score = 0
        name = column.lower()
        if any(token in name for token in ("target", "label", "class", "outcome", "sales", "price", "profit", "revenue", "score")):
            score += 10
        if pd.api.types.is_numeric_dtype(df[column]):
            score += 3
        if 2 <= unique_count <= max(20, row_count // 10):
            score += 4
        if unique_count < row_count:
            score += 2
        ranked.append((score, column))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in ranked[:5]]


def infer_task_type(df: pd.DataFrame, target_column: str) -> str:
    target = df[target_column]
    unique_count = target.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    if unique_count <= min(12, max(3, len(df) // 20)):
        return "classification"
    return "regression"


def _strongest_correlation(corr: pd.DataFrame) -> tuple[str, str, float] | None:
    if corr.empty or len(corr.columns) < 2:
        return None
    best_pair: tuple[str, str, float] | None = None
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            magnitude = abs(float(value))
            if best_pair is None or magnitude > abs(best_pair[2]):
                best_pair = (left, right, float(value))
    return best_pair
