from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:  # pragma: no cover - dependency might not be present on first boot
    ColumnTransformer = None


@dataclass
class TrainingOutcome:
    task_type: str
    target_column: str
    best_model: str
    best_metric: str
    candidates: list[dict[str, Any]]
    metrics_table: list[dict[str, Any]]
    evaluation_notes: list[str]
    dropped_features: list[str]


def can_train_models() -> bool:
    return ColumnTransformer is not None


def train_and_evaluate(df: pd.DataFrame, target_column: str, random_state: int = 42) -> TrainingOutcome:
    if not can_train_models():
        raise RuntimeError("scikit-learn is not installed for model training.")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in the dataset.")

    model_df = df.dropna(subset=[target_column]).copy()
    feature_df = model_df.drop(columns=[target_column]).copy()
    target = model_df[target_column].copy()
    task_type = _infer_task_type(target)

    dropped_features = _non_predictive_features(feature_df, target)
    if dropped_features:
        feature_df = feature_df.drop(columns=dropped_features)
    if feature_df.empty or not len(feature_df.columns):
        raise ValueError("No usable feature columns remain after excluding IDs, constants, and empty columns.")

    numeric_features = [column for column in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[column])]
    categorical_features = [column for column in feature_df.columns if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    test_size = _recommended_test_size(task_type, target)
    test_stratify = target if task_type == "classification" and _can_stratify(target, test_size) else None
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        feature_df,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=test_stratify,
    )
    validation_stratify = (
        y_train_validation
        if task_type == "classification" and _can_stratify(y_train_validation, 0.25)
        else None
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        random_state=random_state + 1,
        stratify=validation_stratify,
    )
    candidates = _candidate_models(task_type)

    best_score: float | None = None
    best_model_name = ""
    best_metric = "f1" if task_type == "classification" else "r2"
    validation_scores: dict[str, float] = {}
    for model_name, estimator in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
        pipeline.fit(X_train, y_train)
        validation_metrics = _evaluate_pipeline(task_type, pipeline, X_validation, y_validation)
        score = float(validation_metrics[best_metric])
        validation_scores[model_name] = score
        if best_score is None or score > best_score:
            best_score = score
            best_model_name = model_name

    X_final_train = pd.concat([X_train, X_validation], axis=0)
    y_final_train = pd.concat([y_train, y_validation], axis=0)
    metrics_table: list[dict[str, Any]] = []
    for model_name, estimator in candidates:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", clone(estimator)),
            ]
        )
        pipeline.fit(X_final_train, y_final_train)
        row = {"model": model_name, **_evaluate_pipeline(task_type, pipeline, X_test, y_test)}
        metrics_table.append(row)

    evaluation_notes = _build_evaluation_notes(task_type, metrics_table, best_model_name)

    return TrainingOutcome(
        task_type=task_type,
        target_column=target_column,
        best_model=best_model_name,
        best_metric=best_metric,
        candidates=[
            {
                "model": model_name,
                "reason": _model_reason(task_type, model_name),
                "validation_score": validation_scores[model_name],
            }
            for model_name, _ in candidates
        ],
        metrics_table=metrics_table,
        evaluation_notes=evaluation_notes,
        dropped_features=dropped_features,
    )


def _non_predictive_features(features: pd.DataFrame, target: pd.Series) -> list[str]:
    excluded: list[str] = []
    for column in features.columns:
        series = features[column]
        normalized = re.sub(r"\s+", "_", str(column).strip().lower())
        is_identifier_name = (
            normalized in {"id", "uuid", "index", "row_number"}
            or normalized.endswith("_id")
            or normalized.endswith("_uuid")
            or bool(re.fullmatch(r"unnamed(?::_?\d+)?", normalized))
        )
        is_empty_or_constant = series.nunique(dropna=True) <= 1
        is_target_copy = series.reset_index(drop=True).equals(target.reset_index(drop=True))
        is_unique_text_key = (
            not pd.api.types.is_numeric_dtype(series)
            and len(series) > 0
            and series.nunique(dropna=True) / len(series) >= 0.98
        )
        if is_identifier_name or is_empty_or_constant or is_target_copy or is_unique_text_key:
            excluded.append(column)
    return excluded


def _infer_task_type(target: pd.Series) -> str:
    unique_count = target.nunique(dropna=True)
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    if unique_count <= min(12, max(3, len(target) // 20)):
        return "classification"
    return "regression"


def _recommended_test_size(task_type: str, target: pd.Series) -> float:
    base = 0.2
    row_count = max(len(target), 1)
    if task_type != "classification":
        return base
    class_count = max(target.nunique(dropna=True), 1)
    minimum_fraction = class_count / row_count
    return min(0.5, max(base, minimum_fraction))


def _can_stratify(target: pd.Series, test_size: float) -> bool:
    row_count = len(target)
    class_count = target.nunique(dropna=True)
    if row_count < 4 or class_count < 2 or int(target.value_counts().min()) < 2:
        return False
    test_rows = math.ceil(row_count * test_size)
    train_rows = row_count - test_rows
    return test_rows >= class_count and train_rows >= class_count


def _candidate_models(task_type: str):
    if task_type == "classification":
        return [
            ("Logistic Regression", LogisticRegression(max_iter=200)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("Gradient Boosting Classifier", GradientBoostingClassifier(random_state=42)),
        ]
    return [
        ("Linear Regression", LinearRegression()),
        ("Random Forest Regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ("Gradient Boosting Regressor", GradientBoostingRegressor(random_state=42)),
    ]


def _evaluate_pipeline(task_type: str, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = pipeline.predict(X_test)
    if task_type == "classification":
        average = "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average=average, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, average=average, zero_division=0)),
        }
        if y_test.nunique(dropna=True) == 2 and hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
            except ValueError:
                pass
        return metrics

    mse = float(mean_squared_error(y_test, predictions))
    return {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": mse,
        "rmse": mse ** 0.5,
        "r2": float(r2_score(y_test, predictions)),
    }


def _model_reason(task_type: str, model_name: str) -> str:
    reasons = {
        "Logistic Regression": "Strong baseline for interpretable classification.",
        "Random Forest Classifier": "Handles nonlinear patterns and mixed feature types well.",
        "Gradient Boosting Classifier": "Useful when interactions and ranking quality matter.",
        "Linear Regression": "Clear baseline for continuous targets and linear trends.",
        "Random Forest Regressor": "Captures nonlinear relationships without heavy tuning.",
        "Gradient Boosting Regressor": "Often performs well on tabular regression tasks.",
    }
    return reasons.get(model_name, f"Candidate {task_type} model for tabular data.")


def _build_evaluation_notes(task_type: str, metrics_table: list[dict[str, Any]], best_model_name: str) -> list[str]:
    if not metrics_table:
        return ["No model metrics were generated."]
    best_row = next((row for row in metrics_table if row["model"] == best_model_name), metrics_table[0])
    if task_type == "classification":
        return [
            f"Best classifier: {best_model_name} with F1-score {best_row.get('f1', 0.0):.3f}.",
            f"Accuracy landed at {best_row.get('accuracy', 0.0):.3f}, which is a good headline metric but should be read beside recall and precision.",
        ]
    return [
        f"Best regressor: {best_model_name} with R^2 {best_row.get('r2', 0.0):.3f}.",
        f"RMSE for the selected model is {best_row.get('rmse', 0.0):.3f}, which gives the typical prediction error scale.",
    ]
