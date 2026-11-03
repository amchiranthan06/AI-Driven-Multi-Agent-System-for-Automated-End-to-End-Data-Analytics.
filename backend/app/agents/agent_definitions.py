from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent, previous_outputs
from app.agents.state import AgentResult, AnalyticsState
from app.data_pipeline import clean_dataframe, infer_task_type, load_dataframe, summarize_dataframe
from app.dataset_store import load_training_sample
from app.ml_pipeline import can_train_models, train_and_evaluate


def _format_bullets(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines if line)


class DataPreprocessingAgent(BaseAgent):
    name = "data_preprocessing"
    title = "Data Preprocessing Agent"
    system_prompt = ""

    async def run(self, state: AnalyticsState) -> AgentResult:
        if state.get("dataset_path") and state.get("metadata"):
            metadata = state["metadata"]
            sample_df = load_training_sample(state["dataset_path"])
            cleaned, cleaning_summary = clean_dataframe(sample_df)
            metadata["cleaning_summary"] = cleaning_summary
            metadata["cleaned_row_count"] = int(metadata.get("row_count", len(cleaned)))
            metadata["training_sample_csv"] = cleaned.to_csv(index=False)
            metadata["cleaned_dataset_preview"] = cleaned.head(20).to_csv(index=False)
            metadata["cleaned_dtypes"] = {column: str(dtype) for column, dtype in cleaned.dtypes.items()}
            lines = [
                f"Profiled uploaded dataset exactly enough for critique: {metadata['row_count']} rows and {metadata['column_count']} columns.",
                f"Numeric columns: {', '.join(metadata['numeric_columns'][:6]) or 'none detected'}.",
                f"Categorical columns: {', '.join(metadata['categorical_columns'][:6]) or 'none detected'}.",
                f"Columns with missing values: {len(metadata['missing_by_column'])}.",
                f"Training/evaluation will use a bounded sample of {len(cleaned)} rows to protect local memory.",
                "Missing-value imputation is fitted inside each training split to avoid test-data leakage.",
            ]
            if metadata.get("missing_by_column"):
                worst = sorted(metadata["missing_by_column"].items(), key=lambda item: item[1], reverse=True)[:3]
                lines.append("Worst missingness: " + ", ".join(f"{column}={count}" for column, count in worst) + ".")
            lines.append(f"Likely target columns: {', '.join(metadata['target_suggestions']) or 'no clear target suggestion yet'}.")
            return {"agent": self.name, "title": self.title, "status": "complete", "output": _format_bullets(lines)}

        csv_text = state.get("dataset_preview", "").strip()
        if not csv_text:
            return {"agent": self.name, "title": self.title, "status": "error", "output": "- No CSV data was uploaded."}

        df = load_dataframe(csv_text)
        metadata = summarize_dataframe(df)
        state["metadata"] = metadata
        cleaned, cleaning_summary = clean_dataframe(df)
        state.setdefault("metadata", {})["cleaning_summary"] = cleaning_summary
        state["metadata"]["cleaned_row_count"] = int(len(cleaned))
        state["metadata"]["cleaned_dataset_csv"] = cleaned.to_csv(index=False)
        state["metadata"]["cleaned_dataset_preview"] = cleaned.head(20).to_csv(index=False)
        state["metadata"]["cleaned_dtypes"] = {column: str(dtype) for column, dtype in cleaned.dtypes.items()}

        lines = [
            f"Loaded {metadata['row_count']} rows and {metadata['column_count']} columns.",
            f"Numeric columns: {', '.join(metadata['numeric_columns'][:6]) or 'none detected'}.",
            f"Categorical columns: {', '.join(metadata['categorical_columns'][:6]) or 'none detected'}.",
            f"Missing-value columns: {len(metadata['missing_by_column'])}; duplicate rows: {metadata['duplicate_rows']}.",
        ]
        lines.extend(metadata.get("pattern_notes", [])[:3])
        if cleaning_summary["dropped_duplicates"]:
            lines.append(f"Removed {cleaning_summary['dropped_duplicates']} duplicate rows before modeling.")
        lines.append(f"Likely target columns: {', '.join(metadata['target_suggestions']) or 'no clear target suggestion yet'}.")
        lines.append(f"Missing cells retained for split-local imputation: {cleaning_summary['remaining_missing']}.")
        return {"agent": self.name, "title": self.title, "status": "complete", "output": _format_bullets(lines)}

    def build_prompt(self, state: AnalyticsState) -> str:
        raise NotImplementedError

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        return reason


class ModelTrainingAgent(BaseAgent):
    name = "model_training"
    title = "Model Selection and Training Agent"
    system_prompt = ""

    async def run(self, state: AnalyticsState) -> AgentResult:
        metadata = state.setdefault("metadata", {})
        cleaned_csv = metadata.get("training_sample_csv") or metadata.get("cleaned_dataset_csv") or state.get("dataset_preview", "")
        df = load_dataframe(cleaned_csv)

        target_column = state.get("target_column") or metadata.get("target_suggestions", [None])[0]
        if not target_column:
            return {
                "agent": self.name,
                "title": self.title,
                "status": "waiting",
                "output": _format_bullets(["No target column was selected, so model training is paused."]),
            }

        metadata["target_column"] = target_column
        metadata["task_type"] = infer_task_type(df, target_column)

        if not can_train_models():
            summary = {
                "task_type": metadata["task_type"],
                "best_model": "Pending scikit-learn install",
                "best_metric": "n/a",
                "target_column": target_column,
                "candidates": [],
                "metrics_table": [],
            }
            state["model_summary"] = summary
            lines = [
                f"Target column selected: {target_column}.",
                f"Detected task type: {metadata['task_type']}.",
                "Model training is scaffolded, but scikit-learn is not installed in the backend runtime yet.",
            ]
            return {"agent": self.name, "title": self.title, "status": "fallback", "output": _format_bullets(lines)}

        outcome = train_and_evaluate(df, target_column)
        state["model_summary"] = {
            "task_type": outcome.task_type,
            "best_model": outcome.best_model,
            "best_metric": outcome.best_metric,
            "target_column": outcome.target_column,
            "candidates": outcome.candidates,
            "metrics_table": outcome.metrics_table,
            "dropped_features": outcome.dropped_features,
        }

        lines = [
            f"Target column: {outcome.target_column}.",
            f"Detected task type: {outcome.task_type}.",
            f"Excluded non-predictive features: {', '.join(outcome.dropped_features) or 'none'}.",
            f"Candidate models: {', '.join(item['model'] for item in outcome.candidates)}.",
            f"Best model selected on validation data: {outcome.best_model}.",
        ]
        return {"agent": self.name, "title": self.title, "status": "complete", "output": _format_bullets(lines)}

    def build_prompt(self, state: AnalyticsState) -> str:
        raise NotImplementedError

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        return reason


class ModelEvaluationAgent(BaseAgent):
    name = "model_evaluation"
    title = "Model Evaluation Agent"
    system_prompt = ""

    async def run(self, state: AnalyticsState) -> AgentResult:
        summary = state.get("model_summary", {})
        if not summary:
            return {"agent": self.name, "title": self.title, "status": "waiting", "output": _format_bullets(["No model summary is available yet for evaluation."])}

        metrics_table = summary.get("metrics_table", [])
        if not metrics_table:
            return {
                "agent": self.name,
                "title": self.title,
                "status": "fallback",
                "output": _format_bullets(
                    [
                        f"Prepared evaluation stage for target '{summary.get('target_column', 'unknown')}'.",
                        "Metrics table will populate once scikit-learn is available for backend training.",
                    ]
                ),
            }

        best_row = next((row for row in metrics_table if row["model"] == summary.get("best_model")), metrics_table[0])
        lines = [f"Best model: {summary.get('best_model')}."]
        if summary.get("task_type") == "classification":
            lines.extend(
                [
                    f"Accuracy: {best_row.get('accuracy', 0.0):.3f}.",
                    f"Precision / Recall / F1: {best_row.get('precision', 0.0):.3f} / {best_row.get('recall', 0.0):.3f} / {best_row.get('f1', 0.0):.3f}.",
                ]
            )
        else:
            lines.extend(
                [
                    f"R^2: {best_row.get('r2', 0.0):.3f}.",
                    f"MAE / RMSE: {best_row.get('mae', 0.0):.3f} / {best_row.get('rmse', 0.0):.3f}.",
                ]
            )
        return {"agent": self.name, "title": self.title, "status": "complete", "output": _format_bullets(lines)}

    def build_prompt(self, state: AnalyticsState) -> str:
        raise NotImplementedError

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        return reason


class VisualizationAgent(BaseAgent):
    name = "visualization"
    title = "Visualization Agent"
    system_prompt = (
        "Recommend chart types only. Do not report metric values, percentages, feature rankings, "
        "class proportions, or conclusions about the data. Use only facts present in the supplied context."
    )

    async def run(self, state: AnalyticsState) -> AgentResult:
        return {
            "agent": self.name,
            "title": self.title,
            "status": "complete",
            "output": self.recommendations(state),
        }

    def recommendations(self, state: AnalyticsState) -> str:
        model_summary = state.get("model_summary", {})
        task_type = model_summary.get("task_type")
        quality_chart = (
            "a confusion matrix and class-level metrics"
            if task_type == "classification"
            else "actual-versus-predicted values and residuals"
            if task_type == "regression"
            else "task-appropriate prediction-quality charts"
        )
        return (
            "Recommended visualizations: missing values by column, key feature distributions, "
            f"candidate-model comparison, and {quality_chart}."
        )

    def build_prompt(self, state: AnalyticsState) -> str:
        model_summary = state.get("model_summary", {})
        metadata = state.get("metadata", {})
        columns = metadata.get("columns", [])
        return (
            f"Question: {str(state.get('question', ''))[:250]}\n"
            f"Target column: {model_summary.get('target_column', state.get('target_column', 'not selected'))}\n"
            f"Task type: {model_summary.get('task_type', 'not determined')}\n"
            f"Available columns: {', '.join(columns[:12])}\n\n"
            "Give exactly three concise chart-type recommendations. Do not include numbers, results, or reasoning."
        )

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        return self.recommendations(state)


class InsightAgent(BaseAgent):
    name = "insight"
    title = "Insight Agent"
    system_prompt = ""

    async def run(self, state: AnalyticsState) -> AgentResult:
        return self._build_result(state)

    def _build_result(self, state: AnalyticsState) -> AgentResult:
        summary = state.get("model_summary", {})
        metadata = state.get("metadata", {})
        metrics_table = summary.get("metrics_table", [])
        best_model = summary.get("best_model", "not trained")
        best_row = next((row for row in metrics_table if row.get("model") == best_model), {})
        task_type = summary.get("task_type")
        target = summary.get("target_column", state.get("target_column", "not selected"))
        row_count = int(metadata.get("row_count", 0) or 0)
        sample_csv = metadata.get("training_sample_csv", "")
        sample_rows = 0
        class_distribution = ""
        if sample_csv:
            sample_df = load_dataframe(sample_csv)
            sample_rows = len(sample_df)
            if task_type == "classification" and target in sample_df.columns:
                counts = sample_df[target].dropna().value_counts()
                if len(counts) > 1:
                    labels = [
                        f"{str(label)[:32]}: {count} ({count / counts.sum() * 100:.1f}%)"
                        for label, count in counts.head(6).items()
                    ]
                    class_distribution = "; ".join(labels)

        lines = [
            f"- Task: {task_type or 'not determined'} for target '{target}'.",
            f"- Selected model: {best_model}; selection used the validation split, while final metrics below use held-out test data.",
        ]
        if best_row and task_type == "classification":
            test_metrics = (
                f"Accuracy {best_row.get('accuracy', 0):.3f}; weighted precision "
                f"{best_row.get('precision', 0):.3f}; recall {best_row.get('recall', 0):.3f}; "
                f"F1 {best_row.get('f1', 0):.3f}"
            )
            if "roc_auc" in best_row:
                test_metrics += f"; ROC-AUC {best_row['roc_auc']:.3f}"
            lines.append(f"- Held-out test results: {test_metrics}.")
            selection_metric = "validation F1"
            next_step = "Inspect the confusion matrix and per-class recall, then compare against a dummy baseline."
        elif best_row and task_type == "regression":
            lines.append(
                f"- Held-out test results: R-squared {best_row.get('r2', 0):.3f}; "
                f"MAE {best_row.get('mae', 0):.3f}; RMSE {best_row.get('rmse', 0):.3f}."
            )
            selection_metric = "validation R-squared"
            next_step = "Inspect residuals and compare MAE/RMSE with the target scale and a dummy baseline."
        else:
            lines.append("- Held-out test metrics are not available for this run.")
            selection_metric = "validation score"
            next_step = "Confirm the target column and complete model evaluation before interpreting results."

        candidates = summary.get("candidates", [])
        candidate_scores = [
            item for item in candidates
            if isinstance(item.get("validation_score"), (int, float))
        ]
        if candidate_scores:
            ranked = sorted(candidate_scores, key=lambda item: item["validation_score"], reverse=True)
            comparison = "; ".join(
                f"{item['model']}: {item['validation_score']:.3f}"
                for item in ranked
            )
            lines.append(f"- Candidate comparison ({selection_metric}): {comparison}.")

        missing = metadata.get("missing_by_column", {})
        if missing:
            missing_info = []
            for column, count in sorted(missing.items(), key=lambda item: item[1], reverse=True)[:3]:
                share = (count / row_count * 100) if row_count else 0
                missing_info.append(f"{column}: {count}/{row_count} ({share:.1f}%)")
            lines.append("- Data quality: missing values by column: " + "; ".join(missing_info) + ".")
        else:
            lines.append("- Data quality: no missing values were recorded in the dataset profile.")

        dropped = summary.get("dropped_features", [])
        lines.append(
            "- Modeling sample: "
            + (f"{sample_rows} of {row_count} rows were used." if sample_rows and row_count else "row count was not recorded.")
            + (f" Excluded features: {', '.join(dropped)}." if dropped else " No non-predictive features were excluded.")
        )
        if class_distribution:
            lines.append(f"- Target-class distribution in the modeling sample: {class_distribution}.")
        lines.append(f"- Recommended next step: {next_step}")

        return {
            "agent": self.name,
            "title": self.title,
            "status": "complete",
            "output": "\n".join(lines),
        }

    def build_prompt(self, state: AnalyticsState) -> str:
        return "Summarize verified model metrics and dataset-quality facts from the shared workflow state."

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        return self._build_result(state)["output"]


class ReportAgent(BaseAgent):
    name = "report"
    title = "Report Agent"
    system_prompt = "You produce concise executive analytics reports with next actions."

    def build_prompt(self, state: AnalyticsState) -> str:
        return (
            f"Question: {state['question']}\n"
            f"Prior work:\n{previous_outputs(state, max_chars=3400)}\n\n"
            "Write a brief report with the dataset condition, selected model, verified evaluation result, "
            "and one next action. Use only facts and numbers present in the supplied results."
        )

    def fallback(self, state: AnalyticsState, reason: str) -> str:
        summary = state.get("model_summary", {})
        metadata = state.get("metadata", {})
        lines = [
            "Executive summary:",
            f"- Dataset size: {metadata.get('row_count', 'unknown')} rows and {metadata.get('column_count', 'unknown')} columns.",
            f"- Target column: {summary.get('target_column', state.get('target_column', 'not selected'))}.",
            f"- Selected model: {summary.get('best_model', 'not trained yet')}.",
            "- Next action: review the evaluation charts and validate whether the target is business-relevant before deployment.",
        ]
        return "\n".join(lines)


AGENTS = [
    DataPreprocessingAgent(),
    ModelTrainingAgent(),
    ModelEvaluationAgent(),
    VisualizationAgent(),
    InsightAgent(),
    ReportAgent(),
]
