from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
import textwrap
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from app.agents.base import previous_outputs
from app.agents.state import AnalyticsState, ChartArtifact
from app.core.paths import GENERATED_CHARTS_DIR
from app.dataset_store import load_training_sample
from app.llm.ollama_client import ollama_client


SAFE_BUILTINS = {
    "abs": abs,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "zip": zip,
}


async def generate_chart_artifacts(state: AnalyticsState) -> list[ChartArtifact]:
    if state.get("dataset_path"):
        try:
            csv_text = load_training_sample(state["dataset_path"], max_rows=20_000).to_csv(index=False)
        except Exception:
            csv_text = ""
    else:
        csv_text = (
            state.get("metadata", {}).get("cleaned_dataset_csv")
            or state.get("dataset_preview", "")
        ).strip()
    if not csv_text:
        return []

    run_dir = GENERATED_CHARTS_DIR / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _schema_driven_charts(csv_text, run_dir, state)
    artifacts.extend(_evaluation_charts(state, run_dir, len(artifacts) + 1))
    if artifacts:
        return artifacts

    code = await _generate_chart_code(state, csv_text)
    artifacts = _execute_chart_code(code, csv_text, run_dir) if code else []
    artifacts.extend(_evaluation_charts(state, run_dir, len(artifacts) + 1))
    return artifacts


async def _generate_chart_code(state: AnalyticsState, csv_text: str) -> str:
    schema_summary = _schema_summary(csv_text)
    model_summary = state.get("model_summary", {})
    prompt = textwrap.dedent(
        f"""
        You are writing Python chart code for a CSV dataset.
        Question:
        {state.get("question", "")}

        Prior analysis:
        {previous_outputs(state)}

        Model summary:
        {model_summary}

        Dataset schema summary:
        {schema_summary}

        Dataset preview:
        {csv_text[:6000]}

        Write Python code only in one fenced ```python``` block.
        The runtime already provides:
        - df: pandas DataFrame
        - plt: matplotlib.pyplot
        - output_dir: pathlib.Path

        Requirements:
        - Do not import anything.
        - Create 1 to 3 useful charts based on the actual dataframe columns and likely analysis intent.
        - Prefer line charts for time series, grouped bars for categorical comparisons, scatter plots for two numeric variables, histograms for distributions, and box plots for outliers.
        - Save each chart as a PNG inside output_dir.
        - Populate chart_manifest as a list of dicts with keys: filename, title, caption.
        - Call plt.close() after each save.
        """
    ).strip()

    try:
        response = await ollama_client.generate(
            prompt=prompt,
            system="You write compact, executable Python visualization code for pandas dataframes.",
        )
    except Exception:
        return ""
    return _extract_python_block(response)


def _extract_python_block(text: str) -> str:
    match = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip() if "chart_manifest" in text else ""


def _execute_chart_code(code: str, csv_text: str, output_dir: Path) -> list[ChartArtifact]:
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        return []

    chart_manifest: list[dict] = []
    globals_dict = {
        "__builtins__": SAFE_BUILTINS,
        "chart_manifest": chart_manifest,
        "df": df,
        "output_dir": output_dir,
        "pd": pd,
        "plt": plt,
    }

    try:
        exec(compile(code, "<chart_code>", "exec"), globals_dict, globals_dict)
    except Exception:
        return []

    artifacts: list[ChartArtifact] = []
    for item in chart_manifest:
        filename = str(item.get("filename", "")).strip()
        if not filename:
            continue
        file_path = output_dir / filename
        if not file_path.exists():
            continue
        artifacts.append(
            {
                "title": str(item.get("title", "Generated chart")).strip(),
                "caption": str(item.get("caption", "")).strip(),
                "image_url": f"/generated_charts/{output_dir.name}/{filename}",
            }
        )
    return artifacts


def _schema_driven_charts(csv_text: str, output_dir: Path, state: AnalyticsState) -> list[ChartArtifact]:
    try:
        df = pd.read_csv(StringIO(csv_text))
    except Exception:
        return []

    df = _normalize_dataframe(df)
    artifacts: list[ChartArtifact] = []
    metadata = state.get("metadata", {})
    model_summary = state.get("model_summary", {})
    target_column = model_summary.get("target_column") or metadata.get("target_column") or state.get("target_column")
    chart_exclusions = []
    for column in df.columns:
        normalized = re.sub(r"\s+", "_", str(column).strip().lower())
        is_id = normalized in {"id", "uuid", "index", "row_number"} or normalized.endswith(("_id", "_uuid"))
        is_empty_or_constant = df[column].nunique(dropna=True) <= 1
        if column != target_column and (is_id or is_empty_or_constant):
            chart_exclusions.append(column)
    if chart_exclusions:
        df = df.drop(columns=chart_exclusions)
    numeric_columns = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    datetime_columns = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    missing_counts = df.isna().sum().sort_values(ascending=False)
    missing_counts = missing_counts[missing_counts > 0].head(12)
    if not missing_counts.empty:
        plt.figure(figsize=(10, 5.5))
        missing_counts.sort_values().plot(kind="barh", color="#b4553a")
        plt.title("Columns with missing values")
        plt.xlabel("missing cells")
        plt.ylabel("")
        _apply_chart_style()
        artifacts.append(
            _save_chart(
                output_dir,
                f"chart_{len(artifacts)+1}.png",
                "Missing value summary",
                "Highlights columns with the highest missing-value counts before modeling.",
            )
        )

    if target_column in df.columns and len(artifacts) < 4:
        target = df[target_column].dropna()
        if not target.empty and (target_column in categorical_columns or target.nunique(dropna=True) <= 20):
            counts = target.astype(str).value_counts().head(15).sort_values()
            plt.figure(figsize=(10, 5.5))
            counts.plot(kind="barh", color="#185e69")
            plt.title(f"Target distribution: {target_column}")
            plt.xlabel("records")
            plt.ylabel("")
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"Target distribution: {target_column}",
                    f"Class balance for target column {target_column}.",
                )
            )
        elif target_column in numeric_columns:
            plt.figure(figsize=(10, 5.5))
            target.plot(kind="hist", bins=_safe_bins(len(target)), color="#185e69", edgecolor="#ffffff")
            plt.title(f"Target distribution: {target_column}")
            plt.xlabel(target_column)
            plt.ylabel("records")
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"Target distribution: {target_column}",
                    f"Distribution of continuous target column {target_column}.",
                )
            )

    if target_column in df.columns and numeric_columns and len(artifacts) < 4:
        feature_col = _best_numeric_feature(df, numeric_columns, target_column)
        if feature_col:
            if target_column in categorical_columns or df[target_column].nunique(dropna=True) <= 12:
                plot_df = df[[target_column, feature_col]].dropna()
                if not plot_df.empty:
                    top_classes = plot_df[target_column].astype(str).value_counts().head(8).index
                    plot_df = plot_df[plot_df[target_column].astype(str).isin(top_classes)]
                    if not plot_df.empty:
                        plt.figure(figsize=(10, 5.5))
                        plot_df.assign(**{target_column: plot_df[target_column].astype(str)}).boxplot(
                            column=feature_col,
                            by=target_column,
                            grid=False,
                            rot=20,
                        )
                        plt.suptitle("")
                        plt.title(f"{feature_col} by {target_column}")
                        plt.xlabel(target_column)
                        plt.ylabel(feature_col)
                        _apply_chart_style()
                        artifacts.append(
                            _save_chart(
                                output_dir,
                                f"chart_{len(artifacts)+1}.png",
                                f"{feature_col} by {target_column}",
                                f"Compares a high-signal numeric feature against target classes.",
                            )
                        )
            elif target_column in numeric_columns and feature_col != target_column:
                scatter = df[[feature_col, target_column]].dropna()
                if not scatter.empty:
                    plt.figure(figsize=(10, 5.5))
                    plt.scatter(scatter[feature_col], scatter[target_column], alpha=0.55, color="#185e69", edgecolors="none")
                    plt.title(f"{target_column} vs {feature_col}")
                    plt.xlabel(feature_col)
                    plt.ylabel(target_column)
                    _apply_chart_style()
                    artifacts.append(
                        _save_chart(
                            output_dir,
                            f"chart_{len(artifacts)+1}.png",
                            f"{target_column} vs {feature_col}",
                            f"Shows the relationship between {feature_col} and target {target_column}.",
                        )
                    )

    if len(numeric_columns) >= 3 and len(artifacts) < 4:
        corr_columns = _correlation_columns(df, numeric_columns, target_column)
        corr = df[corr_columns].corr(numeric_only=True)
        if not corr.empty:
            plt.figure(figsize=(8.5, 7))
            image = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            plt.colorbar(image, fraction=0.046, pad=0.04)
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
            plt.title("Numeric correlation heatmap")
            for i in range(len(corr.index)):
                for j in range(len(corr.columns)):
                    value = corr.iloc[i, j]
                    if pd.notna(value):
                        plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="#111827", fontsize=8)
            plt.tight_layout()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    "Numeric correlation heatmap",
                    "Shows correlation strength between important numeric columns.",
                )
            )

    if datetime_columns and numeric_columns and len(artifacts) < 4:
        time_col = datetime_columns[0]
        metric_col = _first_non_target(numeric_columns, target_column) or numeric_columns[0]
        time_series = (
            df[[time_col, metric_col]]
            .dropna()
            .sort_values(time_col)
            .groupby(time_col, as_index=False)[metric_col]
            .sum()
        )
        if not time_series.empty:
            plt.figure(figsize=(10, 5.5))
            plt.plot(time_series[time_col], time_series[metric_col], color="#1e5b66", linewidth=2)
            plt.title(f"{metric_col} over time")
            plt.xlabel(time_col)
            plt.ylabel(metric_col)
            plt.xticks(rotation=30, ha="right")
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"{metric_col} over time",
                    f"Trend of {metric_col} across {time_col}.",
                )
            )

    if categorical_columns and numeric_columns and len(artifacts) < 4:
        cat_col = _best_categorical_column(df, categorical_columns)
        metric_candidates = [_first_non_target(numeric_columns, target_column) or numeric_columns[0]]
        aggregated = (
            df[[cat_col] + metric_candidates]
            .dropna(subset=[cat_col])
            .groupby(cat_col, dropna=False)[metric_candidates]
            .sum()
            .sort_values(metric_candidates[0], ascending=False)
            .head(12)
        )
        if not aggregated.empty:
            plt.figure(figsize=(10, 5.5))
            aggregated.sort_values(metric_candidates[0]).plot(kind="barh", ax=plt.gca(), color="#1e5b66")
            plt.title(f"{', '.join(metric_candidates)} by {cat_col}")
            plt.xlabel("value")
            plt.ylabel("")
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"{metric_candidates[0]} by {cat_col}",
                    f"Category comparison showing aggregated {', '.join(metric_candidates)} across {cat_col}.",
                )
            )

    if len(numeric_columns) >= 2 and len(artifacts) < 4:
        x_col = _first_non_target(numeric_columns, target_column) or numeric_columns[0]
        y_col = next((column for column in numeric_columns if column != x_col), numeric_columns[1])
        scatter = df[[x_col, y_col]].dropna()
        if not scatter.empty:
            plt.figure(figsize=(10, 5.5))
            plt.scatter(scatter[x_col], scatter[y_col], alpha=0.55, color="#1e5b66", edgecolors="none")
            plt.title(f"{y_col} vs {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"{y_col} vs {x_col}",
                    f"Relationship between numeric columns {x_col} and {y_col}.",
                )
            )

    if numeric_columns and len(artifacts) < 4:
        metric_col = _first_non_target(numeric_columns, target_column) or numeric_columns[0]
        values = df[metric_col].dropna()
        if not values.empty:
            plt.figure(figsize=(10, 5.5))
            values.plot(kind="hist", bins=_safe_bins(len(values)), color="#ae563a", edgecolor="#ffffff")
            plt.title(f"Distribution of {metric_col}")
            plt.xlabel(metric_col)
            plt.ylabel("records")
            _apply_chart_style()
            artifacts.append(
                _save_chart(
                    output_dir,
                    f"chart_{len(artifacts)+1}.png",
                    f"Distribution of {metric_col}",
                    f"Distribution chart for {metric_col} to highlight spread and skew.",
                )
            )

    return artifacts


def _evaluation_charts(state: AnalyticsState, output_dir: Path, start_index: int) -> list[ChartArtifact]:
    summary = state.get("model_summary", {})
    metrics_table = summary.get("metrics_table", [])
    if not metrics_table:
        return []

    df = pd.DataFrame(metrics_table)
    artifacts: list[ChartArtifact] = []
    task_type = summary.get("task_type")

    if task_type == "classification" and "f1" in df.columns:
        plt.figure(figsize=(10, 5.5))
        metric_cols = [column for column in ["accuracy", "precision", "recall", "f1", "roc_auc"] if column in df.columns]
        df.plot(x="model", y=metric_cols, kind="bar", ax=plt.gca(), color=["#185e69", "#2d7f9f", "#c06f3a", "#6b4aa3", "#31785f"][: len(metric_cols)])
        plt.title("Classification model comparison")
        plt.xlabel("model")
        plt.ylabel("score")
        plt.ylim(0, 1.05)
        plt.xticks(rotation=20, ha="right")
        _apply_chart_style()
        artifacts.append(
            _save_chart(
                output_dir,
                f"chart_{start_index}.png",
                "Classification model comparison",
                "Compares accuracy, precision, recall, and F1-score across trained classifiers.",
            )
        )
    elif task_type == "regression" and "r2" in df.columns:
        plt.figure(figsize=(10, 5.5))
        df.plot(x="model", y=["r2", "rmse", "mae"], kind="bar", ax=plt.gca(), color=["#185e69", "#c06f3a", "#6b4aa3"])
        plt.title("Regression model comparison")
        plt.xlabel("model")
        plt.ylabel("score")
        plt.xticks(rotation=20, ha="right")
        _apply_chart_style()
        artifacts.append(
            _save_chart(
                output_dir,
                f"chart_{start_index}.png",
                "Regression model comparison",
                "Compares R^2, RMSE, and MAE across trained regressors.",
            )
        )

    return artifacts


def _schema_summary(csv_text: str) -> str:
    try:
        df = _normalize_dataframe(pd.read_csv(StringIO(csv_text)))
    except Exception:
        return "Unable to read dataframe."

    parts: list[str] = [f"Rows: {len(df)}, Columns: {len(df.columns)}"]
    for column in df.columns[:12]:
        dtype = str(df[column].dtype)
        sample = ", ".join(df[column].dropna().astype(str).head(3).tolist())
        parts.append(f"- {column}: {dtype}; sample values: {sample}")
    return "\n".join(parts)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            stripped = normalized[column].astype(str).str.replace(r"[$,%]", "", regex=True).str.strip()
            numeric_version = pd.to_numeric(stripped, errors="coerce")
            if numeric_version.notna().mean() > 0.8:
                normalized[column] = numeric_version
                continue
            datetime_version = pd.to_datetime(normalized[column], errors="coerce")
            if datetime_version.notna().mean() > 0.8:
                normalized[column] = datetime_version
    return normalized


def _best_categorical_column(df: pd.DataFrame, categorical_columns: list[str]) -> str:
    ranked = sorted(
        categorical_columns,
        key=lambda column: df[column].nunique(dropna=True)
    )
    for column in ranked:
        unique_count = df[column].nunique(dropna=True)
        if 1 < unique_count <= 20:
            return column
    return ranked[0]


def _first_non_target(columns: list[str], target_column: str | None) -> str | None:
    return next((column for column in columns if column != target_column), None)


def _best_numeric_feature(df: pd.DataFrame, numeric_columns: list[str], target_column: str | None) -> str | None:
    candidates = [column for column in numeric_columns if column != target_column]
    if not candidates:
        return None
    if target_column in numeric_columns:
        correlations = df[candidates + [target_column]].corr(numeric_only=True)[target_column].abs().drop(labels=[target_column], errors="ignore")
        correlations = correlations.dropna().sort_values(ascending=False)
        if not correlations.empty:
            return str(correlations.index[0])
    variances = df[candidates].var(numeric_only=True).dropna().sort_values(ascending=False)
    if not variances.empty:
        return str(variances.index[0])
    return candidates[0]


def _correlation_columns(df: pd.DataFrame, numeric_columns: list[str], target_column: str | None) -> list[str]:
    if target_column in numeric_columns:
        correlations = df[numeric_columns].corr(numeric_only=True)[target_column].abs().drop(labels=[target_column], errors="ignore")
        selected = correlations.dropna().sort_values(ascending=False).head(7).index.tolist()
        return [target_column, *selected]
    variances = df[numeric_columns].var(numeric_only=True).dropna().sort_values(ascending=False)
    return variances.head(8).index.tolist() or numeric_columns[:8]


def _safe_bins(row_count: int) -> int:
    return min(30, max(8, int(row_count ** 0.5)))


def _apply_chart_style() -> None:
    axis = plt.gca()
    axis.grid(axis="y", color="#d8e1ea", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)
    axis.spines["left"].set_color("#c8d2dd")
    axis.spines["bottom"].set_color("#c8d2dd")
    axis.tick_params(colors="#344052")
    plt.tight_layout()


def _save_chart(output_dir: Path, filename: str, title: str, caption: str) -> ChartArtifact:
    plt.savefig(output_dir / filename, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    return {
        "title": title,
        "caption": caption,
        "image_url": f"/generated_charts/{output_dir.name}/{filename}",
    }
