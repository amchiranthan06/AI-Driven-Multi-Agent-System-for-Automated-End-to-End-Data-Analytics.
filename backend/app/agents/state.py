from typing import Any, TypedDict


class AgentResult(TypedDict):
    agent: str
    title: str
    status: str
    output: str


class ChartArtifact(TypedDict):
    title: str
    caption: str
    image_url: str


class ModelMetrics(TypedDict, total=False):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    mae: float
    mse: float
    rmse: float
    r2: float


class ModelSummary(TypedDict, total=False):
    task_type: str
    best_model: str
    best_metric: str
    target_column: str
    candidates: list[dict[str, Any]]
    metrics_table: list[dict[str, Any]]
    dropped_features: list[str]


class AnalyticsState(TypedDict, total=False):
    question: str
    dataset_id: str
    dataset_path: str
    dataset_preview: str
    target_column: str
    metadata: dict[str, Any]
    results: list[AgentResult]
    charts: list[ChartArtifact]
    model_summary: ModelSummary
