import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  Database,
  Gauge,
  Layers3,
  Play,
  RefreshCw,
  Server,
  Sparkles,
  Table2,
  Trash2,
  Upload,
} from "lucide-react";
import { API_BASE_URL, getOllamaStatus, streamAnalytics, uploadDataset } from "./services/api";
import { AgentTimeline } from "./components/AgentTimeline";
import { DataChart } from "./components/DataChart";
import "./styles.css";

const DATASET_ANALYSIS_PROMPT =
  "Run the full pipeline on the uploaded dataset: analyze the data, detect patterns, handle missing values, select and train a suitable machine learning model for the chosen target column, evaluate it, generate charts, and summarize the final insights.";

export default function App() {
  const [question, setQuestion] = useState("");
  const [datasetId, setDatasetId] = useState("");
  const [datasetProfile, setDatasetProfile] = useState(null);
  const [columns, setColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState("");
  const [manualTargetSelection, setManualTargetSelection] = useState(false);
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState([]);
  const [charts, setCharts] = useState([]);
  const [hasAnalyzedDataset, setHasAnalyzedDataset] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [datasetName, setDatasetName] = useState("No dataset loaded");
  const [finalState, setFinalState] = useState(null);

  const statusLabel = useMemo(() => {
    if (!status) return "Checking";
    if (!status.server_available) return "Ollama offline";
    if (!status.model_available) return "Model pending";
    return "Ready";
  }, [status]);
  const effectiveTargetColumn = manualTargetSelection ? targetColumn : "";
  const runComplete = results.length >= 6 && !isRunning;
  const canShowVisualization = hasAnalyzedDataset && runComplete;
  const rowCount = datasetProfile?.row_count || 0;
  const columnCount = datasetProfile?.column_count || columns.length || 0;
  const completedAgents = results.length;
  const modelSummary = finalState?.model_summary;

  async function refreshStatus() {
    setError("");
    try {
      setStatus(await getOllamaStatus());
    } catch (err) {
      setError(err.message);
    }
  }

  async function runWorkflow(runQuestion, { markAnalyzed = false } = {}) {
    setIsRunning(true);
    setResults([]);
    setCharts([]);
    setFinalState(null);
    setError("");

    try {
      await streamAnalytics({
        question: runQuestion,
        datasetId,
        targetColumn: effectiveTargetColumn,
        onResult: (result) => {
          setResults((current) => [...current, result]);
        },
        onComplete: (payload) => {
          setFinalState(payload);
          const resolvedCharts = (payload.charts || []).map((chart) => ({
            ...chart,
            image_url: chart.image_url.startsWith("http") ? chart.image_url : `${API_BASE_URL}${chart.image_url}`,
          }));
          setCharts(resolvedCharts);
        },
      });
      if (markAnalyzed) {
        setHasAnalyzedDataset(true);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsRunning(false);
      refreshStatus();
    }
  }

  function analyzeDataset() {
    runWorkflow(DATASET_ANALYSIS_PROMPT, { markAnalyzed: true });
  }

  function queryDataset() {
    runWorkflow(question, { markAnalyzed: false });
  }

  function uploadCsv(event) {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsRunning(true);
    setError("");
    uploadDataset(file)
      .then((profile) => {
        setDatasetId(profile.dataset_id);
        setDatasetProfile(profile);
        setDatasetName(profile.filename || file.name);
        const parsedColumns = profile.columns || [];
        setColumns(parsedColumns);
        setTargetColumn(profile.target_suggestions?.[0] || parsedColumns[0] || "");
        setManualTargetSelection(false);
        setQuestion("");
        setResults([]);
        setCharts([]);
        setHasAnalyzedDataset(false);
        setFinalState(null);
      })
      .catch((err) => {
        setError(err.message);
      })
      .finally(() => {
        setIsRunning(false);
      });
      setQuestion("");
      event.target.value = "";
  }

  function clearDataset() {
    setDatasetId("");
    setDatasetProfile(null);
    setDatasetName("No dataset loaded");
    setQuestion("");
    setResults([]);
    setCharts([]);
    setHasAnalyzedDataset(false);
    setColumns([]);
    setTargetColumn("");
    setManualTargetSelection(false);
    setFinalState(null);
  }

  useEffect(() => {
    refreshStatus();
  }, []);

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Local multi-agent analytics</p>
            <h1>MAS Analytics Console</h1>
            <p className="hero-copy">
              Upload a CSV, run the deployed agents, review machine learning results, and generate a report locally.
            </p>
          </div>
          <div className={`status-pill ${status?.model_available ? "ready" : "pending"}`}>
            <Server size={18} />
            <span>{statusLabel}</span>
          </div>
        </header>

        <section className="dashboard-grid">
          <article className="metric-card">
            <div className="metric-icon teal">
              <Table2 size={20} />
            </div>
            <div>
              <span>Dataset size</span>
              <strong>{datasetId ? `${rowCount.toLocaleString()} x ${columnCount}` : "Not loaded"}</strong>
            </div>
          </article>
          <article className="metric-card">
            <div className="metric-icon blue">
              <Layers3 size={20} />
            </div>
            <div>
              <span>Agent progress</span>
              <strong>{completedAgents} / 6 complete</strong>
            </div>
          </article>
          <article className="metric-card">
            <div className="metric-icon amber">
              <BrainCircuit size={20} />
            </div>
            <div>
              <span>Selected model</span>
              <strong>{modelSummary?.best_model || "Pending run"}</strong>
            </div>
          </article>
          <article className="metric-card">
            <div className="metric-icon violet">
              <Gauge size={20} />
            </div>
            <div>
              <span>Best metric</span>
              <strong>{modelSummary?.best_metric || "Pending run"}</strong>
            </div>
          </article>
        </section>

        <section className="control-grid">
          <div className="input-panel">
            <div className="panel-title-row">
              <div>
                <p className="eyebrow small">Input workspace</p>
                <h2>Dataset Setup</h2>
              </div>
              <span className="mini-badge">{datasetId ? "Uploaded" : "Waiting for CSV"}</span>
            </div>

            <div className="field-heading upload-row">
              <div>
                <label htmlFor="csv-upload">Dataset</label>
                <p className="dataset-name">{datasetName}</p>
              </div>
              <div className="dataset-tools">
                <label className="upload-button" htmlFor="csv-upload">
                  <Upload size={16} />
                  <span>Upload CSV</span>
                </label>
                <input id="csv-upload" type="file" accept=".csv,text/csv" onChange={uploadCsv} />
                <button className="clear-button" type="button" onClick={clearDataset} disabled={!datasetId}>
                  <Trash2 size={16} />
                  <span>Clear</span>
                </button>
              </div>
            </div>

            <div className="dataset-summary">
              <p>{datasetId ? "Dataset uploaded to backend storage. Large-file analysis will use compact profiles and bounded samples, not raw CSV text in the browser." : "Upload a CSV to start the end-to-end analytics flow."}</p>
              {datasetProfile ? (
                <dl className="profile-list">
                  <div>
                    <dt>Rows</dt>
                    <dd>{datasetProfile.row_count?.toLocaleString?.() || datasetProfile.row_count}</dd>
                  </div>
                  <div>
                    <dt>Columns</dt>
                    <dd>{datasetProfile.column_count}</dd>
                  </div>
                  <div>
                    <dt>Suggested targets</dt>
                    <dd>{datasetProfile.target_suggestions?.join(", ") || "none"}</dd>
                  </div>
                </dl>
              ) : null}
            </div>

            <div className="workflow-strip" aria-label="Pipeline stages">
              {[
                "Preprocess",
                "Train",
                "Evaluate",
                "Visualize",
                "Insights",
                "Report",
              ].map((step, index) => (
                <div
                  className={`workflow-step ${completedAgents > index ? "done" : isRunning && completedAgents === index ? "live" : ""}`}
                  key={step}
                >
                  <span>{index + 1}</span>
                  <p>{step}</p>
                </div>
              ))}
            </div>

            <label className="toggle-row" htmlFor="manual-target-selection">
              <span>Choose target manually</span>
              <input
                id="manual-target-selection"
                type="checkbox"
                checked={manualTargetSelection}
                onChange={(event) => setManualTargetSelection(event.target.checked)}
                disabled={!columns.length || isRunning}
              />
            </label>

            {manualTargetSelection ? (
              <>
                <label htmlFor="target-column">Target Column</label>
                <select
                  id="target-column"
                  value={targetColumn}
                  onChange={(event) => setTargetColumn(event.target.value)}
                  disabled={!columns.length || isRunning}
                >
                  {columns.length ? (
                    columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))
                  ) : (
                    <option value="">Upload a dataset first</option>
                  )}
                </select>
              </>
            ) : (
              <div className="dataset-summary">
                <p>The system will suggest a target column automatically. Turn on manual selection only if you want to override it.</p>
              </div>
            )}

            <div className="actions">
              <button className="primary" onClick={analyzeDataset} disabled={isRunning || !datasetId || (manualTargetSelection && !targetColumn)}>
                {isRunning ? <Activity size={18} /> : <Play size={18} />}
                <span>{isRunning && !hasAnalyzedDataset ? "Running Pipeline" : "Analyze, Train, and Evaluate"}</span>
              </button>
              <button className="secondary" onClick={refreshStatus} disabled={isRunning}>
                <RefreshCw size={18} />
                <span>Refresh</span>
              </button>
            </div>

            <div className="query-heading">
              <div>
                <label htmlFor="question">Query Dataset</label>
                <p>Ask follow-up questions after the full pipeline completes.</p>
              </div>
              <Sparkles size={18} />
            </div>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={4}
              disabled={!hasAnalyzedDataset || isRunning}
              placeholder={
                hasAnalyzedDataset
                  ? "Ask a follow-up question after the full analysis and modeling run."
                  : "Run the full dataset pipeline first to unlock querying."
              }
            />

            <div className="actions">
              <button
                className="primary"
                onClick={queryDataset}
                disabled={isRunning || !hasAnalyzedDataset || question.trim().length < 3}
              >
                {isRunning && hasAnalyzedDataset ? <Activity size={18} /> : <Play size={18} />}
                <span>{isRunning && hasAnalyzedDataset ? "Querying" : "Run Query"}</span>
              </button>
            </div>
          </div>

          <aside className="system-panel">
            <div className="panel-heading">
              <Database size={20} />
              <h2>System Readiness</h2>
            </div>
            <dl>
              <div>
                <dt>Ollama server</dt>
                <dd>{status?.server_available ? "Available" : "Not reachable"}</dd>
              </div>
              <div>
                <dt>Configured model</dt>
                <dd>{status?.model || "qwen3:4b"}</dd>
              </div>
              <div>
                <dt>LLM state</dt>
                <dd>{status?.model_available ? "Installed" : "Not downloaded"}</dd>
              </div>
              <div>
                <dt>Prediction target</dt>
                <dd>{manualTargetSelection ? targetColumn || "Not selected" : "Auto-select"}</dd>
              </div>
            </dl>
            <p className="detail">{status?.detail || "Waiting for backend status."}</p>
            <p className="detail workflow-note">
              {hasAnalyzedDataset
                ? "Pipeline finished. You can now query the analyzed and modeled dataset."
                : "Step 1: upload a CSV, choose the target, and run the pipeline."}
            </p>
            {error ? <p className="error">{error}</p> : null}

            <div className="quick-facts">
              <div>
                <BarChart3 size={18} />
                <span>{charts.length ? `${charts.length} chart${charts.length > 1 ? "s" : ""} generated` : "Charts after analysis"}</span>
              </div>
              <div>
                <Sparkles size={18} />
                <span>{runComplete ? "Report ready" : "Report pending"}</span>
              </div>
            </div>
          </aside>
        </section>

        <AgentTimeline results={results} isRunning={isRunning} />
        {finalState?.model_summary ? (
          <section className="chart-panel">
            <div className="section-heading">
              <h2>Model Summary</h2>
            </div>
            <div className="timeline">
              <article className="agent-card complete">
                <div className="agent-card-header">
                  <div>
                    <h3>{finalState.model_summary.best_model || "Pending model selection"}</h3>
                    <p>{finalState.model_summary.task_type || "unknown task"}</p>
                  </div>
                </div>
                <p className="agent-output">
                  Target column: {finalState.model_summary.target_column || targetColumn || "auto-detected"}
                  {"\n"}
                  Best metric: {finalState.model_summary.best_metric || "n/a"}
                </p>
              </article>
            </div>
          </section>
        ) : null}
        <DataChart charts={charts} canShow={canShowVisualization} />
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
