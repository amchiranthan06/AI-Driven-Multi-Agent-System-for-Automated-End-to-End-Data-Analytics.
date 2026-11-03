# Research Experiment Protocol

Run from the backend environment with the local Ollama model available at the URL in `backend/.env`:

```powershell
cd backend
.\.venv\Scripts\python.exe ..\scripts\run_research_experiments.py
```

The experiment dependencies are `flaml[automl]==2.7.0` and `psutil==7.2.2`, in addition to `backend/requirements.txt`. The runner writes a timestamped directory under `output/experiments/` and does not overwrite previous runs.

Narrative generation uses the configured local Ollama model, context size from backend settings, a 1,024-token generation cap, and temperature 0.2. The client accepts only a marked final answer and removes any preceding tagged reasoning; truncated responses fall back to deterministic text.

## Datasets

| Dataset | Task | Target | Source used by runner |
| --- | --- | --- | --- |
| Wisconsin Diagnostic Breast Cancer (WDBC) | Classification | `diagnosis` | Existing project CSV; the `id` and empty `Unnamed: 32` columns are excluded from modeling |
| Wine | Classification | `target` | scikit-learn dataset loader |
| Iris | Classification | `target` | scikit-learn dataset loader |
| Diabetes | Regression | `target` | scikit-learn dataset loader |
| California Housing | Regression | `target` | scikit-learn dataset loader; downloaded into the run cache |

The local one-million-row crop-yield CSV is a separate processing stress test. Its provenance is not recorded, so it is not included in predictive-accuracy summaries or described as a public benchmark.

## Model Comparison

Each benchmark dataset is evaluated on three seeded 80/20 holdouts (`17`, `42`, and `2026`). Duplicate rows are removed before modeling. The target is separated before feature preprocessing. Identifier, constant, all-empty, and exact target-copy columns are excluded. Numeric median imputation, numeric scaling, categorical most-frequent imputation, and one-hot encoding are fitted within the training partition.

The project selector evaluates Logistic Regression, Random Forest, and Gradient Boosting on a validation split carved from the outer training partition. It chooses by weighted F1 for classification or R-squared for regression, refits the candidates on the full outer training partition, and reports metrics on the untouched outer test partition. Baselines use the same outer split: a prior/mean Dummy estimator and FLAML AutoML with an eight-second search budget per dataset and seed. FLAML optimizes weighted F1 for classification and R-squared for regression. The budget is a search budget, not a guaranteed wall-clock limit.

Classification metrics are accuracy, weighted precision, weighted recall, weighted F1, and macro F1. Regression metrics are MAE, RMSE, and R-squared. The summary reports means and sample standard deviations across the three seeded holdouts; these are repeated holdouts, not k-fold cross-validation.

## Report Faithfulness

For each public dataset, the runner invokes the actual six-agent LangGraph workflow once and compares its final report with an LLM-only response from the same local model. The LLM-only prompt contains a 12-row CSV excerpt and does not contain the backend model metrics. Numeric claims are extracted from each report and matched against backend facts: dataset dimensions, duplicate and missing counts, and selected-model test metrics. Values are matched with a tolerance for normal rounding.

Only reports whose Report Agent status is `complete` are scored as generated reports. Deterministic fallback text is excluded and recorded with an undefined faithfulness score. The LLM-only run records Ollama completion reason, token counts, response length, and presence of thinking tags to help diagnose empty or truncated outputs; hidden reasoning text is not written to the experiment logs.

This automatic score is a numeric-value match rate, not a semantic proof that each number is attached to the correct metric. Review `faithfulness_claims.csv` and manually verify the claim text and `matched_fact` before describing the value as report faithfulness in a paper. If a report contains no numeric claims, its score is undefined rather than zero or 100%.

## Large-File Stress Test

The runner profiles the local one-million-row CSV with 100,000-row Pandas chunks, then invokes the complete graph. The system's current training loader uses the first 50,000 rows as its bounded modeling sample. Benchmark datasets are released before this stress test. The runner records profiling and end-to-end elapsed time, and samples Python backend-process RSS every 50 ms. This measurement does not include Ollama's separate process or GPU VRAM; report it as backend-process memory only. Dataset provenance is not recorded, so this is a processing stress fixture, not an accuracy benchmark.

## Refreshed Comparison and Reproducible Stress Fixture

The refreshed selector/baseline comparison uses five repeated 80/20 holdouts (seeds `17`, `42`, `2026`, `123`, and `777`) and FLAML search budgets of 8 and 60 seconds. Run it with:

```powershell
backend\.venv\Scripts\python.exe scripts\run_model_budget_sweep.py
```

Each trial is appended to `model_trials.jsonl`; the summary is written to `model_summary.csv`. The FLAML budget is a search budget, not a hard wall-clock limit. This script omits report generation and does not require Ollama.

The replacement stress fixture is generated from NumPy's seeded random generator and includes a sequential `id` field. Run it with:

```powershell
backend\.venv\Scripts\python.exe scripts\run_synthetic_stress.py --output-dir output\experiments\synthetic_run --rows 1000000 --seed 42
```

It profiles the CSV in 100,000-row chunks, verifies that the modeling feature filter excludes `id`, and runs the full workflow with the normal 50,000-row modeling cap. Report the synthetic file only as a processing stress fixture, not as a public accuracy benchmark. Backend RSS excludes Ollama and GPU memory.

## Output Files

- `benchmark_summary.csv`: per-dataset, per-method mean and standard deviation across seeded holdouts.
- `trial_results.jsonl`: individual model/seed metrics, runtime, memory, and status.
- `faithfulness_summary.csv`: per-dataset grounded and LLM-only numeric match rates with claim counts.
- `faithfulness_claims.csv`: extracted claim text, matched fact, and automatic match decision for manual audit.
- `agent_reports.json`: workflow and LLM-only report text plus graph status and computed fact maps.
- `dataset_manifest.csv`: dataset names, sources, dimensions, targets, and task types.
- `large_file_stress.json`: chunked profile and end-to-end large-file measurements.
- `run_metadata.json`: seeds, tool versions/setting summary, and holdout note.
