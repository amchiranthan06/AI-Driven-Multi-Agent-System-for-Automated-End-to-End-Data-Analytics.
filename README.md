# AI Driven MAS for End-to-End Data Analytics

Local multi-agent analytics scaffold using FastAPI, LangGraph-style orchestration, Ollama, and React.

The app is ready to run with a local Ollama model. The current default is `qwen3:4b`, which is a better fit for a 6 GB VRAM laptop than an 8B model.

## Project Layout

```text
backend/
  app/
    agents/          # Six analytics agents and graph runner
    api/             # FastAPI routes
    core/            # Settings and shared helpers
    llm/             # Ollama client
    dataset_store.py # Local CSV storage and profiling
    main.py          # FastAPI app
  requirements.txt
frontend/
  src/
    components/
    services/
    App.jsx
  package.json
scripts/
  check_ollama.ps1  # Ollama availability check
```

## Quick Start

### First-time setup

Install Python, Node.js, and Ollama. From the project root, prepare the local environments:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..\frontend
npm ci
ollama pull qwen3:4b
```

The launcher serves Ollama on `127.0.0.1:11435`. It uses Ollama's configured model directory by default. Set `MAS_OLLAMA_MODELS_DIR` before launching only if you need a custom directory.

### One-Click Launch

On Windows, start Ollama, FastAPI, and the React frontend together by double-clicking:

```text
start_project.bat
```

The launcher requires the prepared `backend/.venv` and `frontend/node_modules` folders. It opens service windows and the dashboard:

- Ollama on `127.0.0.1:11435`
- FastAPI on `127.0.0.1:8000`
- React UI on `http://localhost:5173`

### Backend

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

### Frontend

```powershell
cd frontend
npm run dev
```

Open the Vite URL printed by npm, usually `http://localhost:5173`.

## Ollama Model

The target model is:

```text
qwen3:4b
```

You can postpone downloading it. The backend still starts and exposes:

- `GET /health`
- `GET /api/ollama/status`
- `POST /api/analytics/run`
- `GET /api/analytics/stream?question=...`

When you are ready:

```powershell
ollama pull qwen3:4b
```

## Notes

- Ollama prompts are capped to keep the local model stable.
- The server uses SSE streaming so the UI receives agent progress live.
- If Ollama or the configured model is unavailable, agents use local fallback summaries instead of failing the whole workflow.
- Uploaded datasets, generated charts, experiment outputs, and local environment files are excluded from Git; upload or regenerate them locally.
