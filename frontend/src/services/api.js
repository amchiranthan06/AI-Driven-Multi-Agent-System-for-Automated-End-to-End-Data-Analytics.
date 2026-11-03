const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

export async function getOllamaStatus() {
  const response = await fetch(`${API_BASE_URL}/api/ollama/status`);
  if (!response.ok) {
    throw new Error("Could not reach the FastAPI backend.");
  }
  return response.json();
}

export { API_BASE_URL };

export async function uploadDataset(file) {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/analytics/upload`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    let detail = "Dataset upload failed.";
    try {
      const payload = await response.json();
      detail = payload?.detail || detail;
    } catch {
      // keep fallback detail
    }
    throw new Error(detail);
  }

  return response.json();
}

export async function runAnalytics({ question, datasetId, targetColumn }) {
  const response = await fetch(`${API_BASE_URL}/api/analytics/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      dataset_id: datasetId || "",
      target_column: targetColumn || "",
    }),
  });

  if (!response.ok) {
    let detail = "Analytics request failed.";
    try {
      const payload = await response.json();
      detail = payload?.detail
        ? typeof payload.detail === "string"
          ? payload.detail
          : JSON.stringify(payload.detail)
        : detail;
    } catch {
      // keep fallback detail
    }
    throw new Error(detail);
  }

  return response.json();
}

export function streamAnalytics({ question, datasetId, targetColumn, onResult, onComplete }) {
  return new Promise((resolve, reject) => {
    const params = new URLSearchParams({
      question,
      dataset_id: datasetId || "",
      target_column: targetColumn || "",
    });
    const source = new EventSource(`${API_BASE_URL}/api/analytics/stream?${params.toString()}`);

    source.addEventListener("agent_result", (event) => {
      onResult(JSON.parse(event.data));
    });

    source.addEventListener("final_state", (event) => {
      const payload = JSON.parse(event.data);
      onComplete?.(payload);
    });

    source.addEventListener("done", () => {
      source.close();
      resolve();
    });

    source.onerror = () => {
      source.close();
      reject(new Error("Analytics stream failed. Check that FastAPI is running on port 8000."));
    };
  });
}
