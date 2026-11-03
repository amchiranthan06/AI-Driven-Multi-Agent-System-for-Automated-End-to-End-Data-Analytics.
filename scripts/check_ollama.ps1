$ErrorActionPreference = "Stop"

Write-Host "Checking Ollama server..."
try {
  $tags = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 5
} catch {
  Write-Host "Ollama is not reachable at http://localhost:11434"
  exit 1
}

$model = "qwen3:4b"
$names = @($tags.models | ForEach-Object { $_.name })
if ($names -contains $model) {
  Write-Host "Model is installed: $model"
} else {
  Write-Host "Ollama is running, but the model is not installed yet: $model"
  Write-Host "Later, run: ollama pull $model"
}
