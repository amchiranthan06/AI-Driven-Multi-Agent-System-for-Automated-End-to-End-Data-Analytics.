from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
GENERATED_CHARTS_DIR = APP_DIR / "generated_charts"
GENERATED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR = APP_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
