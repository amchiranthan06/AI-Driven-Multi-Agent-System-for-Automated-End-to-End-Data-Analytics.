from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.analytics import router as analytics_router
from app.api.ollama import router as ollama_router
from app.core.config import settings
from app.core.paths import GENERATED_CHARTS_DIR

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ollama_router, prefix="/api/ollama", tags=["ollama"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["analytics"])
app.mount("/generated_charts", StaticFiles(directory=GENERATED_CHARTS_DIR), name="generated_charts")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}
