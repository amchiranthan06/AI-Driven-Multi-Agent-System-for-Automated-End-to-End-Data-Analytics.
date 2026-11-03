from fastapi import APIRouter

from app.llm.ollama_client import ollama_client

router = APIRouter()


@router.get("/status")
async def ollama_status() -> dict[str, bool | str]:
    status = await ollama_client.status()
    return {
        "server_available": status.server_available,
        "model_available": status.model_available,
        "model": status.model,
        "detail": status.detail,
    }
