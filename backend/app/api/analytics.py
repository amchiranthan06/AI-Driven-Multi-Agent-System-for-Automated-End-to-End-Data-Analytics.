from __future__ import annotations

import json

from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import run_analytics, stream_analytics_state
from app.dataset_store import compact_profile_text, dataset_path, profile_csv_file, save_uploaded_dataset
from app.visualization import generate_chart_artifacts

router = APIRouter()


class AnalyticsRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    dataset_id: str | None = Field(default=None, max_length=80)
    dataset_preview: str | None = None
    target_column: str | None = Field(default=None, max_length=200)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    return await save_uploaded_dataset(file)


@router.post("/run")
async def run(request: AnalyticsRequest) -> dict:
    state = await run_analytics(
        question=request.question,
        dataset_preview=request.dataset_preview,
        target_column=request.target_column,
        dataset_id=request.dataset_id,
    )
    return dict(state)


@router.get("/stream")
async def stream(
    question: str = Query(min_length=3, max_length=2000),
    dataset_id: str | None = Query(default=None, max_length=80),
    dataset_preview: str | None = Query(default=None),
    target_column: str | None = Query(default=None, max_length=200),
) -> StreamingResponse:
    async def event_source():
        state = {
            "question": question,
            "dataset_id": dataset_id or "",
            "dataset_preview": dataset_preview or "",
            "target_column": target_column or "",
            "results": [],
        }
        if dataset_id:
            path = dataset_path(dataset_id)
            profile = profile_csv_file(path)
            profile["dataset_id"] = dataset_id
            state["dataset_path"] = str(path)
            state["metadata"] = profile
            state["dataset_preview"] = compact_profile_text(profile)
        async for result in stream_analytics_state(state):
            state["results"].append(result)
            yield f"event: agent_result\ndata: {json.dumps(result)}\n\n"
        state["charts"] = await generate_chart_artifacts(state)
        yield f"event: final_state\ndata: {json.dumps(state)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
