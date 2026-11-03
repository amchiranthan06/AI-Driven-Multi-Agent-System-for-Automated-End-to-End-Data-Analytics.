from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import httpx

from app.core.config import settings


@dataclass(slots=True)
class OllamaStatus:
    server_available: bool
    model_available: bool
    model: str
    detail: str


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_model,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def status(self) -> OllamaStatus:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
        except Exception as exc:
            return OllamaStatus(
                server_available=False,
                model_available=False,
                model=self.model,
                detail=f"Ollama server is not reachable: {exc}",
            )

        data = response.json()
        models = {item.get("name") for item in data.get("models", [])}
        model_available = self.model in models
        detail = "Model is available." if model_available else "Ollama is running, but the configured model is not downloaded yet."
        return OllamaStatus(
            server_available=True,
            model_available=model_available,
            model=self.model,
            detail=detail,
        )

    async def generate(self, prompt: str, system: str | None = None, require_final_marker: bool = False) -> str:
        content, _ = await self.generate_with_diagnostics(
            prompt=prompt,
            system=system,
            require_final_marker=require_final_marker,
        )
        return content

    async def generate_with_diagnostics(
        self,
        prompt: str,
        system: str | None = None,
        require_final_marker: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        trimmed_prompt = prompt[: settings.max_prompt_chars]
        system_prompt = system or "You are a concise analytics assistant."
        if self.model.lower().startswith("qwen3:"):
            trimmed_prompt = f"/no_think\n{trimmed_prompt}"
            system_prompt = f"/no_think\n{system_prompt}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": trimmed_prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": settings.ollama_num_ctx,
                "num_predict": settings.ollama_num_predict,
                "temperature": 0.2,
            },
        }

        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "").strip()
        thinking = message.get("thinking", "")
        diagnostics = {
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "response_chars": len(content),
            "thinking_chars": len(thinking),
            "think_tags_present": bool(thinking) or bool(re.search(r"</?think>", content, flags=re.IGNORECASE)),
        }
        if data.get("done_reason") == "length":
            diagnostics["failure_reason"] = "truncated_at_generation_limit"
            return "", diagnostics
        cleaned = _clean_thinking_output(content, require_final_marker=require_final_marker)
        if not cleaned:
            diagnostics["failure_reason"] = "empty_or_missing_final_marker"
        return cleaned, diagnostics


def _clean_thinking_output(content: str, require_final_marker: bool = False) -> str:
    opening = re.search(r"<think>", content, flags=re.IGNORECASE)
    closing = list(re.finditer(r"</think>", content, flags=re.IGNORECASE))
    if closing:
        content = content[closing[-1].end():]
    elif opening:
        content = content[:opening.start()]

    marker = re.search(r"(?:^|\n)\s*(?:Final answer|Answer|Final):\s*", content, flags=re.IGNORECASE)
    if marker:
        content = content[marker.end():]
    elif require_final_marker:
        return ""
    else:
        code_block = re.search(r"```python\s*|```\s*", content, flags=re.IGNORECASE)
        if code_block:
            content = content[code_block.start():]

    content = re.sub(r"<\/?think>", "", content, flags=re.IGNORECASE).strip()
    return content


ollama_client = OllamaClient()
