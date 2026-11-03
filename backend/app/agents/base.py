from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.agents.state import AgentResult, AnalyticsState
from app.llm.ollama_client import OllamaClient, ollama_client

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    name: str
    title: str
    system_prompt: str

    def __init__(self, llm: OllamaClient = ollama_client) -> None:
        self.llm = llm

    async def run(self, state: AnalyticsState) -> AgentResult:
        status = await self.llm.status()
        if not status.model_available:
            return {
                "agent": self.name,
                "title": self.title,
                "status": "fallback",
                "output": self.fallback(state, status.detail),
            }

        prompt = (
            f"{self.build_prompt(state)}\n\n"
            "Return a concise result in 3 bullet points or fewer. "
            "Do not include hidden reasoning, chain-of-thought, or <think> sections. "
            "Start the visible response with 'Final:'."
        )
        output, diagnostics = await self.llm.generate_with_diagnostics(
            prompt=prompt,
            system=self.system_prompt,
            require_final_marker=True,
        )
        status = "complete" if output else "fallback"
        if not output:
            logger.warning(
                "Suppressed %s response: %s (done_reason=%s, response_chars=%s)",
                self.name,
                diagnostics.get("failure_reason", "unknown"),
                diagnostics.get("done_reason", "unknown"),
                diagnostics.get("response_chars", 0),
            )
        return {
            "agent": self.name,
            "title": self.title,
            "status": status,
            "output": output or self.fallback(state, "The model response was suppressed because it lacked a final-answer marker or was truncated."),
        }

    @abstractmethod
    def build_prompt(self, state: AnalyticsState) -> str:
        raise NotImplementedError

    @abstractmethod
    def fallback(self, state: AnalyticsState, reason: str) -> str:
        raise NotImplementedError


def previous_outputs(state: AnalyticsState, max_chars: int = 4200) -> str:
    results = state.get("results", [])
    if not results:
        return "No previous agent output yet."
    entries = []
    for item in results:
        output = str(item.get("output", "")).strip()
        if len(output) > 800:
            output = output[:797].rsplit(" ", 1)[0] + "..."
        entries.append(f"{item.get('title', 'Agent')}:\n{output}")
    combined = "\n\n".join(entries)
    if len(combined) <= max_chars:
        return combined
    return combined[: max_chars - 29].rsplit("\n", 1)[0] + "\n[Earlier details abbreviated.]"
