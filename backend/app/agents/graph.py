from __future__ import annotations

from collections.abc import AsyncIterator

from app.agents.agent_definitions import AGENTS
from app.agents.state import AgentResult, AnalyticsState
from app.dataset_store import compact_profile_text, dataset_path, profile_csv_file
from app.visualization import generate_chart_artifacts

try:
    from langgraph.graph import END, START, StateGraph
except ImportError:  # Keeps local inspection usable before dependencies are installed.
    END = START = StateGraph = None


async def run_analytics(
    question: str,
    dataset_preview: str | None = None,
    target_column: str | None = None,
    dataset_id: str | None = None,
) -> AnalyticsState:
    state: AnalyticsState = {
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

    if StateGraph is not None:
        graph = build_graph()
        state = await graph.ainvoke(state)
        state["charts"] = await generate_chart_artifacts(state)
        return state

    async for result in stream_analytics_state(state):
        state["results"].append(result)

    state["charts"] = await generate_chart_artifacts(state)
    return state


async def stream_analytics(
    question: str,
    dataset_preview: str | None = None,
    target_column: str | None = None,
    dataset_id: str | None = None,
) -> AsyncIterator[AgentResult]:
    state: AnalyticsState = {
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
        yield result


async def stream_analytics_state(state: AnalyticsState) -> AsyncIterator[AgentResult]:
    for agent in AGENTS:
        yield await agent.run(state)


def build_graph():
    workflow = StateGraph(AnalyticsState)
    previous_node = START

    for agent in AGENTS:
        node_name = agent.name

        async def run_agent(current_state: AnalyticsState, agent_instance=agent) -> AnalyticsState:
            result = await agent_instance.run(current_state)
            return {**current_state, "results": current_state.get("results", []) + [result]}

        workflow.add_node(node_name, run_agent)
        workflow.add_edge(previous_node, node_name)
        previous_node = node_name

    workflow.add_edge(previous_node, END)
    return workflow.compile()
