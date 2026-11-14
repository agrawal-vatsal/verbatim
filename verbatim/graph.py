"""The query pipeline as a LangGraph StateGraph.

    START -> extract --(no valid company / quarter without FY)--> END
                  \\-> retrieve --(no candidates)--> END
                             \\-> rerank -> synthesize -> log -> END

The stage functions are passed in (ChatManager passes its bound methods), so this module
only owns the flow: which stage runs next and where the pipeline stops early. Presentation
(printing the answer) stays with the caller.
"""

import warnings
from collections.abc import Callable
from typing import Any, Literal, TypedDict, cast

# Imported first on purpose: on import, langchain_core installs filters that force its own
# deprecation warnings to show. Our narrower ignore below must be added after those to win.
import langchain_core  # noqa: F401

from verbatim.reranker import RerankerResult

with warnings.catch_warnings():
    # langgraph's checkpoint serializer triggers a LangChain pending-deprecation warning on
    # import. We don't use checkpointing, and it would print on every CLI start.
    warnings.filterwarnings("ignore", message=".*allowed_objects.*")
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.state import CompiledStateGraph

StopStage = Literal["extract", "retrieve"]


class PipelineState(TypedDict, total=False):
    question: str
    extracted: dict[str, Any] | None
    candidates: list[dict[str, Any]]
    rerank: RerankerResult
    answer: str
    # Set when the pipeline ends early; absent on a full run.
    stopped_at: StopStage


ExtractFn = Callable[[str], dict[str, Any] | None]
RetrieveFn = Callable[[dict[str, Any]], list[dict[str, Any]]]
RerankFn = Callable[[str, list[dict[str, Any]]], RerankerResult]
SynthesizeFn = Callable[[str, list[dict[str, Any]]], str]
LogFn = Callable[[str, dict[str, Any], list[dict[str, Any]], list[str], list[str], float], None]
PipelineGraph = CompiledStateGraph[PipelineState, None, PipelineState, PipelineState]


def build_pipeline_graph(
    extract: ExtractFn,
    retrieve: RetrieveFn,
    rerank: RerankFn,
    synthesize: SynthesizeFn,
    log: LogFn,
) -> PipelineGraph:
    def extract_node(state: PipelineState) -> PipelineState:
        extracted = extract(state["question"])
        if extracted is None:
            return {"extracted": None, "stopped_at": "extract"}
        return {"extracted": extracted}

    def retrieve_node(state: PipelineState) -> PipelineState:
        candidates = retrieve(_extracted(state))
        if not candidates:
            return {"candidates": [], "stopped_at": "retrieve"}
        return {"candidates": candidates}

    def rerank_node(state: PipelineState) -> PipelineState:
        return {"rerank": rerank(_extracted(state)["search_query"], state["candidates"])}

    def synthesize_node(state: PipelineState) -> PipelineState:
        return {"answer": synthesize(state["question"], state["rerank"]["chunks"])}

    def log_node(state: PipelineState) -> PipelineState:
        result = state["rerank"]
        log(
            state["question"], _extracted(state), result["chunks"],
            result["pre_ce_ids"], result["post_ce_ids"], result["displacement_score"],
        )
        return {}

    def continue_unless_stopped(next_node: str) -> Callable[[PipelineState], str]:
        return lambda state: END if "stopped_at" in state else next_node

    graph = StateGraph(PipelineState)
    graph.add_node("extract", extract_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("log", log_node)

    graph.add_edge(START, "extract")
    graph.add_conditional_edges("extract", continue_unless_stopped("retrieve"), ["retrieve", END])
    graph.add_conditional_edges("retrieve", continue_unless_stopped("rerank"), ["rerank", END])
    graph.add_edge("rerank", "synthesize")
    graph.add_edge("synthesize", "log")
    graph.add_edge("log", END)
    return graph.compile()


def run_graph(graph: PipelineGraph, question: str) -> PipelineState:
    """Run the pipeline for one question. Every call starts from a fresh state."""
    # invoke() is typed as returning dict[str, Any]; the graph's output schema is PipelineState.
    return cast(PipelineState, graph.invoke({"question": question}))


def _extracted(state: PipelineState) -> dict[str, Any]:
    extracted = state.get("extracted")
    assert extracted is not None, "stage ran after extraction stopped the pipeline"
    return extracted
