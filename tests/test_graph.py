from dataclasses import dataclass, field
from typing import Any

import pytest

from verbatim.graph import PipelineState, _extracted, build_pipeline_graph, run_graph
from verbatim.reranker import RerankerResult

EXTRACTED: dict[str, Any] = {"company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2", "search_query": "NIM guidance"}
CANDIDATES: list[dict[str, Any]] = [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}]


@dataclass
class Stages:
    """Fake stage functions that record the order and arguments of every call."""

    extracted: dict[str, Any] | None = field(default_factory=lambda: dict(EXTRACTED))
    candidates: list[dict[str, Any]] = field(default_factory=lambda: list(CANDIDATES))
    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def extract(self, question: str) -> dict[str, Any] | None:
        self.calls.append(("extract", (question,)))
        return self.extracted

    def retrieve(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append(("retrieve", (extracted,)))
        return self.candidates

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> RerankerResult:
        self.calls.append(("rerank", (query, candidates)))
        picked = list(reversed(candidates))[:1]
        return RerankerResult(
            chunks=picked, pre_ce_ids=[c["id"] for c in candidates],
            post_ce_ids=[c["id"] for c in picked], displacement_score=0.5,
        )

    def synthesize(self, question: str, chunks: list[dict[str, Any]]) -> str:
        self.calls.append(("synthesize", (question, chunks)))
        return "answer"

    def log(self, *args: Any) -> None:
        self.calls.append(("log", args))

    def run(self, question: str = "What about NIM?") -> PipelineState:
        graph = build_pipeline_graph(self.extract, self.retrieve, self.rerank, self.synthesize, self.log)
        return run_graph(graph, question)

    @property
    def order(self) -> list[str]:
        return [name for name, _ in self.calls]


def test_full_run_calls_every_stage_in_order_with_the_right_inputs() -> None:
    stages = Stages()
    state = stages.run("What about NIM?")

    assert stages.order == ["extract", "retrieve", "rerank", "synthesize", "log"]
    calls = dict(stages.calls)
    assert calls["extract"] == ("What about NIM?",)
    assert calls["retrieve"] == (EXTRACTED,)
    assert calls["rerank"] == ("NIM guidance", CANDIDATES)
    assert calls["synthesize"] == ("What about NIM?", [CANDIDATES[1]])
    assert calls["log"] == ("What about NIM?", EXTRACTED, [CANDIDATES[1]], ["a", "b"], ["b"], 0.5)

    assert state["answer"] == "answer"
    assert state["extracted"] == EXTRACTED
    assert state["candidates"] == CANDIDATES
    assert state["rerank"]["post_ce_ids"] == ["b"]
    assert "stopped_at" not in state


def test_stops_after_extract_when_guardrails_reject() -> None:
    stages = Stages(extracted=None)
    state = stages.run()

    assert stages.order == ["extract"]
    assert state["stopped_at"] == "extract"
    assert state["extracted"] is None
    assert "answer" not in state


def test_stops_after_retrieve_when_nothing_found() -> None:
    stages = Stages(candidates=[])
    state = stages.run()

    assert stages.order == ["extract", "retrieve"]
    assert state["stopped_at"] == "retrieve"
    assert state["candidates"] == []
    assert "answer" not in state


def test_graph_shape() -> None:
    stages = Stages()
    drawn = build_pipeline_graph(stages.extract, stages.retrieve, stages.rerank, stages.synthesize, stages.log).get_graph()

    assert set(drawn.nodes) == {"__start__", "extract", "retrieve", "rerank", "synthesize", "log", "__end__"}
    edges = {(e.source, e.target, e.conditional) for e in drawn.edges}
    assert edges == {
        ("__start__", "extract", False),
        ("extract", "retrieve", True),
        ("extract", "__end__", True),
        ("retrieve", "rerank", True),
        ("retrieve", "__end__", True),
        ("rerank", "synthesize", False),
        ("synthesize", "log", False),
        ("log", "__end__", False),
    }


def test_each_run_starts_from_fresh_state() -> None:
    stages = Stages()
    graph = build_pipeline_graph(stages.extract, stages.retrieve, stages.rerank, stages.synthesize, stages.log)
    run_graph(graph, "first")
    stages.extracted = None

    second = run_graph(graph, "second")

    assert second["stopped_at"] == "extract"
    assert "answer" not in second and "candidates" not in second


def test_extracted_guard() -> None:
    with pytest.raises(AssertionError, match="after extraction stopped"):
        _extracted({"question": "q", "extracted": None})
