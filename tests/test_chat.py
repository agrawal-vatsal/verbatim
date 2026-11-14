from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from scripts import chat
from tests.conftest import run_main
from verbatim.reranker import RerankerResult

CANDIDATES: list[dict[str, Any]] = [
    {"id": "a", "content": "both", "rrf_score": 0.033, "found_by_vector": True, "found_by_keyword": True},
    {"id": "b", "content": "vec", "rrf_score": 0.020, "found_by_vector": True, "found_by_keyword": False},
    {"id": "c", "content": "kw", "rrf_score": 0.010, "found_by_vector": False, "found_by_keyword": True},
]
EXTRACTED: dict[str, Any] = {"company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2", "search_query": "NIM guidance"}


@dataclass
class Fakes:
    extracted: dict[str, Any] = field(default_factory=lambda: dict(EXTRACTED))
    candidates: list[dict[str, Any]] = field(default_factory=lambda: list(CANDIDATES))
    log_error: Exception | None = None
    search_calls: list[dict[str, Any]] = field(default_factory=list)
    logged: list[dict[str, Any]] = field(default_factory=list)
    synth_calls: list[tuple[str, list[dict[str, Any]]]] = field(default_factory=list)
    embedded: list[list[str]] = field(default_factory=list)

    # Database
    def get_unique_companies(self) -> list[str]:
        return ["HDFC_Bank", "Infosys"]

    def search_hybrid_rrf(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.search_calls.append(kwargs)
        return self.candidates

    def log_query(self, data: dict[str, Any]) -> None:
        if self.log_error:
            raise self.log_error
        self.logged.append(data)

    # QueryProcessor
    def process_query(self, user_input: str) -> dict[str, Any]:
        return self.extracted

    # LLMReranker: reverse order, keep 2
    def rerank(self, query: str, chunks: list[dict[str, Any]], top_n: int = 5) -> RerankerResult:
        picked = list(reversed(chunks))[:2]
        return RerankerResult(
            chunks=picked,
            pre_ce_ids=[c["id"] for c in chunks],
            post_ce_ids=[c["id"] for c in picked],
            displacement_score=0.42,
        )

    # Synthesizer
    def generate_answer(self, question: str, chunks: list[dict[str, Any]]) -> str:
        self.synth_calls.append((question, chunks))
        return "the answer"

    # get_embeddings
    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.embedded.append(texts)
        return [[0.1, 0.2]]


def _patch(monkeypatch: pytest.MonkeyPatch, fakes: Fakes, prefix: dict[str, str]) -> None:
    for name, target in prefix.items():
        if name == "get_embeddings":
            monkeypatch.setattr(target, fakes.get_embeddings)
        else:
            monkeypatch.setattr(target, lambda *_a, **_k: fakes)


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    f = Fakes()
    _patch(monkeypatch, f, {
        "Database": "scripts.chat.Database",
        "QueryProcessor": "scripts.chat.QueryProcessor",
        "LLMReranker": "scripts.chat.LLMReranker",
        "Synthesizer": "scripts.chat.Synthesizer",
        "get_embeddings": "scripts.chat.get_embeddings",
    })
    return f


# --- track_latency ------------------------------------------------------------

def test_track_latency_records_ms_and_returns_result() -> None:
    class Thing:
        @chat.track_latency
        def work(self, x: int) -> int:
            return x * 2

    thing = Thing()
    assert thing.work(21) == 42
    latencies: dict[str, int] = thing.latencies  # type: ignore[attr-defined]
    assert set(latencies) == {"work"} and latencies["work"] >= 0


# --- _extract guardrails ----------------------------------------------------------

@pytest.mark.parametrize("company", [None, "", "null"])
def test_extract_rejects_missing_company(fakes: Fakes, company: Any, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.extracted = {"company": company, "search_query": "x"}
    assert chat.ChatManager()._extract("q") is None
    assert "Specify a company. Options: HDFC_Bank, Infosys" in capsys.readouterr().out


def test_extract_rejects_quarter_without_fy(fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.extracted = {"company": "HDFC_Bank", "quarter": "Q3", "fy": None, "search_query": "x"}
    assert chat.ChatManager()._extract("q") is None
    assert "Financial Year" in capsys.readouterr().out


def test_extract_passes_valid_query_and_records_latency(fakes: Fakes) -> None:
    manager = chat.ChatManager()
    assert manager._extract("q") == EXTRACTED
    assert "_extract" in manager.latencies


# --- _retrieve ------------------------------------------------------------------

def test_retrieve_embeds_search_query_and_counts_signals(fakes: Fakes) -> None:
    manager = chat.ChatManager()
    result = manager._retrieve(EXTRACTED)

    assert result == CANDIDATES
    assert fakes.embedded == [["NIM guidance"]]
    call = fakes.search_calls[0]
    assert call["query_text"] == "NIM guidance"
    assert call["query_embedding"] == [0.1, 0.2]
    assert (call["company"], call["fy"], call["quarter"]) == ("HDFC_Bank", "FY26", "Q2")
    assert call["final_limit"] == chat.ChatManager.CANDIDATE_POOL
    assert manager.search_stats == {"overlap": 1, "vector_only": 1, "keyword_only": 1}


def test_retrieve_with_no_candidates_leaves_stats_empty(fakes: Fakes) -> None:
    fakes.candidates = []
    manager = chat.ChatManager()
    assert manager._retrieve(EXTRACTED) == []
    assert manager.search_stats == {}


# --- run_pipeline -----------------------------------------------------------------

def test_run_pipeline_happy_path(fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    chat.ChatManager().run_pipeline("What about NIM?")

    question, chunks = fakes.synth_calls[0]
    assert question == "What about NIM?"
    assert [c["id"] for c in chunks] == ["c", "b"]

    log = fakes.logged[0]
    assert log["question"] == "What about NIM?"
    assert log["refined_query"] == "NIM guidance"
    assert (log["company_filter"], log["fy_filter"], log["quarter_filter"]) == ("HDFC_Bank", "FY26", "Q2")
    assert log["top_distance"] == 0.010  # rrf_score of the top reranked chunk
    assert log["pre_ce_chunk_ids"] == ["a", "b", "c"]
    assert log["post_ce_chunk_ids"] == ["c", "b"]
    assert log["reranker_displacement"] == 0.42
    assert (log["search_overlap_count"], log["vector_signal"], log["keyword_signal"]) == (1, 1, 1)
    stages = ["processing_time_ms", "retrieval_time_ms", "rerank_time_ms", "synthesis_time_ms"]
    assert log["response_time_ms"] == sum(log[s] for s in stages)

    out = capsys.readouterr().out
    assert "VERBATIM RESPONSE" in out
    assert "Context: HDFC_Bank | FY26" in out
    assert "the answer" in out


def test_run_pipeline_stops_when_extraction_fails(fakes: Fakes) -> None:
    fakes.extracted = {"company": None}
    chat.ChatManager().run_pipeline("q")
    assert fakes.search_calls == [] and fakes.synth_calls == [] and fakes.logged == []


def test_run_pipeline_stops_when_nothing_retrieved(fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.candidates = []
    chat.ChatManager().run_pipeline("q")
    assert fakes.synth_calls == [] and fakes.logged == []
    assert "No relevant data found" in capsys.readouterr().out


def test_run_pipeline_survives_telemetry_failure(fakes: Fakes, capsys: pytest.CaptureFixture[str]) -> None:
    fakes.log_error = RuntimeError("db down")
    chat.ChatManager().run_pipeline("q")
    out = capsys.readouterr().out
    assert "Telemetry logging failed: db down" in out
    assert "the answer" in out


def test_log_interaction_with_no_chunks_logs_null_top_score(fakes: Fakes) -> None:
    chat.ChatManager()._log_interaction("q", EXTRACTED, [], [], [], 0.0)
    assert fakes.logged[0]["top_distance"] is None


# --- interactive loop (python -m scripts.chat) ------------------------------------------

def _scripted_input(answers: list[Any]) -> Callable[[str], str]:
    it: Iterator[Any] = iter(answers)

    def fake_input(_prompt: str) -> str:
        value = next(it)
        if isinstance(value, BaseException):
            raise value
        return str(value)

    return fake_input


@pytest.fixture
def main_fakes(monkeypatch: pytest.MonkeyPatch) -> Fakes:
    # run_main re-imports names from their source modules, so patch those.
    f = Fakes()
    _patch(monkeypatch, f, {
        "Database": "verbatim.db.Database",
        "QueryProcessor": "verbatim.processor.QueryProcessor",
        "LLMReranker": "verbatim.reranker.LLMReranker",
        "Synthesizer": "verbatim.synthesizer.Synthesizer",
        "get_embeddings": "scripts.ingest_data.get_embeddings",
    })
    return f


@pytest.mark.parametrize("terminator", ["exit", "QUIT", KeyboardInterrupt(), EOFError()])
def test_main_loop_answers_until_terminated(
    main_fakes: Fakes, monkeypatch: pytest.MonkeyPatch, terminator: Any
) -> None:
    monkeypatch.setattr("builtins.input", _scripted_input(["", "   ", "What about NIM?", terminator]))
    run_main("scripts.chat")
    assert [q for q, _ in main_fakes.synth_calls] == ["What about NIM?"]
