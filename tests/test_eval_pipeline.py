import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from evals import pipeline, synthesis
from evals.harness import load_cases
from tests.conftest import run_main
from verbatim.reranker import RerankerResult

H = "HDFC_Bank"
SCOPE: dict[str, Any] = {"company": H, "fy": "FY26", "quarter": "Q2"}


def _chunk(page: int, fy: str = "FY26", quarter: str = "Q2", company: str = H) -> dict[str, Any]:
    return {"id": f"id-{page}", "company": company, "fy": fy, "quarter": quarter, "page_number": page,
            "content": f"content {page}", "rrf_score": 0.01}


def _cite(page: int, fy: str = "FY26", quarter: str = "Q2") -> str:
    return f"[HDFC_Bank, {fy}, {quarter}, Page {page}]"


@dataclass
class FakeManager:
    extracted: dict[str, Any] | None = field(default_factory=lambda: {**SCOPE, "search_query": "nim margin"})
    candidates: list[dict[str, Any]] = field(default_factory=lambda: [_chunk(1), _chunk(2), _chunk(3)])
    answer: str = f"Margins were stable {_cite(3)}."
    latencies: dict[str, int] = field(default_factory=lambda: {"stale": 1})
    calls: list[str] = field(default_factory=list)

    def _extract(self, user_input: str) -> dict[str, Any] | None:
        self.calls.append("extract")
        self.latencies["_extract"] = 5
        return self.extracted

    def _retrieve(self, extracted: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append("retrieve")
        return self.candidates

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> RerankerResult:
        self.calls.append(f"rerank:{query}")
        picked = list(reversed(candidates))[:2]
        return RerankerResult(chunks=picked, pre_ce_ids=[], post_ce_ids=[], displacement_score=0.3)

    def _synthesize(self, user_input: str, chunks: list[dict[str, Any]]) -> str:
        self.calls.append("synthesize")
        return self.answer


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {"id": "p1", "flow": "scoped_fy_quarter", "question": "NIM in Q2 FY26?",
                            "expected_scope": dict(SCOPE), "expect_stop": None, "keywords": ["margin"]}
    case.update(overrides)
    return case


# --- graders --------------------------------------------------------------------------

@pytest.mark.parametrize(("value", "expected"), [(None, None), ("", None), ("null", None), ("None", None), (" FY26 ", "FY26")])
def test_normalize_value(value: Any, expected: str | None) -> None:
    assert pipeline.normalize_value(value) == expected


def test_expected_stops() -> None:
    assert pipeline.expected_stops({}) == {None}
    assert pipeline.expected_stops({"expect_stop": "extract"}) == {"extract"}
    assert pipeline.expected_stops({"expect_stop": ["extract", "retrieve"]}) == {"extract", "retrieve"}
    with pytest.raises(ValueError, match="unknown stage"):
        pipeline.expected_stops({"expect_stop": "synthesize"})


def test_extraction_scope_score() -> None:
    assert pipeline.extraction_scope_score({"company": "hdfc_bank", "fy": "FY26", "quarter": "null"},
                                           {"company": H, "fy": "FY26", "quarter": None}) == 1.0
    assert pipeline.extraction_scope_score({"company": H, "fy": "FY25", "quarter": "Q2"}, SCOPE) == pytest.approx(2 / 3)
    assert pipeline.extraction_scope_score({"company": H}, {}) == 1.0


def test_chunks_in_scope_checks_only_non_null_fields() -> None:
    assert pipeline.chunks_in_scope([_chunk(1), _chunk(2)], SCOPE) == 1.0
    assert pipeline.chunks_in_scope([_chunk(1), _chunk(2, quarter="Q1")], SCOPE) == 0.0
    fy_only = {"company": H, "fy": "FY26", "quarter": None}
    assert pipeline.chunks_in_scope([_chunk(1, quarter="Q1"), _chunk(2)], fy_only) == 1.0
    assert pipeline.chunks_in_scope([_chunk(1, company="ICICI_Bank")], fy_only) == 0.0


def test_citations_in_scope() -> None:
    assert pipeline.citations_in_scope(f"x {_cite(1)} y [HDFC Bank, FY26, Q2, Page 2]", SCOPE) == 1.0
    assert pipeline.citations_in_scope(f"x {_cite(1, quarter='Q1')}", SCOPE) == 0.0
    assert pipeline.citations_in_scope("x [ICICI_Bank, FY26, Q2, Page 1]", SCOPE) == 0.0
    assert pipeline.citations_in_scope("no citations", SCOPE) == 1.0


# --- evaluate_case -----------------------------------------------------------------------

def test_full_run_passes_and_records_details() -> None:
    manager = FakeManager()
    result = pipeline.evaluate_case(_case(), manager)

    assert result.passed, result.scores
    assert manager.calls == ["extract", "retrieve", "rerank:nim margin", "synthesize"]
    assert result.scores["stop_stage_correct"] == 1.0
    assert result.scores["extraction_scope"] == 1.0
    assert result.details["final_sources"] == [[H, "FY26", "Q2", 3], [H, "FY26", "Q2", 2]]
    assert result.details["latency_ms"] == {"_extract": 5}  # reset per case, stale entry dropped
    assert result.details["stopped_at"] is None
    assert result.details["displacement"] == 0.3


def test_keyword_recall_is_soft() -> None:
    result = pipeline.evaluate_case(_case(keywords=["nonexistent"]), FakeManager())
    assert result.passed and result.scores["keyword_recall"] == 0.0


@pytest.mark.parametrize(
    ("manager_kwargs", "failed_score"),
    [
        ({"extracted": {**SCOPE, "fy": "FY25", "search_query": "q"}}, "extraction_scope"),
        ({"candidates": [_chunk(1, quarter="Q1"), _chunk(2), _chunk(3)]}, "scope_respected_candidates"),
        ({"candidates": [_chunk(1), _chunk(2), _chunk(3, quarter="Q1")], "answer": f"x {_cite(2)}"}, "scope_respected_chunks"),
        ({"answer": f"x {_cite(3)} {_cite(9, quarter='Q1')}"}, "scope_respected_citations"),
        ({"answer": f"x {_cite(1)}"}, "citations_valid"),  # page 1 was retrieved but dropped by the reranker
        ({"answer": "Margins were stable."}, "citation_present"),
        ({"answer": "   "}, "answer_nonempty"),
        ({"answer": synthesis.REFUSAL_TEXT}, "refusal_correct"),
        ({"candidates": []}, "stop_stage_correct"),
        ({"extracted": None}, "stop_stage_correct"),
    ],
)
def test_failures(manager_kwargs: dict[str, Any], failed_score: str) -> None:
    result = pipeline.evaluate_case(_case(), FakeManager(**manager_kwargs))
    assert not result.passed
    assert result.scores[failed_score] < 1.0


def test_expected_stop_at_extract() -> None:
    manager = FakeManager(extracted=None)
    result = pipeline.evaluate_case(_case(expect_stop="extract", expected_scope={}), manager)
    assert result.passed and result.details["stopped_at"] == "extract"
    assert manager.calls == ["extract"]
    assert set(result.scores) == {"stop_stage_correct"}


def test_expected_stop_at_retrieve_or_extract() -> None:
    case = _case(expect_stop=["extract", "retrieve"], expected_scope={})
    assert pipeline.evaluate_case(case, FakeManager(candidates=[])).passed
    assert pipeline.evaluate_case(case, FakeManager(extracted=None)).passed
    assert not pipeline.evaluate_case(case, FakeManager()).passed  # ran to completion instead of stopping


def test_not_found_case() -> None:
    case = _case(expect_not_found=True, keywords=[])
    refused = pipeline.evaluate_case(case, FakeManager(answer=synthesis.REFUSAL_TEXT))
    assert refused.passed and "citation_present" not in refused.scores
    answered = pipeline.evaluate_case(case, FakeManager())
    assert not answered.passed and answered.scores["refusal_correct"] == 0.0


# --- with the real ChatManager (fake dependencies) -------------------------------------------

@dataclass
class ChatDeps:
    extracted: dict[str, Any] = field(default_factory=lambda: {**SCOPE, "search_query": "nim margin"})
    logged: list[dict[str, Any]] = field(default_factory=list)
    companies: list[str] = field(default_factory=lambda: [H])

    def get_unique_companies(self) -> list[str]:
        return self.companies

    def search_hybrid_rrf(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [_chunk(1), _chunk(2), _chunk(3)]

    def log_query(self, data: dict[str, Any]) -> None:
        self.logged.append(data)

    def process_query(self, user_input: str) -> dict[str, Any]:
        return self.extracted

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_n: int = 5) -> RerankerResult:
        return RerankerResult(chunks=chunks[:2], pre_ce_ids=[], post_ce_ids=[], displacement_score=0.0)

    def generate_answer(self, question: str, chunks: list[dict[str, Any]]) -> str:
        return f"Stable margins {_cite(chunks[0]['page_number'])}."


@pytest.fixture
def chat_deps(monkeypatch: pytest.MonkeyPatch) -> ChatDeps:
    deps = ChatDeps()
    for name in ("Database", "QueryProcessor", "LLMReranker", "Synthesizer"):
        monkeypatch.setattr(f"scripts.chat.{name}", lambda *_a, _d=deps, **_k: _d)
    monkeypatch.setattr("scripts.chat.get_embeddings", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(pipeline, "Database", lambda: deps)
    return deps


def test_evaluate_case_with_real_chat_manager_never_logs(chat_deps: ChatDeps) -> None:
    from scripts.chat import ChatManager

    result = pipeline.evaluate_case(_case(), ChatManager())

    assert result.passed, result.scores
    assert set(result.details["latency_ms"]) == {"_extract", "_retrieve", "_rerank", "_synthesize"}
    assert chat_deps.logged == []  # evals must not write to query_logs


def test_real_chat_manager_guardrail_stops(chat_deps: ChatDeps) -> None:
    from scripts.chat import ChatManager

    chat_deps.extracted = {"company": "Q3", "quarter": "Q3", "fy": None, "search_query": "x"}
    result = pipeline.evaluate_case(_case(expect_stop="extract", expected_scope={}), ChatManager())
    assert result.passed and result.details["stopped_at"] == "extract"


# --- preflight + main -------------------------------------------------------------------------

def test_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    class Down:
        def get_unique_companies(self) -> list[str]:
            raise ConnectionError("refused")

    monkeypatch.setattr(pipeline, "Database", Down)
    problem = pipeline.preflight()
    assert problem is not None and "not reachable (ConnectionError: refused)" in problem
    assert pipeline.SETUP_HINT in problem

    monkeypatch.setattr(pipeline, "Database", lambda: ChatDeps(companies=[]))
    problem = pipeline.preflight()
    assert problem is not None and "No transcripts are ingested" in problem

    monkeypatch.setattr(pipeline, "Database", lambda: ChatDeps())
    assert pipeline.preflight() is None


@pytest.mark.parametrize("missing", ["DATABASE_URL", "OPENAI_API_KEY", "GEMINI_API_KEY"])
def test_main_requires_env(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    monkeypatch.delenv(missing)
    with pytest.raises(SystemExit, match=missing):
        pipeline.main([])


def test_main_exits_when_db_not_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "Database", lambda: ChatDeps(companies=[]))
    with pytest.raises(SystemExit, match="make db-up && make db-init && make download && make ingest"):
        pipeline.main([])


@pytest.fixture
def eval_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    datasets, results = tmp_path / "datasets", tmp_path / "results"
    datasets.mkdir()
    monkeypatch.setattr(pipeline, "DATASETS_DIR", datasets)
    monkeypatch.setattr(pipeline, "RESULTS_DIR", results)
    return datasets, results


def test_main_runs_cases_and_reports(chat_deps: ChatDeps, eval_dirs: tuple[Path, Path]) -> None:
    datasets, results = eval_dirs
    cases = [_case(id="ok"), _case(id="guard", expect_stop="extract", expected_scope={})]
    (datasets / "pipeline.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))

    assert pipeline.main(["--threshold", "0.5"]) == 0  # "guard" fails (pipeline doesn't stop): 1/2 passes
    assert pipeline.main([]) == 1  # default threshold 0.8

    report = json.loads(sorted(results.glob("*_pipeline.json"))[-1].read_text())
    assert report["summary"]["cases"] == 2 and report["summary"]["passed"] == 1
    assert chat_deps.logged == []


# --- python -m entry points (both eval modules) --------------------------------------------------

def test_module_entry_points(
    chat_deps: ChatDeps, in_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run_main re-imports names from their source modules, so patch those.
    monkeypatch.setattr("verbatim.db.Database", lambda: chat_deps)
    monkeypatch.setattr("verbatim.synthesizer.Synthesizer", lambda: chat_deps)
    datasets = in_tmp_dir / "evals" / "datasets"
    datasets.mkdir(parents=True)
    (datasets / "pipeline.jsonl").write_text(json.dumps(_case(id="ok")) + "\n")
    syn_case = {"id": "s", "question": "q", "expect": "any", "chunks": [_chunk(1)]}
    (datasets / "synthesis.jsonl").write_text(json.dumps(syn_case) + "\n")

    for module in ("evals.pipeline", "evals.synthesis"):
        monkeypatch.setattr("sys.argv", [module])  # argparse would otherwise see pytest's arguments
        with pytest.raises(SystemExit) as exc:
            run_main(module)
        assert exc.value.code == 0, module
    assert len(list((in_tmp_dir / "evals" / "results").glob("*.json"))) == 2


# --- the real dataset ------------------------------------------------------------------------------

def test_pipeline_dataset_is_well_formed() -> None:
    cases = load_cases("pipeline.jsonl")
    assert len(cases) >= 15
    assert {"guardrail", "scoped_fy_quarter", "scoped_fy", "unscoped", "not_found"} <= {c["flow"] for c in cases}

    manifest = json.loads(Path("manifest.json").read_text())["transcripts"]
    ingested = {(t["company"], t["fy"], t["quarter"]) for t in manifest}
    ingested_fys = {(c, fy) for c, fy, _ in ingested}

    for case in cases:
        cid = case["id"]
        assert isinstance(case["question"], str) and case["question"], cid
        stops = pipeline.expected_stops(case)
        scope = case.get("expected_scope", {})
        assert set(scope) <= {"company", "fy", "quarter"}, cid
        if None in stops:
            assert scope.get("company"), f"{cid}: full-run cases must name the expected company"
            company, fy, quarter = scope["company"], scope.get("fy"), scope.get("quarter")
            if fy and quarter:
                assert (company, fy, quarter) in ingested, f"{cid}: {company} {fy} {quarter} not in manifest.json"
            elif fy:
                assert (company, fy) in ingested_fys, cid
        else:
            assert not case.get("expect_not_found"), cid
