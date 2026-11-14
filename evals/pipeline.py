"""End-to-end eval of the query pipeline against the real database.

Drives ChatManager's stages (_extract, _retrieve, _rerank, _synthesize) for each
case and checks guardrails, scope filtering and citations. It deliberately skips
_log_interaction / run_pipeline so evals never write to query_logs.

Needs: make db-up && make db-init && make download && make ingest
Run:   make eval-pipeline   (paid: OpenAI + Gemini calls per case)
"""

import argparse
from typing import Any, Protocol

from evals import harness
from evals.harness import CaseResult, finish, load_cases, require_env, run_cases
from evals.synthesis import (
    citation_key,
    citations_valid,
    is_refusal,
    keyword_recall,
    normalize_company,
    parse_citations,
)
from verbatim.db import Database
from verbatim.reranker import RerankerResult

NAME = "pipeline"
DEFAULT_THRESHOLD = 0.8
DATASETS_DIR = harness.DATASETS_DIR
RESULTS_DIR = harness.RESULTS_DIR

SETUP_HINT = "Run: make db-up && make db-init && make download && make ingest"
STAGES = ("extract", "retrieve")  # stages that can stop the pipeline early


class ManagerLike(Protocol):
    latencies: dict[str, int]

    def _extract(self, user_input: str) -> dict[str, Any] | None: ...

    def _retrieve(self, extracted: dict[str, Any]) -> list[dict[str, Any]]: ...

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> RerankerResult: ...

    def _synthesize(self, user_input: str, chunks: list[dict[str, Any]]) -> str: ...


# --- Graders -----------------------------------------------------------------------

def normalize_value(value: Any) -> str | None:
    """Treat None, "", "null" and "None" (all of which the extractor may emit) as missing."""
    if value is None or str(value).strip().lower() in ("", "null", "none"):
        return None
    return str(value).strip()


def expected_stops(case: dict[str, Any]) -> set[str | None]:
    """expect_stop may be null (full run), a stage name, or a list of acceptable outcomes."""
    raw = case.get("expect_stop")
    values = raw if isinstance(raw, list) else [raw]
    for v in values:
        if v is not None and v not in STAGES:
            raise ValueError(f"unknown stage {v!r}")
    return set(values)


def extraction_scope_score(extracted: dict[str, Any], expected: dict[str, Any]) -> float:
    """Fraction of expected company/fy/quarter fields the extractor got right (case-insensitive)."""
    if not expected:
        return 1.0
    hits = 0
    for key, want in expected.items():
        got, want_n = normalize_value(extracted.get(key)), normalize_value(want)
        hits += (got.lower() if got else None) == (want_n.lower() if want_n else None)
    return hits / len(expected)


def _scope_filters(expected: dict[str, Any]) -> dict[str, str]:
    return {k: v for k, v in ((k, normalize_value(v)) for k, v in expected.items()) if v is not None}


def chunks_in_scope(chunks: list[dict[str, Any]], expected: dict[str, Any]) -> float:
    """1.0 if every chunk matches the non-null company/fy/quarter in the expected scope."""
    filters = _scope_filters(expected)
    ok = all(str(c.get(k, "")).lower() == v.lower() for c in chunks for k, v in filters.items())
    return 1.0 if ok else 0.0


def citations_in_scope(answer: str, expected: dict[str, Any]) -> float:
    filters = _scope_filters(expected)
    citations, _ = parse_citations(answer)
    for c in citations:
        company, fy, quarter, _page = citation_key(c)
        actual = {"company": company, "fy": fy, "quarter": quarter}
        for key, want in filters.items():
            want_n = normalize_company(want) if key == "company" else want.upper()
            if actual[key] != want_n:
                return 0.0
    return 1.0


# --- Case evaluation ---------------------------------------------------------------

def evaluate_case(case: dict[str, Any], manager: ManagerLike) -> CaseResult:
    question: str = case["question"]
    expected_scope: dict[str, Any] = case.get("expected_scope", {})
    stops = expected_stops(case)
    manager.latencies = {}

    scores: dict[str, float] = {}
    details: dict[str, Any] = {"flow": case.get("flow")}
    stopped: str | None = None
    answer = ""

    extracted = manager._extract(question)
    details["extracted"] = extracted
    if extracted is None:
        stopped = "extract"
    else:
        if expected_scope:
            scores["extraction_scope"] = extraction_scope_score(extracted, expected_scope)
        candidates = manager._retrieve(extracted)
        details["candidate_count"] = len(candidates)
        if not candidates:
            stopped = "retrieve"
        else:
            reranked = manager._rerank(str(extracted["search_query"]), candidates)
            chunks = reranked["chunks"]
            answer = manager._synthesize(question, chunks)
            citations, malformed = parse_citations(answer)
            details.update({
                "final_sources": [[c["company"], c["fy"], c["quarter"], c["page_number"]] for c in chunks],
                "displacement": reranked["displacement_score"],
                "answer": answer,
                "citations": [list(c) for c in citations],
                "malformed_citations": malformed,
            })
            scores["scope_respected_candidates"] = chunks_in_scope(candidates, expected_scope)
            scores["scope_respected_chunks"] = chunks_in_scope(chunks, expected_scope)
            scores["scope_respected_citations"] = citations_in_scope(answer, expected_scope)
            scores["citations_valid"] = citations_valid(citations, malformed, chunks)
            scores["answer_nonempty"] = 1.0 if answer.strip() else 0.0
            if case.get("expect_not_found"):
                scores["refusal_correct"] = 1.0 if is_refusal(answer) else 0.0
            else:
                scores["refusal_correct"] = 0.0 if is_refusal(answer) else 1.0
                scores["citation_present"] = 1.0 if citations else 0.0
            if case.get("keywords"):
                scores["keyword_recall"] = keyword_recall(answer, case["keywords"])  # soft: reported only

    scores["stop_stage_correct"] = 1.0 if stopped in stops else 0.0
    details["stopped_at"] = stopped
    details["latency_ms"] = dict(manager.latencies)

    passed = all(v == 1.0 for k, v in scores.items() if k != "keyword_recall")
    return CaseResult(case_id=case["id"], passed=passed, scores=scores, details=details)


# --- Pre-flight + entry point ---------------------------------------------------------

def preflight() -> str | None:
    """Return a human-readable problem with the DB, or None when it's ready."""
    try:
        companies = Database().get_unique_companies()
    except Exception as e:  # noqa: BLE001 - any failure here means "not set up"
        return f"Database is not reachable ({type(e).__name__}: {e}). {SETUP_HINT}"
    if not companies:
        return f"No transcripts are ingested (transcript_chunks is empty). {SETUP_HINT}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline eval against the real DB (paid).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum pass rate")
    args = parser.parse_args(argv)

    require_env("DATABASE_URL", "OPENAI_API_KEY", "GEMINI_API_KEY")
    problem = preflight()
    if problem:
        raise SystemExit(problem)

    # Imported here: scripts.chat builds an OpenAI client at import time, which needs the key checked above.
    from scripts.chat import ChatManager

    manager = ChatManager()
    cases = load_cases("pipeline.jsonl", DATASETS_DIR)
    print(f"Running {len(cases)} pipeline cases")
    results = run_cases(cases, lambda case: evaluate_case(case, manager))
    return finish(NAME, results, args.threshold, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
