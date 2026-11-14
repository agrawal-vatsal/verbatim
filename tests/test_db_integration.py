"""verbatim.db against a real Postgres + pgvector, after applying the real migrations.

The fake connection in the other tests can't catch a wrong column name, a broken
migration or a type mismatch; these can. They run only when TEST_DATABASE_URL points at
a throwaway database (CI starts one; see .github/workflows/build.yml). Never point it
at a database you care about: these tests write to it.
"""

import os
import uuid
from collections.abc import Iterator

import pytest

from scripts.init_db import enable_extension
from scripts.migrate import run_migrations
from verbatim.db import Database

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="set TEST_DATABASE_URL to a throwaway Postgres + pgvector database"
)

DIM = 1536


def one_hot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


@pytest.fixture(scope="module")
def db() -> Iterator[Database]:
    assert TEST_DATABASE_URL is not None
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    try:
        enable_extension()
        run_migrations()
        run_migrations()  # migrate.py re-runs every file, so migrations must be idempotent
        yield Database()
    finally:
        if previous is None:
            del os.environ["DATABASE_URL"]
        else:
            os.environ["DATABASE_URL"] = previous


@pytest.fixture
def company(db: Database) -> str:
    """A company name unique to this test, so tests don't see each other's rows."""
    name = f"Test_{uuid.uuid4().hex[:8]}"
    chunks = [
        {"company": name, "fy": "FY26", "quarter": "Q1", "page_number": 1,
         "content": "Net interest margin was stable at 3.4 percent this quarter."},
        {"company": name, "fy": "FY26", "quarter": "Q1", "page_number": 2,
         "content": "Deposit growth was strong, led by current and savings accounts."},
        {"company": name, "fy": "FY26", "quarter": "Q2", "page_number": 5,
         "content": "Credit costs moderated and gross NPA improved to 1.3 percent."},
    ]
    assert db.insert_transcript_chunks(chunks, [one_hot(0), one_hot(1), one_hot(2)]) == 3
    return name


def test_insert_and_lookup(db: Database, company: str) -> None:
    assert db.has_transcript_data(company, "FY26", "Q1")
    assert db.has_transcript_data(company, "FY26", "Q2")
    assert not db.has_transcript_data(company, "FY25", "Q1")
    assert company in db.get_unique_companies()


def test_search_similar_chunks_ranks_by_cosine_distance(db: Database, company: str) -> None:
    results = db.search_similar_chunks(one_hot(1), limit=3, company=company)
    assert results[0]["content"].startswith("Deposit growth")
    assert results[0]["distance"] == 0.0
    assert results[0]["metadata"] == f"{company} FY26 Q1 (p. 2)"


def test_search_hybrid_rrf_fuses_vector_and_keyword_hits(db: Database, company: str) -> None:
    # The vector points at the deposits chunk; the text matches the NIM chunk.
    rows = db.search_hybrid_rrf(
        query_text="interest margin", query_embedding=one_hot(1), company=company, final_limit=10
    )
    by_page = {r["page_number"]: r for r in rows}

    assert by_page[2]["found_by_vector"]
    assert by_page[1]["found_by_keyword"]
    assert all(r["company"] == company for r in rows)
    assert all(isinstance(r["id"], uuid.UUID) for r in rows)
    scores = [float(r["rrf_score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_search_hybrid_rrf_respects_fy_and_quarter_filters(db: Database, company: str) -> None:
    rows = db.search_hybrid_rrf(
        query_text="credit costs NPA", query_embedding=one_hot(2), company=company, fy="FY26", quarter="Q1"
    )
    assert rows, "expected Q1 chunks from the vector search"
    assert {r["quarter"] for r in rows} == {"Q1"}


def test_search_hybrid_rrf_is_scoped_to_company(db: Database, company: str) -> None:
    other = f"Test_{uuid.uuid4().hex[:8]}"
    rows = db.search_hybrid_rrf(query_text="interest margin", query_embedding=one_hot(0), company=other)
    assert rows == []


def test_log_query_round_trips_into_stats(db: Database, company: str) -> None:
    ids = [str(r["id"]) for r in db.search_hybrid_rrf(
        query_text="margin", query_embedding=one_hot(0), company=company, final_limit=3
    )]
    before = db.get_system_stats()["queries_24h"]

    # Same shape ChatManager._log_interaction sends, including UUID strings for the uuid[] columns.
    db.log_query({
        "question": "What about NIM?", "refined_query": "net interest margin",
        "company_filter": company, "fy_filter": "FY26", "quarter_filter": "Q1",
        "top_distance": 0.0328, "processing_time_ms": 300, "retrieval_time_ms": 80,
        "rerank_time_ms": 400, "synthesis_time_ms": 1800, "response_time_ms": 2580,
        "search_overlap_count": 1, "vector_signal": 1, "keyword_signal": 1,
        "pre_ce_chunk_ids": ids, "post_ce_chunk_ids": ids[:1], "reranker_displacement": 0.25,
    })

    stats = db.get_system_stats()
    assert stats["queries_24h"] == before + 1
    assert stats["total_chunks"] >= 3
    assert (company, 3) in stats["company_dist"]
