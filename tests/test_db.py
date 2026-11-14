from typing import Any

import pytest
from psycopg.rows import dict_row

from tests.conftest import FakeDB
from verbatim.db import Database


def test_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        Database()


def test_connects_with_database_url(fake_db: FakeDB, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h:5433/verbatim")
    fake_db.fetchall_results.append([])
    Database().get_unique_companies()
    assert fake_db.conn_strs == ["postgresql://u:p@h:5433/verbatim"]


def test_insert_transcript_chunks_copies_rows_and_commits(fake_db: FakeDB) -> None:
    chunks = [
        {"company": "HDFC_Bank", "fy": "FY25", "quarter": "Q3", "page_number": 1, "content": "a"},
        {"company": "HDFC_Bank", "fy": "FY25", "quarter": "Q3", "page_number": 2, "content": "b"},
    ]
    count = Database().insert_transcript_chunks(chunks, [[0.1, 0.2], [0.3, 0.4]])

    assert count == 2
    assert fake_db.copy_sql is not None and fake_db.copy_sql.startswith("COPY transcript_chunks")
    assert fake_db.copied_rows == [
        ("HDFC_Bank", "FY25", "Q3", 1, "a", "[0.1,0.2]"),
        ("HDFC_Bank", "FY25", "Q3", 2, "b", "[0.3,0.4]"),
    ]
    assert fake_db.commits == 1


@pytest.mark.parametrize(("row", "expected"), [((1,), True), (None, False)])
def test_has_transcript_data(fake_db: FakeDB, row: Any, expected: bool) -> None:
    fake_db.fetchone_results.append(row)
    assert Database().has_transcript_data("HDFC_Bank", "FY25", "Q3") is expected
    assert fake_db.executed[0][1] == ("HDFC_Bank", "FY25", "Q3")


def test_get_unique_companies(fake_db: FakeDB) -> None:
    fake_db.fetchall_results.append([("HDFC_Bank",), ("Infosys",)])
    assert Database().get_unique_companies() == ["HDFC_Bank", "Infosys"]


def test_search_similar_chunks_without_filters(fake_db: FakeDB) -> None:
    fake_db.fetchall_results.append([("text", "HDFC_Bank", "FY25", "Q3", 7, 0.123456)])
    results = Database().search_similar_chunks([0.5, 0.25], limit=3)

    sql, params = fake_db.executed[0]
    assert "AND company" not in sql
    assert params == ("[0.5,0.25]", 3)
    assert results == [{"content": "text", "metadata": "HDFC_Bank FY25 Q3 (p. 7)", "distance": 0.1235}]


def test_search_similar_chunks_with_all_filters(fake_db: FakeDB) -> None:
    fake_db.fetchall_results.append([])
    Database().search_similar_chunks([1.0], limit=5, company="Infosys", fy="FY26", quarter="Q1")

    sql, params = fake_db.executed[0]
    assert "AND company = %s AND fy = %s AND quarter = %s" in sql
    assert params == ("[1.0]", "Infosys", "FY26", "Q1", 5)


def test_log_query_maps_fields_and_defaults(fake_db: FakeDB) -> None:
    Database().log_query({"question": "q", "company_filter": "HDFC_Bank", "reranker_displacement": 0.5})

    sql, params = fake_db.executed[0]
    assert "INSERT INTO query_logs" in sql
    assert len(params) == 17
    assert params[0] == "q"
    assert params[2] == "HDFC_Bank"
    # search_overlap_count, vector_signal, keyword_signal default to 0, not None
    assert params[11:14] == (0, 0, 0)
    assert params[16] == 0.5
    assert fake_db.commits == 1


def test_get_system_stats_with_no_queries_returns_zeros(fake_db: FakeDB) -> None:
    fake_db.fetchone_results += [(10,), (0,) + (None,) * 17]
    fake_db.fetchall_results.append([("HDFC_Bank", 10)])

    stats = Database().get_system_stats()

    assert stats["total_chunks"] == 10
    assert stats["company_dist"] == [("HDFC_Bank", 10)]
    assert stats["queries_24h"] == 0
    assert stats["latency"]["avg"] == {"p": 0, "r": 0, "rr": 0, "s": 0, "t": 0}
    assert stats["distance"] == {"p50": 0.0, "p95": 0.0}
    assert stats["displacement"] == {"p50": 0.0, "p95": 0.0}


def test_get_system_stats_maps_row_columns(fake_db: FakeDB) -> None:
    row = (5, 10, 20, 30, 40, 100, 11, 21, 31, 41, 104, 0.3, 0.45, 1.5, 2.5, 3.5, 0.2, 0.7)
    fake_db.fetchone_results += [(10,), row]
    fake_db.fetchall_results.append([])

    stats = Database().get_system_stats()

    assert stats["queries_24h"] == 5
    assert stats["latency"]["avg"] == {"p": 10, "r": 20, "rr": 30, "s": 40, "t": 100}
    assert stats["latency"]["p95"] == {"p": 11, "r": 21, "rr": 31, "s": 41, "t": 104}
    assert stats["distance"] == {"p50": 0.3, "p95": 0.45}
    assert stats["retrieval_signal"] == {"overlap": 1.5, "vector_only": 2.5, "keyword_only": 3.5}
    assert stats["displacement"] == {"p50": 0.2, "p95": 0.7}


def test_search_hybrid_rrf_builds_or_query_and_params(fake_db: FakeDB) -> None:
    rows = [{"id": "x", "content": "c", "rrf_score": 0.03}]
    fake_db.fetchall_results.append(rows)

    result = Database().search_hybrid_rrf(
        query_text="net interest margin",
        query_embedding=[0.1],
        company="HDFC_Bank",
        fy="FY26",
        quarter=None,
        k_limit=20,
        weight_vec=0.7,
        weight_kw=0.3,
        final_limit=5,
    )

    assert result == rows
    assert fake_db.row_factories == [dict_row]
    _, params = fake_db.executed[0]
    fts = "net OR interest OR margin"
    assert params == (
        [0.1], "HDFC_Bank", "FY26", "FY26", None, None, [0.1], 20,
        fts, "HDFC_Bank", "FY26", "FY26", None, None, fts, fts, 20,
        0.7, 0.3, 5,
    )
