from pathlib import Path
from typing import Any

import pytest

from scripts import init_db, migrate, stats
from tests.conftest import FakeDB, run_main

# --- check_db (runs at import time) ------------------------------------------------


def test_check_db_prints_server_version(fake_db: FakeDB, capsys: pytest.CaptureFixture[str]) -> None:
    fake_db.fetchone_results.append(("PostgreSQL 16.4",))
    run_main("scripts.check_db")
    out = capsys.readouterr().out
    assert "Connecting to: postgresql://" in out
    assert "PostgreSQL 16.4" in out
    assert fake_db.executed[0][0] == "SELECT version()"


def test_check_db_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(AssertionError, match="DATABASE_URL"):
        run_main("scripts.check_db")


# --- init_db ------------------------------------------------------------------------

def test_init_db_enables_pgvector(fake_db: FakeDB, capsys: pytest.CaptureFixture[str]) -> None:
    run_main("scripts.init_db")
    assert fake_db.executed == [("CREATE EXTENSION IF NOT EXISTS vector;", None)]
    assert "pgvector extension enabled" in capsys.readouterr().out


def test_init_db_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(AssertionError, match="DATABASE_URL"):
        init_db.enable_extension()


# --- migrate ------------------------------------------------------------------------

def test_migrate_applies_files_in_order_in_one_transaction(fake_db: FakeDB, in_tmp_dir: Path) -> None:
    (in_tmp_dir / "migrations").mkdir()
    (in_tmp_dir / "migrations" / "0002_second.sql").write_text("SELECT 2;")
    (in_tmp_dir / "migrations" / "0001_first.sql").write_text("SELECT 1;")
    (in_tmp_dir / "migrations" / "notes.txt").write_text("ignored")

    run_main("scripts.migrate")

    assert [sql for sql, _ in fake_db.executed] == ["SELECT 1;", "SELECT 2;"]
    assert fake_db.commits == 1


def test_repo_migrations_are_all_applied(fake_db: FakeDB) -> None:
    migrate.run_migrations()
    applied = [sql for sql, _ in fake_db.executed]
    expected = [p.read_text() for p in sorted(Path("migrations").glob("*.sql"))]
    assert applied == expected and len(applied) >= 7


def test_migrate_requires_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(AssertionError, match="DATABASE_URL"):
        migrate.run_migrations()


# --- stats --------------------------------------------------------------------------

def _stats(**overrides: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "total_chunks": 1284,
        "company_dist": [("HDFC_Bank", 1284)],
        "queries_24h": 3,
        "latency": {
            "avg": {"p": 320, "r": 85, "rr": 410, "s": 1840, "t": 2655},
            "p95": {"p": 480, "r": 130, "rr": 680, "s": 2500, "t": 3790},
        },
        "distance": {"p50": 0.3142, "p95": 0.456},
        "retrieval_signal": {"overlap": 3.2, "vector_only": 12.4, "keyword_only": 4.4},
        "displacement": {"p50": 0.312, "p95": 0.684},
    }
    s.update(overrides)
    return s


class _StatsDB:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def get_system_stats(self) -> dict[str, Any]:
        return self.data


def test_stats_dashboard_renders_all_sections(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(stats, "Database", lambda: _StatsDB(_stats()))
    stats.display_stats()
    out = capsys.readouterr().out

    assert "320ms" in out and "3790ms" in out
    assert "Median (P50):  0.3142 🟢" in out
    assert "Worst  (P95):  0.4560 🟢" in out
    assert "Both engines:  3.2  (16%)" in out
    assert "Vector only:   12.4  (62%)" in out
    assert "Keyword only:  4.4  (22%)" in out
    assert "Median (P50):  0.3120 🟢" in out
    assert "HDFC_Bank" in out and "1284 chunks" in out


def test_stats_dashboard_warning_states_and_no_signal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = _stats(
        distance={"p50": 0.45, "p95": 0.6},
        retrieval_signal={"overlap": 0.0, "vector_only": 0.0, "keyword_only": 0.0},
        displacement={"p50": 0.05, "p95": 0.1},
    )
    monkeypatch.setattr(stats, "Database", lambda: _StatsDB(data))
    stats.display_stats()
    out = capsys.readouterr().out

    assert "Median (P50):  0.4500 🟡" in out
    assert "Worst  (P95):  0.6000 🔴" in out
    assert "(N/A)" in out
    assert "Median (P50):  0.0500 🟡" in out


def test_stats_main(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr("verbatim.db.Database", lambda: _StatsDB(_stats()))
    run_main("scripts.stats")
    assert "VERBATIM OBSERVABILITY DASHBOARD" in capsys.readouterr().out
