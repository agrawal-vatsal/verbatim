import importlib.util
import os
import runpy
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import dotenv
import psycopg
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

# Never let tests read the developer's real .env (real keys, real DATABASE_URL).
# This runs before any project module is imported, so their `from dotenv import load_dotenv` gets the no-op.
dotenv.load_dotenv = lambda *args, **kwargs: False

# scripts.ingest_data builds an OpenAI client at import time, which fails without a key.
# Tests never make real API calls, so placeholders are enough.
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")
os.environ.setdefault("GEMINI_API_KEY", "test-placeholder")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:1/test")


# --- Fake psycopg -------------------------------------------------------------

@dataclass
class FakeDB:
    """Stands in for psycopg.connect: records SQL and replays queued results."""

    fetchone_results: list[Any] = field(default_factory=list)
    fetchall_results: list[Any] = field(default_factory=list)
    executed: list[tuple[str, Any]] = field(default_factory=list)
    copy_sql: str | None = None
    copied_rows: list[tuple[Any, ...]] = field(default_factory=list)
    row_factories: list[Any] = field(default_factory=list)
    conn_strs: list[str] = field(default_factory=list)
    commits: int = 0

    def connect(self, conn_str: str) -> "FakeConnection":
        self.conn_strs.append(conn_str)
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, db: FakeDB) -> None:
        self.db = db

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def cursor(self, row_factory: Any = None) -> "FakeCursor":
        self.db.row_factories.append(row_factory)
        return FakeCursor(self.db)

    def commit(self) -> None:
        self.db.commits += 1


class FakeCursor:
    def __init__(self, db: FakeDB) -> None:
        self.db = db

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.db.executed.append((sql, params))

    def fetchone(self) -> Any:
        return self.db.fetchone_results.pop(0)

    def fetchall(self) -> Any:
        return self.db.fetchall_results.pop(0)

    def copy(self, sql: str) -> "FakeCopy":
        self.db.copy_sql = sql
        return FakeCopy(self.db)


class FakeCopy:
    def __init__(self, db: FakeDB) -> None:
        self.db = db

    def __enter__(self) -> "FakeCopy":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def write_row(self, row: tuple[Any, ...]) -> None:
        self.db.copied_rows.append(row)


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> FakeDB:
    db = FakeDB()
    monkeypatch.setattr(psycopg, "connect", db.connect)
    monkeypatch.setattr("verbatim.db.register_vector", lambda conn: None)
    return db


# --- Fake chat model ------------------------------------------------------------

class RecordingChatModel(FakeListChatModel):
    """FakeListChatModel that also records the messages it was sent."""

    seen: list[Any] = Field(default_factory=list)

    def _call(self, *args: Any, **kwargs: Any) -> str:
        self.seen.append(args[0])
        return super()._call(*args, **kwargs)


# --- Running scripts as __main__ ---------------------------------------------------

def run_main(module: str) -> dict[str, Any]:
    """Execute a module's file as `python -m module` would, inside the test process.

    Runs the file by path rather than with runpy.run_module: pytest's assertion-rewriting
    import hook claims modules named test_*.py (scripts/test_parser.py, ...), and
    run_module can't load code through it.
    """
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, f"cannot find {module}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return runpy.run_path(spec.origin, run_name="__main__")


@pytest.fixture
def in_tmp_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
