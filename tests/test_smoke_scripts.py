"""scripts/test_parser.py and scripts/test_retrieval.py are manual smoke scripts, not pytest
tests. Import them as modules so pytest doesn't collect their test_* functions."""

from pathlib import Path
from typing import Any

import pytest

import scripts.test_parser as parser_script
import scripts.test_retrieval as retrieval_script
from tests.conftest import run_main


class _Page:
    def extract_text(self) -> str:
        return "Good evening, everyone."


class _PDF:
    pages = [_Page()]

    def __enter__(self) -> "_PDF":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def test_parser_reports_missing_file(in_tmp_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser_script.test_parse_single_page()
    assert "File not found: data/raw/transcripts/HDFC_Bank_FY25_Q3.pdf" in capsys.readouterr().out


def test_parser_prints_first_page(
    in_tmp_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pdf = in_tmp_dir / "data" / "raw" / "transcripts" / "HDFC_Bank_FY25_Q3.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.touch()
    monkeypatch.setattr("pdfplumber.open", lambda path: _PDF())

    run_main("scripts.test_parser")

    out = capsys.readouterr().out
    assert "Content of HDFC_Bank_FY25_Q3.pdf (Page 1)" in out
    assert "Good evening, everyone." in out


class _RetrievalDB:
    calls: list[tuple[list[float], int]] = []

    def search_similar_chunks(self, query_embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        _RetrievalDB.calls.append((query_embedding, limit))
        return [{"distance": 0.21, "metadata": "HDFC_Bank FY25 Q3 (p. 4)", "content": "x" * 400}]


@pytest.fixture
def retrieval_fakes(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    embedded: list[list[str]] = []

    def fake_embeddings(texts: list[str]) -> list[list[float]]:
        embedded.append(texts)
        return [[0.3]]

    _RetrievalDB.calls = []
    for target in ("verbatim.db.Database", "scripts.test_retrieval.Database"):
        monkeypatch.setattr(target, _RetrievalDB)
    for target in ("scripts.ingest_data.get_embeddings", "scripts.test_retrieval.get_embeddings"):
        monkeypatch.setattr(target, fake_embeddings)
    return embedded


def test_retrieval_prints_top_chunks(retrieval_fakes: list[list[str]], capsys: pytest.CaptureFixture[str]) -> None:
    retrieval_script.ask_verbatim("NIM guidance?")
    out = capsys.readouterr().out

    assert retrieval_fakes == [["NIM guidance?"]]
    assert _RetrievalDB.calls == [([0.3], 3)]
    assert "RESULT 1 | Distance: 0.21" in out
    assert "Source: HDFC_Bank FY25 Q3 (p. 4)" in out
    assert f"Content: {'x' * 300}..." in out


@pytest.mark.parametrize(
    ("argv", "expected_question"),
    [
        (["test_retrieval.py", "What", "about", "deposits?"], "What about deposits?"),
        (["test_retrieval.py"], "What are the key highlights of HDFC Bank's Q3 performance?"),
    ],
)
def test_retrieval_main_uses_argv_or_default(
    retrieval_fakes: list[list[str]], monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_question: str
) -> None:
    monkeypatch.setattr("sys.argv", argv)
    run_main("scripts.test_retrieval")
    assert retrieval_fakes == [[expected_question]]
