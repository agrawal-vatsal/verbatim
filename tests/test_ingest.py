from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import ingest_all, ingest_data
from scripts.ingest_data import parse_metadata
from tests.conftest import run_main


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("HDFC_Bank_FY25_Q3.pdf", {"company": "HDFC_Bank", "fy": "FY25", "quarter": "Q3"}),
        ("Infosys_FY26_Q1.pdf", {"company": "Infosys", "fy": "FY26", "quarter": "Q1"}),
        ("Larsen_and_Toubro_FY24_Q4.pdf", {"company": "Larsen_and_Toubro", "fy": "FY24", "quarter": "Q4"}),
    ],
)
def test_parse_metadata(filename: str, expected: dict[str, str]) -> None:
    assert parse_metadata(Path("data/raw/transcripts") / filename) == expected


# --- Fakes ----------------------------------------------------------------------

@dataclass
class FakePage:
    page_number: int
    text: str | None

    def extract_text(self) -> str | None:
        return self.text


class FakePDF:
    def __init__(self, pages: list[FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "FakePDF":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _fake_pdfplumber(monkeypatch: pytest.MonkeyPatch, pages: list[FakePage]) -> list[Any]:
    opened: list[Any] = []

    def fake_open(path: Any) -> FakePDF:
        opened.append(path)
        return FakePDF(pages)

    monkeypatch.setattr("pdfplumber.open", fake_open)
    return opened


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, input: list[str], model: str) -> SimpleNamespace:
        self.calls.append({"input": input, "model": model})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(i), 0.5]) for i in range(len(input))])


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.embeddings = FakeEmbeddings()
        FakeOpenAI.instances.append(self)


@dataclass
class FakeIngestDB:
    existing: set[tuple[str, str, str]] = field(default_factory=set)
    inserted: list[tuple[list[dict[str, Any]], list[list[float]]]] = field(default_factory=list)

    def has_transcript_data(self, company: str, fy: str, quarter: str) -> bool:
        return (company, fy, quarter) in self.existing

    def insert_transcript_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
        self.inserted.append((chunks, embeddings))
        return len(chunks)


# --- get_chunks_from_pdf ----------------------------------------------------------------

def test_get_chunks_splits_per_page_and_keeps_page_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    long_text = " ".join(f"sentence{i} about net interest margin." for i in range(60))  # > 800 chars
    opened = _fake_pdfplumber(monkeypatch, [
        FakePage(1, "Opening remarks."),
        FakePage(2, None),  # scanned / empty page is skipped
        FakePage(3, long_text),
    ])
    meta = {"company": "HDFC_Bank", "fy": "FY25", "quarter": "Q3"}

    chunks = ingest_data.get_chunks_from_pdf(Path("x.pdf"), meta)

    assert opened == [Path("x.pdf")]
    assert chunks[0] == {**meta, "page_number": 1, "content": "Opening remarks."}
    page3 = [c for c in chunks if c["page_number"] == 3]
    assert len(page3) > 1
    assert all(len(c["content"]) <= 800 for c in page3)
    assert {c["page_number"] for c in chunks} == {1, 3}
    assert all(c["company"] == "HDFC_Bank" and c["fy"] == "FY25" and c["quarter"] == "Q3" for c in chunks)


def test_get_chunks_of_empty_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_pdfplumber(monkeypatch, [FakePage(1, ""), FakePage(2, None)])
    assert ingest_data.get_chunks_from_pdf(Path("x.pdf"), {"company": "c", "fy": "f", "quarter": "q"}) == []


# --- get_embeddings ---------------------------------------------------------------------

def test_get_embeddings_uses_small_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOpenAI()
    monkeypatch.setattr(ingest_data, "client", fake)

    vectors = ingest_data.get_embeddings(["a", "b"])

    assert vectors == [[0.0, 0.5], [1.0, 0.5]]
    assert fake.embeddings.calls == [{"input": ["a", "b"], "model": "text-embedding-3-small"}]


# --- ingest_all ----------------------------------------------------------------------------

@pytest.fixture
def ingest_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeIngestDB:
    db = FakeIngestDB(existing={("Done", "FY25", "Q1")})
    monkeypatch.setattr(ingest_all, "Database", lambda: db)

    def fake_chunks(path: Path, meta: dict[str, str]) -> list[dict[str, Any]]:
        if meta["company"] == "Broken":
            raise RuntimeError("corrupt pdf")
        if meta["company"] == "Empty":
            return []
        return [{**meta, "page_number": 1, "content": "c1"}, {**meta, "page_number": 2, "content": "c2"}]

    monkeypatch.setattr(ingest_all, "get_chunks_from_pdf", fake_chunks)
    monkeypatch.setattr(ingest_all, "get_embeddings", lambda texts: [[0.1] for _ in texts])
    return db


def test_ingest_all_handles_new_existing_empty_and_broken(
    ingest_fakes: FakeIngestDB, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ["Done_FY25_Q1", "New_FY26_Q2", "Empty_FY26_Q1", "Broken_FY24_Q4", "Other_FY26_Q3"]:
        (tmp_path / f"{name}.pdf").touch()
    (tmp_path / "notes.txt").touch()

    ingest_all.ingest_all_transcripts(str(tmp_path))
    out = capsys.readouterr().out

    inserted = sorted(chunks[0]["company"] for chunks, _ in ingest_fakes.inserted)
    assert inserted == ["New", "Other"]
    assert "Found 5 PDFs" in out
    assert "Skipping: Done_FY25_Q1.pdf" in out
    assert "No text extracted from Empty_FY26_Q1.pdf" in out
    assert "Error processing Broken_FY24_Q4.pdf: corrupt pdf" in out
    assert "New files added:  2" in out
    assert "Total new chunks: 4" in out


def test_ingest_all_with_no_pdfs(ingest_fakes: FakeIngestDB, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ingest_all.ingest_all_transcripts(str(tmp_path))
    assert "No PDF files found" in capsys.readouterr().out
    assert ingest_fakes.inserted == []


def test_ingest_all_main_reads_default_dir(
    monkeypatch: pytest.MonkeyPatch, in_tmp_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("verbatim.db.Database", FakeIngestDB)
    run_main("scripts.ingest_all")
    assert "No PDF files found in data/raw/transcripts" in capsys.readouterr().out


# --- ingest_data __main__ smoke run --------------------------------------------------------

@pytest.fixture
def ingest_main_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeIngestDB:
    db = FakeIngestDB()
    FakeOpenAI.instances.clear()
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr("verbatim.db.Database", lambda: db)
    _fake_pdfplumber(monkeypatch, [FakePage(i, f"page {i} text") for i in range(1, 6)])
    return db


def test_ingest_data_main_stores_only_three_chunks(ingest_main_fakes: FakeIngestDB, in_tmp_dir: Path) -> None:
    pdf = in_tmp_dir / "data" / "raw" / "transcripts" / "HDFC_Bank_FY25_Q3.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.touch()

    run_main("scripts.ingest_data")

    (chunks, embeddings), = ingest_main_fakes.inserted
    assert [c["content"] for c in chunks] == ["page 1 text", "page 2 text", "page 3 text"]
    assert len(embeddings) == 3
    assert FakeOpenAI.instances[-1].embeddings.calls[0]["input"] == ["page 1 text", "page 2 text", "page 3 text"]


def test_ingest_data_main_without_sample_file_does_nothing(ingest_main_fakes: FakeIngestDB, in_tmp_dir: Path) -> None:
    run_main("scripts.ingest_data")
    assert ingest_main_fakes.inserted == []
