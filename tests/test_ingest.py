from pathlib import Path

import pytest

from scripts.ingest_data import parse_metadata


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
