import pdfplumber
from pathlib import Path


def test_parse_single_page() -> None:
    pdf_path = Path("data/raw/transcripts/HDFC_Bank_FY25_Q3.pdf")

    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return

    with pdfplumber.open(pdf_path) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()

        print(f"--- Content of {pdf_path.name} (Page 1) ---")
        print(text)
        print("--------------------------------------------")


if __name__ == "__main__":
    test_parse_single_page()
