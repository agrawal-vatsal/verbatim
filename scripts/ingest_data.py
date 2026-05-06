import pdfplumber
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def parse_metadata(file_path: Path):
    """
    Extracts company, fy, and quarter from filename: Company_FYXX_QX.pdf
    """
    # Example: HDFC_Bank_FY25_Q3.pdf -> ['HDFC_Bank', 'FY25', 'Q3']
    parts = file_path.stem.split('_')

    # Handling the case where company name might have underscores
    quarter = parts[-1]
    fy = parts[-2]
    company = "_".join(parts[:-2])

    return {
        "company": company,
        "fy": fy,
        "quarter": quarter
    }


def get_chunks_from_pdf(file_path: Path, metadata: Dict) -> List[Dict]:
    """
    Extracts text from PDF and uses LangChain to create
    semantic chunks while preserving page numbers.
    """
    chunks = []

    # RecursiveCharacterTextSplitter tries to split at natural boundaries
    # (newlines, then periods, then spaces) to keep thoughts together.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            # Split the text of the CURRENT page only
            page_texts = text_splitter.split_text(text)

            for t in page_texts:
                chunks.append(
                    {
                        "company": metadata["company"],
                        "fy": metadata["fy"],
                        "quarter": metadata["quarter"],
                        "page_number": page.page_number,
                        "content": t
                    }
                )

    return chunks


# Keep your __main__ test block to verify the new logic
if __name__ == "__main__":
    test_path = Path("data/raw/transcripts/HDFC_Bank_FY26_Q1.pdf")
    if test_path.exists():
        # Re-use your parse_metadata from earlier
        meta = parse_metadata(test_path)
        sample_chunks = get_chunks_from_pdf(test_path, meta)
        print(f"✅ Extracted {len(sample_chunks)} semantic chunks.")
        pass
