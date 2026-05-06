import pdfplumber
from typing import List, Dict
from pathlib import Path


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
    Extracts text from PDF and returns a list of chunk objects with metadata.
    """
    chunks = []
    chunk_size = 800
    chunk_overlap = 100

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            # Simple sliding window chunking
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunk_text = text[start:end]

                chunks.append(
                    {
                        "company": metadata["company"],
                        "fy": metadata["fy"],
                        "quarter": metadata["quarter"],
                        "page_number": page.page_number,
                        "content": chunk_text
                    }
                )

                # Move window forward by (size - overlap)
                start += (chunk_size - chunk_overlap)

                # If the remaining text is very small, just stop
                if len(text) - start < chunk_overlap:
                    break

    return chunks


# Test it out
if __name__ == "__main__":
    test_path = Path("data/raw/transcripts/HDFC_Bank_FY25_Q4.pdf")
    if test_path.exists():
        meta = parse_metadata(test_path)
        sample_chunks = get_chunks_from_pdf(test_path, meta)
        print(f"✅ Extracted {len(sample_chunks)} chunks from {test_path.name}")
        print(f"First chunk sample: {sample_chunks[0]['content'][:100]}...")