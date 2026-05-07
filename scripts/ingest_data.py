import os

import pdfplumber
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from verbatim.db import Database

load_dotenv()

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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Sends a list of strings to OpenAI and returns a list of 1536-dim vectors.
    """
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    # Extract the vectors from the response object
    return [e.embedding for e in response.data]


# Keep your __main__ test block to verify the new logic
if __name__ == "__main__":
    db = Database()
    # Using your existing sample file
    pdf_path = Path("data/raw/transcripts/HDFC_Bank_FY25_Q3.pdf")

    if pdf_path.exists():
        print(f"🚀 Smoke testing ingestion for {pdf_path.name}...")

        meta = parse_metadata(pdf_path)
        all_chunks = get_chunks_from_pdf(pdf_path, meta)

        # --- THE SMOKE TEST SLICE ---
        # Only take the first 3 chunks to save costs
        test_chunks = all_chunks[:3]
        print(f"📦 Selected {len(test_chunks)} chunks for testing.")

        # 4. Generate Embeddings (only for these 3)
        texts = [c["content"] for c in test_chunks]
        print("🧠 Calling OpenAI for embeddings...")
        embeddings = get_embeddings(texts)

        # 5. Save to DB
        count = db.insert_transcript_chunks(test_chunks, embeddings)
        print(f"🏁 Successfully stored {count} test chunks in the database.")