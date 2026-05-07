import os
from pathlib import Path
from verbatim.db import Database
from scripts.ingest_data import parse_metadata, get_chunks_from_pdf, get_embeddings


def ingest_all_transcripts(directory_path: str):
    """
    Iterates through a folder of transcripts, checks for existing data,
    and ingests new files into the vector database.
    """
    db = Database()
    folder = Path(directory_path)

    # Collect all PDF files from the directory
    pdf_files = sorted(list(folder.glob("*.pdf")))

    if not pdf_files:
        print(f"📁 No PDF files found in {directory_path}")
        return

    print(f"📂 Found {len(pdf_files)} PDFs. Starting ingestion pipeline...")

    total_files_processed = 0
    total_chunks_stored = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        # 1. Extract Metadata from Filename
        meta = parse_metadata(pdf_path)

        # 2. Idempotency Check: Skip if this specific transcript is already in DB
        if db.has_transcript_data(meta["company"], meta["fy"], meta["quarter"]):
            print(f"[{i}/{len(pdf_files)}] ⏭️  Skipping: {pdf_path.name} (Data already exists)")
            continue

        print(f"[{i}/{len(pdf_files)}] 🚀 Processing: {pdf_path.name}...")

        try:
            # 3. Extract Text & Create Semantic Chunks (using LangChain logic)
            chunks = get_chunks_from_pdf(pdf_path, meta)

            if not chunks:
                print(f"   ⚠️ No text extracted from {pdf_path.name}. Moving on.")
                continue

            # 4. Generate Embeddings via OpenAI (Batching the whole file)
            texts = [c["content"] for c in chunks]
            print(f"   🧠 Generating {len(texts)} embeddings...")
            embeddings = get_embeddings(texts)

            # 5. Save to Database using the High-Speed COPY method
            count = db.insert_transcript_chunks(chunks, embeddings)

            total_files_processed += 1
            total_chunks_stored += count
            print(f"   ✅ Stored {count} chunks.")

        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {str(e)}")
            continue

    print("\n" + "=" * 30)
    print("✨ INGESTION SUMMARY")
    print(f"📁 New files added:  {total_files_processed}")
    print(f"📦 Total new chunks: {total_chunks_stored}")
    print("=" * 30)


if __name__ == "__main__":
    # Ensure the path points to your raw transcript data
    TRANSCRIPT_DIR = "data/raw/transcripts"
    ingest_all_transcripts(TRANSCRIPT_DIR)