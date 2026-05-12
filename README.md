# Verbatim

A RAG (Retrieval-Augmented Generation) system for querying financial earnings call transcripts. Ask natural language questions about a company's quarterly results and get precise, cited answers grounded strictly in the source documents.

```
💬 Question: What did HDFC Bank say about their NIM guidance for FY26?

=============== VERBATIM RESPONSE ===============
Context: HDFC_Bank | FY26

HDFC Bank management indicated that they expect the NIM to stabilise and
gradually improve over the coming quarters as the CD ratio normalises...
[HDFC_Bank, FY26, Q2, Page 14]
==================================================
```

## How it works

Each question goes through a four-stage pipeline:

```
User Question
      │
      ▼
┌─────────────────┐
│  1. Extraction  │  GPT-4o-mini parses the question into structured filters
│                 │  (company, FY, quarter) and a retrieval-optimised query
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Retrieval   │  Hybrid search — 70% vector (pgvector cosine) +
│                 │  30% keyword (PostgreSQL FTS) fused via Weighted RRF
└────────┬────────┘
         │  20 candidate chunks
         ▼
┌─────────────────┐
│  3. Reranking   │  Gemini 2.5 Flash (RankGPT pattern) selects the
│                 │  top-5 most relevant chunks from the 20 candidates
└────────┬────────┘
         │  5 precise chunks
         ▼
┌─────────────────┐
│  4. Synthesis   │  GPT-4o generates a cited answer using only the
│                 │  retrieved chunks — no hallucination, strict citations
└─────────────────┘
```

Every query is logged to the database with split-stage latencies, retrieval signal breakdown (vector-only / keyword-only / overlap), and a **reranker displacement score** that measures how much the LLM reranker promoted lower-ranked chunks.

## Tech stack

| Component | Technology |
|---|---|
| Database | PostgreSQL 16 + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Query extraction | OpenAI GPT-4o-mini via LangChain |
| Reranking | Google Gemini 2.5 Flash Lite |
| Synthesis | OpenAI GPT-4o |
| PDF parsing | pdfplumber + LangChain RecursiveCharacterTextSplitter |
| Package manager | uv |
| Type checking | mypy (strict) |

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Docker](https://docs.docker.com/get-docker/) (for PostgreSQL)
- OpenAI API key
- Google Gemini API key

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/agrawal-vatsal/verbatim.git
cd verbatim
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
DATABASE_URL=postgresql://user:user@localhost:5433/verbatim
OPENAI_API_KEY=sk-proj-...
GEMINI_API_KEY=AIza...
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

This starts a `pgvector/pgvector:pg16` container on port **5433** (to avoid conflicts with any local PostgreSQL on 5432). Data is persisted in a named Docker volume.

### 4. Initialise the database

Enable the pgvector extension:

```bash
uv run python -m scripts.init_db
```

Apply all schema migrations:

```bash
uv run python -m scripts.migrate
```

### 5. Load transcript data

Download the PDFs listed in `manifest.json`:

```bash
uv run python -m scripts.download_data
```

Ingest all downloaded PDFs into the vector database:

```bash
uv run python -m scripts.ingest_all
```

Ingestion is idempotent — re-running it skips any transcripts already in the database.

### 6. Start chatting

```bash
uv run python -m scripts.chat
```

Type your question at the prompt. Type `exit` or `quit` to stop.

## Adding your own transcripts

1. Open `manifest.json` and add an entry:

```json
{
  "company": "Infosys",
  "fy": "FY26",
  "quarter": "Q1",
  "url": "https://..."
}
```

2. Re-run the download and ingest steps:

```bash
uv run python -m scripts.download_data
uv run python -m scripts.ingest_all
```

If you already have PDFs locally, place them in `data/raw/transcripts/` following the naming convention `Company_FYXX_QX.pdf` (e.g. `Infosys_FY26_Q1.pdf`) and run only the ingest step.

## Observability dashboard

View system statistics including per-stage latencies, retrieval signal breakdown, reranker effectiveness, and content coverage:

```bash
uv run python -m scripts.stats
```

```
════════════════════════════════════════════════════════════
              📊 VERBATIM OBSERVABILITY DASHBOARD
════════════════════════════════════════════════════════════

⏳ LATENCY PIPELINE (24H)
   Phase           |      Average |   P95 (Tail)
   ---------------------------------------------
   🧠 Processing   |        320ms |        480ms
   🔍 Retrieval    |         85ms |        130ms
   🔀 Reranking    |        410ms |        680ms
   ✍️  Synthesis    |       1840ms |       2500ms
   ---------------------------------------------
   🚀 TOTAL E2E    |       2655ms |       3790ms

🎯 RAG QUALITY (DISTANCE)
   • Median (P50):  0.3142 🟢
   • Worst  (P95):  0.4560 🟢 (Lower is better)

🔀 RETRIEVAL SIGNAL (AVG CHUNKS PER QUERY)
   • Both engines:  3.2  (16%)
   • Vector only:   12.4  (62%)
   • Keyword only:  4.4  (22%)

🔁 RERANKER DISPLACEMENT (0=no change, 1=full reorder)
   • Median (P50):  0.3120 🟢
   • Tail   (P95):  0.6840

🏢 CONTENT COVERAGE
   • HDFC_Bank              1284 chunks
```

**Reading the dashboard:**

- **Reranking row**: shows what the Gemini reranker call costs in latency; if P95 is high relative to synthesis it may not be worth the trade-off.
- **Retrieval signal**: shows whether hybrid search is balanced. High vector-only % means keyword search isn't contributing much. High overlap % means both engines agree, which is a strong relevance signal.
- **Reranker displacement**: measures how much the LLM reranker actually reorders chunks. P50 near 0 means it rubber-stamps the RRF ranking (wasted API call); P50 above ~0.15 means it's genuinely promoting different chunks.

## Project structure

```
verbatim/
├── verbatim/               # Core library
│   ├── db.py               # Database layer (PostgreSQL + pgvector)
│   ├── processor.py        # Query extraction (GPT-4o-mini via LangChain)
│   ├── reranker.py         # LLM reranker (Gemini RankGPT) + displacement metric
│   └── synthesizer.py      # Answer synthesis (GPT-4o)
│
├── scripts/
│   ├── chat.py             # Main CLI — ChatManager orchestrator
│   ├── download_data.py    # Downloads PDFs from manifest.json
│   ├── ingest_data.py      # PDF parsing and embedding generation
│   ├── ingest_all.py       # Batch ingestion for a directory of PDFs
│   ├── init_db.py          # Enables the pgvector extension
│   ├── migrate.py          # Applies SQL migrations in order
│   ├── stats.py            # Observability dashboard
│   └── check_db.py         # Quick DB connectivity check
│
├── migrations/             # SQL schema migrations (applied in numbered order)
├── prompts/                # Prompt templates
│   ├── extraction.txt      # Metadata extraction + query optimisation
│   ├── reranking.txt       # RankGPT reranking prompt
│   └── synthesis.txt       # Cited answer generation
│
├── data/raw/transcripts/   # Downloaded PDF files (gitignored)
├── manifest.json           # List of transcripts to download
├── docker-compose.yaml     # PostgreSQL + pgvector service
├── pyproject.toml
└── .env.example
```

## Querying tips

- **Always specify a company.** The system requires a company name to scope the search. If you omit it, it will list available options.
- **FY and quarter are optional filters.** Omitting them searches across all available periods for that company.
- **Specifying a quarter requires a FY.** `Q3` alone is ambiguous — provide `FY25 Q3`.
- **Ask about what was said, not facts.** This system is grounded in earnings call transcripts, so questions like "What did management say about X?" or "What guidance was given for Y?" work best.

## Development

Type-check the codebase:

```bash
uv run mypy .
```

Verify database connectivity:

```bash
uv run python -m scripts.check_db
```
