---
name: verbatim-artifacts
description: Create shareable artifacts about the Verbatim RAG system - architecture and pipeline diagrams, the database schema, eval reports and observability dashboards - grounded in this repo's code and data, then publish them as claude.ai Artifacts. Use when the user asks to diagram, visualize, map, explain or report on Verbatim (its architecture, LangGraph pipeline, ingestion, schema, a query's path, eval results or stats), or to turn any of those into a page, artifact or something to share.
---

# Verbatim artifacts

Turn this repository into artifacts people can open and share. Every fact in an artifact comes from the code, the migrations, `evals/results/` or the database; never invent components, numbers or results.

## 1. Pick the artifact

| Request | Artifact | How to build it |
|---|---|---|
| System or component map | Architecture diagram | Archify `architecture` (section 2) |
| How one question flows through the pipeline | Pipeline diagram | Archify `workflow` from `verbatim/graph.py`; `sequence` for one request across components |
| Ingestion (PDF to vectors) | Data-flow diagram | Archify `dataflow` from `scripts/ingest_data.py` and `scripts/ingest_all.py` |
| Database tables | ERD | Archify `erd` from `migrations/*.sql` (apply them in order; later files alter earlier tables) |
| Eval results | Report page | Hand-built page (section 3) |
| Latency, retrieval-signal and reranker stats | Dashboard page | Hand-built page (section 3) |
| A question answered as a page | Explainer | Archify `explainer` |

If the request fits none of these, build a page as in section 3.

## 2. Diagrams: use Archify when it's installed

Check whether Archify is available: the `archify` skill appears in your skill list, or `~/.claude/skills/archify/bin/archify.mjs` exists and `node` is on the PATH.

**Archify available.** Invoke the `archify` skill and follow its instructions exactly. It owns authoring, validation and delivery, and it writes to `.archify/` (gitignored). Give it:
- the diagram type from the table above
- repository evidence: the source files to trace, and `--repo-root` set to this repo so nodes link to code
- the facts below, so the diagram uses this project's real names

Facts to trace in the code (re-read the code first; it is the source of truth, not this list):
- Query pipeline: a LangGraph `StateGraph` in `verbatim/graph.py`: extract → retrieve → rerank → synthesize → log. Conditional edges end the run after extract (no valid company, or a quarter without an FY) and after retrieve (no candidates). `ChatManager` in `scripts/chat.py` supplies the stage functions and prints the answer. If `verbatim/graph.py` doesn't exist on this branch, diagram the orchestration in `ChatManager.run_pipeline` instead.
- Extract: `verbatim/processor.py`, gpt-4o-mini through LangChain, `prompts/extraction.txt`
- Retrieve: `Database.search_hybrid_rrf` in `verbatim/db.py`, one SQL query fusing pgvector cosine search and Postgres full-text search with weighted RRF
- Rerank: `verbatim/reranker.py`, Gemini (`google-genai`), `prompts/reranking.txt`; falls back to RRF order on any error
- Synthesize: `verbatim/synthesizer.py`, gpt-4o-mini through LangChain, `prompts/synthesis.txt`
- Log: `Database.log_query` writes per-stage latencies and reranker displacement to `query_logs`; `scripts/stats.py` reads them back
- Storage: Postgres 16 + pgvector (`docker-compose.yaml`, port 5433), tables `transcript_chunks` and `query_logs`
- External services: OpenAI (embeddings `text-embedding-3-small`, chat `gpt-4o-mini`) and Google Gemini

Report what the diagram leaves out (for example evals, CI or the CLI scripts) when a reader might expect it.

**Archify not available.** Load the `artifact-diagramming` skill and draw the diagram as inline SVG in a page built as in section 3. Say in your reply that Archify wasn't used. Don't install Archify or Node yourself; offer it instead (`npx skills add https://archify.si --skill archify --agent claude-code --global --copy` needs Node).

## 3. Pages: eval reports and dashboards

Load the `artifact-design` skill before writing the page, and `dataviz` before any chart. Write the HTML to `.artifacts/<slug>.html` (gitignored).

- **Eval report.** Read the newest file per eval in `evals/results/` (`<timestamp>_<eval>.json`: `summary` plus per-case `results` with `scores`, `details`, `error`). Show the pass rate against `summary.threshold`, each score, and the failing cases with their reason. Separate quality failures from infrastructure errors: rate limits, exhausted quota or credits, and auth errors mean the case never ran, so label them as not evaluated rather than failed. No results file means no report; say which `make eval-*` to run (it costs money, so don't run it unasked).
- **Observability dashboard.** The data comes from `Database.get_system_stats()` (what `make stats` prints); it needs the database (`make db-up`). `top_distance` actually stores the top RRF score (higher is better) even though `scripts/stats.py` labels it a distance; label it correctly. `query_logs.question` holds users' questions: ask before putting raw questions in an artifact; aggregates are fine.

## 4. Publish

Use the Artifact tool when it is available:
- An Archify diagram is already a finished standalone page. Read the whole generated HTML before publishing it, then publish that file with a short `description` and `icon` `diagram`. Don't rewrite Archify's HTML.
- A page from section 3: publish `.artifacts/<slug>.html` with an `icon` such as `chart` or `report`.
- To update an artifact made earlier in the conversation, publish the same file path again so it keeps its URL.

Without the Artifact tool, give the absolute path to the local HTML file instead.

Never put secrets in an artifact: no `.env` values, API keys or connection strings with passwords (the docker-compose default is fine to name, never to embed as a working credential).

## 5. Reply

Lead with the link and a descriptive title. Add one or two sentences on the key takeaway when the user wants to understand the system, and state the abstraction level or anything deliberately left out. Don't recap commands or validation steps; if asked, say in one line what was checked (for example "Archify's automated checks passed; not visually reviewed").
