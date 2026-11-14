.DEFAULT_GOAL := help
.PHONY: help install sync test coverage typecheck lint fix format check \
        db-up db-down db-init db-check stats chat download ingest clean \
        eval eval-extraction eval-reranking eval-synthesis eval-pipeline

RUN := uv run

help: ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

# --- Setup -------------------------------------------------------------------

install: ## First-time setup: exact locked deps + .env from template
	uv sync --locked
	@test -f .env || { cp .env.example .env; echo "Created .env from .env.example - fill in your API keys."; }

sync: ## Sync the env with pyproject.toml (updates uv.lock if needed)
	uv sync

# --- Quality -----------------------------------------------------------------

test: ## Run the offline test suite with coverage (fails under 90%)
	$(RUN) pytest --cov

coverage: ## HTML coverage report in htmlcov/
	$(RUN) pytest --cov --cov-report=html
	@echo "Open htmlcov/index.html"

typecheck: ## Type-check with mypy (strict)
	$(RUN) mypy .

lint: ## Lint with ruff
	$(RUN) ruff check .

fix: ## Auto-fix lint issues
	$(RUN) ruff check --fix .

format: ## Reformat code with ruff format
	$(RUN) ruff format .

check: lint typecheck test ## Lint + typecheck + tests (run before pushing)

# --- Database (Docker) -------------------------------------------------------

db-up: ## Start Postgres + pgvector on port 5433
	docker compose up -d

db-down: ## Stop Postgres (data volume is kept)
	docker compose down

db-init: ## Enable pgvector and apply all migrations
	$(RUN) python -m scripts.init_db
	$(RUN) python -m scripts.migrate

db-check: ## Verify DB connectivity
	$(RUN) python -m scripts.check_db

# --- App ---------------------------------------------------------------------

stats: ## Show the observability dashboard
	$(RUN) python -m scripts.stats

chat: ## Interactive chat (paid: OpenAI + Gemini)
	$(RUN) python -m scripts.chat

download: ## Download transcripts listed in manifest.json
	$(RUN) python -m scripts.download_data

ingest: ## Embed and load downloaded PDFs (paid: OpenAI)
	$(RUN) python -m scripts.ingest_all

# --- Evals (real, paid LLM calls; never part of `check`) ----------------------

eval: eval-extraction eval-reranking eval-synthesis ## Run all component evals (paid: OpenAI + Gemini)

eval-extraction: ## Eval query extraction (paid: OpenAI)
	$(RUN) python -m evals.extraction

eval-reranking: ## Eval the Gemini reranker (paid: Gemini)
	$(RUN) python -m evals.reranking

eval-synthesis: ## Eval cited answer synthesis (paid: OpenAI)
	$(RUN) python -m evals.synthesis

eval-pipeline: ## End-to-end eval against the DB (paid; needs db-up + ingested data)
	$(RUN) python -m evals.pipeline

clean: ## Remove tool caches
	rm -rf .mypy_cache .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
