import os

# scripts.ingest_data builds an OpenAI client at import time, which fails without a key.
# Tests never make real API calls, so a placeholder is enough.
os.environ.setdefault("OPENAI_API_KEY", "test-placeholder")
