import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def enable_extension() -> None:
    # Ensure your .env has a DATABASE_URL or similar connection string
    conn_str = os.getenv("DATABASE_URL")
    assert conn_str is not None, "DATABASE_URL must be set"

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ pgvector extension enabled.")


if __name__ == "__main__":
    enable_extension()
