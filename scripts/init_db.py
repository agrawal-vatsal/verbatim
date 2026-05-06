import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def enable_extension():
    # Ensure your .env has a DATABASE_URL or similar connection string
    conn_str = os.getenv("DATABASE_URL")

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✅ pgvector extension enabled.")


if __name__ == "__main__":
    enable_extension()
