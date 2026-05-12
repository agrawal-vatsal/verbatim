import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

db_url = os.getenv("DATABASE_URL")
assert db_url is not None, "DATABASE_URL must be set"
print(f"Connecting to: {db_url}")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        row = cur.fetchone()
        assert row is not None
        print(row[0])
