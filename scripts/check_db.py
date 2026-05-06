import os
from dotenv import load_dotenv
import psycopg

load_dotenv()

db_url = os.getenv("DATABASE_URL")
print(f"Connecting to: {db_url}")

with psycopg.connect(db_url) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(cur.fetchone()[0])
