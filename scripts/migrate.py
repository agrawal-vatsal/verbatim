import os
import psycopg
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def run_migrations():
    conn_str = os.getenv("DATABASE_URL")
    migration_dir = Path("migrations")

    # Get all .sql files and sort them to ensure 0001 runs before 0002
    sql_files = sorted(migration_dir.glob("*.sql"))

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            for sql_file in sql_files:
                print(f"Applying migration: {sql_file.name}...")
                with open(sql_file, "r") as f:
                    cur.execute(f.read())
            conn.commit()
    print("✅ All migrations applied successfully.")


if __name__ == "__main__":
    run_migrations()
