import os
import psycopg
from typing import List, Dict
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


class Database:
    def __init__(self):
        self.conn_str = os.getenv("DATABASE_URL")

    def _get_connection(self):
        """Internal helper to get a connection and register the vector type."""
        conn = psycopg.connect(self.conn_str)
        register_vector(conn)
        return conn

    def insert_transcript_chunks(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Performs a high-speed batch insert using COPY."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                data_to_insert = [
                    (
                        c["company"],
                        c["fy"],
                        c["quarter"],
                        c["page_number"],
                        c["content"],
                        # MANUALLY FORMAT: Convert list [0.1, 0.2] -> string "[0.1,0.2]"
                        f"[{','.join(map(str, emb))}]"
                    )
                    for c, emb in zip(chunks, embeddings)
                ]

                with cur.copy(
                        "COPY transcript_chunks (company, fy, quarter, page_number, content, embedding) FROM STDIN"
                ) as copy:
                    for row in data_to_insert:
                        copy.write_row(row)

                conn.commit()
        return len(chunks)

    def has_transcript_data(self, company: str, fy: str, quarter: str) -> bool:
        """Checks if data for a specific transcript already exists in the DB."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1 FROM transcript_chunks 
                    WHERE company = %s AND fy = %s AND quarter = %s 
                    LIMIT 1
                    """,
                    (company, fy, quarter)
                )
                return cur.fetchone() is not None
