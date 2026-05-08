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

    def get_unique_companies(self) -> List[str]:
        """Returns a list of all company names present in the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT company FROM transcript_chunks;")
                return [row[0] for row in cur.fetchall()]

    def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 5,
        company: str = None,
        fy: str = None,
        quarter: str = None
    ):
        """
        Finds relevant chunks with optional hard metadata filters.
        """
        formatted_embedding = f"[{','.join(map(str, query_embedding))}]"

        # Base Query
        query = """
            SELECT 
                content, company, fy, quarter, page_number,
                embedding <=> %s AS distance
            FROM transcript_chunks
            WHERE 1=1
        """
        params = [formatted_embedding]

        # Dynamically add filters if they are provided
        if company:
            query += " AND company = %s"
            params.append(company)
        if fy:
            query += " AND fy = %s"
            params.append(fy)
        if quarter:
            query += " AND quarter = %s"
            params.append(quarter)

        # Add Ordering and Limit
        query += " ORDER BY distance ASC LIMIT %s;"
        params.append(limit)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()

                results = []
                for r in rows:
                    results.append(
                        {
                            "content": r[0],
                            "metadata": f"{r[1]} {r[2]} {r[3]} (p. {r[4]})",
                            "distance": round(float(r[5]), 4)
                        }
                    )
                return results

    def log_query(self, data: dict):
        """Persists granular RAG telemetry to the database."""
        sql = """
            INSERT INTO query_logs 
            (question, refined_query, company_filter, fy_filter, quarter_filter, 
             top_distance, processing_time_ms, retrieval_time_ms, synthesis_time_ms, response_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql, (
                        data['question'],
                        data.get('refined_query'),
                        data.get('company_filter'),
                        data.get('fy_filter'),
                        data.get('quarter_filter'),
                        data.get('top_distance'),
                        data.get('processing_time_ms'),
                        data.get('retrieval_time_ms'),
                        data.get('synthesis_time_ms'),
                        data.get('response_time_ms')
                    )
                    )
                conn.commit()

    def get_system_stats(self) -> dict:
        """Fetches metrics including Median (P50) and P95 tail latencies."""
        stats = {}
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Basic Counts
                cur.execute("SELECT count(*) FROM transcript_chunks;")
                stats['total_chunks'] = cur.fetchone()[0]

                cur.execute(
                    "SELECT company, count(*) FROM transcript_chunks GROUP BY company ORDER BY count DESC;"
                    )
                stats['company_dist'] = cur.fetchall()

                # 2. Granular Performance Metrics (Last 24h)
                cur.execute(
                    """
                    SELECT 
                        count(*),
                        -- Averages
                        AVG(processing_time_ms), AVG(retrieval_time_ms), 
                        AVG(synthesis_time_ms), AVG(response_time_ms),
                        -- P95 Latencies
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_time_ms),
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY retrieval_time_ms),
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY synthesis_time_ms),
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms),
                        -- Distance Percentiles
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY top_distance),
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY top_distance)
                    FROM query_logs 
                    WHERE created_at > now() - interval '1 day';
                """
                    )
                row = cur.fetchone()

                stats['queries_24h'] = row[0] or 0
                stats['latency'] = {
                    "avg": {"p": row[1] or 0, "r": row[2] or 0, "s": row[3] or 0, "t": row[4] or 0},
                    "p95": {"p": row[5] or 0, "r": row[6] or 0, "s": row[7] or 0, "t": row[8] or 0}
                }
                stats['distance'] = {
                    "p50": float(row[9] or 0),
                    "p95": float(row[10] or 0)
                }
        return stats
