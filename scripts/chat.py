import time
from typing import Dict, List, Optional, Tuple
from verbatim.db import Database
from verbatim.processor import QueryProcessor
from verbatim.synthesizer import Synthesizer
from scripts.ingest_data import get_embeddings


class ChatManager:
    def __init__(self):
        self.db = Database()
        self.processor = QueryProcessor(self.db)
        self.synthesizer = Synthesizer()

    def run_pipeline(self, user_input: str):
        """Orchestrates the RAG phases and logs results."""

        # 1. Extraction Phase
        extracted, p_time = self._timed_step(self.processor.process_query, user_input)

        # 2. Validation / Guardrails
        if not self._validate_extraction(extracted):
            return

        # 3. Retrieval Phase
        chunks, r_time = self._timed_step(self._perform_retrieval, extracted)

        if not chunks:
            print("❌ No relevant data found for those filters.")
            return

        # 4. Synthesis Phase
        answer, s_time = self._timed_step(self.synthesizer.generate_answer, user_input, chunks)

        # 5. Observability / Logging
        self._log_interaction(user_input, extracted, chunks, p_time, r_time, s_time)

        # 6. Final Output
        self._print_response(answer, extracted)

    def _timed_step(self, func, *args, **kwargs) -> Tuple[any, int]:
        """Utility to measure execution time of any method in ms."""
        start = time.time()
        result = func(*args, **kwargs)
        duration = int((time.time() - start) * 1000)
        return result, duration

    def _validate_extraction(self, extracted: Dict) -> bool:
        """Handles logic-gate guardrails."""
        company = extracted.get("company")
        fy = extracted.get("fy")
        quarter = extracted.get("quarter")

        if not company or company == "null":
            valid = ", ".join(self.db.get_unique_companies())
            print(f"❌ ERROR: Please specify a company. Available: {valid}")
            return False

        if quarter and not fy:
            print(f"❌ ERROR: I need the Financial Year to look up {quarter} data.")
            return False

        return True

    def _perform_retrieval(self, extracted: Dict) -> List[Dict]:
        """Handles vectorization and DB search."""
        query_vector = get_embeddings([extracted.get("search_query")])[0]
        return self.db.search_similar_chunks(
            query_vector,
            limit=5,
            company=extracted.get("company"),
            fy=extracted.get("fy"),
            quarter=extracted.get("quarter")
        )

    def _log_interaction(self, question, extracted, chunks, p_time, r_time, s_time):
        """Bundles telemetry and sends to DB."""
        self.db.log_query(
            {
                "question": question,
                "refined_query": extracted.get("search_query"),
                "company_filter": extracted.get("company"),
                "fy_filter": extracted.get("fy"),
                "quarter_filter": extracted.get("quarter"),
                "top_distance": chunks[0]['distance'] if chunks else None,
                "processing_time_ms": p_time,
                "retrieval_time_ms": r_time,
                "synthesis_time_ms": s_time,
                "response_time_ms": p_time + r_time + s_time
            }
        )

    def _print_response(self, answer: str, extracted: Dict):
        """Clean UI for the final result."""
        ctx = f"{extracted['company']} | {extracted.get('fy', 'All Years')}"
        print(f"\n{'=' * 10} ANSWER ({ctx}) {'=' * 10}")
        print(answer)
        print(f"{'=' * 40}\n")


def main():
    manager = ChatManager()
    user_input = input("💬 Ask Verbatim: ")
    if user_input.strip():
        manager.run_pipeline(user_input)


if __name__ == "__main__":
    main()