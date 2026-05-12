import time
import functools
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast
from verbatim.db import Database
from verbatim.processor import QueryProcessor
from verbatim.synthesizer import Synthesizer
from scripts.ingest_data import get_embeddings

F = TypeVar('F', bound=Callable[..., Any])


def track_latency(func: F) -> F:
    """
    Decorator that measures execution time in milliseconds
    and stores it in the instance's 'latencies' dictionary.
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)
        self.latencies[func.__name__] = int((time.time() - start_time) * 1000)
        return result

    return cast(F, wrapper)


class ChatManager:
    # ---------------------------------------------------------
    # Retrieval Configuration
    # ---------------------------------------------------------
    K_LIMIT = 20
    WEIGHT_VEC = 0.7
    WEIGHT_KW = 0.3
    FINAL_LIMIT = 5

    # ---------------------------------------------------------

    def __init__(self) -> None:
        self.db = Database()
        self.processor = QueryProcessor(self.db)
        self.synthesizer = Synthesizer()
        self.latencies: dict[str, int] = {}  # State storage for the decorator
        self.search_stats: dict[str, int] = {}

    @track_latency
    def _extract(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Phase 1: Metadata extraction and query rewriting."""
        extracted = self.processor.process_query(user_input)

        # Validation Guardrails
        if not extracted.get("company") or extracted["company"] == "null":
            companies = ", ".join(self.db.get_unique_companies())
            print(f"❌ Error: Please specify a company. Available: {companies}")
            return None

        if extracted.get("quarter") and not extracted.get("fy"):
            print("❌ Error: Financial Year (FY) is required when searching a specific Quarter.")
            return None

        return extracted

    @track_latency
    def _retrieve(self, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes hybrid search and calculates the retrieval signal distribution
        (Overlap vs Vector-only vs Keyword-only).
        """
        search_query: str = cast(str, extracted.get("search_query", ""))
        query_vector = get_embeddings([search_query])[0]

        chunks = self.db.search_hybrid_rrf(
            query_text=search_query,
            query_embedding=query_vector,
            company=extracted["company"],
            fy=extracted.get("fy"),
            quarter=extracted.get("quarter"),
            k_limit=self.K_LIMIT,
            weight_vec=self.WEIGHT_VEC,
            weight_kw=self.WEIGHT_KW,
            final_limit=self.FINAL_LIMIT
        )

        # Calculate Retrieval Stats
        if chunks:
            overlap = [c for c in chunks if c['found_by_vector'] and c['found_by_keyword']]
            pure_v = [c for c in chunks if c['found_by_vector'] and not c['found_by_keyword']]
            pure_k = [c for c in chunks if c['found_by_keyword'] and not c['found_by_vector']]

            self.search_stats = {
                "overlap_count": len(overlap),
                "vector_signal": len(pure_v),
                "keyword_signal": len(pure_k)
            }
        else:
            self.search_stats = {"overlap_count": 0, "vector_signal": 0, "keyword_signal": 0}

        return chunks

    @track_latency
    def _synthesize(self, user_input: str, chunks: List[Dict[str, Any]]) -> str:
        """Phase 3: Context-aware answer generation."""
        return self.synthesizer.generate_answer(user_input, chunks)

    def _log_interaction(self, question: str, extracted: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
        """Persists performance data and search mode distribution to the database."""
        total_time = sum(self.latencies.values())

        # Pack the telemetry payload
        telemetry: dict[str, Any] = {
            "question": question,
            "refined_query": extracted.get("search_query"),
            "company_filter": extracted.get("company"),
            "fy_filter": extracted.get("fy"),
            "quarter_filter": extracted.get("quarter"),
            "top_distance": chunks[0].get('rrf_score') if chunks else None,
            # Timing (from @track_latency decorator)
            "processing_time_ms": self.latencies.get('_extract', 0),
            "retrieval_time_ms": self.latencies.get('_retrieve', 0),
            "synthesis_time_ms": self.latencies.get('_synthesize', 0),
            "response_time_ms": total_time,
            # Search Distribution Signal
            "search_overlap_count": self.search_stats.get("overlap_count", 0),
            "vector_signal": self.search_stats.get("vector_signal", 0),
            "keyword_signal": self.search_stats.get("keyword_signal", 0)
        }

        # Save to Postgres
        self.db.log_query(telemetry)

    def _display_result(self, answer: str, extracted: Dict[str, Any]) -> None:
        """Formats the final output for the console."""
        metadata = f" {extracted['company']} | {extracted.get('fy', 'All Years')} "
        print(f"\n{metadata.center(60, '=')}")
        print(answer)
        print("=" * 60 + "\n")

    def run_pipeline(self, user_input: str) -> None:
        """Main orchestrator for the RAG pipeline flow."""
        self.latencies = {}  # Clear latencies for the new request

        # 1. Extraction
        extracted = self._extract(user_input)
        if not extracted:
            return

        # 2. Retrieval
        chunks = self._retrieve(extracted)
        if not chunks:
            print("⚠️ No relevant documents found for the given parameters.")
            return

        # 3. Synthesis
        answer = self._synthesize(user_input, chunks)

        # 4. Telemetry & Display
        self._log_interaction(user_input, extracted, chunks)
        self._display_result(answer, extracted)


# --- Entry Point ---

def main() -> None:
    manager = ChatManager()
    print("🏦 Verbatim Financial Intelligence Engine | 2026")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            prompt = input("💬 Ask about a company: ").strip()
            if prompt.lower() in ['exit', 'quit']:
                break
            if prompt:
                manager.run_pipeline(prompt)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
