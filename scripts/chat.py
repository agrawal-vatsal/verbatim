import time
import functools
from typing import Dict, List, Optional, Any, Callable, TypeVar, cast
from dotenv import load_dotenv

# Project Imports
from verbatim.db import Database
from verbatim.processor import QueryProcessor
from verbatim.synthesizer import Synthesizer
from verbatim.reranker import LLMReranker, RerankerResult
from scripts.ingest_data import get_embeddings

# Load API Keys from .env
load_dotenv()

# Type variable for the decorator to preserve function signatures
F = TypeVar('F', bound=Callable[..., Any])


def track_latency(func: F) -> F:
    """Decorator to record method execution time in milliseconds."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        result = func(self, *args, **kwargs)

        # Ensure latencies dict exists on the instance
        if not hasattr(self, 'latencies'):
            self.latencies = {}

        self.latencies[func.__name__] = int((time.time() - start_time) * 1000)
        return result

    return cast(F, wrapper)


class ChatManager:
    # ---------------------------------------------------------
    # Configuration: The Funnel Strategy
    # ---------------------------------------------------------
    CANDIDATE_POOL: int = 20  # Funnel Phase 1
    FINAL_TOP_K: int = 5  # Funnel Phase 2

    WEIGHT_VEC: float = 1
    WEIGHT_KW: float = 1

    # ---------------------------------------------------------

    def __init__(self) -> None:
        self.db: Database = Database()
        self.processor: QueryProcessor = QueryProcessor(self.db)
        self.reranker: LLMReranker = LLMReranker()
        self.synthesizer: Synthesizer = Synthesizer()
        self.latencies: Dict[str, int] = {}
        self.search_stats: Dict[str, int] = {}

    @track_latency
    def _extract(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Phase 1: Metadata extraction and guardrails."""
        extracted: Dict[str, Any] = self.processor.process_query(user_input)

        company: Optional[str] = extracted.get("company")
        if not company or company == "null":
            valid: str = ", ".join(self.db.get_unique_companies())
            print(f"❌ Error: Specify a company. Options: {valid}")
            return None

        if extracted.get("quarter") and not extracted.get("fy"):
            print("❌ Error: Please provide the Financial Year (FY) for that quarter.")
            return None

        return extracted

    @track_latency
    def _retrieve(self, extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Phase 2: Hybrid Retrieval (The 'Fast' Stage)."""
        search_query: str = cast(str, extracted["search_query"])
        # get_embeddings typically returns a list of lists
        query_vector: List[float] = get_embeddings([search_query])[0]

        candidates: List[Dict[str, Any]] = self.db.search_hybrid_rrf(
            query_text=search_query,
            query_embedding=query_vector,
            company=cast(str, extracted["company"]),
            fy=extracted.get("fy"),
            quarter=extracted.get("quarter"),
            k_limit=20,
            weight_vec=self.WEIGHT_VEC,
            weight_kw=self.WEIGHT_KW,
            final_limit=self.CANDIDATE_POOL
        )

        if candidates:
            overlap = [c for c in candidates if
                       c.get('found_by_vector') and c.get('found_by_keyword')]
            pure_v = [c for c in candidates if
                      c.get('found_by_vector') and not c.get('found_by_keyword')]
            pure_k = [c for c in candidates if
                      c.get('found_by_keyword') and not c.get('found_by_vector')]

            self.search_stats = {
                "overlap": len(overlap),
                "vector_only": len(pure_v),
                "keyword_only": len(pure_k)
            }

        return candidates

    @track_latency
    def _rerank(self, query: str, candidates: List[Dict[str, Any]]) -> RerankerResult:
        """Phase 3: LLM + Cross-Encoder Reranking (The 'Smart' Stage)."""
        return self.reranker.rerank(query, candidates, top_n=self.FINAL_TOP_K)

    @track_latency
    def _synthesize(self, user_input: str, chunks: List[Dict[str, Any]]) -> str:
        """Phase 4: Final Synthesis."""
        return self.synthesizer.generate_answer(user_input, chunks)

    def _log_interaction(
            self,
            question: str,
            extracted: Dict[str, Any],
            chunks: List[Dict[str, Any]],
            pre_ce_ids: List[str],
            post_ce_ids: List[str],
            displacement_score: float,
        ) -> None:
        """Phase 5: Telemetry Persistence."""
        total_time: int = sum(self.latencies.values())

        # Safe extraction of the top score
        top_score: Optional[float] = chunks[0].get('rrf_score') if chunks else None

        telemetry: Dict[str, Any] = {
            "question": question,
            "refined_query": extracted.get("search_query"),
            "company_filter": extracted.get("company"),
            "fy_filter": extracted.get("fy"),
            "quarter_filter": extracted.get("quarter"),
            "top_distance": top_score,
            "processing_time_ms": self.latencies.get('_extract', 0),
            "retrieval_time_ms": self.latencies.get('_retrieve', 0),
            "rerank_time_ms": self.latencies.get('_rerank', 0),
            "synthesis_time_ms": self.latencies.get('_synthesize', 0),
            "response_time_ms": total_time,
            "search_overlap_count": self.search_stats.get("overlap", 0),
            "vector_signal": self.search_stats.get("vector_only", 0),
            "keyword_signal": self.search_stats.get("keyword_only", 0),
            "pre_ce_chunk_ids": pre_ce_ids,
            "post_ce_chunk_ids": post_ce_ids,
            "reranker_displacement": displacement_score,
        }
        try:
            self.db.log_query(telemetry)
        except Exception as e:
            print(f"⚠️ Telemetry logging failed: {e}")

    def _display_result(self, answer: str, extracted: Dict[str, Any]) -> None:
        """Phase 5: UI Output."""
        company: str = cast(str, extracted.get('company', 'Unknown'))
        fy: str = cast(str, extracted.get('fy', 'All Time'))

        print(f"\n{'=' * 15} VERBATIM RESPONSE {'=' * 15}")
        print(f"Context: {company} | {fy}")
        print("-" * 50)
        print(answer)
        print(f"{'=' * 50}\n")

    def run_pipeline(self, user_input: str) -> None:
        """Main Orchestrator."""
        self.latencies = {}

        extracted = self._extract(user_input)
        if extracted is None:
            return

        candidates = self._retrieve(extracted)
        if not candidates:
            print("⚠️ No relevant data found.")
            return

        rerank_result: RerankerResult = self._rerank(cast(str, extracted["search_query"]), candidates)
        refined_chunks = rerank_result["chunks"]

        answer = self._synthesize(user_input, refined_chunks)

        self._log_interaction(
            user_input, extracted, refined_chunks,
            rerank_result["pre_ce_ids"], rerank_result["post_ce_ids"],
            rerank_result["displacement_score"],
        )
        self._display_result(answer, extracted)


if __name__ == "__main__":
    manager = ChatManager()
    while True:
        try:
            prompt: str = input("💬 Question: ").strip()
            if prompt.lower() in ['exit', 'quit']:
                break
            if prompt:
                manager.run_pipeline(prompt)
        except (KeyboardInterrupt, EOFError):
            break