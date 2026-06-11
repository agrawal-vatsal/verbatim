from typing import Dict, List, Optional, Any, cast
from dotenv import load_dotenv

# Project Imports
from verbatim.db import Database
from verbatim.processor import QueryProcessor
from verbatim.synthesizer import Synthesizer
from verbatim.reranker import LLMReranker, RerankerResult
from verbatim.utils import PipelineAbort, PipelineState, runnable_step, track_latency
from scripts.ingest_data import get_embeddings

# Load API Keys from .env
load_dotenv()


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

    @runnable_step
    @track_latency
    def _extract(self, state: PipelineState) -> PipelineState:
        """Phase 1: Metadata extraction and guardrails."""
        extracted: Dict[str, Any] = self.processor.process_query(state["user_input"])

        company: Optional[str] = extracted.get("company")
        if not company or company == "null":
            valid: str = ", ".join(self.db.get_unique_companies())
            raise PipelineAbort(f"❌ Error: Specify a company. Options: {valid}")

        if extracted.get("quarter") and not extracted.get("fy"):
            raise PipelineAbort("❌ Error: Please provide the Financial Year (FY) for that quarter.")

        return {**state, "extracted": extracted}

    @runnable_step
    @track_latency
    def _retrieve(self, state: PipelineState) -> PipelineState:
        """Phase 2: Hybrid Retrieval (The 'Fast' Stage)."""
        extracted = state["extracted"]
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

        if not candidates:
            raise PipelineAbort("⚠️ No relevant data found.")

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

        return {**state, "candidates": candidates}

    @runnable_step
    @track_latency
    def _rerank(self, state: PipelineState) -> PipelineState:
        """Phase 3: LLM + Cross-Encoder Reranking (The 'Smart' Stage)."""
        rerank_result: RerankerResult = self.reranker.rerank(
            state["extracted"]["search_query"],
            state["candidates"],
            top_n=self.FINAL_TOP_K
        )
        return {**state, "rerank_result": rerank_result}

    @runnable_step
    @track_latency
    def _synthesize(self, state: PipelineState) -> PipelineState:
        """Phase 4: Final Synthesis."""
        answer: str = self.synthesizer.generate_answer(
            state["user_input"],
            state["rerank_result"]["chunks"]
        )
        return {**state, "answer": answer}

    @runnable_step
    def _log_interaction(self, state: PipelineState) -> PipelineState:
        """Phase 5: Telemetry Persistence."""
        extracted = state["extracted"]
        rerank_result: RerankerResult = state["rerank_result"]
        chunks = rerank_result["chunks"]
        total_time: int = sum(self.latencies.values())

        # Safe extraction of the top score
        top_score: Optional[float] = chunks[0].get('rrf_score') if chunks else None

        telemetry: Dict[str, Any] = {
            "question": state["user_input"],
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
            "pre_ce_chunk_ids": rerank_result["pre_ce_ids"],
            "post_ce_chunk_ids": rerank_result["post_ce_ids"],
            "reranker_displacement": rerank_result["displacement_score"],
        }
        try:
            self.db.log_query(telemetry)
        except Exception as e:
            print(f"⚠️ Telemetry logging failed: {e}")

        return state

    @runnable_step
    def _display_result(self, state: PipelineState) -> None:
        """Phase 6: UI Output."""
        extracted = state["extracted"]
        company: str = cast(str, extracted.get('company', 'Unknown'))
        fy: str = cast(str, extracted.get('fy', 'All Time'))

        print(f"\n{'=' * 15} VERBATIM RESPONSE {'=' * 15}")
        print(f"Context: {company} | {fy}")
        print("-" * 50)
        print(state["answer"])
        print(f"{'=' * 50}\n")

    def run_pipeline(self, user_input: str) -> None:
        """Main Orchestrator."""
        self.latencies = {}

        pipeline = (
            self._extract
            | self._retrieve
            | self._rerank
            | self._synthesize
            | self._log_interaction
            | self._display_result
        )

        try:
            pipeline.invoke(PipelineState(user_input=user_input))
        except PipelineAbort as e:
            print(str(e))


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
