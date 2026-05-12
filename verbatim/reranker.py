import json
import os
from pathlib import Path
from typing import List, Dict, Any, TypedDict

from google import genai
from google.genai import types


class RerankerResult(TypedDict):
    chunks: List[Dict[str, Any]]
    pre_ce_ids: List[int]   # chunk IDs in LLM-ranked order, before cross encoder
    post_ce_ids: List[int]  # chunk IDs after cross encoder reranking


class LLMReranker:
    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_name
        self.prompt_template = Path("prompts/reranking.txt").read_text()

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> RerankerResult:
        """
        Uses an LLM to rank chunks by relevance using the RankGPT pattern,
        then applies cross-encoder re-scoring on the top results.
        """
        if not chunks:
            return RerankerResult(chunks=[], pre_ce_ids=[], post_ce_ids=[])

        llm_ordered = self._llm_rank(query, chunks, top_n)
        pre_ce_ids = [int(c['id']) for c in llm_ordered]

        cross_encoded = self._cross_encode(query, llm_ordered)
        post_ce_ids = [int(c['id']) for c in cross_encoded]

        return RerankerResult(chunks=cross_encoded, pre_ce_ids=pre_ce_ids, post_ce_ids=post_ce_ids)

    def _llm_rank(self, query: str, chunks: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """Reranks using the RankGPT pattern."""
        # We only send content to save tokens, keeping the index for mapping
        items = "\n".join(f"[{i}] {c['content']}" for i, c in enumerate(chunks))
        prompt = self.prompt_template.format(query=query, chunks=items)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            ordered_indices = json.loads(response.text)
            reranked = [chunks[i] for i in ordered_indices if i < len(chunks)]
            return reranked[:top_n]
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Falling back to original order.")
            return chunks[:top_n]

    def _cross_encode(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cross-encoder re-scoring on the LLM-ranked top results."""
        # Replace with your cross-encoder implementation
        return chunks
