import json
import os
from pathlib import Path
from typing import List, Dict, Any, TypedDict

from google import genai
from google.genai import types


def _weighted_displacement(positions: List[int], n: int, k: int) -> float:
    """
    Measures how much the LLM reranker promoted lower-RRF-ranked chunks,
    accounting for both which chunks were selected and where they were placed.

    Each selected chunk's original RRF rank is weighted by its position in
    the LLM output — the chunk ranked #1 by the LLM carries more weight than #5.

    Score = 0 : LLM returned the top-K chunks in their original RRF order (no uplift).
    Score = 1 : LLM placed the most-displaced chunks first (maximum uplift).
    """
    if k == 0 or n <= k:
        return 0.0

    d = k * (k + 1) / 2                              # sum of weights: K + (K-1) + ... + 1
    weights = [(k - j) / d for j in range(k)]        # LLM pos 0 → highest weight

    weighted_mean = sum(w * r for w, r in zip(weights, positions))
    min_wm = sum(weights[j] * j for j in range(k))              # top-K in RRF order
    max_wm = sum(weights[j] * (n - 1 - j) for j in range(k))   # bottom-K, most displaced first

    if max_wm <= min_wm:
        return 0.0
    return round(max(0.0, min(1.0, (weighted_mean - min_wm) / (max_wm - min_wm))), 4)


class RerankerResult(TypedDict):
    chunks: List[Dict[str, Any]]
    pre_ce_ids: List[str]       # chunk UUIDs in RRF order, before LLM reranking
    post_ce_ids: List[str]      # chunk UUIDs after LLM reranking
    displacement_score: float   # 0 = top-K unchanged, 1 = bottom-K fully promoted


class LLMReranker:
    def __init__(self, model_name: str = "gemini-2.5-flash-lite") -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name = model_name
        self.prompt_template = Path("prompts/reranking.txt").read_text()

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> RerankerResult:
        """
        Captures the RRF-ordered candidates, then uses an LLM to rerank them.
        pre_ce_ids = all candidate IDs in original RRF order (before reranking).
        post_ce_ids = top-N IDs after LLM reranking.
        displacement_score = normalised measure of how far the LLM promoted lower-ranked chunks.
        """
        if not chunks:
            return RerankerResult(chunks=[], pre_ce_ids=[], post_ce_ids=[], displacement_score=0.0)

        # Capture the incoming RRF order before any reranking
        pre_ce_ids = [str(c['id']) for c in chunks]

        reranked = self._llm_rank(query, chunks, top_n)
        post_ce_ids = [str(c['id']) for c in reranked]

        n, k = len(pre_ce_ids), len(post_ce_ids)
        positions = [pre_ce_ids.index(uid) for uid in post_ce_ids if uid in pre_ce_ids]
        displacement_score = _weighted_displacement(positions, n, k)

        return RerankerResult(chunks=reranked, pre_ce_ids=pre_ce_ids, post_ce_ids=post_ce_ids, displacement_score=displacement_score)

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
            if response.text is None:
                raise ValueError("Empty response from reranker model")
            ordered_indices, _ = json.JSONDecoder().raw_decode(response.text.strip())
            reranked = [chunks[i] for i in ordered_indices if i < len(chunks)]
            return reranked[:top_n]
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Falling back to original order.")
            return chunks[:top_n]
