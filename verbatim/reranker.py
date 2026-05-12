import json
import os
from typing import List, Dict, Any

from google.generativeai import configure as genai_configure, GenerativeModel, GenerationConfig


class LLMReranker:
    def __init__(self, model_name: str = "gemini-3-flash") -> None:
        genai_configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = GenerativeModel(model_name)

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Uses an LLM to rank chunks by relevance using the RankGPT pattern.
        """
        if not chunks:
            return []

        # 1. Prepare the prompt with a numbered list of chunks
        # We only send content to save tokens, keeping the index for mapping
        items = [f"[{i}] {c['content']}" for i, c in enumerate(chunks)]
        prompt = f"""
        I will provide you with a user query and a list of financial transcript chunks. 
        Your task is to rank the chunks based on how relevant they are to the query.

        User Query: "{query}"

        Chunks:
        {chr(10).join(items)}

        Instructions:
        - Rank the chunks from most relevant to least relevant.
        - Return ONLY a JSON list of the integers representing the chunk indices.
        - Example Output: [4, 0, 12, 2, 7]
        """

        try:
            # 2. Call the small LLM
            response = self.model.generate_content(
                prompt,
                generation_config=GenerationConfig(response_mime_type="application/json")
            )

            # 3. Parse the ordered indices
            ordered_indices = json.loads(response.text)

            # 4. Rebuild the chunk list based on the new order
            reranked = [chunks[i] for i in ordered_indices if i < len(chunks)]

            # Return only the top requested
            return reranked[:top_n]

        except Exception as e:
            print(f"⚠️ Reranking failed: {e}. Falling back to original order.")
            return chunks[:top_n]