from types import SimpleNamespace
from typing import Any

import pytest

from verbatim.reranker import LLMReranker, _weighted_displacement


def _chunks(n: int) -> list[dict[str, Any]]:
    return [{"id": f"id-{i}", "content": f"chunk {i}"} for i in range(n)]


class _FakeModels:
    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text = text
        self.error = error

    def generate_content(self, **_: Any) -> SimpleNamespace:
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


def _reranker(models: _FakeModels) -> LLMReranker:
    # Skip __init__ so no Gemini client or prompt file is needed.
    reranker = LLMReranker.__new__(LLMReranker)
    reranker.client = SimpleNamespace(models=models)  # type: ignore[assignment]
    reranker.model_name = "fake"
    reranker.prompt_template = "{query}\n{chunks}"
    return reranker


# --- _weighted_displacement -------------------------------------------------

def test_displacement_zero_when_top_k_kept_in_order() -> None:
    assert _weighted_displacement([0, 1, 2, 3, 4], n=20, k=5) == 0.0


def test_displacement_one_when_bottom_k_promoted_most_displaced_first() -> None:
    assert _weighted_displacement([19, 18, 17, 16, 15], n=20, k=5) == 1.0


def test_displacement_reversed_top_k_is_small_but_nonzero() -> None:
    # weighted mean 40/15, min 20/15, max 265/15 -> (40-20)/(265-20)
    assert _weighted_displacement([4, 3, 2, 1, 0], n=20, k=5) == pytest.approx(20 / 245, abs=1e-4)


@pytest.mark.parametrize(("n", "k"), [(20, 0), (5, 5), (3, 5)])
def test_displacement_degenerate_sizes_return_zero(n: int, k: int) -> None:
    assert _weighted_displacement(list(range(k)), n=n, k=k) == 0.0


# --- LLMReranker.rerank -----------------------------------------------------

def test_rerank_empty_input() -> None:
    result = _reranker(_FakeModels(text="[]")).rerank("q", [])
    assert result == {"chunks": [], "pre_ce_ids": [], "post_ce_ids": [], "displacement_score": 0.0}


def test_rerank_applies_llm_order_and_truncates() -> None:
    chunks = _chunks(10)
    result = _reranker(_FakeModels(text="[9, 8, 7, 6, 5, 4]")).rerank("q", chunks, top_n=3)

    assert result["post_ce_ids"] == ["id-9", "id-8", "id-7"]
    assert result["pre_ce_ids"] == [c["id"] for c in chunks]
    assert result["displacement_score"] == 1.0


def test_rerank_drops_out_of_range_indices() -> None:
    result = _reranker(_FakeModels(text="[42, 1, 0]")).rerank("q", _chunks(3), top_n=5)
    assert result["post_ce_ids"] == ["id-1", "id-0"]


def test_rerank_tolerates_trailing_text_after_json() -> None:
    result = _reranker(_FakeModels(text="[2, 0, 1]\nsome trailing explanation")).rerank("q", _chunks(3))
    assert result["post_ce_ids"] == ["id-2", "id-0", "id-1"]


@pytest.mark.parametrize(
    "models",
    [_FakeModels(error=RuntimeError("boom")), _FakeModels(text=None), _FakeModels(text="not json")],
)
def test_rerank_falls_back_to_rrf_order_on_failure(models: _FakeModels) -> None:
    result = _reranker(models).rerank("q", _chunks(8), top_n=5)
    assert result["post_ce_ids"] == [f"id-{i}" for i in range(5)]
    assert result["displacement_score"] == 0.0
