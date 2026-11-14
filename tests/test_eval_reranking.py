import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import harness, reranking
from evals.harness import load_cases
from tests.conftest import run_main
from verbatim.reranker import LLMReranker

CASES = load_cases("reranking.jsonl")
CASES_BY_ID = {c["id"]: c for c in CASES}
CASES_BY_QUERY = {c["query"]: c for c in CASES}
PROMPT = Path("prompts/reranking.txt").read_text()


class FakeModels:
    """Stands in for genai `client.models`; `respond` maps the prompt to response text or raises."""

    def __init__(self, respond: Callable[[str], str | None]) -> None:
        self.respond = respond
        self.prompts: list[str] = []

    def generate_content(self, **kwargs: Any) -> SimpleNamespace:
        self.prompts.append(kwargs["contents"])
        return SimpleNamespace(text=self.respond(kwargs["contents"]))


def _reranker(respond: Callable[[str], str | None]) -> LLMReranker:
    # Real LLMReranker (real prompt, parsing and fallback logic), fake Gemini client.
    r = LLMReranker.__new__(LLMReranker)
    r.client = SimpleNamespace(models=FakeModels(respond))  # type: ignore[assignment]
    r.model_name = "fake"
    r.prompt_template = PROMPT
    return r


def _always(text: str | None) -> Callable[[str], str | None]:
    return lambda _prompt: text


def _raise(_prompt: str) -> str:
    raise RuntimeError("quota exceeded")


def _oracle(prompt: str) -> str:
    """Perfect ranker for dataset cases: relevant first, forbidden last."""
    match = re.search(r'User Query: "(.*)"', prompt)
    assert match, "query not found in prompt"
    case = CASES_BY_QUERY[match.group(1)]
    ids = [c["id"] for c in case["candidates"]]
    forbidden = set(case.get("forbidden_first", []))
    order = [r for r in case["relevant"]] + [i for i in ids if i not in case["relevant"] and i not in forbidden]
    order += [i for i in ids if i in forbidden]
    return json.dumps([ids.index(i) for i in order])


# --- ranking_metrics ------------------------------------------------------------------

def test_metrics_perfect_ranking() -> None:
    m = reranking.ranking_metrics(["a", "b", "x", "y"], {"a", "b"}, 4)
    assert m == {"recall_at_k": 1.0, "precision_at_k": 0.5, "mrr": 1.0, "ndcg_at_k": 1.0}


def test_metrics_nothing_relevant_in_top_k() -> None:
    m = reranking.ranking_metrics(["x", "y", "a"], {"a"}, 2)
    assert m == {"recall_at_k": 0.0, "precision_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}


def test_metrics_partial_and_positional() -> None:
    m = reranking.ranking_metrics(["x", "a", "y", "b"], {"a", "b", "c"}, 3)
    assert m["recall_at_k"] == pytest.approx(1 / 3)
    assert m["precision_at_k"] == pytest.approx(1 / 3)
    assert m["mrr"] == 0.5
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["ndcg_at_k"] == pytest.approx((1 / math.log2(3)) / idcg)


def test_metrics_count_repeated_ids_once() -> None:
    m = reranking.ranking_metrics(["a", "a", "a"], {"a", "b"}, 3)
    assert m["recall_at_k"] == 0.5
    assert m["precision_at_k"] == pytest.approx(1 / 3)


def test_metrics_with_k_zero() -> None:
    assert reranking.ranking_metrics([], {"a"}, 0)["precision_at_k"] == 0.0


@pytest.mark.parametrize(("n_relevant", "k", "expected"), [(0, 5, 1.0), (2, 5, 1.0), (2, 1, 0.5), (5, 5, 1.0), (4, 3, 0.75)])
def test_default_min_recall(n_relevant: int, k: int, expected: float) -> None:
    assert reranking.default_min_recall(n_relevant, k) == expected


# --- detect_fallback -------------------------------------------------------------------

@pytest.mark.parametrize(
    ("calls", "fallback"),
    [
        ([], False),
        ([{"text": "[2, 0, 1]"}], False),
        ([{"text": " [2, 0, 1]\nbecause chunk 2 mentions NIM"}], False),
        ([{"text": "[-1, 0]"}], False),
        ([{"error": "RuntimeError: boom"}], True),
        ([{"text": None}], True),
        ([{"text": "I think chunk 2 is best"}], True),
        ([{"text": '{"ranking": [2, 0]}'}], True),
        ([{"text": '["2", "0"]'}], True),
        ([{"text": "[true, 0]"}], True),
        ([{"text": "[-9, 0]"}], True),
        ([{"error": "x"}, {"text": "[0]"}], False),  # only the last call counts
    ],
)
def test_detect_fallback(calls: list[dict[str, Any]], fallback: bool) -> None:
    assert reranking.detect_fallback(calls, n_candidates=3) is fallback


@pytest.mark.parametrize("response", ["not json", None, '{"a": 1}', '["x"]'])
def test_detect_fallback_agrees_with_real_reranker(response: str | None, capsys: pytest.CaptureFixture[str]) -> None:
    r = _reranker(_always(response))
    recorder = reranking.instrument(r)
    chunks = [{"id": str(i), "content": f"chunk {i}"} for i in range(3)]
    r.rerank("q", chunks, top_n=2)
    assert reranking.detect_fallback(recorder.calls, 3)
    assert "Falling back to original order" in capsys.readouterr().out


# --- instrument ------------------------------------------------------------------------

def test_instrument_records_text_and_errors_and_is_idempotent() -> None:
    r = _reranker(_always("[1, 0]"))
    recorder = reranking.instrument(r)
    recorder.generate_content(contents="p1")
    assert recorder.calls == [{"text": "[1, 0]"}]

    again = reranking.instrument(r)
    assert again is recorder and again.calls == []

    failing = _reranker(_raise)
    rec = reranking.instrument(failing)
    with pytest.raises(RuntimeError):
        rec.generate_content(contents="p")
    assert rec.calls == [{"error": "RuntimeError: quota exceeded"}]


# --- evaluate_case ---------------------------------------------------------------------

def test_evaluate_case_rewards_promoting_buried_chunks() -> None:
    case = CASES_BY_ID["nim_buried_at_bottom"]  # relevant at positions 12 and 13 of 14
    result = reranking.evaluate_case(case, _reranker(_oracle))

    assert result.passed
    assert result.scores["recall_at_k"] == 1.0
    assert result.scores["baseline_recall_at_k"] == 0.0
    assert result.scores["lift_recall_at_k"] == 1.0
    assert result.scores["lift_ndcg_at_k"] == 1.0
    assert result.scores["fallback"] == 0.0 and result.scores["complete"] == 1.0
    assert result.scores["displacement"] == 0.6194  # buried chunks promoted to #1-#2, fillers after
    assert result.details["ranked"][:2] == case["relevant"]


def test_evaluate_case_sends_query_and_all_chunks_in_prompt() -> None:
    case = CASES_BY_ID["fewer_candidates_than_top_n"]
    r = _reranker(_oracle)
    reranking.evaluate_case(case, r)
    prompt = r.client.models.inner.prompts[0]  # type: ignore[attr-defined]
    assert f'User Query: "{case["query"]}"' in prompt
    for i, chunk in enumerate(case["candidates"]):
        assert f"[{i}] {chunk['content']}" in prompt


def test_evaluate_case_fails_on_fallback_even_if_order_is_fine() -> None:
    case = CASES_BY_ID["relevant_already_first"]  # input order already has the answer first
    result = reranking.evaluate_case(case, _reranker(_raise))

    assert not result.passed
    assert result.scores["fallback"] == 1.0
    assert result.scores["recall_at_k"] == 1.0
    assert result.details["model_calls"] == [{"error": "RuntimeError: quota exceeded"}]


def test_evaluate_case_fails_when_relevant_chunk_not_promoted() -> None:
    case = CASES_BY_ID["cd_ratio_last_of_twenty"]
    result = reranking.evaluate_case(case, _reranker(_always("[0, 1, 2, 3, 4]")))
    assert not result.passed
    assert result.scores["fallback"] == 0.0
    assert result.scores["recall_at_k"] == 0.0 and result.scores["lift_recall_at_k"] == 0.0


def test_evaluate_case_with_no_relevant_chunks_needs_a_full_top_k() -> None:
    case = CASES_BY_ID["no_relevant_chunk"]
    full = reranking.evaluate_case(case, _reranker(_always("[3, 2, 1, 0, 4]")))
    short = reranking.evaluate_case(case, _reranker(_always("[3]")))

    assert full.passed and "recall_at_k" not in full.scores
    assert not short.passed and short.scores["complete"] == 0.0


def test_evaluate_case_flags_injection_ranked_first() -> None:
    case = CASES_BY_ID["injection_at_top"]  # injection at 0, relevant at 11
    obeyed = reranking.evaluate_case(case, _reranker(_always("[0, 11, 1, 2, 3]")))
    resisted = reranking.evaluate_case(case, _reranker(_oracle))

    assert not obeyed.passed and obeyed.scores["forbidden_first_avoided"] == 0.0
    assert obeyed.scores["recall_at_k"] == 1.0  # recall alone would have hidden the problem
    assert resisted.passed and resisted.scores["forbidden_first_avoided"] == 1.0


def test_evaluate_case_respects_min_recall_when_relevant_exceeds_k() -> None:
    case = CASES_BY_ID["more_relevant_than_top_n"]  # top_n=1, two relevant
    result = reranking.evaluate_case(case, _reranker(_oracle))
    assert result.passed
    assert result.details["min_recall"] == 0.5
    assert result.scores["recall_at_k"] == 0.5


def test_evaluate_case_reports_duplicates() -> None:
    case = CASES_BY_ID["fewer_candidates_than_top_n"]  # relevant "deals" at index 2
    result = reranking.evaluate_case(case, _reranker(_always("[2, 2, 0]")))
    assert result.scores["duplicates"] == 1.0
    assert result.passed  # duplicates are reported, recall is still satisfied


def test_evaluate_case_honours_explicit_min_recall_and_top_n() -> None:
    case = {
        "id": "custom", "flow": "x", "query": "q", "top_n": 2, "min_recall": 0.5,
        "candidates": [{"id": "a", "content": "A"}, {"id": "b", "content": "B"}, {"id": "c", "content": "C"}],
        "relevant": ["b", "c"],
    }
    result = reranking.evaluate_case(case, _reranker(_always("[1, 0, 2]")))
    assert result.passed and result.scores["recall_at_k"] == 0.5
    assert result.details["ranked"] == ["b", "a"]


# --- main --------------------------------------------------------------------------------

@pytest.fixture
def reports_to(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    real_finish = harness.finish

    def finish(name: str, results: list[harness.CaseResult], threshold: float) -> int:
        return real_finish(name, results, threshold, results_dir=tmp_path)

    monkeypatch.setattr(reranking, "finish", finish)
    monkeypatch.setattr(harness, "finish", finish)  # for run_main, which re-imports it
    return tmp_path


def _patch_reranker(monkeypatch: pytest.MonkeyPatch, respond: Callable[[str], str | None]) -> None:
    monkeypatch.setattr("verbatim.reranker.LLMReranker", lambda: _reranker(respond))


def test_main_passes_with_perfect_ranker(
    monkeypatch: pytest.MonkeyPatch, reports_to: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_reranker(monkeypatch, _oracle)

    assert reranking.main() == 0

    (report_path,) = reports_to.glob("*_reranking.json")
    summary = json.loads(report_path.read_text())["summary"]
    assert summary["cases"] == len(CASES) and summary["pass_rate"] == 1.0
    assert summary["scores"]["fallback"] == 0.0
    assert summary["scores"]["lift_recall_at_k"] > 0.5
    assert f"{len(CASES)}/{len(CASES)} passed" in capsys.readouterr().out


def test_main_fails_when_every_call_falls_back(monkeypatch: pytest.MonkeyPatch, reports_to: Path) -> None:
    _patch_reranker(monkeypatch, _raise)
    assert reranking.main() == 1
    (report_path,) = reports_to.glob("*_reranking.json")
    assert json.loads(report_path.read_text())["summary"]["scores"]["fallback"] == 1.0


def test_main_requires_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        reranking.main()


def test_build_reranker_constructs_real_class(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("verbatim.reranker.genai.Client", lambda api_key: SimpleNamespace(models=None))
    assert isinstance(reranking.build_reranker(), LLMReranker)


def test_runs_as_module(monkeypatch: pytest.MonkeyPatch, reports_to: Path) -> None:
    _patch_reranker(monkeypatch, _oracle)
    with pytest.raises(SystemExit) as exit_info:
        run_main("evals.reranking")
    assert exit_info.value.code == 0


# --- dataset ---------------------------------------------------------------------------------

def test_dataset_schema() -> None:
    assert len(CASES) >= 15
    for case in CASES:
        assert {"id", "flow", "query", "candidates", "relevant"} <= case.keys(), case["id"]
        ids = [c["id"] for c in case["candidates"]]
        assert ids and len(ids) == len(set(ids)), case["id"]
        assert all(c["content"].strip() for c in case["candidates"]), case["id"]
        assert set(case["relevant"]) <= set(ids), case["id"]
        assert set(case.get("forbidden_first", [])) <= set(ids) - set(case["relevant"]), case["id"]
        assert int(case.get("top_n", reranking.DEFAULT_TOP_N)) >= 1, case["id"]


def test_dataset_mostly_buries_relevant_chunks() -> None:
    """Most cases must need promotion: the incoming top-k misses at least one relevant chunk."""
    buried = 0
    for case in CASES:
        ids = [c["id"] for c in case["candidates"]]
        k = min(int(case.get("top_n", reranking.DEFAULT_TOP_N)), len(ids))
        if case["relevant"] and not set(case["relevant"]) <= set(ids[:k]):
            buried += 1
    assert buried >= len(CASES) * 0.7


def test_dataset_covers_every_flow() -> None:
    assert {c["flow"] for c in CASES} >= {
        "buried", "near_duplicate", "numeric", "few_candidates", "no_relevant", "injection",
        "long_chunks", "relevant_exceeds_k", "control",
    }
    assert any(len(c["content"]) > 1000 for case in CASES for c in case["candidates"])


def test_instrument_keeps_the_real_client_alive() -> None:
    """google-genai closes its transport when the Client is garbage-collected; the wrapper must keep it."""
    import gc

    class ClosingModels:
        closed = False

        def generate_content(self, **_: Any) -> SimpleNamespace:
            if self.closed:
                raise RuntimeError("Cannot send a request, as the client has been closed.")
            return SimpleNamespace(text="[0]")

    class ClosingClient:
        def __init__(self) -> None:
            self.models = ClosingModels()

        def __del__(self) -> None:
            self.models.closed = True

    reranker = LLMReranker.__new__(LLMReranker)
    reranker.client = ClosingClient()  # type: ignore[assignment]
    reranker.model_name = "fake"
    reranker.prompt_template = "{query}\n{chunks}"

    recorder = reranking.instrument(reranker)
    gc.collect()
    reranker.rerank("q", [{"id": "a", "content": "x"}], top_n=1)

    assert recorder.calls == [{"text": "[0]"}]


class _QuotaError(Exception):
    code = 429


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_QuotaError("429 RESOURCE_EXHAUSTED ... Please retry in 5.25s."), 6.25),
        (RuntimeError("429 RESOURCE_EXHAUSTED, no hint"), reranking.DEFAULT_RETRY_DELAY_S),
        (_QuotaError("Please retry in 500s."), reranking.MAX_RETRY_DELAY_S),
        (RuntimeError("500 internal error"), None),
        (_QuotaError("RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, retry in 30s"), None),
        (ValueError("bad json"), None),
    ],
)
def test_rate_limit_delay(error: Exception, expected: float | None) -> None:
    assert reranking.rate_limit_delay(error) == expected


class _FlakyModels:
    def __init__(self, failures: int) -> None:
        self.failures = failures

    def generate_content(self, **_: Any) -> SimpleNamespace:
        if self.failures:
            self.failures -= 1
            raise _QuotaError("RESOURCE_EXHAUSTED. Please retry in 2s.")
        return SimpleNamespace(text="[0]")


def test_recording_models_retries_rate_limits_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    models = reranking.RecordingModels(_FlakyModels(failures=2))

    assert models.generate_content(model="m", contents="c").text == "[0]"
    assert slept == [3.0, 3.0]
    assert models.calls == [{"text": "[0]", "retries": 2}]


def test_recording_models_gives_up_after_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    models = reranking.RecordingModels(_FlakyModels(failures=99))

    with pytest.raises(_QuotaError):
        models.generate_content(model="m", contents="c")
    assert len(slept) == reranking.MAX_RATE_LIMIT_RETRIES
    assert models.calls[0]["retries"] == reranking.MAX_RATE_LIMIT_RETRIES
    assert models.calls[0]["error"].startswith("_QuotaError")


def test_recording_models_does_not_retry_other_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: pytest.fail("should not sleep"))

    class Broken:
        def generate_content(self, **_: Any) -> Any:
            raise RuntimeError("500 internal")

    models = reranking.RecordingModels(Broken())
    with pytest.raises(RuntimeError):
        models.generate_content()
    assert models.calls == [{"error": "RuntimeError: 500 internal"}]


def test_each_case_keeps_its_own_model_calls() -> None:
    """The recorder is reused and cleared per case; a case's details must not change afterwards."""
    responses = iter(["[1, 0]", "not json"])
    reranker = _reranker(lambda _prompt: next(responses))
    case = {"id": "c", "query": "q", "candidates": [{"id": "a", "content": "x"}, {"id": "b", "content": "y"}],
            "relevant": ["b"], "top_n": 1}

    first = reranking.evaluate_case(case, reranker)
    reranking.evaluate_case({**case, "id": "d"}, reranker)

    assert first.details["model_calls"] == [{"text": "[1, 0]"}]
