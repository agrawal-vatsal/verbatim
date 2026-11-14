"""Eval: LLM reranking (verbatim.reranker.LLMReranker, real Gemini).

Each case in evals/datasets/reranking.jsonl has:
  id, flow, query
  candidates: [{id, content}] in the incoming (RRF) order
  relevant: ids of the chunks that answer the query (may be empty)
  top_n (optional, default 5)
  min_recall (optional, default min(1, k / len(relevant)))
  forbidden_first (optional): ids that must not be ranked first (e.g. a prompt-injection chunk)

Metrics per case are computed for the reranked order and for the incoming order
(the baseline), plus lift = reranked - baseline. LLMReranker silently falls back to
the incoming order when the model errors or returns unparseable output; the eval
records every model call to detect that, and reports it as the `fallback` score
(its mean is the fallback rate).

A case passes when there was no fallback, recall@k >= min_recall (or, with no
relevant chunks, the reranker returned a full top-k), and no forbidden chunk is first.

Run: make eval-reranking (paid).
"""

import json
import math
import re
import time
from typing import TYPE_CHECKING, Any, cast

from evals.harness import CaseResult, finish, load_cases, require_env, run_cases

if TYPE_CHECKING:
    from verbatim.reranker import LLMReranker

NAME = "reranking"
THRESHOLD = 0.85
DEFAULT_TOP_N = 5
MAX_RATE_LIMIT_RETRIES = 5
DEFAULT_RETRY_DELAY_S = 6.0
MAX_RETRY_DELAY_S = 60.0


# --- Model-call recording (fallback detection) -------------------------------------

def rate_limit_delay(error: Exception) -> float | None:
    """Seconds to wait before retrying if `error` is a rate limit (HTTP 429), else None.

    Gemini's 429 message says "Please retry in 5.2s"; honour that (plus a second of slack).
    Daily-quota 429s (quotaId ...PerDay...) are not retried.
    """
    text = str(error)
    if getattr(error, "code", None) != 429 and "RESOURCE_EXHAUSTED" not in text:
        return None
    if "PerDay" in text:  # daily quota: waiting seconds won't help
        return None
    match = re.search(r"retry in ([\d.]+)s", text)
    delay = float(match.group(1)) + 1 if match else DEFAULT_RETRY_DELAY_S
    return min(delay, MAX_RETRY_DELAY_S)


class RecordingModels:
    """Wraps `client.models`: records each generate_content call's text or error.

    Rate-limit errors are retried here, below LLMReranker, so a quota hiccup doesn't show up
    as a reranking failure. Production LLMReranker does not retry; this eval measures ranking
    quality, not quota. Retries are recorded on the call (only when there were any).
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        retries = 0
        while True:
            try:
                response = self.inner.generate_content(**kwargs)
            except Exception as e:
                delay = rate_limit_delay(e)
                if delay is not None and retries < MAX_RATE_LIMIT_RETRIES:
                    retries += 1
                    time.sleep(delay)
                    continue
                self.calls.append({"error": f"{type(e).__name__}: {e}", **({"retries": retries} if retries else {})})
                raise
            self.calls.append({"text": response.text, **({"retries": retries} if retries else {})})
            return response


class RecordingClient:
    def __init__(self, inner_client: Any) -> None:
        # Keep the real client alive: google-genai closes its HTTP transport when the Client
        # is garbage-collected, which breaks `models` even though we still hold a reference to it.
        self.inner = inner_client
        self.models = RecordingModels(inner_client.models)


def instrument(reranker: "LLMReranker") -> RecordingModels:
    """Route the reranker's model calls through a recorder (idempotent); returns it with calls cleared."""
    client: Any = reranker.client
    if isinstance(client, RecordingClient):
        client.models.calls.clear()
        return client.models
    recording = RecordingClient(client)
    reranker.client = cast(Any, recording)
    return recording.models


def detect_fallback(calls: list[dict[str, Any]], n_candidates: int) -> bool:
    """True when the last model call would have made LLMReranker fall back to the input order.

    Mirrors LLMReranker._llm_rank: an exception, a None response, output that
    json.raw_decode can't read, or anything that isn't a list of usable int indices.
    """
    if not calls:
        return False
    last = calls[-1]
    if "error" in last or last.get("text") is None:
        return True
    try:
        value, _ = json.JSONDecoder().raw_decode(str(last["text"]).strip())
    except ValueError:
        return True
    if not isinstance(value, list):
        return True
    return not all(isinstance(i, int) and not isinstance(i, bool) and i >= -n_candidates for i in value)


# --- Ranking metrics -----------------------------------------------------------------

def ranking_metrics(ranked_ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """recall@k, precision@k, MRR and binary nDCG@k of `ranked_ids` against `relevant`."""
    top = ranked_ids[:k]
    hits = [i for i, cid in enumerate(top) if cid in relevant]
    # Count each relevant chunk once even if the model repeated an index.
    found = {top[i] for i in hits}
    dcg = sum(1 / math.log2(i + 2) for i in hits)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {
        "recall_at_k": len(found) / len(relevant),
        "precision_at_k": len(found) / k if k else 0.0,
        "mrr": 1 / (hits[0] + 1) if hits else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def default_min_recall(n_relevant: int, k: int) -> float:
    return min(1.0, k / n_relevant) if n_relevant else 1.0


# --- Runner --------------------------------------------------------------------------

def evaluate_case(case: dict[str, Any], reranker: "LLMReranker") -> CaseResult:
    candidates: list[dict[str, Any]] = case["candidates"]
    relevant = {str(r) for r in case["relevant"]}
    top_n = int(case.get("top_n", DEFAULT_TOP_N))
    k = min(top_n, len(candidates))

    recorder = instrument(reranker)
    result = reranker.rerank(case["query"], candidates, top_n=top_n)
    ranked = result["post_ce_ids"]
    baseline = [str(c["id"]) for c in candidates]

    fallback = detect_fallback(recorder.calls, len(candidates))
    complete = len(ranked) == k
    scores: dict[str, float] = {
        "fallback": float(fallback),
        "complete": float(complete),
        "duplicates": float(len(set(ranked)) != len(ranked)),
        "displacement": result["displacement_score"],
    }
    details: dict[str, Any] = {
        "query": case["query"],
        "relevant": sorted(relevant),
        "ranked": ranked,
        "baseline_top_k": baseline[:k],
        "model_calls": list(recorder.calls),  # copy: the recorder is cleared for the next case
    }

    if relevant:
        reranked_metrics = ranking_metrics(ranked, relevant, k)
        baseline_metrics = ranking_metrics(baseline, relevant, k)
        scores.update(reranked_metrics)
        for name, value in baseline_metrics.items():
            scores[f"baseline_{name}"] = value
            scores[f"lift_{name}"] = round(reranked_metrics[name] - value, 4)
        min_recall = float(case.get("min_recall", default_min_recall(len(relevant), k)))
        quality_ok = reranked_metrics["recall_at_k"] >= min_recall
        details["min_recall"] = min_recall
    else:
        quality_ok = complete

    forbidden = {str(f) for f in case.get("forbidden_first", [])}
    first_ok = not (ranked and ranked[0] in forbidden)
    if forbidden:
        scores["forbidden_first_avoided"] = float(first_ok)

    passed = not fallback and quality_ok and first_ok
    return CaseResult(case_id=case["id"], passed=passed, scores=scores, details=details)


def build_reranker() -> "LLMReranker":
    from verbatim.reranker import LLMReranker

    return LLMReranker()


def main() -> int:
    require_env("GEMINI_API_KEY")
    reranker = build_reranker()
    cases = load_cases(f"{NAME}.jsonl")
    print(f"Running {len(cases)} {NAME} cases...")
    results = run_cases(cases, lambda case: evaluate_case(case, reranker))
    return finish(NAME, results, THRESHOLD)


if __name__ == "__main__":
    raise SystemExit(main())
