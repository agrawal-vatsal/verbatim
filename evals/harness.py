"""Shared plumbing for the LLM evals in this package.

Each eval module (evals.extraction, evals.reranking, ...) loads its cases with
`load_cases`, turns each case into a `CaseResult`, and ends with
`raise SystemExit(finish(name, results, threshold))`.

Evals make real, paid API calls, so they are never part of `make check`.
"""

import json
import os
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DATASETS_DIR = Path("evals/datasets")
RESULTS_DIR = Path("evals/results")


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    # Numeric metrics for this case, e.g. {"company_correct": 1.0, "recall@5": 0.67}.
    scores: dict[str, float] = field(default_factory=dict)
    # Anything useful for debugging a failure: model output, expected values, etc.
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: int = 0


def require_env(*names: str) -> None:
    """Load .env, then exit with a clear message if any required variable is still missing."""
    load_dotenv()
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}. Set them in .env (see .env.example).")


def load_cases(filename: str, datasets_dir: Path = DATASETS_DIR) -> list[dict[str, Any]]:
    """Load a JSONL dataset; every case must have a unique "id"."""
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate((datasets_dir / filename).read_text().splitlines(), 1):
        if not line.strip():
            continue
        case = json.loads(line)
        if "id" not in case:
            raise ValueError(f"{filename}:{line_no}: case is missing an 'id'")
        cases.append(case)
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{filename}: duplicate case ids")
    return cases


MAX_IDENTICAL_ERRORS = 3


def run_cases(
    cases: Iterable[dict[str, Any]],
    evaluate: Callable[[dict[str, Any]], CaseResult],
    max_identical_errors: int = MAX_IDENTICAL_ERRORS,
) -> list[CaseResult]:
    """Run `evaluate` on every case; an exception fails that case instead of aborting the run.

    If `max_identical_errors` cases in a row fail with the same error (an exhausted quota, a
    bad key), the rest are skipped: every further call would fail the same way and still cost.
    """
    results: list[CaseResult] = []
    for case in cases:
        recent = [r.error for r in results[-max_identical_errors:]]
        if len(recent) == max_identical_errors and recent[0] is not None and len(set(recent)) == 1:
            print(f"\nAborting: the last {max_identical_errors} cases failed with the same error; "
                  "remaining cases are skipped.")
            break
        start = time.perf_counter()
        try:
            result = evaluate(case)
        except Exception as e:  # noqa: BLE001 - one broken case must not hide the rest
            result = CaseResult(case_id=case["id"], passed=False, error=f"{type(e).__name__}: {e}")
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.case_id}" + (f"  ({result.error})" if result.error else ""))
        results.append(result)
    return results


def summarize(results: list[CaseResult]) -> dict[str, Any]:
    """Pass rate, mean of every score key, and latency stats."""
    if not results:
        return {"cases": 0, "pass_rate": 0.0, "scores": {}, "errors": 0}
    score_keys = sorted({k for r in results for k in r.scores})
    latencies = [r.latency_ms for r in results]
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / len(results), 4),
        "errors": sum(r.error is not None for r in results),
        "scores": {
            k: round(statistics.mean(r.scores[k] for r in results if k in r.scores), 4) for k in score_keys
        },
        "latency_ms": {"p50": int(statistics.median(latencies)), "max": max(latencies)},
    }


def write_report(
    name: str, results: list[CaseResult], summary: dict[str, Any], results_dir: Path | None = None
) -> Path:
    # Resolved at call time so tests can redirect reports by patching evals.harness.RESULTS_DIR.
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = results_dir / f"{stamp}_{name}.json"
    path.write_text(json.dumps({"eval": name, "summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    return path


def finish(name: str, results: list[CaseResult], threshold: float, results_dir: Path | None = None) -> int:
    """Print the summary, write the JSON report, and return a process exit code."""
    summary = summarize(results)
    summary["threshold"] = threshold
    report = write_report(name, results, summary, results_dir)

    print(f"\n=== {name}: {summary.get('passed', 0)}/{summary['cases']} passed "
          f"({summary['pass_rate']:.0%}, threshold {threshold:.0%}) ===")
    for key, value in summary["scores"].items():
        print(f"  {key:<28} {value:.4f}")
    print(f"  report: {report}")

    return 0 if summary["cases"] and summary["pass_rate"] >= threshold else 1
