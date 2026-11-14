import json
from pathlib import Path
from typing import Any

import pytest

from evals.harness import CaseResult, finish, load_cases, require_env, run_cases, summarize


def test_require_env_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "x")
    require_env("A_KEY")


def test_require_env_lists_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setenv("B_KEY", "")
    with pytest.raises(SystemExit, match="A_KEY, B_KEY"):
        require_env("A_KEY", "B_KEY")


def test_load_cases_skips_blank_lines(tmp_path: Path) -> None:
    (tmp_path / "d.jsonl").write_text('{"id": "a", "x": 1}\n\n{"id": "b"}\n')
    assert load_cases("d.jsonl", tmp_path) == [{"id": "a", "x": 1}, {"id": "b"}]


@pytest.mark.parametrize(
    ("content", "error"),
    [('{"x": 1}\n', "d.jsonl:1: case is missing an 'id'"), ('{"id": "a"}\n{"id": "a"}\n', "duplicate case ids")],
)
def test_load_cases_validates_ids(tmp_path: Path, content: str, error: str) -> None:
    (tmp_path / "d.jsonl").write_text(content)
    with pytest.raises(ValueError, match=error):
        load_cases("d.jsonl", tmp_path)


def test_run_cases_records_results_and_isolates_errors(capsys: pytest.CaptureFixture[str]) -> None:
    def evaluate(case: dict[str, Any]) -> CaseResult:
        if case["id"] == "boom":
            raise KeyError("missing")
        return CaseResult(case_id=case["id"], passed=case["id"] == "good", scores={"s": 1.0})

    results = run_cases([{"id": "good"}, {"id": "bad"}, {"id": "boom"}], evaluate)

    assert [(r.case_id, r.passed) for r in results] == [("good", True), ("bad", False), ("boom", False)]
    assert results[2].error == "KeyError: 'missing'"
    assert all(r.latency_ms >= 0 for r in results)
    out = capsys.readouterr().out
    assert "[PASS] good" in out and "[FAIL] bad" in out and "[FAIL] boom  (KeyError: 'missing')" in out


def test_summarize_means_only_over_cases_with_that_score() -> None:
    results = [
        CaseResult("a", True, {"acc": 1.0, "recall": 0.5}, latency_ms=10),
        CaseResult("b", False, {"acc": 0.0}, latency_ms=30),
        CaseResult("c", False, error="x", latency_ms=20),
    ]
    summary = summarize(results)
    assert summary["cases"] == 3
    assert summary["passed"] == 1
    assert summary["pass_rate"] == pytest.approx(0.3333)
    assert summary["errors"] == 1
    assert summary["scores"] == {"acc": 0.5, "recall": 0.5}
    assert summary["latency_ms"] == {"p50": 20, "max": 30}


def test_summarize_empty() -> None:
    assert summarize([]) == {"cases": 0, "pass_rate": 0.0, "scores": {}, "errors": 0}


@pytest.mark.parametrize(("threshold", "exit_code"), [(0.5, 0), (0.51, 1)])
def test_finish_writes_report_and_applies_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], threshold: float, exit_code: int
) -> None:
    results = [CaseResult("a", True, {"acc": 1.0}), CaseResult("b", False, {"acc": 0.0}, details={"why": "x"})]

    assert finish("demo", results, threshold, results_dir=tmp_path) == exit_code

    (report,) = tmp_path.glob("*_demo.json")
    data = json.loads(report.read_text())
    assert data["eval"] == "demo"
    assert data["summary"]["threshold"] == threshold
    assert data["results"][1]["details"] == {"why": "x"}
    out = capsys.readouterr().out
    assert "=== demo: 1/2 passed (50%" in out
    assert "acc" in out


def test_finish_with_no_cases_fails(tmp_path: Path) -> None:
    assert finish("empty", [], 0.0, results_dir=tmp_path) == 1


def test_finish_default_results_dir_is_resolved_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.harness.RESULTS_DIR", tmp_path / "out")
    finish("late", [CaseResult("a", True)], 0.5)
    assert len(list((tmp_path / "out").glob("*_late.json"))) == 1


def test_run_cases_aborts_after_identical_errors(capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    def evaluate(case: dict[str, Any]) -> CaseResult:
        calls.append(case["id"])
        raise RuntimeError("429 no credits")

    results = run_cases([{"id": str(i)} for i in range(10)], evaluate)

    assert calls == ["0", "1", "2"]
    assert len(results) == 3
    assert "Aborting: the last 3 cases failed with the same error" in capsys.readouterr().out


def test_run_cases_keeps_going_when_errors_differ_or_cases_pass() -> None:
    outcomes = iter([RuntimeError("a"), RuntimeError("b"), RuntimeError("a"), None, RuntimeError("a"), None])

    def evaluate(case: dict[str, Any]) -> CaseResult:
        outcome = next(outcomes)
        if outcome:
            raise outcome
        return CaseResult(case["id"], passed=False)

    assert len(run_cases([{"id": str(i)} for i in range(6)], evaluate)) == 6
