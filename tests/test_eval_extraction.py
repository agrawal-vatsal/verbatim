import json
import re
from pathlib import Path
from typing import Any

import pytest

from evals import extraction, harness
from evals.harness import load_cases
from tests.conftest import RecordingChatModel, run_main

CASES = load_cases("extraction.jsonl")


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "c1",
        "flow": "full",
        "question": "What did HDFC Bank say about NIM in Q2 FY26?",
        "expected": {"company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2"},
        "search_query_keywords": [["margin", "nim"]],
    }
    case.update(overrides)
    return case


def _output(**overrides: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "company": "HDFC_Bank",
        "fy": "FY26",
        "quarter": "Q2",
        "search_query": "net interest margin NIM outlook profitability",
    }
    out.update(overrides)
    return out


def _ideal_output(case: dict[str, Any]) -> dict[str, Any]:
    """What a perfect model would return for a dataset case."""
    out: dict[str, Any] = {"company": None, "fy": None, "quarter": None}
    for name, value in case["expected"].items():
        out[name] = value[0] if isinstance(value, list) else value
    groups = case.get("search_query_keywords", [])
    keywords = [g if isinstance(g, str) else g[0] for g in groups]
    out["search_query"] = " ".join(keywords) or "management commentary outlook"
    return out


# --- field_matches ---------------------------------------------------------------

@pytest.mark.parametrize(
    ("actual", "expected", "ok"),
    [
        ("HDFC_Bank", "HDFC_Bank", True),
        ("hdfc_bank", "HDFC_Bank", False),
        (None, None, True),
        ("null", None, False),  # the string "null" would become a literal DB filter
        ("ICICI_Bank", ["HDFC_Bank", "ICICI_Bank", None], True),
        (None, ["HDFC_Bank", None], True),
        ("Infosys", ["HDFC_Bank", "ICICI_Bank"], False),
    ],
)
def test_field_matches(actual: Any, expected: Any, ok: bool) -> None:
    assert extraction.field_matches(actual, expected) is ok


# --- keyword_recall ------------------------------------------------------------------

def test_keyword_recall() -> None:
    query = "Net Interest Margin outlook and deposit repricing"
    assert extraction.keyword_recall(query, []) == 1.0
    assert extraction.keyword_recall(query, ["margin"]) == 1.0
    assert extraction.keyword_recall(query, [["nim", "MARGIN"]]) == 1.0
    assert extraction.keyword_recall(query, ["margin", "attrition"]) == 0.5
    assert extraction.keyword_recall(query, [["tcv", "deal"]]) == 0.0


# --- query_leaks -----------------------------------------------------------------------

def test_query_leaks_flags_periods_company_words_and_extras() -> None:
    leaks = extraction.query_leaks("HDFC margins Q3 FY26 fy2025 last quarter", "HDFC_Bank", ["last quarter"])
    assert leaks == ["Q3", "FY26", "fy2025", "HDFC", "last quarter"]


def test_query_leaks_ignores_generic_name_words_and_substrings() -> None:
    # "bank"/"and" are generic; "Larsen" must match as a whole word only; "Q5" isn't a quarter.
    assert extraction.query_leaks("banking and lending Q5 larsenite", "Larsen_and_Toubro") == []
    assert extraction.query_leaks("deposit growth", None) == []


# --- grade ---------------------------------------------------------------------------------

def test_grade_passes_correct_output() -> None:
    result = extraction.grade(_case(), _output())
    assert result.passed
    assert result.scores == {
        "json_valid": 1.0,
        "schema_valid": 1.0,
        "company_correct": 1.0,
        "fy_correct": 1.0,
        "quarter_correct": 1.0,
        "search_query_nonempty": 1.0,
        "search_query_keyword_recall": 1.0,
        "search_query_clean": 1.0,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("company", "ICICI_Bank"), ("company", None), ("fy", "FY25"), ("fy", "null"), ("quarter", "Q3")],
)
def test_grade_fails_on_any_wrong_field(field: str, value: Any) -> None:
    result = extraction.grade(_case(), _output(**{field: value}))
    assert not result.passed
    assert result.scores[f"{field}_correct"] == 0.0


def test_grade_only_checks_expected_fields() -> None:
    result = extraction.grade(_case(expected={"company": "HDFC_Bank"}), _output(fy="FY99", quarter="Q4"))
    assert result.passed
    assert "fy_correct" not in result.scores and "quarter_correct" not in result.scores


@pytest.mark.parametrize("output", [["not", "a", "dict"], {"company": "HDFC_Bank", "fy": "FY26"}, None])
def test_grade_fails_on_bad_schema(output: Any) -> None:
    result = extraction.grade(_case(), output)
    assert not result.passed
    assert result.scores == {"json_valid": 1.0, "schema_valid": 0.0}


def test_grade_requires_search_query_when_company_found() -> None:
    result = extraction.grade(_case(), _output(search_query="   "))
    assert not result.passed
    assert result.scores["search_query_nonempty"] == 0.0


@pytest.mark.parametrize("search_query", ["", None])
def test_grade_allows_empty_search_query_without_company(search_query: Any) -> None:
    case = _case(expected={"company": None}, search_query_keywords=[])
    result = extraction.grade(case, _output(company=None, search_query=search_query))
    assert result.passed
    assert "search_query_keyword_recall" not in result.scores


def test_grade_treats_string_null_company_as_not_found() -> None:
    case = _case(expected={"company": [None, "null"]})
    result = extraction.grade(case, _output(company="null", search_query=""))
    assert result.passed


def test_grade_reports_leaks_and_low_keyword_recall_without_failing() -> None:
    case = _case(forbidden_in_query=["outlook"])
    result = extraction.grade(case, _output(search_query="HDFC deposits Q2 FY26 outlook"))
    assert result.passed  # fields are right; query quality is reported, not gated
    assert result.scores["search_query_keyword_recall"] == 0.0
    assert result.scores["search_query_clean"] == 0.0
    assert result.details["search_query_leaks"] == ["Q2", "FY26", "HDFC", "outlook"]


# --- evaluate_case through the real QueryProcessor (fake chat model) --------------------------------

def _patch_chat(monkeypatch: pytest.MonkeyPatch, responses: list[str]) -> RecordingChatModel:
    model = RecordingChatModel(responses=responses)
    monkeypatch.setattr("verbatim.processor.ChatOpenAI", lambda **_: model)
    return model


def test_evaluate_case_with_real_prompt_and_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _patch_chat(monkeypatch, [json.dumps(_output())])
    result = extraction.evaluate_case(_case(), extraction.build_processor())

    assert result.passed
    system, user = model.seen[0]
    assert "VALID COMPANIES: HDFC_Bank, ICICI_Bank, Infosys, Larsen_and_Toubro" in system.content
    assert user.content == _case()["question"]


def test_evaluate_case_scores_unparseable_output_as_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_chat(monkeypatch, ["Sure! The company is HDFC Bank."])
    result = extraction.evaluate_case(_case(), extraction.build_processor())

    assert not result.passed
    assert result.scores == {"json_valid": 0.0, "schema_valid": 0.0}
    assert result.error is not None and "OutputParserException" in result.error


def test_stub_company_db_returns_a_copy() -> None:
    db = extraction.StubCompanyDB()
    db.get_unique_companies().append("Mutated")
    assert db.get_unique_companies() == extraction.COMPANIES


# --- main ----------------------------------------------------------------------------------

@pytest.fixture
def reports_to(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    real_finish = harness.finish

    def finish(name: str, results: list[harness.CaseResult], threshold: float) -> int:
        return real_finish(name, results, threshold, results_dir=tmp_path)

    monkeypatch.setattr(extraction, "finish", finish)
    monkeypatch.setattr(harness, "finish", finish)  # for run_main, which re-imports it
    return tmp_path


def test_main_passes_with_perfect_model(
    monkeypatch: pytest.MonkeyPatch, reports_to: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_chat(monkeypatch, [json.dumps(_ideal_output(c)) for c in CASES])

    assert extraction.main() == 0

    (report_path,) = reports_to.glob("*_extraction.json")
    report = json.loads(report_path.read_text())
    assert report["summary"]["cases"] == len(CASES)
    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["scores"]["search_query_clean"] == 1.0
    assert f"{len(CASES)}/{len(CASES)} passed" in capsys.readouterr().out


def test_main_fails_below_threshold(monkeypatch: pytest.MonkeyPatch, reports_to: Path) -> None:
    wrong = json.dumps({"company": "Infosys", "fy": "FY99", "quarter": "Q4", "search_query": "x"})
    _patch_chat(monkeypatch, [wrong])
    assert extraction.main() == 1


def test_main_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        extraction.main()


def test_runs_as_module(monkeypatch: pytest.MonkeyPatch, reports_to: Path) -> None:
    _patch_chat(monkeypatch, [json.dumps(_ideal_output(c)) for c in CASES])
    with pytest.raises(SystemExit) as exit_info:
        run_main("evals.extraction")
    assert exit_info.value.code == 0


# --- dataset ---------------------------------------------------------------------------------

def test_dataset_schema() -> None:
    assert len(CASES) >= 30
    allowed_company = set(extraction.COMPANIES) | {None}
    for case in CASES:
        assert {"id", "flow", "question", "expected"} <= case.keys(), case["id"]
        assert case["question"].strip(), case["id"]
        expected = case["expected"]
        assert expected and set(expected) <= {"company", "fy", "quarter"}, case["id"]
        for value in expected.get("company", []) if isinstance(expected.get("company"), list) else [expected.get("company")]:
            assert value in allowed_company, case["id"]
        assert expected.get("fy") is None or re.fullmatch(r"FY\d{2}", expected["fy"]), case["id"]
        assert expected.get("quarter") is None or re.fullmatch(r"Q[1-4]", expected["quarter"]), case["id"]
        for group in case.get("search_query_keywords", []):
            assert isinstance(group, str) or (isinstance(group, list) and all(isinstance(g, str) for g in group))


def test_dataset_covers_every_flow() -> None:
    flows = {c["flow"] for c in CASES}
    assert flows >= {
        "full", "phrasing", "abbreviation", "typo", "fy_only", "quarter_only", "no_company",
        "not_in_list", "multi_company", "relative_time", "injection", "chitchat",
    }
