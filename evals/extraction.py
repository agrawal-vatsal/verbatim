"""Eval: query extraction (verbatim.processor.QueryProcessor, real gpt-4o-mini).

Each case in evals/datasets/extraction.jsonl has:
  id, flow, question
  expected: subset of {company, fy, quarter}; a value, null, or a list of acceptable values
  search_query_keywords (optional): groups of alternatives; each group needs one hit in search_query
  forbidden_in_query (optional): extra phrases that must not appear in search_query

A case passes when the output is a valid JSON object with all four fields, every
expected field matches, and search_query is non-empty whenever a company is found.
Keyword recall and search_query cleanliness are reported as scores.

Run: make eval-extraction (paid).
"""

import re
from typing import Any, Protocol, cast

from evals.harness import CaseResult, finish, load_cases, require_env, run_cases

NAME = "extraction"
THRESHOLD = 0.9
COMPANIES = ["HDFC_Bank", "ICICI_Bank", "Infosys", "Larsen_and_Toubro"]
FIELDS = ("company", "fy", "quarter", "search_query")
# Words of a canonical company name too generic to flag when they show up in search_query.
_GENERIC_NAME_WORDS = {"bank", "and"}
_PERIOD_PATTERN = re.compile(r"\bFY\s?\d{2,4}\b|\bQ[1-4]\b", re.IGNORECASE)


class Processor(Protocol):
    def process_query(self, user_input: str) -> dict[str, Any]: ...


class StubCompanyDB:
    """Supplies the valid-company list to QueryProcessor without Postgres."""

    def get_unique_companies(self) -> list[str]:
        return list(COMPANIES)


# --- Graders ---------------------------------------------------------------------

def field_matches(actual: Any, expected: Any) -> bool:
    """Exact match; a list in `expected` means any of those values is acceptable.

    The string "null" is NOT treated as null: the pipeline would pass it to the DB
    as a literal filter (fy/quarter), so it is a real defect.
    """
    if isinstance(expected, list):
        return any(field_matches(actual, e) for e in expected)
    return bool(actual == expected)


def keyword_recall(search_query: str, groups: list[str | list[str]]) -> float:
    """Fraction of keyword groups with at least one alternative in search_query (case-insensitive)."""
    if not groups:
        return 1.0
    query = search_query.lower()
    hits = 0
    for group in groups:
        alternatives = [group] if isinstance(group, str) else group
        if any(alt.lower() in query for alt in alternatives):
            hits += 1
    return hits / len(groups)


def query_leaks(search_query: str, company: str | None, extra: list[str] | None = None) -> list[str]:
    """Terms the prompt forbids in search_query: FY/quarter tokens, company name words, extras."""
    leaks = [m.group(0) for m in _PERIOD_PATTERN.finditer(search_query)]
    if company:
        for word in company.split("_"):
            if word.lower() not in _GENERIC_NAME_WORDS and re.search(rf"\b{re.escape(word)}\b", search_query, re.I):
                leaks.append(word)
    for phrase in extra or []:
        if phrase.lower() in search_query.lower():
            leaks.append(phrase)
    return leaks


def grade(case: dict[str, Any], output: Any) -> CaseResult:
    expected: dict[str, Any] = case["expected"]
    scores: dict[str, float] = {"json_valid": 1.0}
    details: dict[str, Any] = {"question": case["question"], "expected": expected, "output": output}

    schema_ok = isinstance(output, dict) and all(f in output for f in FIELDS)
    scores["schema_valid"] = float(schema_ok)
    if not schema_ok:
        return CaseResult(case_id=case["id"], passed=False, scores=scores, details=details)

    out = cast(dict[str, Any], output)
    fields_ok = True
    for name in ("company", "fy", "quarter"):
        if name in expected:
            ok = field_matches(out[name], expected[name])
            scores[f"{name}_correct"] = float(ok)
            fields_ok = fields_ok and ok

    search_query = out["search_query"] if isinstance(out["search_query"], str) else ""
    company_found = out["company"] not in (None, "null", "")
    nonempty_ok = bool(search_query.strip()) or not company_found
    scores["search_query_nonempty"] = float(nonempty_ok)

    groups = case.get("search_query_keywords", [])
    if groups:
        scores["search_query_keyword_recall"] = keyword_recall(search_query, groups)
    leaks = query_leaks(search_query, out["company"] if company_found else None, case.get("forbidden_in_query"))
    scores["search_query_clean"] = float(not leaks)
    if leaks:
        details["search_query_leaks"] = leaks

    return CaseResult(case_id=case["id"], passed=fields_ok and nonempty_ok, scores=scores, details=details)


# --- Runner ----------------------------------------------------------------------

def evaluate_case(case: dict[str, Any], processor: Processor) -> CaseResult:
    try:
        output = processor.process_query(case["question"])
    except Exception as e:  # noqa: BLE001 - a parse failure is a graded outcome, not a crash
        return CaseResult(
            case_id=case["id"],
            passed=False,
            scores={"json_valid": 0.0, "schema_valid": 0.0},
            details={"question": case["question"], "expected": case["expected"]},
            error=f"{type(e).__name__}: {e}",
        )
    return grade(case, output)


def build_processor() -> Processor:
    from verbatim.processor import QueryProcessor

    return QueryProcessor(cast(Any, StubCompanyDB()))


def main() -> int:
    require_env("OPENAI_API_KEY")
    processor = build_processor()
    cases = load_cases(f"{NAME}.jsonl")
    print(f"Running {len(cases)} {NAME} cases...")
    results = run_cases(cases, lambda case: evaluate_case(case, processor))
    return finish(NAME, results, THRESHOLD)


if __name__ == "__main__":
    raise SystemExit(main())
