"""Eval for answer synthesis (verbatim.synthesizer.Synthesizer, real gpt-4o-mini).

Each case gives a question plus hand-written transcript chunks and says what a
correct answer looks like. Deterministic graders decide pass/fail; an optional
LLM judge (--judge) adds a faithfulness score for information only.

Run: make eval-synthesis   (paid: one OpenAI call per case, two with --judge)
"""

import argparse
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from evals import harness
from evals.harness import CaseResult, finish, load_cases, require_env, run_cases
from verbatim.synthesizer import Synthesizer

NAME = "synthesis"
DEFAULT_THRESHOLD = 0.85
DATASETS_DIR = harness.DATASETS_DIR
RESULTS_DIR = harness.RESULTS_DIR

# Must match the fallback sentence prompts/synthesis.txt tells the model to use.
REFUSAL_TEXT = "The provided transcripts do not contain sufficient information to answer this question."
_REFUSAL_RE = re.compile(r"\b(?:do|does)\s+not\s+contain\s+(?:sufficient|enough)\s+information\b", re.IGNORECASE)

EXPECTATIONS = ("answer", "partial", "refuse", "any")


class Citation(NamedTuple):
    company: str
    fy: str
    quarter: str
    page: int


class SynthesizerLike(Protocol):
    def generate_answer(self, question: str, chunks: list[dict[str, Any]]) -> str: ...


Judge = Callable[[str, str, str], dict[str, Any]]


# --- Citation parsing -------------------------------------------------------------

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
# One citation inside a bracket: "HDFC_Bank, FY26, Q2, Page 14" (also "p. 14", "Pages 14-15", or a bare "14").
_CITATION_RE = re.compile(
    r"^\s*(?P<company>[^,]+?)\s*,\s*(?P<fy>FY\s?\d{2})\s*,\s*(?P<quarter>Q\s?[1-4])\s*,\s*"
    r"(?:pages?|pg\.?|p\.?)?\s*(?P<page>\d+)(?:\s*(?:-|–|&|and)\s*(?P<page2>\d+))?\s*$",
    re.IGNORECASE,
)


def normalize_company(name: str) -> str:
    return re.sub(r"[\s_\-]+", "_", name.strip()).lower()


def parse_citations(answer: str) -> tuple[list[Citation], list[str]]:
    """Return (citations, malformed) found in square brackets in the answer.

    Several citations may share one bracket, separated by ";". Bracketed text that
    mentions an FY but doesn't match the citation format is reported as malformed.
    """
    citations: list[Citation] = []
    malformed: list[str] = []
    for bracket in _BRACKET_RE.findall(answer):
        for part in bracket.split(";"):
            m = _CITATION_RE.match(part)
            if m is None:
                if re.search(r"\bFY\s?\d{2}\b", part, re.IGNORECASE):
                    malformed.append(part.strip())
                continue
            company = m.group("company").strip()
            fy = m.group("fy").replace(" ", "").upper()
            quarter = m.group("quarter").replace(" ", "").upper()
            pages = [int(m.group("page"))] + ([int(m.group("page2"))] if m.group("page2") else [])
            citations.extend(Citation(company, fy, quarter, p) for p in pages)
    return citations, malformed


def source_key(company: str, fy: str, quarter: str, page: int) -> tuple[str, str, str, int]:
    return normalize_company(company), fy.upper(), quarter.upper(), int(page)


def chunk_key(chunk: dict[str, Any]) -> tuple[str, str, str, int]:
    return source_key(chunk["company"], chunk["fy"], chunk["quarter"], chunk["page_number"])


def citation_key(c: Citation) -> tuple[str, str, str, int]:
    return source_key(c.company, c.fy, c.quarter, c.page)


# --- Graders --------------------------------------------------------------------------

def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def citations_valid(citations: list[Citation], malformed: list[str], chunks: list[dict[str, Any]]) -> float:
    """1.0 if every citation points at a provided chunk and none are malformed."""
    if malformed:
        return 0.0
    provided = {chunk_key(c) for c in chunks}
    return 1.0 if all(citation_key(c) in provided for c in citations) else 0.0


def required_sources_cited(citations: list[Citation], required: Iterable[list[Any]]) -> float:
    """Fraction of required [company, fy, quarter, page] sources that the answer cites."""
    required_keys = [source_key(*r) for r in required]
    if not required_keys:
        return 1.0
    cited = {citation_key(c) for c in citations}
    return sum(k in cited for k in required_keys) / len(required_keys)


def forbidden_sources_absent(citations: list[Citation], forbidden: Iterable[list[Any]]) -> float:
    forbidden_keys = {source_key(*f) for f in forbidden}
    return 0.0 if any(citation_key(c) in forbidden_keys for c in citations) else 1.0


_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:,\d+)*(?:\.\d+)?")
_PERIOD_RE = re.compile(r"\b(?:FY\s?\d{2,4}|Q\s?[1-4]|H[12])\b", re.IGNORECASE)
_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def extract_numbers(text: str) -> set[float]:
    """Numeric values in text, ignoring citations, period labels (FY26, Q2) and list markers.

    Commas are dropped (so "₹1,200 crore", "1200" and Indian "25,28,000" all compare
    by value) and "1.50 per cent" equals "1.5%".
    """
    text = _BRACKET_RE.sub(" ", text)
    text = _PERIOD_RE.sub(" ", text)
    text = _LIST_MARKER_RE.sub("", text)
    return {float(n.replace(",", "")) for n in _NUMBER_RE.findall(text)}


def allowed_numbers(question: str, chunks: list[dict[str, Any]]) -> set[float]:
    """Numbers an answer may legitimately contain: anything in the chunks or the question,
    page numbers, and calendar years implied by FY labels (FY26 -> 2026)."""
    allowed = extract_numbers(question)
    for chunk in chunks:
        allowed |= extract_numbers(chunk["content"])
        allowed.add(float(chunk["page_number"]))
    for label in re.findall(r"FY\s?(\d{2})\b", question + " " + " ".join(c["fy"] for c in chunks), re.IGNORECASE):
        allowed.add(2000.0 + int(label))
    return allowed


def ungrounded_numbers(answer: str, question: str, chunks: list[dict[str, Any]]) -> list[float]:
    return sorted(extract_numbers(answer) - allowed_numbers(question, chunks))


def keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lowered = answer.lower()
    return sum(k.lower() in lowered for k in keywords) / len(keywords)


def forbidden_absent(answer: str, forbidden: list[str]) -> float:
    lowered = answer.lower()
    return 0.0 if any(f.lower() in lowered for f in forbidden) else 1.0


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Same layout Synthesizer uses, so the judge sees what the model saw."""
    return "\n\n".join(
        f"Source: {c['company']} | {c['fy']} {c['quarter']} | Page {c['page_number']}\nContent: {c['content']}"
        for c in chunks
    )


# --- Optional LLM judge ---------------------------------------------------------------

JUDGE_PROMPT_PATH = Path("prompts/judge_faithfulness.txt")


def make_judge(llm: Any) -> Judge:
    chain = ChatPromptTemplate.from_messages([("system", JUDGE_PROMPT_PATH.read_text())]) | llm | JsonOutputParser()

    def judge(question: str, context: str, answer: str) -> dict[str, Any]:
        verdict: dict[str, Any] = chain.invoke({"question": question, "context": context, "answer": answer})
        return verdict

    return judge


# --- Case evaluation ------------------------------------------------------------------

def evaluate_case(case: dict[str, Any], synthesizer: SynthesizerLike, judge: Judge | None = None) -> CaseResult:
    question: str = case["question"]
    chunks: list[dict[str, Any]] = case["chunks"]
    expect: str = case.get("expect", "answer")
    if expect not in EXPECTATIONS:
        raise ValueError(f"unknown expect {expect!r}")

    answer = synthesizer.generate_answer(question, chunks)
    citations, malformed = parse_citations(answer)
    ungrounded = ungrounded_numbers(answer, question, chunks)
    refused = is_refusal(answer)

    scores: dict[str, float] = {
        "citations_valid": citations_valid(citations, malformed, chunks),
        "numbers_grounded": 0.0 if ungrounded else 1.0,
        "forbidden_absent": forbidden_absent(answer, case.get("forbidden", [])),
    }
    if case.get("forbidden_sources"):
        scores["forbidden_sources_absent"] = forbidden_sources_absent(citations, case["forbidden_sources"])
    if expect in ("answer", "partial"):
        scores["citation_present"] = 1.0 if citations else 0.0
        scores["required_sources_cited"] = required_sources_cited(citations, case.get("required_sources", []))
        if case.get("keywords"):
            scores["keyword_recall"] = keyword_recall(answer, case["keywords"])
    if expect == "answer":
        scores["refusal_correct"] = 0.0 if refused else 1.0
    if expect == "refuse":
        scores["refusal_correct"] = 1.0 if refused and not citations else 0.0

    passed = all(scores[k] == 1.0 for k in scores if k != "keyword_recall") and scores.get("keyword_recall", 1.0) >= 0.5

    details: dict[str, Any] = {
        "flow": case.get("flow"),
        "expect": expect,
        "answer": answer,
        "citations": [list(c) for c in citations],
        "malformed_citations": malformed,
        "ungrounded_numbers": ungrounded,
    }

    if judge is not None:
        try:
            verdict = judge(question, format_context(chunks), answer)
            scores["judge_faithful"] = 1.0 if verdict.get("faithful") is True else 0.0
            details["judge"] = verdict
        except Exception as e:  # noqa: BLE001 - the judge is informational; never fail a case on it
            details["judge_error"] = f"{type(e).__name__}: {e}"

    return CaseResult(case_id=case["id"], passed=passed, scores=scores, details=details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eval cited answer synthesis (paid: OpenAI).")
    parser.add_argument("--judge", action="store_true", help="also score faithfulness with an LLM judge (extra cost)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="minimum pass rate")
    args = parser.parse_args(argv)

    require_env("OPENAI_API_KEY")
    synthesizer = Synthesizer()
    judge = make_judge(ChatOpenAI(model="gpt-4o-mini", temperature=0)) if args.judge else None

    cases = load_cases("synthesis.jsonl", DATASETS_DIR)
    print(f"Running {len(cases)} synthesis cases" + (" with LLM judge" if judge else ""))
    results = run_cases(cases, lambda case: evaluate_case(case, synthesizer, judge))
    return finish(NAME, results, args.threshold, results_dir=RESULTS_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
