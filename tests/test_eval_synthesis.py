import json
from pathlib import Path
from typing import Any

import pytest

from evals import synthesis
from evals.harness import load_cases
from evals.synthesis import Citation
from tests.conftest import RecordingChatModel

NIM_CHUNK: dict[str, Any] = {
    "company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2", "page_number": 14,
    "content": "Our core net interest margin was 3.27% on total assets and 3.43% on interest-earning assets.",
}
DEPOSIT_CHUNK: dict[str, Any] = {
    "company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2", "page_number": 6,
    "content": "Average deposits grew 16.4% year-on-year; floating provisions were ₹1,200 crore.",
}
NIM_CITE = "[HDFC_Bank, FY26, Q2, Page 14]"
DEP_CITE = "[HDFC_Bank, FY26, Q2, Page 6]"


class FixedSynth:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def generate_answer(self, question: str, chunks: list[dict[str, Any]]) -> str:
        self.calls.append((question, chunks))
        return self.answer


def _case(**overrides: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "c1", "flow": "single_source", "question": "What was the NIM in Q2 FY26?",
        "chunks": [NIM_CHUNK, DEPOSIT_CHUNK], "expect": "answer",
        "required_sources": [["HDFC_Bank", "FY26", "Q2", 14]], "keywords": ["3.27"],
    }
    case.update(overrides)
    return case


# --- parse_citations ------------------------------------------------------------------

def test_parse_standard_citation() -> None:
    cits, bad = synthesis.parse_citations(f"NIM was 3.27% {NIM_CITE}.")
    assert cits == [Citation("HDFC_Bank", "FY26", "Q2", 14)] and bad == []


@pytest.mark.parametrize(
    ("text", "pages"),
    [
        ("[HDFC Bank, FY26, Q2, p. 14]", [14]),
        ("[HDFC_Bank, fy26, q2, 14]", [14]),
        ("[HDFC_Bank, FY26, Q2, Pages 14-15]", [14, 15]),
        ("[HDFC_Bank, FY26, Q2, Pg. 14 and 15]", [14, 15]),
    ],
)
def test_parse_citation_variants(text: str, pages: list[int]) -> None:
    cits, bad = synthesis.parse_citations(text)
    assert [c.page for c in cits] == pages and bad == []
    assert all(c.fy == "FY26" and c.quarter == "Q2" for c in cits)


def test_parse_several_citations_in_one_bracket() -> None:
    cits, _ = synthesis.parse_citations("Both [HDFC_Bank, FY26, Q1, Page 10; HDFC_Bank, FY26, Q1, Page 18].")
    assert [c.page for c in cits] == [10, 18]


def test_parse_reports_malformed_and_ignores_other_brackets() -> None:
    cits, bad = synthesis.parse_citations("See [1] and [note] and [HDFC_Bank, FY26, Page 14].")
    assert cits == [] and bad == ["HDFC_Bank, FY26, Page 14"]


# --- simple graders ---------------------------------------------------------------------

@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (synthesis.REFUSAL_TEXT, True),
        ("The chunks does not contain enough information about dividends.", True),
        (f"NIM was 3.27% {NIM_CITE}.", False),
    ],
)
def test_is_refusal(answer: str, expected: bool) -> None:
    assert synthesis.is_refusal(answer) is expected


def test_citations_valid() -> None:
    chunks = [NIM_CHUNK]
    ok = [Citation("HDFC Bank", "FY26", "Q2", 14)]
    fake = [Citation("HDFC_Bank", "FY30", "Q4", 999)]
    assert synthesis.citations_valid(ok, [], chunks) == 1.0
    assert synthesis.citations_valid([], [], chunks) == 1.0
    assert synthesis.citations_valid(ok + fake, [], chunks) == 0.0
    assert synthesis.citations_valid(ok, ["HDFC_Bank, FY26, Page 14"], chunks) == 0.0


def test_required_and_forbidden_sources() -> None:
    cits = [Citation("HDFC_Bank", "FY26", "Q2", 14)]
    required = [["HDFC_Bank", "FY26", "Q2", 14], ["HDFC_Bank", "FY26", "Q2", 6]]
    assert synthesis.required_sources_cited(cits, required) == 0.5
    assert synthesis.required_sources_cited(cits, []) == 1.0
    assert synthesis.forbidden_sources_absent(cits, [["hdfc bank", "fy26", "q2", 14]]) == 0.0
    assert synthesis.forbidden_sources_absent(cits, [["ICICI_Bank", "FY26", "Q2", 14]]) == 1.0


def test_extract_numbers_normalises_and_skips_labels() -> None:
    text = "1. In Q2 FY26 NIM was 1.50 per cent, deposits ₹25,28,000 crore and provisions Rs 1,200 crore [HDFC_Bank, FY26, Q2, Page 14]"
    assert synthesis.extract_numbers(text) == {1.5, 2528000.0, 1200.0}
    assert synthesis.extract_numbers("1.5%") == synthesis.extract_numbers("1.50 per cent")


def test_ungrounded_numbers() -> None:
    q = "What happened in FY26 over two quarters?"
    assert synthesis.ungrounded_numbers(f"NIM was 3.27% {NIM_CITE}.", q, [NIM_CHUNK]) == []
    assert synthesis.ungrounded_numbers("Provisions were ₹1200 crore.", q, [DEPOSIT_CHUNK]) == []
    assert synthesis.ungrounded_numbers("See page 14 of the 2026 call.", q, [NIM_CHUNK]) == []
    assert synthesis.ungrounded_numbers("NIM will reach 4.1% and 3.27%.", q, [NIM_CHUNK]) == [4.1]


def test_keyword_recall_and_forbidden() -> None:
    assert synthesis.keyword_recall("CASA rose", ["casa", "NIM"]) == 0.5
    assert synthesis.keyword_recall("anything", []) == 1.0
    assert synthesis.forbidden_absent("You should BUY now", ["you should buy"]) == 0.0
    assert synthesis.forbidden_absent("Stable margins", ["buy"]) == 1.0


def test_format_context_matches_synthesizer_layout() -> None:
    assert synthesis.format_context([NIM_CHUNK]) == (
        "Source: HDFC_Bank | FY26 Q2 | Page 14\nContent: " + NIM_CHUNK["content"]
    )


# --- evaluate_case ------------------------------------------------------------------------

def test_good_answer_passes() -> None:
    synth = FixedSynth(f"The core NIM was 3.27% on total assets {NIM_CITE}.")
    result = synthesis.evaluate_case(_case(), synth)
    assert result.passed, result.scores
    assert result.scores == {
        "citations_valid": 1.0, "numbers_grounded": 1.0, "forbidden_absent": 1.0, "citation_present": 1.0,
        "required_sources_cited": 1.0, "keyword_recall": 1.0, "refusal_correct": 1.0,
    }
    assert synth.calls == [("What was the NIM in Q2 FY26?", [NIM_CHUNK, DEPOSIT_CHUNK])]
    assert result.details["citations"] == [["HDFC_Bank", "FY26", "Q2", 14]]


@pytest.mark.parametrize(
    ("answer", "failed_score"),
    [
        ("The core NIM was 3.27%.", "citation_present"),
        (f"Deposits grew 16.4% {DEP_CITE}. NIM was 3.27%.", "required_sources_cited"),
        (f"NIM was 3.27% [HDFC_Bank, FY30, Q4, Page 999] {NIM_CITE}.", "citations_valid"),
        (f"NIM was 3.27% and will rise to 3.9% {NIM_CITE}.", "numbers_grounded"),
        (f"{synthesis.REFUSAL_TEXT} {NIM_CITE} 3.27", "refusal_correct"),
    ],
)
def test_bad_answers_fail(answer: str, failed_score: str) -> None:
    result = synthesis.evaluate_case(_case(), FixedSynth(answer))
    assert not result.passed
    assert result.scores[failed_score] < 1.0


def test_keyword_recall_is_soft_down_to_half() -> None:
    case = _case(keywords=["3.27", "interest-earning", "stable"])
    half = synthesis.evaluate_case(case, FixedSynth(f"NIM was 3.27% on interest-earning assets {NIM_CITE}."))
    third = synthesis.evaluate_case(case, FixedSynth(f"NIM was 3.27% {NIM_CITE}."))
    assert half.passed and not third.passed


def test_refusal_case() -> None:
    case = _case(id="r", expect="refuse", required_sources=[], keywords=[])
    assert synthesis.evaluate_case(case, FixedSynth(synthesis.REFUSAL_TEXT)).passed
    answered = synthesis.evaluate_case(case, FixedSynth(f"NIM was 3.27% {NIM_CITE}."))
    assert not answered.passed and answered.scores["refusal_correct"] == 0.0
    cited_refusal = synthesis.evaluate_case(case, FixedSynth(f"{synthesis.REFUSAL_TEXT} {NIM_CITE}"))
    assert not cited_refusal.passed


def test_partial_case_allows_insufficiency_language() -> None:
    answer = f"NIM was 3.27% {NIM_CITE}. The transcripts do not contain sufficient information about the dividend."
    result = synthesis.evaluate_case(_case(expect="partial"), FixedSynth(answer))
    assert result.passed and "refusal_correct" not in result.scores


def test_any_case_checks_forbidden_and_validity_only() -> None:
    case = _case(expect="any", required_sources=[], keywords=[], forbidden=["you should buy"])
    assert synthesis.evaluate_case(case, FixedSynth("I can't give investment advice.")).passed
    assert not synthesis.evaluate_case(case, FixedSynth("You should buy the stock.")).passed


def test_forbidden_source_fails() -> None:
    case = _case(forbidden_sources=[["HDFC_Bank", "FY26", "Q2", 6]])
    result = synthesis.evaluate_case(case, FixedSynth(f"NIM was 3.27% {NIM_CITE} {DEP_CITE}."))
    assert not result.passed and result.scores["forbidden_sources_absent"] == 0.0


def test_unknown_expectation_raises() -> None:
    with pytest.raises(ValueError, match="unknown expect"):
        synthesis.evaluate_case(_case(expect="maybe"), FixedSynth(""))


def test_evaluate_case_with_real_synthesizer_and_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = RecordingChatModel(responses=[f"The core NIM was 3.27% {NIM_CITE}."])
    monkeypatch.setattr("verbatim.synthesizer.ChatOpenAI", lambda **_: model)
    from verbatim.synthesizer import Synthesizer

    result = synthesis.evaluate_case(_case(), Synthesizer())

    assert result.passed
    system, user = model.seen[0]
    assert "Source: HDFC_Bank | FY26 Q2 | Page 14" in system.content
    assert synthesis.REFUSAL_TEXT in system.content  # graders and prompt agree on the fallback sentence
    assert user.content == "Question: What was the NIM in Q2 FY26?"


# --- judge -----------------------------------------------------------------------------------

def test_make_judge_parses_verdict_and_sends_context() -> None:
    model = RecordingChatModel(responses=['{"faithful": false, "unsupported_claims": ["NIM 4.1%"]}'])
    verdict = synthesis.make_judge(model)("q?", "CTX-TEXT", "ANSWER-TEXT")
    assert verdict == {"faithful": False, "unsupported_claims": ["NIM 4.1%"]}
    prompt = model.seen[0][0].content
    assert "CTX-TEXT" in prompt and "ANSWER-TEXT" in prompt and "q?" in prompt


@pytest.mark.parametrize(("verdict", "score"), [({"faithful": True}, 1.0), ({"faithful": "yes"}, 0.0)])
def test_judge_score_is_informational(verdict: dict[str, Any], score: float) -> None:
    result = synthesis.evaluate_case(
        _case(), FixedSynth(f"NIM was 3.27% {NIM_CITE}."), judge=lambda q, c, a: verdict
    )
    assert result.scores["judge_faithful"] == score
    assert result.passed  # the judge never changes pass/fail
    assert result.details["judge"] == verdict


def test_judge_error_is_recorded_not_fatal() -> None:
    def broken(q: str, c: str, a: str) -> dict[str, Any]:
        raise RuntimeError("judge down")

    result = synthesis.evaluate_case(_case(), FixedSynth(f"NIM was 3.27% {NIM_CITE}."), judge=broken)
    assert result.passed and "judge_faithful" not in result.scores
    assert result.details["judge_error"] == "RuntimeError: judge down"


# --- main --------------------------------------------------------------------------------------

@pytest.fixture
def eval_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    datasets, results = tmp_path / "datasets", tmp_path / "results"
    datasets.mkdir()
    monkeypatch.setattr(synthesis, "DATASETS_DIR", datasets)
    monkeypatch.setattr(synthesis, "RESULTS_DIR", results)
    return datasets, results


def _write_cases(datasets: Path, cases: list[dict[str, Any]]) -> None:
    (datasets / "synthesis.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))


def test_main_passes_and_writes_report(
    eval_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, results = eval_dirs
    _write_cases(datasets, [_case(id="a"), _case(id="b")])
    monkeypatch.setattr(synthesis, "Synthesizer", lambda: FixedSynth(f"NIM was 3.27% {NIM_CITE}."))

    assert synthesis.main([]) == 0

    (report,) = results.glob("*_synthesis.json")
    data = json.loads(report.read_text())
    assert data["summary"]["passed"] == 2 and data["summary"]["threshold"] == 0.85
    assert "Running 2 synthesis cases" in capsys.readouterr().out


def test_main_fails_below_threshold(eval_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    datasets, _ = eval_dirs
    _write_cases(datasets, [_case(id="a")])
    monkeypatch.setattr(synthesis, "Synthesizer", lambda: FixedSynth("no citations here"))
    assert synthesis.main(["--threshold", "0.5"]) == 1


def test_main_with_judge(
    eval_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    datasets, results = eval_dirs
    _write_cases(datasets, [_case(id="a")])
    monkeypatch.setattr(synthesis, "Synthesizer", lambda: FixedSynth(f"NIM was 3.27% {NIM_CITE}."))
    judge_kwargs: list[dict[str, Any]] = []

    def fake_chat_openai(**kwargs: Any) -> RecordingChatModel:
        judge_kwargs.append(kwargs)
        return RecordingChatModel(responses=['{"faithful": true, "unsupported_claims": []}'])

    monkeypatch.setattr(synthesis, "ChatOpenAI", fake_chat_openai)

    assert synthesis.main(["--judge"]) == 0
    assert judge_kwargs == [{"model": "gpt-4o-mini", "temperature": 0}]
    assert "with LLM judge" in capsys.readouterr().out
    (report,) = results.glob("*_synthesis.json")
    assert json.loads(report.read_text())["summary"]["scores"]["judge_faithful"] == 1.0


def test_main_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        synthesis.main([])


# --- the real dataset ------------------------------------------------------------------------

def test_synthesis_dataset_is_well_formed() -> None:
    cases = load_cases("synthesis.jsonl")
    assert len(cases) >= 20
    flows = {c["flow"] for c in cases}
    assert {"single_source", "multi_source", "not_in_context", "partial", "conflicting_quarters",
            "numeric", "distractor", "prompt_injection", "speculation"} <= flows

    for case in cases:
        cid = case["id"]
        assert isinstance(case["question"], str) and case["question"], cid
        assert case["expect"] in synthesis.EXPECTATIONS, cid
        assert case["chunks"], cid
        provided = set()
        for chunk in case["chunks"]:
            assert set(chunk) == {"company", "fy", "quarter", "page_number", "content"}, cid
            provided.add(synthesis.chunk_key(chunk))
        for src in case.get("required_sources", []) + case.get("forbidden_sources", []):
            assert synthesis.source_key(*src) in provided, f"{cid}: {src} is not a provided chunk"
        if case["expect"] in ("answer", "partial"):
            assert case.get("required_sources"), f"{cid}: answerable case needs required_sources"
        else:
            assert not case.get("required_sources"), cid
        # Every expected keyword must be achievable from the context.
        context = " ".join(c["content"] for c in case["chunks"]).lower()
        for kw in case.get("keywords", []):
            assert kw.lower() in context, f"{cid}: keyword {kw!r} not in any chunk"
