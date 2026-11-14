"""QueryProcessor, Synthesizer and LLMReranker construction, with the real prompt files
and output parsers but a fake chat model in place of OpenAI."""

import json
from typing import Any

import pytest

from tests.conftest import RecordingChatModel
from verbatim import processor, reranker, synthesizer


def _install_fake_chat(
    monkeypatch: pytest.MonkeyPatch, module: Any, response: str
) -> tuple[RecordingChatModel, list[dict[str, Any]]]:
    """Replace module.ChatOpenAI; return the fake model and the kwargs it was built with."""
    model = RecordingChatModel(responses=[response])
    created: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> RecordingChatModel:
        created.append(kwargs)
        return model

    monkeypatch.setattr(module, "ChatOpenAI", factory)
    return model, created


class _StubDB:
    def get_unique_companies(self) -> list[str]:
        return ["HDFC_Bank", "Infosys"]


def test_query_processor_extracts_json_and_injects_companies(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"company": "HDFC_Bank", "fy": "FY26", "quarter": None, "search_query": "NIM guidance"}
    model, created = _install_fake_chat(monkeypatch, processor, json.dumps(payload))

    qp = processor.QueryProcessor(_StubDB())  # type: ignore[arg-type]
    result = qp.process_query("What did HDFC say about NIM?")

    assert result == payload
    assert created == [{"model": "gpt-4o-mini", "temperature": 0}]
    system, user = model.seen[0]
    assert "VALID COMPANIES: HDFC_Bank, Infosys" in system.content
    assert user.content == "What did HDFC say about NIM?"


def test_query_processor_parses_fenced_json(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chat(monkeypatch, processor, '```json\n{"company": null}\n```')
    assert processor.QueryProcessor(_StubDB()).process_query("hi") == {"company": None}  # type: ignore[arg-type]


def test_synthesizer_formats_context_and_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model, _ = _install_fake_chat(monkeypatch, synthesizer, "Answer [HDFC_Bank, FY26, Q2, Page 14]")
    chunks = [
        {"company": "HDFC_Bank", "fy": "FY26", "quarter": "Q2", "page_number": 14, "content": "NIM will improve"},
        {"company": "HDFC_Bank", "fy": "FY26", "quarter": "Q1", "page_number": 3, "content": "CD ratio"},
    ]

    answer = synthesizer.Synthesizer().generate_answer("NIM outlook?", chunks)

    assert answer == "Answer [HDFC_Bank, FY26, Q2, Page 14]"
    system, user = model.seen[0]
    assert "Source: HDFC_Bank | FY26 Q2 | Page 14\nContent: NIM will improve" in system.content
    assert "Source: HDFC_Bank | FY26 Q1 | Page 3\nContent: CD ratio" in system.content
    assert user.content == "Question: NIM outlook?"


def test_llm_reranker_init_uses_gemini_key_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    keys: list[str] = []

    def fake_client(api_key: str) -> object:
        keys.append(api_key)
        return object()

    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr("verbatim.reranker.genai.Client", fake_client)

    r = reranker.LLMReranker()

    assert keys == ["gem-key"]
    assert r.model_name == "gemini-2.5-flash-lite"
    assert "{query}" in r.prompt_template and "{chunks}" in r.prompt_template
