import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.conftest import run_main

BASE = "https://example.test"


def _entry(company: str, path: str) -> dict[str, str]:
    return {"company": company, "fy": "FY26", "quarter": "Q1", "url": f"{BASE}/{path}"}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/ok.pdf":
        assert request.headers["User-Agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, content=b"%PDF-1.4 ok")
    if path == "/redirect.pdf":
        return httpx.Response(302, headers={"Location": f"{BASE}/ok.pdf"})
    if path == "/missing.pdf":
        return httpx.Response(404)
    if path == "/loop.pdf":
        return httpx.Response(302, headers={"Location": f"{BASE}/loop.pdf"})
    raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    real_client = httpx.Client
    client_kwargs: list[dict[str, Any]] = []

    def client(**kwargs: Any) -> httpx.Client:
        client_kwargs.append(kwargs)
        return real_client(transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr("time.sleep", lambda _s: None)
    return client_kwargs


def test_download_transcripts_handles_every_outcome(
    in_tmp_dir: Path, mock_http: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = {"transcripts": [
        _entry("Existing", "ok.pdf"),
        _entry("Fresh", "ok.pdf"),
        _entry("Redirected", "redirect.pdf"),
        _entry("Missing", "missing.pdf"),
        _entry("Looping", "loop.pdf"),
        _entry("Offline", "offline.pdf"),
    ]}
    (in_tmp_dir / "manifest.json").write_text(json.dumps(manifest))
    out_dir = in_tmp_dir / "data" / "raw" / "transcripts"
    out_dir.mkdir(parents=True)
    (out_dir / "Existing_FY26_Q1.pdf").write_bytes(b"already here")

    run_main("scripts.download_data")
    out = capsys.readouterr().out

    assert (out_dir / "Existing_FY26_Q1.pdf").read_bytes() == b"already here"
    assert "Skipping Existing_FY26_Q1.pdf, already exists." in out
    assert (out_dir / "Fresh_FY26_Q1.pdf").read_bytes() == b"%PDF-1.4 ok"
    assert (out_dir / "Redirected_FY26_Q1.pdf").read_bytes() == b"%PDF-1.4 ok"
    assert "Failed to download Missing_FY26_Q1.pdf: 404" in out
    assert "Failed Looping_FY26_Q1.pdf: Too many redirects" in out
    assert "Unexpected error for Offline_FY26_Q1.pdf: connection refused" in out
    for name in ("Missing", "Looping", "Offline"):
        assert not (out_dir / f"{name}_FY26_Q1.pdf").exists()

    assert mock_http[0]["follow_redirects"] is True
    assert mock_http[0]["max_redirects"] == 20


def test_download_creates_output_dir_and_handles_empty_manifest(
    in_tmp_dir: Path, mock_http: list[dict[str, Any]]
) -> None:
    (in_tmp_dir / "manifest.json").write_text("{}")
    run_main("scripts.download_data")
    assert (in_tmp_dir / "data" / "raw" / "transcripts").is_dir()
