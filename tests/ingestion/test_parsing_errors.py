"""Parser errors say what went wrong once, without re-wrapping their own message."""

from types import SimpleNamespace

import pytest

from core.exceptions import ParsingError
from ingestion import parsing
from ingestion.parsing import ParserFactory, URLParser


def test_an_empty_page_is_reported_once(monkeypatch):
    page = SimpleNamespace(
        text="<html><body></body></html>",
        headers={},
        status_code=200,
        raise_for_status=lambda: None,
    )
    monkeypatch.setattr(parsing.requests, "get", lambda url, timeout: page)

    with pytest.raises(ParsingError) as excinfo:
        URLParser().parse("https://example.org/empty")

    assert str(excinfo.value) == "URL parsing resulted in empty content: https://example.org/empty"


def test_an_unsupported_source_lists_every_supported_kind():
    with pytest.raises(ParsingError, match="SuttaCentral"):
        ParserFactory(llamaparse_api_key="unused").get_parser("notes.txt")
