"""
Document parsing with Strategy pattern.

Each parser implements the Parser protocol from core.interfaces.
ParserFactory automatically selects the appropriate parser based on source type.
"""

import logging
import os
import re
from typing import Optional

import requests
from markdownify import markdownify as md

from config.settings import ParsingSettings
from core.exceptions import ConfigurationError, ParsingError
from core.interfaces import Parser, ParseResult
from ingestion.markdown_utils import extract_first_h1
from ingestion.suttacentral import SuttaCentralParser

logger = logging.getLogger(__name__)


# ============================================================================
# CONCRETE PARSER IMPLEMENTATIONS
# ============================================================================


class URLParser:
    """Parser for web URLs using requests + markdownify."""

    URL_PATTERN = re.compile(r"^https?://", re.IGNORECASE)

    def can_parse(self, source: str) -> bool:
        """Check if source is a valid HTTP/HTTPS URL."""
        return bool(self.URL_PATTERN.match(source))

    def parse(self, source: str) -> ParseResult:
        """
        Fetch URL and convert HTML to markdown.

        Args:
            source: HTTP or HTTPS URL

        Returns:
            ParseResult with markdown content

        Raises:
            ParsingError: If fetch or conversion fails
        """
        logger.info("Parsing URL: %s", source)
        try:
            response = requests.get(source, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ParsingError(f"Failed to fetch URL {source}: {e}") from e

        markdown_text = md(response.text, heading_style="ATX")
        if not markdown_text.strip():
            raise ParsingError(f"URL parsing resulted in empty content: {source}")

        return ParseResult(
            content=markdown_text,
            title=extract_first_h1(markdown_text, max_lines=20),
            metadata={
                "source_url": source,
                "content_type": response.headers.get("content-type"),
                "status_code": response.status_code,
            },
        )


class PDFParser:
    """Parser for PDF files using LlamaParse."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PDF parser with LlamaParse API key.

        Args:
            api_key: LlamaParse API key (defaults to parsing settings /
                LLAMAPARSE_API environment variable)

        Raises:
            ConfigurationError: If API key is not provided or found in settings
        """
        self.api_key = api_key or ParsingSettings().llamaparse_api_key
        if not self.api_key:
            raise ConfigurationError(
                "LLAMAPARSE_API key required for PDF parsing. "
                "Set LLAMAPARSE_API environment variable or pass api_key parameter."
            )

    def can_parse(self, source: str) -> bool:
        """Check if source is a PDF file."""
        return source.lower().endswith(".pdf")

    def parse(self, source: str) -> ParseResult:
        """
        Parse PDF file using LlamaParse.

        Args:
            source: Path to PDF file

        Returns:
            ParseResult with markdown content

        Raises:
            ParsingError: If PDF parsing fails or file not found
        """
        if not os.path.exists(source):
            raise ParsingError(f"PDF file not found: {source}")

        logger.info("Parsing PDF: %s", source)
        try:
            # Lazy import: LlamaParse pulls a heavy (and currently fragile) dependency
            # chain, so it must not be imported unless a PDF is actually being parsed.
            from llama_cloud_services import LlamaParse

            parser = LlamaParse(
                api_key=self.api_key,
                parse_mode="parse_page_with_llm",
                high_res_ocr=True,
                adaptive_long_table=True,
                outlined_table_extraction=True,
                output_tables_as_HTML=True,
            )

            result = parser.parse(source)
            markdown_documents = result.get_markdown_documents(split_by_page=True)
        except Exception as e:
            raise ParsingError(f"PDF parsing failed for {source}: {e}") from e

        full_markdown = "\n".join(doc.text for doc in markdown_documents)
        if not full_markdown.strip():
            raise ParsingError(f"PDF parsing resulted in empty content: {source}")

        logger.info("Parsed PDF %s (%s pages)", source, len(markdown_documents))
        return ParseResult(
            content=full_markdown,
            title=extract_first_h1(full_markdown, max_lines=20),
            metadata={"source_file": source, "page_count": len(markdown_documents)},
        )


# ============================================================================
# PARSER FACTORY
# ============================================================================


class ParserFactory:
    """
    Factory for creating appropriate parser based on source type.

    Automatically selects parser using can_parse() checks.
    Parsers are tried in order until one matches.
    """

    def __init__(self, llamaparse_api_key: Optional[str] = None):
        """
        Initialize factory with available parsers.

        Args:
            llamaparse_api_key: API key for PDF parsing (optional, uses env var if not provided)
        """
        self._parsers = [
            SuttaCentralParser(),
            URLParser(),
        ]

        # Only add PDF parser if API key is available
        try:
            self._parsers.append(PDFParser(api_key=llamaparse_api_key))
        except ConfigurationError as e:
            logger.warning("PDF parser not available: %s", e)

    def get_parser(self, source: str) -> Parser:
        """
        Get appropriate parser for source.

        Args:
            source: File path or URL

        Returns:
            Parser instance that can handle this source

        Raises:
            ParsingError: If no parser can handle this source
        """
        for parser in self._parsers:
            if parser.can_parse(source):
                logger.debug("Selected %s for source %s", type(parser).__name__, source)
                return parser

        raise ParsingError(
            f"No parser available for source: {source}. "
            "Supported types: SuttaCentral references (sc:<uid>/<author>) and reading "
            "URLs, HTTP/HTTPS URLs, PDF files (.pdf)"
        )

    def parse(self, source: str) -> ParseResult:
        """
        Convenience method: automatically select parser and parse source.

        Args:
            source: File path or URL

        Returns:
            ParseResult with content and metadata

        Raises:
            ParsingError: If no parser found or parsing fails
        """
        parser = self.get_parser(source)
        return parser.parse(source)
