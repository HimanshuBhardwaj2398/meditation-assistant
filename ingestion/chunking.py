"""Split a document's markdown into chunks for embedding.

``MarkdownChunker.chunk`` runs four steps:

1. Split on markdown headers, so each chunk starts inside one section.
2. Split sections longer than ``max_size`` characters semantically, at the
   sentence boundaries where a small local embedding model
   (``ChunkingConfig.model``) sees the meaning shift.
3. Merge chunks shorter than ``min_size`` into their neighbours, without
   exceeding ``max_size``.
4. Record each chunk's header paths, level map, title and sizes in its
   metadata (``ingestion.toc`` builds the table of contents from these).

All sizes are in characters.
"""

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from core.exceptions import ChunkingError, ConfigurationError
from ingestion.markdown_utils import extract_first_h1

if TYPE_CHECKING:
    from config.settings import ChunkingSettings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Chunker configuration. Sizes are in characters."""

    max_size: int = 2000
    min_size: int = 700
    max_header_level: int = 6
    enable_semantic: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    tiny_chunk_threshold: int = 50
    model: str = "BAAI/bge-small-en-v1.5"

    def __post_init__(self) -> None:
        if self.max_size <= self.min_size:
            raise ConfigurationError("max_size must be greater than min_size")

    @classmethod
    def from_settings(
        cls,
        chunking: Optional["ChunkingSettings"] = None,
        model: Optional[str] = None,
    ) -> "ChunkingConfig":
        """Build a ChunkingConfig from env-driven settings (CHUNKING_* / EMBEDDING_*).

        This is the bridge that makes the documented environment variables
        actually reach the chunker; without it callers silently run on the
        dataclass defaults above.
        """
        from config.settings import ChunkingSettings, EmbeddingSettings

        settings = chunking or ChunkingSettings()
        return cls(
            max_size=settings.max_size,
            min_size=settings.min_size,
            max_header_level=settings.max_header_level,
            enable_semantic=settings.enable_semantic,
            enable_parallel=settings.enable_parallel,
            max_workers=settings.max_workers,
            tiny_chunk_threshold=settings.tiny_chunk_threshold,
            model=model or EmbeddingSettings().huggingface_model,
        )


@dataclass
class ChunkingStats:
    """Processing statistics."""

    total_chunks: int = 0
    processing_time: float = 0.0
    avg_chunk_size: float = 0.0


_embeddings_by_model: Dict[str, HuggingFaceEmbeddings] = {}
_embeddings_lock = threading.Lock()


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """The local embeddings model, loaded once per process and shared.

    Loading takes seconds, so concurrent chunkers wait on one lock rather
    than each loading their own copy.

    Raises:
        ChunkingError: The model could not be loaded.
    """
    with _embeddings_lock:
        if model_name not in _embeddings_by_model:
            logger.info("Loading embeddings model %s", model_name)
            start = time.perf_counter()
            try:
                _embeddings_by_model[model_name] = HuggingFaceEmbeddings(model_name=model_name)
            except Exception as e:
                raise ChunkingError(f"Failed to load embeddings model {model_name}: {e}") from e
            logger.info("Loaded %s in %.2fs", model_name, time.perf_counter() - start)
        return _embeddings_by_model[model_name]


class MarkdownChunker:
    """Chunks one document's markdown (see the module docstring)."""

    def __init__(
        self,
        text: str,
        config: Optional[ChunkingConfig] = None,
        title: Optional[str] = None,
    ):
        if not text or not text.strip():
            raise ChunkingError("Text cannot be empty")

        self.text = text.strip()
        self.config = config or ChunkingConfig()
        self.title = title or self._extract_title()
        self.embeddings = get_embeddings(self.config.model) if self.config.enable_semantic else None

    def _extract_title(self) -> str:
        """Extract title from first H1 header."""
        return extract_first_h1(self.text, max_lines=10) or "Untitled"

    def _extract_headers(self, text: str) -> Dict[str, str]:
        """Extract current header context from text."""
        headers = {}
        current_headers = [None] * 7  # levels 0-6

        for match in re.finditer(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE):
            level = len(match.group(1))
            header = match.group(2).strip()
            current_headers[level] = header
            # Clear deeper levels
            for i in range(level + 1, 7):
                current_headers[i] = None

        for level in range(1, 7):
            if current_headers[level]:
                headers[f"Header {level}"] = current_headers[level]

        return headers

    def chunk(self) -> Tuple[List[Document], ChunkingStats]:
        """Run the four chunking steps (see the module docstring).

        Raises:
            ChunkingError: A step failed.
        """
        start_time = time.perf_counter()

        try:
            chunks = self._split_by_headers()
            after_headers = len(chunks)
            if self.config.enable_semantic:
                chunks = self._split_oversized_chunks(chunks)
            after_semantic = len(chunks)
            chunks = self._combine_small_chunks(chunks)
            chunks = self._add_final_metadata(chunks)

            processing_time = time.perf_counter() - start_time
            total_words = sum(len(c.page_content.split()) for c in chunks)
            stats = ChunkingStats(
                total_chunks=len(chunks),
                processing_time=processing_time,
                avg_chunk_size=total_words / len(chunks) if chunks else 0,
            )
            logger.info(
                "Chunked %r: %s header sections -> %s after semantic split -> %s chunks in %.2fs",
                self.title,
                after_headers,
                after_semantic,
                len(chunks),
                processing_time,
            )
            return chunks, stats

        except Exception as e:
            raise ChunkingError(f"Chunking failed: {e}") from e

    def _split_by_headers(self) -> List[Document]:
        """Split text by markdown headers."""
        headers_to_split = [
            (f"{'#' * i}", f"Header {i}") for i in range(1, self.config.max_header_level + 1)
        ]

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split, strip_headers=False
        )
        docs = splitter.split_text(self.text)

        # Ensure all docs have metadata (don't filter here, let combine step handle tiny chunks)
        for doc in docs:
            if not doc.metadata:
                doc.metadata = self._extract_headers(doc.page_content)

        return docs

    def _split_oversized_chunks(self, chunks: List[Document]) -> List[Document]:
        """Split chunks longer than max_size semantically, keeping document order."""
        oversized = [chunk for chunk in chunks if len(chunk.page_content) > self.config.max_size]
        if not oversized:
            return chunks

        if self.config.enable_parallel and len(oversized) > 1:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                pieces = list(executor.map(self._semantic_split, oversized))
        else:
            pieces = [self._semantic_split(chunk) for chunk in oversized]

        # _semantic_split never raises and returns at least the chunk itself.
        replacements = iter(pieces)
        ordered: List[Document] = []
        for chunk in chunks:
            if len(chunk.page_content) > self.config.max_size:
                ordered.extend(next(replacements))
            else:
                ordered.append(chunk)
        return ordered

    def _semantic_split(self, chunk: Document) -> List[Document]:
        """Split a single chunk using semantic chunking.

        If the split fails, the oversized chunk is kept whole and marked
        ``semantic_split_failed``, so degraded chunks can be found in the
        stored chunk metadata instead of passing as normal ones.
        """
        try:
            if not self.embeddings:
                return [chunk]

            splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
            docs = splitter.create_documents([chunk.page_content])

            # Preserve original metadata
            for doc in docs:
                doc.metadata.update(chunk.metadata)
                doc.metadata["is_semantic_split"] = True

            return docs if len(docs) > 1 else [chunk]

        except Exception as e:
            logger.warning(
                "Semantic split failed; keeping a %s-character chunk whole: %s",
                len(chunk.page_content),
                e,
            )
            flagged = {**chunk.metadata, "semantic_split_failed": True}
            return [Document(page_content=chunk.page_content, metadata=flagged)]

    def _combine_small_chunks(self, chunks: List[Document]) -> List[Document]:
        """Merge small adjacent chunks, across section boundaries.

        - A tiny chunk (under ``tiny_chunk_threshold``) always joins the next
          one, whatever the size of the result.
        - A chunk of at least ``min_size`` stays as it is.
        - Otherwise the chunk absorbs following ones until it reaches
          ``min_size``, stopping before it would exceed ``max_size``.

        Merged chunks keep every header set they cover in ``all_headers``,
        so the table of contents can recover each section heading.
        """
        merged: List[Document] = []
        start = 0
        while start < len(chunks):
            size = len(chunks[start].page_content)
            if size < self.config.tiny_chunk_threshold and start + 1 < len(chunks):
                end = start + 2
            elif size >= self.config.min_size:
                end = start + 1
            else:
                end = self._end_of_small_run(chunks, start)
            merged.append(self._join(chunks[start:end]))
            start = end
        return merged

    def _end_of_small_run(self, chunks: List[Document], start: int) -> int:
        """Where a merge that starts at a small chunk stops (exclusive index)."""
        size = len(chunks[start].page_content)
        end = start + 1
        while end < len(chunks) and size < self.config.min_size:
            next_size = len(chunks[end].page_content)
            if size + next_size > self.config.max_size:
                break
            size += next_size
            end += 1
        return end

    def _join(self, run: List[Document]) -> Document:
        """One chunk from a run of adjacent chunks (a run of one is returned as is)."""
        if len(run) == 1:
            return run[0]
        metadata = run[0].metadata.copy()
        for chunk in run[1:]:
            metadata = self._merge_metadata(metadata, chunk.metadata)
        metadata["is_combined"] = True
        return Document(
            page_content="\n\n".join(chunk.page_content for chunk in run), metadata=metadata
        )

    def _extract_header_set(self, metadata: Dict) -> Dict[str, str]:
        """Extract just the Header N keys from metadata as a dict."""
        return {k: v for k, v in metadata.items() if k.startswith("Header ")}

    def _build_header_path(self, header_dict: Dict[str, str]) -> Tuple[str, Dict[str, int]]:
        """Build a header path string and level map from a Header N dict.

        Args:
            header_dict: Dict like {"Header 1": "Title", "Header 3": "Section"}

        Returns:
            Tuple of (path_string, level_map):
                path_string: "Title > Section"
                level_map: {"Title": 1, "Section": 3}
        """
        levels = {}
        for key, value in header_dict.items():
            if key.startswith("Header "):
                level = int(key.split()[1])
                levels[level] = value

        if not levels:
            return "", {}

        sorted_levels = sorted(levels.keys())
        path_parts = [levels[level] for level in sorted_levels]
        level_map = {levels[level]: level for level in sorted_levels}

        return " > ".join(path_parts), level_map

    def _merge_metadata(self, meta1: Dict, meta2: Dict) -> Dict:
        """
        Merge metadata from two chunks, accumulating all unique header sets.

        Keeps meta1's Header N keys as the primary headers and collects
        every unique header combination into 'all_headers' so the TOC
        builder can recover all section headings even after merging.
        """
        merged = meta1.copy()

        # Collect all unique header sets from both sides
        existing_headers = list(meta1.get("all_headers", []))
        if not existing_headers:
            h1 = self._extract_header_set(meta1)
            if h1:
                existing_headers.append(h1)

        incoming_headers = list(meta2.get("all_headers", []))
        if not incoming_headers:
            h2 = self._extract_header_set(meta2)
            if h2:
                incoming_headers.append(h2)

        for h_set in incoming_headers:
            if h_set not in existing_headers:
                existing_headers.append(h_set)

        if existing_headers:
            merged["all_headers"] = existing_headers

        # Add non-header metadata from meta2 that isn't already present
        for k, v in meta2.items():
            if not k.startswith("Header ") and k != "all_headers" and k not in merged:
                merged[k] = v

        return merged

    def _add_final_metadata(self, chunks: List[Document]) -> List[Document]:
        """Transform chunk metadata to final format with header paths.

        Converts internal Header N keys and all_headers lists into the
        clean output format: all_header_paths (list of delimited strings)
        and header_level_map (dict of header text to level).

        Removes the intermediate metadata fields (Header N, all_headers).
        """
        for chunk in chunks:
            # Collect header dicts from all_headers (combined) or Header N keys
            all_headers = chunk.metadata.get("all_headers", [])
            if not all_headers:
                header_set = self._extract_header_set(chunk.metadata)
                if header_set:
                    all_headers = [header_set]

            # Build paths and level map
            all_paths = []
            level_map = {}
            for h_dict in all_headers:
                path, lmap = self._build_header_path(h_dict)
                if path and path not in all_paths:
                    all_paths.append(path)
                level_map.update(lmap)

            # Drop the intermediate header keys
            for key in [k for k in chunk.metadata if k.startswith("Header ")] + ["all_headers"]:
                chunk.metadata.pop(key, None)

            # Set new metadata
            chunk.metadata["all_header_paths"] = all_paths
            chunk.metadata["header_level_map"] = level_map
            chunk.metadata["doc_title"] = self.title
            chunk.metadata["word_count"] = len(chunk.page_content.split())
            chunk.metadata["char_count"] = len(chunk.page_content)

        return chunks
