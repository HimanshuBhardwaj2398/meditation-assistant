"""Chunk-merging thresholds are measured in characters, matching config docs.

Historically `_combine_small_chunks` measured the current chunk in *words*
while accumulating *characters*, so `min_size`/`tiny_chunk_threshold`
(documented as characters) silently meant different things in different
comparisons. These tests pin character semantics with cases where the two
interpretations disagree.
"""

from langchain.schema import Document

from ingestion.chunking import Config, MarkdownChunker


def make_chunker(**overrides) -> MarkdownChunker:
    config = Config(enable_semantic=False, enable_parallel=False, **overrides)
    return MarkdownChunker(text="# Title\n\nbody", config=config)


def test_few_long_words_are_not_tiny():
    """5 words / 174 chars with tiny_chunk_threshold=20: not tiny in chars.

    Word-based measuring would call this tiny (5 < 20) and merge it away.
    """
    long_words = ("supercalifragilisticexpialidocious " * 5).strip()
    assert len(long_words.split()) == 5
    assert len(long_words) >= 100

    chunker = make_chunker(max_size=300, min_size=100, tiny_chunk_threshold=20)
    chunks = [
        Document(page_content=long_words, metadata={}),
        Document(page_content="y" * 150, metadata={}),
    ]

    merged = chunker._combine_small_chunks(chunks)

    assert len(merged) == 2


def test_many_short_words_meeting_min_chars_stay_standalone():
    """80 words / ~400 chars with min_size=300: large enough in chars.

    Word-based measuring would call this small (80 < 300) and start merging.
    """
    content = ("word " * 79) + "word"
    assert len(content.split()) == 80
    assert len(content) >= 300

    chunker = make_chunker(max_size=1000, min_size=300, tiny_chunk_threshold=50)
    chunks = [
        Document(page_content=content, metadata={}),
        Document(page_content="z" * 400, metadata={}),
    ]

    merged = chunker._combine_small_chunks(chunks)

    assert len(merged) == 2


def test_combining_accumulates_characters_and_stops_at_min_size():
    chunker = make_chunker(max_size=1000, min_size=180, tiny_chunk_threshold=10)
    chunks = [
        Document(page_content="a" * 100, metadata={}),
        Document(page_content="b" * 90, metadata={}),
        Document(page_content="c" * 150, metadata={}),
    ]

    merged = chunker._combine_small_chunks(chunks)

    assert len(merged) == 2
    # 100 + "\n\n" + 90
    assert len(merged[0].page_content) == 192


def test_a_header_splitter_error_fails_chunking(monkeypatch):
    """A broken header split is a bug, not one big chunk passed off as success."""
    import pytest

    from core.exceptions import ChunkingError
    from ingestion import chunking

    class BrokenSplitter:
        def __init__(self, **kwargs):
            pass

        def split_text(self, text):
            raise RuntimeError("splitter exploded")

    monkeypatch.setattr(chunking, "MarkdownHeaderTextSplitter", BrokenSplitter)

    with pytest.raises(ChunkingError, match="splitter exploded"):
        make_chunker().chunk()


def test_a_failed_semantic_split_keeps_the_chunk_and_flags_it():
    """The oversized chunk survives, marked so degraded chunks can be found."""

    class BrokenEmbeddings:
        def embed_documents(self, texts):
            raise RuntimeError("model unavailable")

    chunker = make_chunker()
    chunker.embeddings = BrokenEmbeddings()
    chunk = Document(
        page_content="One sentence here. Another sentence there. A third one.",
        metadata={"Header 1": "Title"},
    )

    result = chunker._semantic_split(chunk)

    assert [doc.page_content for doc in result] == [chunk.page_content]
    assert result[0].metadata == {"Header 1": "Title", "semantic_split_failed": True}
    assert chunk.metadata == {"Header 1": "Title"}
