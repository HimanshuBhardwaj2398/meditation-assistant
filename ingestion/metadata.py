"""The metadata keys ingestion writes on documents and chunks.

Retrieval, the Streamlit views, the atlas and the eval tooling read these
keys from stored rows, so they are a contract with the existing corpus:
renaming one means migrating every stored chunk and vector.

Document metadata (``documents.doc_metadata``) carries the source's
provenance (``PROVENANCE_KEYS``), the vector collection
(``admission.VECTOR_COLLECTION_KEY``) and the table of contents
(``ingestion.toc``).

Chunk metadata, stored on ``chunks.chunk_metadata`` and on each vector's
``cmetadata``, is described by ``ChunkMetadata``.
"""

from typing import Any, Dict, List, Mapping, Optional, TypedDict

# Source fields stored on the document.
PROVENANCE_KEYS = (
    "source",
    "uid",
    "author_uid",
    "lang",
    "nikaya",
    "nikaya_name",
    "nikaya_english",
    "reading_url",
)

# The subset copied onto every chunk and vector, for retrieval filtering.
CHUNK_TAG_KEYS = ("source", "uid", "author_uid", "lang", "nikaya", "nikaya_name")

# The chunk's ID: the chunk row's uuid and the vector's custom_id. Eval
# ground truth is keyed on it, so a re-embed keeps it.
CHUNK_UUID_KEY = "uuid"
DOCUMENT_ID_KEY = "document_id"


class ChunkMetadata(TypedDict, total=False):
    """What one chunk's metadata holds once ingestion has finished with it."""

    # Written by the chunker (ingestion.chunking).
    all_header_paths: List[str]  # every header path the chunk covers, "A > B"
    header_level_map: Dict[str, int]  # header text -> markdown level
    doc_title: str
    word_count: int
    char_count: int
    is_semantic_split: bool
    is_combined: bool
    semantic_split_failed: bool  # an oversized chunk kept whole

    # Written by the embedding stage (chunk_tags).
    uuid: str  # on vectors only; the chunk row stores it in its own column
    document_id: int
    original_doc_id: int  # legacy duplicate of document_id; retrieval falls back to it
    source_title: str
    source: str
    uid: str
    author_uid: str
    lang: str
    nikaya: str
    nikaya_name: str


def source_provenance(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Source fields shared by freshly parsed and reprocessed documents."""
    return {key: metadata[key] for key in PROVENANCE_KEYS if key in metadata}


def chunk_tags(
    document_id: Optional[int], title: Optional[str], source_metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    """The document-level keys every chunk of a document carries."""
    tags: Dict[str, Any] = {
        key: source_metadata[key] for key in CHUNK_TAG_KEYS if key in source_metadata
    }
    if document_id is not None:
        tags[DOCUMENT_ID_KEY] = document_id
        tags["original_doc_id"] = document_id
    if title:
        tags["source_title"] = title
    return tags
