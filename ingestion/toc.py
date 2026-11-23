"""Build a document's table of contents from its chunks' header paths.

The chunker records, on every chunk, each header path it covers
(``all_header_paths``, e.g. ``"MN 1 > The Setting"``) and the level of each
header (``header_level_map``). Publishing turns those into the table of
contents stored in ``doc_metadata["table_of_contents"]``.
"""

from typing import Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document


class TocEntry(TypedDict):
    id: str
    level: int
    text: str
    parent_id: Optional[str]
    # The first chunk the heading appears in.
    chunk_index: int
    path_from_root: str


class TableOfContents(TypedDict):
    entries: List[TocEntry]
    # The headings as an indented outline, two spaces per level.
    text: str


def build_table_of_contents(chunks: Sequence[Document]) -> TableOfContents:
    """One entry per distinct header path, in the order the chunks reach it.

    A heading is identified by its whole path, so the same heading text under
    two different parents gives two entries.
    """
    entries: List[TocEntry] = []
    entry_id_by_path: Dict[str, str] = {}

    for chunk_index, chunk in enumerate(chunks):
        metadata = chunk.metadata or {}
        level_map = metadata.get("header_level_map", {})

        for header_path in metadata.get("all_header_paths", []):
            segments = [s.strip() for s in header_path.split(" > ") if s.strip()]
            parent_path = ""
            for depth, segment in enumerate(segments, start=1):
                path = f"{parent_path} > {segment}" if parent_path else segment
                if path not in entry_id_by_path:
                    level = level_map.get(segment, depth)
                    entry_id = f"h{level}_{len(entries)}"
                    entries.append(
                        TocEntry(
                            id=entry_id,
                            level=level,
                            text=segment,
                            parent_id=entry_id_by_path.get(parent_path),
                            chunk_index=chunk_index,
                            path_from_root=path,
                        )
                    )
                    entry_id_by_path[path] = entry_id
                parent_path = path

    text = "\n".join(f"{'  ' * (entry['level'] - 1)}{entry['text']}" for entry in entries)
    return TableOfContents(entries=entries, text=text)
