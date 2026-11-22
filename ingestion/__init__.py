"""Document ingestion: parse a source to markdown, chunk it, embed the chunks,
and publish vectors and chunk rows together.

Import from the submodules (``ingestion.orchestrator``, ``ingestion.stages``,
``ingestion.parsing``, ...). The package itself imports nothing, so a light
import such as ``ingestion.suttacentral`` does not load the embedding and
chunking models' dependencies.
"""
