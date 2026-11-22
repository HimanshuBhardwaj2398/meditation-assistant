"""
Core foundational modules for the meditation database.

This package contains central abstractions, exceptions, and interfaces
used throughout the application.
"""

from .exceptions import (
    ChunkingError,
    CollectionError,
    ConfigurationError,
    DatabaseConnectionError,
    DatabaseError,
    DocumentBusyError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbeddingError,
    EmbeddingSyncError,
    MeditationDBError,
    ParsingError,
    PipelineError,
    ProcessingLeaseLostError,
    SchemaValidationError,
    VectorStoreError,
)
from .interfaces import (
    Parser,
    ParseResult,
    PipelineContext,
    PipelineStage,
    StageStatus,
)

__all__ = [
    # Exceptions
    "MeditationDBError",
    "ConfigurationError",
    "PipelineError",
    "ParsingError",
    "ChunkingError",
    "EmbeddingError",
    "DocumentBusyError",
    "ProcessingLeaseLostError",
    "DatabaseError",
    "DocumentNotFoundError",
    "DuplicateDocumentError",
    "SchemaValidationError",
    "VectorStoreError",
    "DatabaseConnectionError",
    "CollectionError",
    "EmbeddingSyncError",
    # Interfaces
    "ParseResult",
    "Parser",
    "StageStatus",
    "PipelineContext",
    "PipelineStage",
]
