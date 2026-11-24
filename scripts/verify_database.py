"""
Verification script for database setup.

Usage:
    poetry run python scripts/verify_database.py                     # all checks
    poetry run python scripts/verify_database.py --consistency-only  # read-only checks
    poetry run python scripts/verify_database.py --delete-orphan-vectors          # dry run
    poetry run python scripts/verify_database.py --delete-orphan-vectors --apply  # delete

The full run includes a CRUD check that writes (and removes) a test document;
``--consistency-only`` skips it and only reads.
"""

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

from config.settings import get_settings
from db.crud import DocumentCRUD
from db.database import get_database, session_scope
from db.schema import DocumentStatus
from db.vector_consistency import check_vector_consistency, delete_orphan_vectors


def verify_connection():
    """Test database connection."""
    print("1. Testing database connection...")
    try:
        with get_database().engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"   ✓ Connected to PostgreSQL: {version[:50]}...")
        return True
    except OperationalError as e:
        print(f"   ✗ Connection failed: {e}")
        return False


def verify_pgvector():
    """Check pgvector extension."""
    print("\n2. Checking pgvector extension...")
    try:
        with get_database().engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector'"))
            if result.fetchone():
                print("   ✓ pgvector extension is installed")
                return True
            else:
                print("   ✗ pgvector extension not found")
                return False
    except Exception as e:
        print(f"   ✗ Error checking pgvector: {e}")
        return False


def verify_tables():
    """Check that all required tables exist."""
    print("\n3. Verifying database tables...")
    inspector = inspect(get_database().engine)
    tables = inspector.get_table_names()

    required_tables = ["documents", "chunks"]
    all_exist = True

    for table in required_tables:
        if table in tables:
            print(f"   ✓ Table '{table}' exists")
        else:
            print(f"   ✗ Table '{table}' missing")
            all_exist = False

    return all_exist


def verify_document_schema():
    """Check Document table has all required columns."""
    print("\n4. Verifying Document table schema...")
    inspector = inspect(get_database().engine)
    columns = {col["name"] for col in inspector.get_columns("documents")}

    required_columns = {
        "id",
        "title",
        "file_path",
        "markdown",
        "doc_metadata",
        "tags",
        "status",
        "status_details",
        "chunks",
        "description",
        "created_at",
        "updated_at",
    }

    missing = required_columns - columns
    if missing:
        print(f"   ✗ Missing columns: {', '.join(missing)}")
        return False
    else:
        print("   ✓ All required columns present")
        return True


def verify_chunk_schema():
    """Check Chunk table has all required columns."""
    print("\n5. Verifying Chunk table schema...")
    inspector = inspect(get_database().engine)
    columns = {col["name"] for col in inspector.get_columns("chunks")}

    required_columns = {
        "id",
        "uuid",
        "document_id",
        "chunk_text",
        "chunk_index",
        "chunk_metadata",
        "created_at",
    }

    missing = required_columns - columns
    if missing:
        print(f"   ✗ Missing columns: {', '.join(missing)}")
        return False
    else:
        print("   ✓ All required columns present")
        return True


def verify_crud_operations():
    """Test basic CRUD operations."""
    print("\n6. Testing CRUD operations...")
    try:
        with session_scope() as session:
            crud = DocumentCRUD(session)

            # Create test document
            doc = crud.create_document(
                title="Test Document",
                file_path="https://example.com/test.pdf",
                tags=["test"],
                doc_metadata={"test": True},
            )
            print(f"   ✓ Created document (ID: {doc.id})")

            # Update status
            crud.update_status(
                document_id=doc.id,
                status=DocumentStatus.PARSING,
                status_details="Test status update",
            )
            print("   ✓ Updated document status")

            # Get by status
            docs = crud.get_documents_by_status(DocumentStatus.PARSING)
            print(f"   ✓ Retrieved {len(docs)} documents by status")

            # Delete test document
            session.delete(doc)
            session.commit()
            print("   ✓ Deleted test document")

        return True
    except Exception as e:
        print(f"   ✗ CRUD operations failed: {e}")
        return False


def verify_settings():
    """Check configuration settings."""
    print("\n7. Verifying configuration settings...")
    settings = get_settings()

    checks = {
        "Database URL": bool(settings.database.url),
        "Voyage API Key": bool(settings.embedding.voyage_api_key),
        "LlamaParse API Key": bool(settings.parsing.llamaparse_api_key),
    }

    all_ok = True
    for name, status in checks.items():
        if status:
            print(f"   ✓ {name} configured")
        else:
            print(f"   ⚠ {name} not set (may be optional)")
            # Don't fail on optional settings

    # Check whether this is a managed/remote database (Neon, etc.)
    if settings.database.is_remote:
        print("   ✓ Using remote (managed) PostgreSQL database")
    else:
        print("   ℹ Using local PostgreSQL database")

    return all_ok


def verify_vector_consistency():
    """Check that chunk rows and vectors still match one to one (read-only)."""
    print("\nChecking chunk rows against vectors...")
    try:
        with session_scope() as session:
            report = check_vector_consistency(session)
    except Exception as e:
        print(f"   ✗ Consistency check failed: {e}")
        return False

    print(f"   {report.chunks} chunk rows, {report.vectors} vectors")
    for label, count in (
        ("pipeline vectors with no chunk row", report.orphan_vectors),
        ("chunk rows with no vector", report.chunks_without_vectors),
        ("vectors naming a deleted document", report.vectors_of_missing_documents),
    ):
        print(f"   {'✓' if count == 0 else '✗'} {label}: {count}")
    if report.foreign_vectors:
        print(
            f"   ℹ {report.foreign_vectors} vectors carry no document_id "
            "(not written by this pipeline; left alone)"
        )
    if report.orphan_vector_sample:
        print(f"     e.g. {report.orphan_vector_sample}")
        print("     Review, then: scripts/verify_database.py --delete-orphan-vectors [--apply]")
    if report.documents_missing_vectors:
        print(
            f"     Documents to re-embed (scripts/ingest.py --reprocess ID "
            f"--reprocess-mode from_embedding): {report.documents_missing_vectors}"
        )
    return report.consistent


def delete_orphan_vectors_command(apply: bool) -> int:
    """Show (and with apply, delete) pipeline vectors whose chunk row is gone."""
    with session_scope() as session:
        report = check_vector_consistency(session)
        print(f"{report.orphan_vectors} pipeline vectors have no chunk row.")
        if report.orphan_vector_sample:
            print(f"e.g. {report.orphan_vector_sample}")
        if not report.orphan_vectors:
            return 0
        if not apply:
            print("Dry run: nothing was deleted. Add --apply to delete them.")
            return 0
        deleted = delete_orphan_vectors(session)
    print(f"Deleted {deleted} vectors.")
    return 0


def main(argv=None):
    """Run the verification checks."""
    parser = argparse.ArgumentParser(description="Verify the database setup and its data.")
    parser.add_argument(
        "--consistency-only",
        action="store_true",
        help="run only the read-only checks (skips the CRUD check, which writes)",
    )
    parser.add_argument(
        "--delete-orphan-vectors",
        action="store_true",
        help="show pipeline vectors that have no chunk row (dry run unless --apply)",
    )
    parser.add_argument("--apply", action="store_true", help="with --delete-orphan-vectors")
    args = parser.parse_args(argv)

    if args.apply and not args.delete_orphan_vectors:
        parser.error("--apply only goes with --delete-orphan-vectors")
    if args.delete_orphan_vectors:
        return delete_orphan_vectors_command(apply=args.apply)

    print("=" * 60)
    print("Database Setup Verification")
    print("=" * 60)

    if args.consistency_only:
        checks = [
            verify_connection(),
            verify_pgvector(),
            verify_tables(),
            verify_vector_consistency(),
        ]
    else:
        checks = [
            verify_connection(),
            verify_pgvector(),
            verify_tables(),
            verify_document_schema(),
            verify_chunk_schema(),
            verify_crud_operations(),
            verify_settings(),
            verify_vector_consistency(),
        ]

    print("\n" + "=" * 60)
    if all(checks):
        print("✓ ALL CHECKS PASSED")
        print("=" * 60)
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Review errors above")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
