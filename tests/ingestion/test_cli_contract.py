"""CLI regression tests: no live databases, model loading, or provider calls."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from db.schema import DocumentStatus


@pytest.fixture
def cli_modules(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("config.logging_config.setup_logging", lambda: None)
    return SimpleNamespace(
        **{
            name: importlib.import_module(f"scripts.{name}")
            for name in ("ingest", "ingest_one", "ingest_batch", "ingest_all")
        }
    )


@pytest.mark.parametrize("titles", [["one"], []])
async def test_batch_rejects_mismatched_titles_before_starting(cli_modules, monkeypatch, titles):
    cli = cli_modules.ingest
    factory = MagicMock()
    monkeypatch.setattr(cli, "IngestionOrchestrator", factory)
    with pytest.raises(ValueError, match="one entry per source"):
        await cli.ingest_batch(["a.pdf", "b.pdf"], titles=titles)
    factory.assert_not_called()


def test_cli_shared_title_processes_every_source(cli_modules, monkeypatch):
    cli = cli_modules.ingest
    processor = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(cli, "ingest_document", processor)
    monkeypatch.setattr(cli, "IngestionOrchestrator", lambda **kwargs: object())
    monkeypatch.setenv("DB_URL", "postgresql://test/test")
    monkeypatch.setattr("sys.argv", ["ingest", "a.pdf", "b.pdf", "--title", "Shared", "--quiet"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert [(call.args[1], call.args[2]) for call in processor.await_args_list] == [
        ("a.pdf", "Shared"),
        ("b.pdf", "Shared"),
    ]


@pytest.mark.parametrize("verbose", [False, True])
async def test_cli_display_title_does_not_override_extraction(cli_modules, verbose):
    orchestrator = SimpleNamespace(
        process=AsyncMock(
            return_value={
                "document_id": 1,
                "title": "Extracted",
                "chunk_count": 1,
                "success": True,
                "errors": {},
                "stage_results": {},
            }
        )
    )
    await cli_modules.ingest.ingest_document(orchestrator, "a.pdf", verbose=verbose)
    orchestrator.process.assert_awaited_once_with(source="a.pdf", title=None)


@pytest.mark.parametrize(
    "status", [DocumentStatus.COMPLETED, DocumentStatus.FAILED, DocumentStatus.PARSING]
)
async def test_batch_distinguishes_completed_from_incomplete(cli_modules, monkeypatch, status):
    cli = cli_modules.ingest_batch
    processor = SimpleNamespace(process=AsyncMock())
    monkeypatch.setattr(cli, "IngestionOrchestrator", lambda **kwargs: processor)
    monkeypatch.setattr(cli, "session_scope", lambda: nullcontext(None))
    monkeypatch.setattr(
        cli,
        "DocumentCRUD",
        lambda session: SimpleNamespace(
            check_duplicate=lambda source: SimpleNamespace(id=7, status=status),
        ),
    )
    success = await cli._ingest_all(["sc:mn1/sujato"], "test", "postgresql://test/test")
    assert success is (status == DocumentStatus.COMPLETED)
    processor.process.assert_not_awaited()


@pytest.mark.parametrize("module_name", ["ingest_one", "ingest_batch"])
@pytest.mark.parametrize("success", [False, True])
def test_single_and_batch_exit_status(cli_modules, monkeypatch, module_name, success):
    cli = getattr(cli_modules, module_name)
    monkeypatch.setenv("NEON_DIRECT_URL", "postgresql://test/test")
    monkeypatch.setattr(cli, "Database", lambda settings: MagicMock())
    monkeypatch.setattr(cli, "set_default_database", lambda database: None)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: SimpleNamespace(
            vector=SimpleNamespace(collection_name="test"),
        ),
    )
    monkeypatch.setattr(
        cli,
        "IngestionOrchestrator",
        lambda **kwargs: SimpleNamespace(
            process=AsyncMock(
                return_value={
                    "document_id": 7,
                    "title": "Title",
                    "chunk_count": 1,
                    "success": success,
                    "stage_results": {},
                    "errors": {"embedding": "failed"} if not success else {},
                }
            ),
        ),
    )
    if module_name == "ingest_batch":
        monkeypatch.setattr(cli, "session_scope", lambda: nullcontext(None))
        monkeypatch.setattr(
            cli,
            "DocumentCRUD",
            lambda session: SimpleNamespace(check_duplicate=lambda source: None),
        )

    if success:
        cli.main(["sc:mn1/sujato"])
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main(["sc:mn1/sujato"])
        assert exc.value.code == 1


async def test_bulk_failure_keeps_stage_and_error_message(cli_modules, monkeypatch):
    cli = cli_modules.ingest_all
    monkeypatch.setattr(cli, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(
        cli,
        "_build_orchestrator",
        lambda url: SimpleNamespace(
            process=AsyncMock(
                return_value={"success": False, "errors": {"embedding": "rate limited"}}
            ),
        ),
    )
    progress = SimpleNamespace(value=0, get_lock=lambda: nullcontext())

    result = await cli._run_shard(0, [("sc:mn1/sujato", None)], "unused", progress)

    assert result["ingested"] == 0
    assert "embedding: rate limited" in result["failures"][0]["error"]
    assert progress.value == 1


@pytest.mark.parametrize("outcome", ["success", "failure", "crash", "missing", "short"])
def test_bulk_main_accounts_for_workers(cli_modules, monkeypatch, tmp_path, outcome):
    from db import database

    cli = cli_modules.ingest_all
    monkeypatch.setenv("NEON_DIRECT_URL", "postgresql://test/test")
    monkeypatch.setattr(database, "Database", lambda settings: MagicMock())
    monkeypatch.setattr(database, "set_default_database", lambda database: None)
    monkeypatch.setattr(cli, "load_catalog", lambda **kwargs: ["sc:mn1/sujato"])
    monkeypatch.setattr(cli, "partition_worklist", lambda sources: (sources, []))
    monkeypatch.setattr(cli, "_PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    results = {}

    class Process:
        exitcode = 9 if outcome == "crash" else 0

        def __init__(self, **kwargs):
            self.worker_id = kwargs["args"][0]

        def start(self):
            if outcome not in ("crash", "missing"):
                results[self.worker_id] = {
                    "ingested": 1 if outcome == "success" else 0,
                    "chunks": 1 if outcome == "success" else 0,
                    "failures": [{"source": "sc:mn1/sujato", "error": "failed"}]
                    if outcome == "failure"
                    else [],
                }

        def is_alive(self):
            return False

        def join(self):
            pass

    monkeypatch.setattr(
        cli.mp,
        "get_context",
        lambda method: SimpleNamespace(
            Value=lambda *args: SimpleNamespace(value=0),
            Manager=lambda: SimpleNamespace(dict=lambda: results),
            Process=Process,
        ),
    )

    if outcome == "success":
        cli.main(["--workers", "1"])
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--workers", "1"])
        assert exc.value.code == 1
        if outcome in ("crash", "missing"):
            assert "<worker 0>" in (tmp_path / "data" / "ingest_failures.json").read_text()
