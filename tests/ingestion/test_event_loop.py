"""Blocking work in a run happens off the event loop.

Parsing, embedding and every database round trip block. Run on the event
loop, they would serialize documents that a caller runs concurrently with
``asyncio.gather``. Each call below records whether it ran on the loop's
thread: a worker thread has no running loop.
"""

import asyncio
from typing import List, Tuple

from db.crud import DocumentCRUD
from tests.ingestion.conftest import SOURCE


def _on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def record_calls(monkeypatch, target, names, calls: List[Tuple[str, bool]]) -> None:
    for name in names:
        original = getattr(target, name)

        def wrapper(*args, __original=original, __name=name, **kwargs):
            calls.append((__name, _on_event_loop()))
            return __original(*args, **kwargs)

        monkeypatch.setattr(target, name, wrapper)


async def test_a_run_does_its_blocking_work_off_the_event_loop(
    orchestrator, parser, vectors, monkeypatch
):
    calls: List[Tuple[str, bool]] = []
    record_calls(monkeypatch, parser, ["parse"], calls)
    record_calls(monkeypatch, vectors, ["compute_embeddings", "stage_vectors"], calls)
    record_calls(
        monkeypatch,
        DocumentCRUD,
        ["insert_if_absent", "claim_for_processing", "set_status_if_owner", "update_status"],
        calls,
    )

    result = await orchestrator.ingest(SOURCE)

    assert result["success"] is True
    assert {name for name, _ in calls} == {
        "parse",
        "compute_embeddings",
        "stage_vectors",
        "insert_if_absent",
        "claim_for_processing",
        "set_status_if_owner",
        "update_status",
    }
    assert [name for name, on_loop in calls if on_loop] == []


async def test_the_callers_progress_callback_stays_on_the_event_loop(orchestrator):
    seen: List[bool] = []

    await orchestrator.ingest(SOURCE, on_stage_update=lambda *_: seen.append(_on_event_loop()))

    assert seen and all(seen)
