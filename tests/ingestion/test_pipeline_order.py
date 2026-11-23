"""PipelineOrchestrator orders stages by their dependencies and rejects bad graphs."""

import pytest

from core.exceptions import PipelineError
from core.interfaces import PipelineContext, PipelineStage, StageName
from ingestion.pipeline import PipelineOrchestrator


class Stage(PipelineStage):
    def __init__(self, name: str, *deps: str):
        self._name = name
        self._deps = list(deps)

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_stages(self):
        return self._deps

    async def execute(self, context: PipelineContext) -> PipelineContext:
        return context.mark_stage_completed(self.name)


def order(pipeline: PipelineOrchestrator) -> list:
    return [stage.name for stage in pipeline._execution_order]


def test_stages_given_out_of_order_run_in_dependency_order():
    pipeline = PipelineOrchestrator(
        [
            Stage("persist", "embed"),
            Stage("embed", "chunk"),
            Stage("chunk", "parse"),
            Stage("parse"),
        ]
    )
    assert order(pipeline) == ["parse", "chunk", "embed", "persist"]


def test_a_stage_with_two_dependencies_runs_after_both():
    pipeline = PipelineOrchestrator([Stage("join", "left", "right"), Stage("left"), Stage("right")])
    assert order(pipeline)[-1] == "join"


def test_a_cycle_is_rejected():
    with pytest.raises(PipelineError, match="Circular dependency"):
        PipelineOrchestrator([Stage("a", "b"), Stage("b", "a")])


def test_a_dependency_missing_from_the_pipeline_is_rejected():
    with pytest.raises(PipelineError, match="not in the pipeline"):
        PipelineOrchestrator([Stage("chunk", "parse")])


def test_the_ingestion_stages_order_without_persistence():
    """With persistence disabled, nothing depends on the missing persistence stage."""
    from ingestion.chunking import Config as ChunkingConfig
    from ingestion.stages import ChunkingStage, EmbeddingStage, ParsingStage

    stages = [
        EmbeddingStage(vector_store_manager=None, vector_only=True),  # type: ignore[arg-type]
        ChunkingStage(ChunkingConfig()),
        ParsingStage(parser_factory=None, database=None),  # type: ignore[arg-type]
    ]
    assert order(PipelineOrchestrator(stages)) == [
        StageName.PARSING,
        StageName.CHUNKING,
        StageName.EMBEDDING,
    ]


async def test_a_stage_whose_dependency_failed_does_not_run():
    class Failing(Stage):
        async def execute(self, context):
            raise RuntimeError("boom")

    context = await PipelineOrchestrator([Failing("parse"), Stage("chunk", "parse")]).execute(
        PipelineContext()
    )
    assert context.stage_results["parse"].value == "failed"
    assert "chunk" not in context.stage_results
