"""
Run pipeline stages in dependency order.

``PipelineOrchestrator`` knows nothing about documents, claims or databases:
it orders the stages it is given, runs each one whose dependencies completed,
and turns a stage's exception into a FAILED stage. ``IngestionOrchestrator``
(``ingestion.orchestrator``) builds the stages and owns everything else.
"""

import inspect
import logging
from graphlib import CycleError, TopologicalSorter
from typing import Awaitable, Callable, List, Optional, Union

from core.exceptions import MeditationDBError, PipelineError, ProcessingLeaseLostError
from core.interfaces import PipelineContext, PipelineStage, StageStatus

logger = logging.getLogger(__name__)

# Called with (stage_name, status) on each stage transition. It may be a
# coroutine function; the pipeline awaits what it returns.
StageCallback = Callable[[str, StageStatus], Union[None, Awaitable[None]]]


class PipelineOrchestrator:
    """
    DAG-based pipeline orchestrator with automatic dependency resolution.

    Stages are executed in topological order based on their dependencies.
    Each stage is idempotent and can be resumed from any point.
    """

    def __init__(self, stages: List[PipelineStage]):
        """
        Initialize orchestrator with pipeline stages.

        Args:
            stages: List of pipeline stages to execute

        Raises:
            PipelineError: If a stage depends on a stage that is not in the
                pipeline, or the dependencies form a cycle
        """
        self._stages = stages
        self._execution_order = self._order(stages)

        logger.debug("Pipeline execution order: %s", [s.name for s in self._execution_order])

    @staticmethod
    def _order(stages: List[PipelineStage]) -> List[PipelineStage]:
        """Sort stages so each comes after the stages it requires."""
        stage_map = {stage.name: stage for stage in stages}

        for stage in stages:
            unknown = [dep for dep in stage.required_stages if dep not in stage_map]
            if unknown:
                raise PipelineError(
                    f"Stage '{stage.name}' requires stages not in the pipeline: {unknown}"
                )

        graph = TopologicalSorter({stage.name: stage.required_stages for stage in stages})
        try:
            return [stage_map[name] for name in graph.static_order()]
        except CycleError as e:
            raise PipelineError(f"Circular dependency detected in pipeline stages: {e.args[1]}")

    async def execute(
        self,
        context: PipelineContext,
        on_stage_update: Optional[StageCallback] = None,
    ) -> PipelineContext:
        """
        Execute pipeline stages in dependency order.

        Args:
            context: Initial pipeline context
            on_stage_update: Optional callback invoked with (stage_name, StageStatus)
                for each stage transition (RUNNING, COMPLETED, FAILED); sync or
                async.

        Returns:
            Final context after all stages
        """
        current_context = context

        async def _notify(stage_name: str, status: StageStatus) -> None:
            if on_stage_update is not None:
                result = on_stage_update(stage_name, status)
                if inspect.isawaitable(result):
                    await result

        for stage in self._execution_order:
            # Skip if already completed
            if stage.should_skip(current_context):
                logger.info("Skipping stage '%s' (already completed)", stage.name)
                continue

            # Check if dependencies are met
            if not stage.can_run(current_context):
                missing = [
                    dep
                    for dep in stage.required_stages
                    if current_context.stage_results.get(dep) != StageStatus.COMPLETED
                ]
                failed = [
                    dep
                    for dep in missing
                    if current_context.stage_results.get(dep) == StageStatus.FAILED
                ]
                if failed:
                    logger.info("Skipping stage '%s': %s failed", stage.name, failed)
                else:
                    logger.warning(
                        "Stage '%s' cannot run. Missing dependencies: %s", stage.name, missing
                    )
                continue

            # Execute stage. Stages raise when they cannot finish; this is the
            # one place an exception becomes a FAILED stage with its message.
            logger.info("Executing stage: %s", stage.name)
            await _notify(stage.name, StageStatus.RUNNING)
            try:
                current_context = await stage.execute(current_context)
            except ProcessingLeaseLostError:
                # Another run owns the document now: stop, and record nothing.
                raise
            except MeditationDBError as e:
                # Expected failures (ParsingError, EmbeddingError, ...) keep their message.
                current_context = current_context.mark_stage_failed(stage.name, str(e))
            except Exception as e:
                logger.error("Unexpected error in stage '%s'", stage.name, exc_info=True)
                current_context = current_context.mark_stage_failed(
                    stage.name, f"Unexpected {type(e).__name__}: {e}"
                )

            if current_context.stage_results.get(stage.name) == StageStatus.FAILED:
                error = current_context.error_messages.get(stage.name, "Unknown error")
                logger.error("Stage '%s' failed: %s", stage.name, error)
                await _notify(stage.name, StageStatus.FAILED)
                # Later stages that depend on this one are skipped below.
            else:
                await _notify(stage.name, StageStatus.COMPLETED)

        return current_context
