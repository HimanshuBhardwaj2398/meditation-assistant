"""
Run pipeline stages in dependency order.

``PipelineOrchestrator`` knows nothing about documents, claims or databases:
it orders the stages it is given, runs each one whose dependencies completed,
and turns a stage's exception into a FAILED stage. ``IngestionOrchestrator``
(``ingestion.orchestrator``) builds the stages and owns everything else.
"""

import logging
from graphlib import CycleError, TopologicalSorter
from typing import Callable, List, Optional

from core.exceptions import MeditationDBError, PipelineError, ProcessingLeaseLostError
from core.interfaces import PipelineContext, PipelineStage, StageStatus

logger = logging.getLogger(__name__)


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

        logger.debug(f"Pipeline execution order: {[s.name for s in self._execution_order]}")

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
        on_stage_update: Optional[Callable[[str, StageStatus], None]] = None,
    ) -> PipelineContext:
        """
        Execute pipeline stages in dependency order.

        Args:
            context: Initial pipeline context
            on_stage_update: Optional callback invoked with (stage_name, StageStatus)
                for each stage transition (RUNNING, COMPLETED, FAILED).

        Returns:
            Final context after all stages
        """
        current_context = context

        def _notify(stage_name: str, status: StageStatus):
            if on_stage_update is not None:
                on_stage_update(stage_name, status)

        for stage in self._execution_order:
            # Skip if already completed
            if stage.should_skip(current_context):
                logger.info(f"Skipping stage '{stage.name}' (already completed)")
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
                    logger.info(f"Skipping stage '{stage.name}': {failed} failed")
                else:
                    logger.warning(
                        f"Stage '{stage.name}' cannot run. Missing dependencies: {missing}"
                    )
                continue

            # Execute stage. Stages raise when they cannot finish; this is the
            # one place an exception becomes a FAILED stage with its message.
            logger.info(f"Executing stage: {stage.name}")
            _notify(stage.name, StageStatus.RUNNING)
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
                logger.error(f"Stage '{stage.name}' failed: {error}")
                _notify(stage.name, StageStatus.FAILED)
                # Later stages that depend on this one are skipped below.
            else:
                _notify(stage.name, StageStatus.COMPLETED)

        return current_context
