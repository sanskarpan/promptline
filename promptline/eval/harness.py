from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from promptline.core.llm import LLMClient
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    """Score plus optional feedback for a single evaluation."""

    score: float
    feedback: str = ""
    per_module: dict[str, str] = {}


#: A metric is any callable that maps (Example, Prediction) to a MetricResult.
#: It may be a plain function or an async coroutine function; the harness
#: awaits it when necessary.
Metric = Callable[[Example, Prediction], "Awaitable[MetricResult] | MetricResult"]

# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget:
    """Mutable rollout/cost budget tracker.

    All attribute access is safe within a single asyncio event loop.
    The :class:`EvalHarness` guards modifications with an ``asyncio.Lock``
    to prevent concurrent over-runs.
    """

    def __init__(
        self,
        max_rollouts: int | None = None,
        max_cost_usd: float | None = None,
    ) -> None:
        self.max_rollouts = max_rollouts
        self.max_cost_usd = max_cost_usd
        self.rollouts_used: int = 0
        self.cost_used: float = 0.0

    def charge(self, rollouts: int = 1, cost: float = 0.0) -> None:
        """Increment consumed rollouts and cost."""
        self.rollouts_used += rollouts
        self.cost_used += cost

    @property
    def exhausted(self) -> bool:
        """True when any limit is reached."""
        if self.max_rollouts is not None and self.rollouts_used >= self.max_rollouts:
            return True
        if self.max_cost_usd is not None and self.cost_used >= self.max_cost_usd:
            return True
        return False

    @property
    def remaining_rollouts(self) -> int | None:
        """Remaining rollout slots, or *None* when there is no rollout cap."""
        if self.max_rollouts is None:
            return None
        return max(0, self.max_rollouts - self.rollouts_used)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ExampleResult(BaseModel):
    """Outcome of evaluating a single training example."""

    example_idx: int
    score: float
    feedback: str
    cost_usd: float
    failed: bool


class EvalReport(BaseModel):
    """Aggregate results of an evaluation run."""

    per_example: list[ExampleResult]
    truncated: bool = False

    @property
    def mean_score(self) -> float:
        """Mean score across all evaluated examples (0.0 when empty)."""
        if not self.per_example:
            return 0.0
        return sum(r.score for r in self.per_example) / len(self.per_example)

    @property
    def total_cost(self) -> float:
        """Total LLM cost across all evaluated examples."""
        return sum(r.cost_usd for r in self.per_example)

    @property
    def n(self) -> int:
        """Number of examples that were actually evaluated."""
        return len(self.per_example)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class EvalHarness:
    """Runs a :class:`PromptProgram` over a list of examples concurrently.

    Parameters
    ----------
    client:
        LLM client used by the program.
    cfg:
        Model and sampling parameters.
    concurrency:
        Maximum number of examples evaluated in parallel.
    """

    def __init__(
        self,
        client: LLMClient,
        cfg: ModelConfig,
        concurrency: int = 8,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.concurrency = concurrency

    async def evaluate(
        self,
        program: PromptProgram,
        candidate: Candidate,
        examples: list[Example],
        metric: Metric,
        budget: Budget | None = None,
    ) -> EvalReport:
        """Evaluate *candidate* on every example and return an :class:`EvalReport`.

        Examples are processed concurrently (up to ``self.concurrency`` at once).
        If *budget* becomes exhausted before all examples are processed, the
        remaining examples are skipped and ``report.truncated`` is set to ``True``.
        Results are always ordered by example index regardless of completion order.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        budget_lock = asyncio.Lock()
        truncated = False
        # Pre-allocated slots so we can fill in by index for deterministic ordering.
        results: list[ExampleResult | None] = [None] * len(examples)

        async def run_one(idx: int, example: Example) -> None:
            nonlocal truncated

            # Atomically check and pre-reserve a rollout slot so that we
            # never exceed max_rollouts even with high concurrency.
            if budget is not None:
                async with budget_lock:
                    if budget.exhausted:
                        truncated = True
                        return
                    # Pre-reserve the rollout count; cost is added after run.
                    budget.rollouts_used += 1

            async with semaphore:
                prediction: Prediction = await program.run(
                    example, candidate, self.client, self.cfg
                )

                if prediction.failed:
                    score: float = 0.0
                    feedback: str = prediction.failure_reason
                else:
                    raw = metric(example, prediction)
                    if inspect.isawaitable(raw):
                        raw = await raw
                    score = raw.score
                    feedback = raw.feedback

                # Charge the cost portion of the budget (rollout already reserved).
                if budget is not None:
                    async with budget_lock:
                        budget.cost_used += prediction.cost_usd

                results[idx] = ExampleResult(
                    example_idx=idx,
                    score=score,
                    feedback=feedback,
                    cost_usd=prediction.cost_usd,
                    failed=prediction.failed,
                )

        tasks = [
            asyncio.create_task(run_one(i, ex)) for i, ex in enumerate(examples)
        ]
        await asyncio.gather(*tasks)

        per_example = [r for r in results if r is not None]
        return EvalReport(per_example=per_example, truncated=truncated)
