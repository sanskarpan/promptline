from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from promptline.core.llm import LLMClient
from promptline.core.program import ModelConfig, Prediction, PromptProgram
from promptline.core.types import Candidate, Example

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
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExhausted(RuntimeError):
    """Raised by optimizers that want to signal budget exhaustion explicitly.

    :class:`EvalHarness.evaluate` does *not* raise this exception itself; it
    truncates the result set and sets ``EvalReport.truncated = True``.
    Optimizers may raise :class:`BudgetExhausted` to break out of their own
    search loops once the budget is spent.
    """


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


class Budget:
    """Mutable rollout/cost budget tracker.

    All attribute access is safe within a single asyncio event loop.
    :meth:`try_reserve` uses an internal :class:`asyncio.Lock` to provide
    an atomic check-and-reserve so concurrent coroutines never over-run the
    cap.
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
        self._lock: asyncio.Lock = asyncio.Lock()

    def charge(self, rollouts: int = 1, cost: float = 0.0) -> None:
        """Increment consumed rollouts and cost."""
        self.rollouts_used += rollouts
        self.cost_used += cost

    async def try_reserve(self, rollouts: int = 1) -> bool:
        """Atomically check budget and reserve *rollouts* if not exhausted.

        Returns ``True`` when the reservation succeeded, ``False`` when the
        budget was already exhausted.  The internal :class:`asyncio.Lock`
        serialises the check-and-reserve so concurrent coroutines cannot both
        observe a non-exhausted budget and double-count.
        """
        async with self._lock:
            if self.exhausted:
                return False
            self.rollouts_used += rollouts
            return True

    def add_cost(self, cost: float) -> None:
        """Charge *cost* USD against the budget."""
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
        remaining examples are skipped and ``report.truncated`` is set to
        ``True``.  Results are always ordered by example index regardless of
        completion order.

        Exceptions raised by ``program.run`` (e.g. :class:`LLMError` after
        retries) are caught per-example and recorded as
        ``ExampleResult(score=0.0, failed=True)`` so a single bad example
        cannot abort the whole evaluation.

        :class:`BudgetExhausted` is *not* raised here; use that exception in
        optimizer loops that want an early exit on budget exhaustion.
        """
        semaphore = asyncio.Semaphore(self.concurrency)
        truncated = False
        # Pre-allocated slots so we can fill in by index for deterministic ordering.
        results: list[ExampleResult | None] = [None] * len(examples)

        async def run_one(idx: int, example: Example) -> None:
            nonlocal truncated

            async with semaphore:
                # Atomically check and pre-reserve a rollout slot INSIDE the
                # semaphore so that cost and rollout counts accumulated by
                # earlier tasks are visible before the next task starts.
                if budget is not None:
                    reserved = await budget.try_reserve(rollouts=1)
                    if not reserved:
                        truncated = True
                        return

                try:
                    prediction: Prediction = await program.run(
                        example, candidate, self.client, self.cfg
                    )
                except Exception as exc:
                    # Isolate the failure: record a zero-score result and
                    # continue with the remaining examples.
                    results[idx] = ExampleResult(
                        example_idx=idx,
                        score=0.0,
                        feedback=f"error: {exc}",
                        cost_usd=0.0,
                        failed=True,
                    )
                    return

                if prediction.failed:
                    score: float = 0.0
                    feedback: str = prediction.failure_reason
                else:
                    raw = metric(example, prediction)
                    if inspect.isawaitable(raw):
                        raw = await raw
                    score = raw.score
                    feedback = raw.feedback

                # Charge the cost portion of the budget while still inside the
                # semaphore so the next task observes the updated cost_used.
                if budget is not None:
                    budget.add_cost(prediction.cost_usd)

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
