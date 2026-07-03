from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field

from pydantic import BaseModel

from promptline.core.llm import LLMCall, LLMClient, LLMResponse, Message
from promptline.core.types import Candidate, Example, Field, Signature

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Models and sampling parameters used during a program run."""

    task_model: str
    reflection_model: str = ""
    judge_model: str = ""
    temperature: float = 0.2
    max_tokens: int = 1024


# ---------------------------------------------------------------------------
# Execution records
# ---------------------------------------------------------------------------


class Trace(BaseModel):
    """Record of a single LLM call made while running a module."""

    module: str
    system_prompt: str
    user_prompt: str
    raw_output: str
    parsed: dict[str, str] | None


class Prediction(BaseModel):
    """Final result returned by :meth:`PromptProgram.run`."""

    outputs: dict[str, str]
    traces: list[Trace]
    cost_usd: float
    failed: bool = False
    failure_reason: str = ""

    @classmethod
    def failure(
        cls,
        reason: str,
        traces: list[Trace],
        cost_usd: float,
    ) -> Prediction:
        """Construct a failed prediction with an explanatory *reason*."""
        return cls(
            outputs={},
            traces=traces,
            cost_usd=cost_usd,
            failed=True,
            failure_reason=reason,
        )


# ---------------------------------------------------------------------------
# Program structure
# ---------------------------------------------------------------------------


@dataclass
class Module:
    """A named unit of a :class:`PromptProgram` with its own :class:`Signature`."""

    name: str
    signature: Signature


@dataclass
class PromptProgram:
    """An ordered sequence of :class:`Module` objects that together solve a task."""

    modules: list[Module] = dc_field(default_factory=list)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def module_names(self) -> list[str]:
        return [m.name for m in self.modules]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def simple(
        cls,
        instruction: str,
        inputs: list[str],
        outputs: list[str],
        name: str = "main",
    ) -> PromptProgram:
        """Build a single-module program from plain field-name lists."""
        sig = Signature(
            instruction=instruction,
            inputs=[Field(name=n) for n in inputs],
            outputs=[Field(name=n) for n in outputs],
        )
        return cls(modules=[Module(name=name, signature=sig)])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        example: Example,
        candidate: Candidate,
        client: LLMClient,
        cfg: ModelConfig,
    ) -> Prediction:
        """Execute all modules in order and return a :class:`Prediction`.

        Parameters
        ----------
        example:
            Holds the initial input fields for the first module.
        candidate:
            Supplies per-module instructions and few-shot demos.
        client:
            LLM client used for every completion call.
        cfg:
            Model names and sampling hyper-parameters.
        """
        traces: list[Trace] = []
        total_cost: float = 0.0
        # Running dict of all field values seen so far; later values win.
        current_inputs: dict[str, str] = dict(example.inputs)
        all_outputs: dict[str, str] = {}

        for module in self.modules:
            state = candidate.modules[module.name]
            sig = module.signature

            # Build effective signature with the candidate's instruction.
            eff_sig = Signature(
                instruction=state.instruction,
                inputs=sig.inputs,
                outputs=sig.outputs,
            )
            system_prompt: str = eff_sig.render_system()

            # ---- Compose messages ----------------------------------------
            messages: list[Message] = [Message(role="system", content=system_prompt)]

            # Few-shot demos: alternating user/assistant pairs.
            for demo in state.demos:
                demo_user = "\n".join(
                    f"{k}: {v}" for k, v in demo.inputs.items()
                )
                demo_asst = "\n".join(
                    f"[[{k}]]: {v}" for k, v in demo.outputs.items()
                )
                messages.append(Message(role="user", content=demo_user))
                messages.append(Message(role="assistant", content=demo_asst))

            # Real user turn: render declared input fields from current state.
            real_user = "\n".join(
                f"{f.name}: {current_inputs[f.name]}"
                for f in sig.inputs
                if f.name in current_inputs
            )
            messages.append(Message(role="user", content=real_user))

            # ---- First LLM call ------------------------------------------
            llm_call = LLMCall(
                model=cfg.task_model,
                messages=tuple(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            resp: LLMResponse = await client.complete(llm_call)
            total_cost += resp.cost_usd
            parsed = eff_sig.parse_output(resp.text)

            traces.append(
                Trace(
                    module=module.name,
                    system_prompt=system_prompt,
                    user_prompt=real_user,
                    raw_output=resp.text,
                    parsed=parsed,
                )
            )

            # ---- Repair attempt if parse failed --------------------------
            if parsed is None:
                repair_messages = messages + [
                    Message(role="assistant", content=resp.text),
                    Message(
                        role="user",
                        content=(
                            "Your output was not in the required format. "
                            "Respond again using exactly the required [[field]]: sections."
                        ),
                    ),
                ]
                repair_call = LLMCall(
                    model=cfg.task_model,
                    messages=tuple(repair_messages),
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                repair_resp: LLMResponse = await client.complete(repair_call)
                total_cost += repair_resp.cost_usd
                parsed = eff_sig.parse_output(repair_resp.text)

                traces.append(
                    Trace(
                        module=module.name,
                        system_prompt=system_prompt,
                        user_prompt=(
                            "Your output was not in the required format. "
                            "Respond again using exactly the required [[field]]: sections."
                        ),
                        raw_output=repair_resp.text,
                        parsed=parsed,
                    )
                )

                if parsed is None:
                    return Prediction.failure(
                        f"unparseable output from module {module.name}",
                        traces,
                        total_cost,
                    )

            # Feed this module's outputs forward.
            current_inputs.update(parsed)
            all_outputs.update(parsed)

        return Prediction(
            outputs=all_outputs,
            traces=traces,
            cost_usd=total_cost,
        )
