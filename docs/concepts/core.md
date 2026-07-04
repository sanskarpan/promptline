# Core concepts

The core library (`promptline/core/`) defines the small set of abstractions everything else builds on.

## PromptProgram, Signature, Module

The unit being optimized is a `PromptProgram` (`promptline.core.program`): an ordered list of `Module`s, each with a name and a `Signature`. A `Signature` (`promptline.core.types`) is DSPy-style structured I/O without the DSPy dependency:

- `instruction` — the system-prompt text (this is what optimizers mutate);
- `inputs` / `outputs` — named `Field`s.

`Signature.render_system()` renders the system message: instruction, input field list, and the output contract ("Answer with each output field as `[[name]]: value`"). `Signature.parse_output()` parses `[[field]]:` sections back into a dict; all declared output fields must be present or parsing fails (single-output signatures fall back to the whole reply).

Most programs are one module, built via:

```python
PromptProgram.simple(instruction=..., inputs=["conversation"], outputs=["answer"], name="support")
```

### Candidates and ModuleState

The program's *structure* is fixed; its *content* lives in a `Candidate`: `{module_name: ModuleState(instruction, demos)}` plus `id`, `parent_ids` (lineage), and the `optimizer` that produced it. `Candidate.seed(...)` creates a root; `candidate.child(...)` records parentage. Few-shot `Demo`s are rendered as alternating user/assistant message pairs before the real input.

### Execution and traces

`await program.run(example, candidate, client, cfg)` executes modules in order, feeding each module's parsed outputs forward as inputs to the next, and returns a `Prediction`:

- `outputs` — merged parsed output fields;
- `traces` — one `Trace` per LLM call (module, system prompt, user prompt, raw output, parsed dict). GEPA's reflection and MIPRO's demo mining consume these;
- `cost_usd`, `failed`, `failure_reason`.

If parsing fails, the program makes exactly one **repair attempt** (re-prompting with `REPAIR_PROMPT`); if that also fails the prediction is returned as `failed=True` — a bad rollout never crashes a run.

`ModelConfig` carries the model roles (`task_model`, `reflection_model`, `judge_model`) and sampling params (`temperature`, `max_tokens`).

## LLMClient

`promptline.core.llm` defines the single interface for all LLM traffic:

```python
class LLMClient(Protocol):
    async def complete(self, call: LLMCall) -> LLMResponse: ...
```

`LLMCall` is a frozen pydantic model (model, messages, temperature, max_tokens, seed) with a stable `key()` — a SHA-256 over its sorted JSON — used as the cache key. `LLMResponse` carries text, token counts, `cost_usd`, and a `cached` flag.

Implementations:

- **`OpenRouterClient`** (`promptline.core.openrouter`) — the real adapter. BYO key via `OPENROUTER_API_KEY`; retries with exponential backoff on 429/5xx and network errors (`max_retries` additional attempts), raises `LLMError` otherwise.
- **`FakeLLMClient`** — deterministic test double. Scripted with either a list of responses (popped in order; exhaustion raises `LLMError`) or a callable `(LLMCall) -> str`. The CLI builds one automatically when the `PROMPTLINE_FAKE_SCRIPT` env var points at a JSON file with `{"responses": [...], "keyed": [{"contains": ..., "response": ...}]}`.

## Cache

`promptline.core.cache` provides `LLMCache` (SQLite table keyed on `LLMCall.key()`) and `CachingClient`, a decorator client that checks the cache before delegating to the inner client and stores every miss. Cache hits come back with `cached=True` and cost nothing.

Because the key covers model + messages + params + seed, reruns and resumed runs replay identical calls for free, and recorded runs double as cassette-style test fixtures. The CLI wires `OpenRouterClient` inside `CachingClient` with the cache at `<registry.path>/cache.db`.

## Configuration

`promptline.core.config` loads `promptline.yaml` into `PromptlineConfig` with sections: `program` (name, instruction, inputs, outputs), `models` (task/reflection/judge), `dataset` (kind: jsonl, path), `budget` (max_rollouts, max_cost_usd), `gate` (alpha, min_examples, certificate, min_kappa), `registry` (path). `promptline init` writes a commented starter file.
