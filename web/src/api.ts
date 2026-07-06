/** Typed client for the Promptline API. */

export const apiBase: string = import.meta.env.VITE_API_BASE ?? "";

// ---------------------------------------------------------------------------
// Types (mirror promptline server / pydantic models)
// ---------------------------------------------------------------------------

export interface ModuleState {
  instruction: string;
  demos: unknown[];
}

export interface ActivePrompt {
  program: string;
  prompt_id: string;
  modules: Record<string, ModuleState>;
  activated_at: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  summary?: Record<string, unknown> | null;
  [key: string]: unknown;
}

export interface RunEvent {
  type:
    | "run_started"
    | "candidate_proposed"
    | "minibatch_scored"
    | "full_eval"
    | "pareto_updated"
    | "merge_attempted"
    | "budget_tick"
    | "run_finished";
  payload: Record<string, unknown>;
  ts: number;
  /**
   * Monotonic per-connection index assigned by {@link openRunEvents}. Resets
   * to 0 on reconnect so the reducer can drop replayed events (the server
   * re-streams a running run from offset 0 on every connection). Undefined
   * for events not sourced from the live SSE stream (e.g. in unit tests).
   */
  seq?: number;
}

export interface StartRunRequest {
  optimizer: string;
  data_path: string;
  budget?: number | null;
}

export interface GateRequestBody {
  program: string;
  incumbent_id?: string;
  candidate_ids: string[];
  dev_path: string;
  val_path: string;
}

export interface CandidateGateResult {
  candidate_id: string;
  mean_delta: number;
  ci_low: number;
  ci_high: number;
  p_value: number;
  holm_significant: boolean;
  dev_mean: number;
  incumbent_dev_mean: number;
}

export interface GateReport {
  program: string;
  incumbent_id: string;
  results: CandidateGateResult[];
  winner_id: string | null;
  val_mean_delta: number | null;
  val_ci_low: number | null;
  val_ci_high: number | null;
  verdict: "promote" | "reject";
  flags: string[];
  spot_samples: Record<string, unknown>[];
  warnings: string[];
  created_at: string;
}

export interface RegistryEntry {
  id: string;
  created_at: string;
  run_id: string;
  mean_score: number | null;
  [key: string]: unknown;
}

export interface CalibrationCertificate {
  judge_name: string;
  criterion: string;
  kappa: number;
  spearman: number;
  n_holdout: number;
  threshold: number;
  passed: boolean;
  degenerate: boolean;
  confusion: number[][];
  label_min: number;
  label_max: number;
  created_at: string;
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
      else if (body) detail = JSON.stringify(body);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const api = {
  activePrompt: (program: string) =>
    request<ActivePrompt>(`/prompts/${encodeURIComponent(program)}/active`),

  startRun: (body: StartRunRequest) =>
    request<{ run_id: string }>("/runs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  listRuns: () => request<RunSummary[]>("/runs"),

  getRun: (runId: string) =>
    request<RunSummary>(`/runs/${encodeURIComponent(runId)}`),

  gate: (body: GateRequestBody) =>
    request<GateReport>("/gate", { method: "POST", body: JSON.stringify(body) }),

  registryList: (program: string) =>
    request<RegistryEntry[]>(`/registry/${encodeURIComponent(program)}`),

  registryActivate: (program: string, promptId: string, gateReport: object = {}) =>
    request<{ program: string; prompt_id: string }>(
      `/registry/${encodeURIComponent(program)}/activate`,
      {
        method: "POST",
        body: JSON.stringify({ prompt_id: promptId, gate_report: gateReport }),
      },
    ),

  registryRollback: (program: string) =>
    request<{ program: string; prompt_id: string }>(
      `/registry/${encodeURIComponent(program)}/rollback`,
      { method: "POST", body: JSON.stringify({}) },
    ),

  certificates: () =>
    request<CalibrationCertificate[]>("/judges/certificates"),
};

/**
 * Coerce an arbitrary parsed SSE payload into a structurally-valid RunEvent, or
 * return null if it is not even an object. A malformed event must never reach
 * the reducer/render: a missing `ts` would make `new Date(NaN)` throw and a
 * missing `type` would make `type.padEnd(...)` throw, blanking the feed. So we
 * default `ts` to 0 and `type` to "unknown" rather than crashing.
 */
export function sanitizeRunEvent(raw: unknown): RunEvent | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const obj = raw as Record<string, unknown>;
  const type = (
    typeof obj.type === "string" ? obj.type : "unknown"
  ) as RunEvent["type"];
  const ts =
    typeof obj.ts === "number" && Number.isFinite(obj.ts) ? obj.ts : 0;
  const payload =
    typeof obj.payload === "object" && obj.payload !== null
      ? (obj.payload as Record<string, unknown>)
      : {};
  return { type, ts, payload };
}

/** Open the SSE stream for a run. Caller must close() the returned source. */
export function openRunEvents(
  runId: string,
  onEvent: (ev: RunEvent) => void,
  onError?: (err: Event) => void,
): EventSource {
  const source = new EventSource(
    `${apiBase}/runs/${encodeURIComponent(runId)}/events`,
  );
  // Per-connection sequence. The server replays a still-running run from the
  // beginning on every (re)connection, so reset to 0 on error/reconnect and let
  // the reducer's high-water mark drop the replayed prefix.
  let seq = 0;
  source.addEventListener("error", () => {
    seq = 0;
  });
  source.addEventListener("run_event", (e) => {
    let parsed: unknown;
    try {
      parsed = JSON.parse((e as MessageEvent).data);
    } catch {
      return; /* skip malformed (non-JSON) lines */
    }
    const ev = sanitizeRunEvent(parsed);
    if (ev === null) return; /* drop non-object events */
    onEvent({ ...ev, seq: seq++ });
  });
  if (onError) source.onerror = onError;
  return source;
}
