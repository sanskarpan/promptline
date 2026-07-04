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

/** Open the SSE stream for a run. Caller must close() the returned source. */
export function openRunEvents(
  runId: string,
  onEvent: (ev: RunEvent) => void,
  onError?: (err: Event) => void,
): EventSource {
  const source = new EventSource(
    `${apiBase}/runs/${encodeURIComponent(runId)}/events`,
  );
  source.addEventListener("run_event", (e) => {
    try {
      onEvent(JSON.parse((e as MessageEvent).data) as RunEvent);
    } catch {
      /* skip malformed lines */
    }
  });
  if (onError) source.onerror = onError;
  return source;
}
