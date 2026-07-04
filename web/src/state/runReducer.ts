/** Pure reducer that folds a stream of RunEvents into dashboard state. */

import type { RunEvent } from "../api";

export const EVENT_LOG_MAX = 500;

export interface ScorePoint {
  /** Monotone index within its series (order of arrival). */
  index: number;
  score: number;
  candidateId: string | null;
  kind: "full_eval" | "minibatch";
}

export interface BudgetState {
  rolloutsUsed: number;
  costUsed: number;
  maxRollouts: number | null;
  maxCostUsd: number | null;
}

export interface LineageNode {
  id: string;
  parents: string[];
  source: string | null;
  score: number | null;
  instruction: string | null;
}

export interface RunState {
  status: "idle" | "running" | "finished";
  bestId: string | null;
  bestScore: number | null;
  fullEvals: ScorePoint[];
  minibatches: ScorePoint[];
  budget: BudgetState;
  /** candidate id -> node, insertion-ordered. */
  nodes: Record<string, LineageNode>;
  /** [parentId, childId] pairs. */
  edges: [string, string][];
  /** Ring buffer of the last EVENT_LOG_MAX events. */
  eventLog: RunEvent[];
  eventCount: number;
}

export const initialRunState: RunState = {
  status: "idle",
  bestId: null,
  bestScore: null,
  fullEvals: [],
  minibatches: [],
  budget: { rolloutsUsed: 0, costUsed: 0, maxRollouts: null, maxCostUsd: null },
  nodes: {},
  edges: [],
  eventLog: [],
  eventCount: 0,
};

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v.length > 0 ? v : null;
}

/** Extract parent ids from the payload variants used across optimizers. */
export function extractParents(payload: Record<string, unknown>): string[] {
  const out: string[] = [];
  const push = (v: unknown) => {
    const s = str(v);
    if (s && !out.includes(s)) out.push(s);
  };
  const many = payload["parents"] ?? payload["parent_ids"];
  if (Array.isArray(many)) many.forEach(push);
  push(payload["parent_id"]);
  return out;
}

export function runReducer(state: RunState, event: RunEvent): RunState {
  const log =
    state.eventLog.length >= EVENT_LOG_MAX
      ? [...state.eventLog.slice(state.eventLog.length - EVENT_LOG_MAX + 1), event]
      : [...state.eventLog, event];
  const next: RunState = {
    ...state,
    eventLog: log,
    eventCount: state.eventCount + 1,
  };
  const p = event.payload ?? {};

  switch (event.type) {
    case "run_started":
      next.status = "running";
      break;

    case "candidate_proposed": {
      const id = str(p["candidate_id"]) ?? str(p["id"]);
      if (!id) break;
      const parents = extractParents(p);
      const node: LineageNode = next.nodes[id] ?? {
        id,
        parents,
        source: str(p["source"]),
        score: null,
        instruction: null,
      };
      const nodes = {
        ...next.nodes,
        [id]: {
          ...node,
          parents,
          source: node.source ?? str(p["source"]),
          instruction: node.instruction ?? str(p["instruction"]),
        },
      };
      const edges = [...next.edges];
      for (const parent of parents) {
        if (!edges.some(([a, b]) => a === parent && b === id)) {
          edges.push([parent, id]);
        }
      }
      next.nodes = nodes;
      next.edges = edges;
      break;
    }

    case "minibatch_scored": {
      const score = num(p["score"]) ?? num(p["mean_score"]);
      if (score === null) break;
      next.minibatches = [
        ...next.minibatches,
        {
          index: next.minibatches.length,
          score,
          candidateId: str(p["candidate_id"]),
          kind: "minibatch",
        },
      ];
      break;
    }

    case "full_eval": {
      const score = num(p["mean_score"]) ?? num(p["score"]);
      const id = str(p["candidate_id"]);
      if (score !== null) {
        next.fullEvals = [
          ...next.fullEvals,
          { index: next.fullEvals.length, score, candidateId: id, kind: "full_eval" },
        ];
      }
      if (id && score !== null) {
        const existing = next.nodes[id] ?? {
          id,
          parents: [],
          source: null,
          score: null,
          instruction: null,
        };
        next.nodes = { ...next.nodes, [id]: { ...existing, score } };
      }
      break;
    }

    case "budget_tick": {
      next.budget = {
        rolloutsUsed: num(p["rollouts_used"]) ?? state.budget.rolloutsUsed,
        costUsed: num(p["cost_used"]) ?? state.budget.costUsed,
        maxRollouts: num(p["max_rollouts"]) ?? state.budget.maxRollouts,
        maxCostUsd: num(p["max_cost_usd"]) ?? state.budget.maxCostUsd,
      };
      break;
    }

    case "run_finished": {
      next.status = "finished";
      next.bestId = str(p["best_id"]) ?? state.bestId;
      next.bestScore = num(p["best_score"]) ?? state.bestScore;
      break;
    }

    default:
      break;
  }
  return next;
}
