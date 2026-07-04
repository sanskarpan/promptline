import { describe, expect, it } from "vitest";
import type { RunEvent } from "../api";
import {
  EVENT_LOG_MAX,
  extractParents,
  initialRunState,
  runReducer,
  type RunState,
} from "../state/runReducer";

function ev(type: RunEvent["type"], payload: Record<string, unknown> = {}): RunEvent {
  return { type, payload, ts: 1700000000 };
}

function fold(events: RunEvent[], start: RunState = initialRunState): RunState {
  return events.reduce(runReducer, start);
}

describe("runReducer", () => {
  it("tracks status through run_started and run_finished", () => {
    let s = fold([ev("run_started", { optimizer: "gepa" })]);
    expect(s.status).toBe("running");
    s = runReducer(s, ev("run_finished", { best_id: "c9", best_score: 0.91 }));
    expect(s.status).toBe("finished");
    expect(s.bestId).toBe("c9");
    expect(s.bestScore).toBe(0.91);
  });

  it("accumulates score series from full_eval and minibatch_scored", () => {
    const s = fold([
      ev("minibatch_scored", { candidate_id: "a", score: 0.5 }),
      ev("minibatch_scored", { candidate_id: "b", mean_score: 0.6 }),
      ev("full_eval", { candidate_id: "a", mean_score: 0.7, n: 50 }),
      ev("full_eval", { candidate_id: "b", mean_score: 0.75, n: 50 }),
    ]);
    expect(s.minibatches.map((p) => p.score)).toEqual([0.5, 0.6]);
    expect(s.fullEvals.map((p) => p.score)).toEqual([0.7, 0.75]);
    expect(s.fullEvals[1].candidateId).toBe("b");
    expect(s.fullEvals.map((p) => p.index)).toEqual([0, 1]);
  });

  it("keeps budget from the last budget_tick", () => {
    const s = fold([
      ev("budget_tick", { rollouts_used: 10, cost_used: 0.1, max_rollouts: 100, max_cost_usd: 5 }),
      ev("budget_tick", { rollouts_used: 42, cost_used: 0.9 }),
    ]);
    expect(s.budget.rolloutsUsed).toBe(42);
    expect(s.budget.costUsed).toBe(0.9);
    // Caps persist from the earlier tick when omitted later.
    expect(s.budget.maxRollouts).toBe(100);
    expect(s.budget.maxCostUsd).toBe(5);
  });

  it("builds lineage nodes and edges from candidate_proposed + full_eval", () => {
    const s = fold([
      ev("candidate_proposed", { candidate_id: "root", instruction: "v0" }),
      ev("candidate_proposed", {
        candidate_id: "child",
        parent_id: "root",
        source: "gradient",
        instruction: "v1",
      }),
      ev("candidate_proposed", { candidate_id: "merge", parents: ["root", "child"] }),
      ev("full_eval", { candidate_id: "child", mean_score: 0.8 }),
    ]);
    expect(Object.keys(s.nodes)).toEqual(["root", "child", "merge"]);
    expect(s.nodes["child"].parents).toEqual(["root"]);
    expect(s.nodes["child"].source).toBe("gradient");
    expect(s.nodes["child"].instruction).toBe("v1");
    expect(s.nodes["child"].score).toBe(0.8);
    expect(s.nodes["merge"].parents).toEqual(["root", "child"]);
    expect(s.edges).toEqual([
      ["root", "child"],
      ["root", "merge"],
      ["child", "merge"],
    ]);
  });

  it("creates a node for full_eval of an unseen candidate", () => {
    const s = fold([ev("full_eval", { candidate_id: "solo", mean_score: 0.4 })]);
    expect(s.nodes["solo"].score).toBe(0.4);
    expect(s.nodes["solo"].parents).toEqual([]);
  });

  it("caps the event log ring buffer at EVENT_LOG_MAX", () => {
    const events = Array.from({ length: EVENT_LOG_MAX + 40 }, (_, i) =>
      ev("minibatch_scored", { score: i }),
    );
    const s = fold(events);
    expect(s.eventLog).toHaveLength(EVENT_LOG_MAX);
    expect(s.eventCount).toBe(EVENT_LOG_MAX + 40);
    // Oldest events were evicted; the last one kept is the newest.
    expect(s.eventLog[0].payload["score"]).toBe(40);
    expect(s.eventLog[EVENT_LOG_MAX - 1].payload["score"]).toBe(EVENT_LOG_MAX + 39);
  });

  it("is pure: does not mutate the previous state", () => {
    const s0 = fold([ev("candidate_proposed", { candidate_id: "a" })]);
    const frozen = JSON.stringify(s0);
    runReducer(s0, ev("full_eval", { candidate_id: "a", mean_score: 1 }));
    runReducer(s0, ev("budget_tick", { rollouts_used: 5 }));
    expect(JSON.stringify(s0)).toBe(frozen);
  });
});

describe("extractParents", () => {
  it("handles parents, parent_ids and parent_id variants", () => {
    expect(extractParents({ parents: ["a", "b"] })).toEqual(["a", "b"]);
    expect(extractParents({ parent_ids: ["a"] })).toEqual(["a"]);
    expect(extractParents({ parent_id: "x" })).toEqual(["x"]);
    expect(extractParents({ parents: ["a"], parent_id: "a" })).toEqual(["a"]);
    expect(extractParents({})).toEqual([]);
  });
});
