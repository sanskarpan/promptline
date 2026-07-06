import { describe, expect, it } from "vitest";
import { sanitizeRunEvent } from "../api";
import { initialRunState, runReducer } from "../state/runReducer";

describe("sanitizeRunEvent", () => {
  it("passes a well-formed event through unchanged", () => {
    const ev = sanitizeRunEvent({
      type: "full_eval",
      ts: 1700000000,
      payload: { candidate_id: "a", mean_score: 0.7 },
    });
    expect(ev).toEqual({
      type: "full_eval",
      ts: 1700000000,
      payload: { candidate_id: "a", mean_score: 0.7 },
    });
  });

  it("drops non-objects", () => {
    expect(sanitizeRunEvent(null)).toBeNull();
    expect(sanitizeRunEvent(42)).toBeNull();
    expect(sanitizeRunEvent("full_eval")).toBeNull();
    expect(sanitizeRunEvent([1, 2, 3])).toBeNull();
  });

  it("defaults a missing/invalid type to 'unknown' instead of crashing render", () => {
    const ev = sanitizeRunEvent({ ts: 1700000000, payload: {} });
    expect(ev?.type).toBe("unknown");
    // Render code does `ev.type.padEnd(18)`; would throw on undefined.
    expect(() => ev!.type.padEnd(18)).not.toThrow();
  });

  it("defaults a missing/invalid ts to 0 instead of crashing render", () => {
    const noTs = sanitizeRunEvent({ type: "budget_tick", payload: {} });
    expect(noTs?.ts).toBe(0);
    const nanTs = sanitizeRunEvent({ type: "budget_tick", ts: NaN, payload: {} });
    expect(nanTs?.ts).toBe(0);
    // Render code does `new Date(ev.ts * 1000).toISOString()`; would throw on NaN.
    expect(() =>
      new Date(noTs!.ts * 1000).toISOString(),
    ).not.toThrow();
  });

  it("defaults a missing payload to an empty object", () => {
    const ev = sanitizeRunEvent({ type: "run_started", ts: 1 });
    expect(ev?.payload).toEqual({});
  });

  it("produces events the reducer folds without throwing", () => {
    // An object with junk fields is sanitized (not dropped) and folds cleanly.
    const ev = sanitizeRunEvent({ payload: null });
    expect(ev).not.toBeNull();
    expect(ev?.payload).toEqual({});
    const ok = sanitizeRunEvent({ type: 5, ts: "bad" });
    expect(ok).toEqual({ type: "unknown", ts: 0, payload: {} });
    expect(() => runReducer(initialRunState, ok!)).not.toThrow();
  });
});
