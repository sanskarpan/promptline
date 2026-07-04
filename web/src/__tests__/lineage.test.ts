import { describe, expect, it } from "vitest";
import { layoutLineage } from "../state/lineage";
import type { LineageNode } from "../state/runReducer";

function node(id: string, parents: string[] = []): LineageNode {
  return { id, parents, source: null, score: null, instruction: null };
}

describe("layoutLineage", () => {
  it("lays out a diamond in 3 layers", () => {
    // root -> a, root -> b, a+b -> merge
    const nodes = {
      root: node("root"),
      a: node("a", ["root"]),
      b: node("b", ["root"]),
      merge: node("merge", ["a", "b"]),
    };
    const edges: [string, string][] = [
      ["root", "a"],
      ["root", "b"],
      ["a", "merge"],
      ["b", "merge"],
    ];
    const layout = layoutLineage(nodes, edges);
    expect(layout.layers).toBe(3);
    const byId = Object.fromEntries(layout.nodes.map((n) => [n.id, n]));
    expect(byId["root"].layer).toBe(0);
    expect(byId["a"].layer).toBe(1);
    expect(byId["b"].layer).toBe(1);
    expect(byId["merge"].layer).toBe(2);
    // a and b stack in distinct rows of the same column.
    expect(byId["a"].row).not.toBe(byId["b"].row);
    expect(byId["a"].x).toBe(byId["b"].x);
    expect(layout.edges).toHaveLength(4);
  });

  it("handles empty input", () => {
    const layout = layoutLineage({}, []);
    expect(layout.nodes).toEqual([]);
    expect(layout.layers).toBe(0);
  });

  it("drops edges that reference unknown nodes", () => {
    const layout = layoutLineage({ a: node("a") }, [["ghost", "a"]]);
    expect(layout.edges).toEqual([]);
    expect(layout.nodes[0].layer).toBe(0);
  });

  it("places a child right of its deepest parent", () => {
    const nodes = {
      r: node("r"),
      mid: node("mid", ["r"]),
      late: node("late", ["r", "mid"]),
    };
    const layout = layoutLineage(nodes, [
      ["r", "mid"],
      ["r", "late"],
      ["mid", "late"],
    ]);
    const byId = Object.fromEntries(layout.nodes.map((n) => [n.id, n]));
    expect(byId["late"].layer).toBe(2);
  });
});
