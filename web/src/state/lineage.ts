/** Layered lineage layout: BFS depth -> column, index within layer -> row. */

import type { LineageNode } from "./runReducer";

export interface LayoutNode {
  id: string;
  layer: number;
  row: number;
  x: number;
  y: number;
}

export interface LayoutEdge {
  from: string;
  to: string;
}

export interface LineageLayout {
  nodes: LayoutNode[];
  edges: LayoutEdge[];
  layers: number;
  maxRows: number;
  width: number;
  height: number;
}

export const NODE_W = 130;
export const NODE_H = 44;
export const GAP_X = 70;
export const GAP_Y = 18;

/**
 * Compute a layered layout. Layer = longest path from any root (so a child
 * always sits right of all its parents); nodes keep insertion order in rows.
 */
export function layoutLineage(
  nodes: Record<string, LineageNode>,
  edges: [string, string][],
): LineageLayout {
  const ids = Object.keys(nodes);
  // parent -> children adjacency, restricted to known nodes.
  const known = new Set(ids);
  const validEdges = edges.filter(([a, b]) => known.has(a) && known.has(b));

  const layerOf: Record<string, number> = {};
  const depth = (id: string, seen: Set<string>): number => {
    if (id in layerOf) return layerOf[id];
    if (seen.has(id)) return 0; // cycle guard
    seen.add(id);
    const parents = nodes[id].parents.filter((p) => known.has(p));
    const d =
      parents.length === 0
        ? 0
        : 1 + Math.max(...parents.map((p) => depth(p, seen)));
    layerOf[id] = d;
    return d;
  };
  for (const id of ids) depth(id, new Set());

  const rows: Record<number, number> = {};
  const out: LayoutNode[] = ids.map((id) => {
    const layer = layerOf[id];
    const row = rows[layer] ?? 0;
    rows[layer] = row + 1;
    return {
      id,
      layer,
      row,
      x: layer * (NODE_W + GAP_X),
      y: row * (NODE_H + GAP_Y),
    };
  });

  const layers = ids.length ? Math.max(...out.map((n) => n.layer)) + 1 : 0;
  const maxRows = ids.length ? Math.max(...Object.values(rows)) : 0;
  return {
    nodes: out,
    edges: validEdges.map(([from, to]) => ({ from, to })),
    layers,
    maxRows,
    width: layers * (NODE_W + GAP_X),
    height: maxRows * (NODE_H + GAP_Y),
  };
}
