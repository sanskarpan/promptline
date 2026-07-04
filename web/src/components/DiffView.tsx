import { diffWordsWithSpace } from "diff";

/** Word-level diff of two instruction texts, green inserts / red deletes. */
export function DiffView({ before, after }: { before: string; after: string }) {
  const parts = diffWordsWithSpace(before, after);
  return (
    <div className="diff-view" data-testid="diff-view">
      {parts.map((part, i) =>
        part.added ? (
          <span key={i} className="diff-ins">
            {part.value}
          </span>
        ) : part.removed ? (
          <span key={i} className="diff-del">
            {part.value}
          </span>
        ) : (
          <span key={i}>{part.value}</span>
        ),
      )}
    </div>
  );
}
