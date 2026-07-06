import { Link, useLocation } from "react-router-dom";
import { Panel } from "../components/Panel";

/**
 * Catch-all for unknown /ui/* routes. Rendered inside <Layout> so the sidebar
 * and nav stay visible instead of stranding the user on React Router's bare
 * default error boundary.
 */
export function NotFound() {
  const { pathname } = useLocation();
  return (
    <Panel title="Not Found">
      <p className="dim" data-testid="not-found">
        No page at <code>{pathname}</code>.
      </p>
      <p>
        <Link to="/ui/runs">← Back to Runs</Link>
      </p>
    </Panel>
  );
}
