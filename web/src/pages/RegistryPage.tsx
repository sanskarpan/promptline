import { useCallback, useEffect, useState } from "react";
import { api, type ActivePrompt, type RegistryEntry } from "../api";
import { Panel } from "../components/Panel";

export function RegistryPage() {
  const [program, setProgram] = useState("");
  const [loadedProgram, setLoadedProgram] = useState("");
  const [entries, setEntries] = useState<RegistryEntry[]>([]);
  const [active, setActive] = useState<ActivePrompt | null>(null);
  const [selected, setSelected] = useState<RegistryEntry | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionMsg, setActionMsg] = useState<string | null>(null);

  const load = useCallback(async (prog: string) => {
    if (!prog) return;
    setError(null);
    setSelected(null);
    try {
      setEntries(await api.registryList(prog));
      setLoadedProgram(prog);
      try {
        setActive(await api.activePrompt(prog));
      } catch {
        setActive(null); // no active prompt yet
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  // Load the first program's versions automatically if a run created one?  No
  // program listing endpoint exists, so the user enters the program name.
  useEffect(() => {
    setActionMsg(null);
  }, [program]);

  const activate = async (entry: RegistryEntry) => {
    if (!window.confirm(`Activate ${entry.id} for ${loadedProgram}?`)) return;
    try {
      await api.registryActivate(loadedProgram, entry.id);
      setActionMsg(`activated ${entry.id}`);
      await load(loadedProgram);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const rollback = async () => {
    if (!window.confirm(`Rollback active prompt for ${loadedProgram}?`)) return;
    try {
      const res = await api.registryRollback(loadedProgram);
      setActionMsg(`rolled back to ${res.prompt_id}`);
      await load(loadedProgram);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <>
      <Panel title="Registry">
        <div className="form-row">
          <div className="field">
            <label>Program</label>
            <input
              value={program}
              onChange={(e) => setProgram(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && load(program)}
              size={24}
            />
          </div>
          <button onClick={() => load(program)}>Load</button>
          {loadedProgram && (
            <button className="danger" onClick={rollback}>
              Rollback
            </button>
          )}
        </div>
        {error && <div className="error-text">{error}</div>}
        {actionMsg && <div className="ok">{actionMsg}</div>}
      </Panel>

      {loadedProgram && (
        <Panel title={`Versions — ${loadedProgram}`}>
          <table>
            <thead>
              <tr>
                <th>Prompt ID</th>
                <th>Created</th>
                <th>Run</th>
                <th>Mean Score</th>
                <th></th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {entries.length === 0 && (
                <tr>
                  <td colSpan={6} className="dim">
                    no versions
                  </td>
                </tr>
              )}
              {entries.map((e) => {
                const isActive = active?.prompt_id === e.id;
                return (
                  <tr
                    key={e.id}
                    className="clickable"
                    onClick={() => setSelected(e)}
                  >
                    <td>{e.id}</td>
                    <td className="dim">{e.created_at}</td>
                    <td className="dim">{e.run_id}</td>
                    <td>
                      {typeof e.mean_score === "number"
                        ? e.mean_score.toFixed(4)
                        : "—"}
                    </td>
                    <td>
                      {isActive && <span className="badge status-pass">ACTIVE</span>}
                    </td>
                    <td>
                      {!isActive && (
                        <button
                          onClick={(ev) => {
                            ev.stopPropagation();
                            activate(e);
                          }}
                        >
                          Activate
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>
      )}

      {selected && (
        <Panel title={`Preview — ${selected.id}`}>
          {active?.prompt_id === selected.id ? (
            Object.entries(active.modules).map(([name, mod]) => (
              <div key={name}>
                <div className="dim" style={{ fontSize: 11, marginTop: 6 }}>
                  MODULE {name.toUpperCase()} ({mod.demos.length} demos)
                </div>
                <pre className="mono-block">{mod.instruction}</pre>
              </div>
            ))
          ) : (
            <div className="dim">
              instruction preview is only served for the active prompt — activate
              this version to inspect it
            </div>
          )}
        </Panel>
      )}
    </>
  );
}
