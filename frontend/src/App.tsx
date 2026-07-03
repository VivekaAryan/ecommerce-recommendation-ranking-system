import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import RecommendShop from "./RecommendShop";
import type {
  JobDetail,
  MetricsResponse,
  SimulatorLogs,
  SystemStatus,
  UserInfo,
} from "./types";
import "./App.css";

type Tab = "overview" | "pipeline" | "recommend" | "metrics" | "simulator";

const PIPELINE_STEPS = [
  { task: "download", title: "Download Dataset", desc: "Download Amazon Reviews 2023 (7 categories, image-backed catalog)" },
  { task: "features", title: "Build Features", desc: "Compute batch user and item feature tables" },
  { task: "measure_skew", title: "Measure Skew", desc: "Compare batch vs online feature distributions" },
  { task: "train_retrieval", title: "Train Retrieval", desc: "Two-tower model + FAISS ANN index" },
  { task: "train_ranker", title: "Train Ranker", desc: "Sequential multi-task + LightGBM hybrid" },
  { task: "simulator", title: "Run Simulator", desc: "Marketplace simulation with propensity logging" },
  { task: "evaluate", title: "Evaluate", desc: "IPS, doubly-robust OPE, interference experiments" },
];

const FULL_PIPELINE = ["download", "features", "measure_skew", "train_retrieval", "train_ranker", "simulator", "evaluate"];

function formatStatValue(value: unknown): string {
  if (typeof value === "number") return value.toLocaleString();
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, entry]) => `${key}: ${typeof entry === "number" ? entry.toLocaleString() : String(entry)}`)
      .join(" · ");
  }
  return String(value ?? "—");
}

function isNestedStat(value: unknown): boolean {
  return Array.isArray(value) || (value !== null && typeof value === "object");
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [jobs, setJobs] = useState<JobDetail[]>([]);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [logs, setLogs] = useState<SimulatorLogs | null>(null);
  const [selectedUser, setSelectedUser] = useState("");
  const [loading, setLoading] = useState(false);
  const [runningTask, setRunningTask] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [usersError, setUsersError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, j, m, l] = await Promise.all([
        api.status(),
        api.jobs(),
        api.metrics(),
        api.simulatorLogs(30),
      ]);
      setStatus(s);
      setJobs(j);
      setMetrics(m);
      setLogs(l);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    }
  }, []);

  const loadUsers = useCallback(async () => {
    try {
      const u = await api.users(30);
      setUsers(u);
      setUsersError(null);
      if (u.length) {
        setSelectedUser((prev) => prev || u[0].user_id);
      }
    } catch (e) {
      setUsers([]);
      const msg = e instanceof Error ? e.message : "Failed to load users";
      const isConnection =
        msg.includes("Failed to fetch") ||
        msg.includes("NetworkError") ||
        msg.includes("ECONNREFUSED") ||
        msg.includes("Network request failed");
      setUsersError(
        isConnection
          ? "Backend not reachable. Start API: python scripts/run_ui.py --reload (port 8000)"
          : msg
      );
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    if (tab === "recommend") loadUsers();
  }, [tab, loadUsers]);

  const runTask = async (task: string) => {
    setRunningTask(task);
    setError(null);
    try {
      await api.runJob(task);
      await refresh();
      if (task === "download") {
        await loadUsers();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Job failed");
    } finally {
      setRunningTask(null);
    }
  };

  const runFullPipeline = async () => {
    setLoading(true);
    setError(null);
    try {
      for (const task of FULL_PIPELINE) {
        setRunningTask(task);
        const { id } = await api.runJob(task);
        let done = false;
        while (!done) {
          await new Promise((r) => setTimeout(r, 1500));
          const job = await api.job(id);
          if (job.status === "completed" || job.status === "failed") {
            done = true;
            if (job.status === "failed") throw new Error(job.error || `${task} failed`);
          }
        }
      }
      await refresh();
      await loadUsers();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Pipeline failed");
    } finally {
      setLoading(false);
      setRunningTask(null);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Recsys Platform Tester</h1>
          <p>Test retrieval, ranking, simulator, and evaluation end-to-end</p>
        </div>
        <span className={`badge ${status?.ready ? "ok" : "warn"}`}>
          {status?.ready ? "● Dataset ready" : "○ Setup required"}
        </span>
      </header>

      <nav className="tabs">
        {(["overview", "pipeline", "recommend", "metrics", "simulator"] as Tab[]).map((t) => (
          <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t === "recommend" ? "Shop" : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>

      {error && <div className="error">{error}</div>}

      {tab === "overview" && (
        <div className="grid grid-2">
          <div className="card">
            <h2>Dataset Stats</h2>
            {status?.dataset ? (
              <div className="stat-grid">
                {Object.entries(status.dataset).map(([k, v]) => (
                  <div
                    key={k}
                    className="stat"
                    style={isNestedStat(v) ? { gridColumn: "1 / -1" } : undefined}
                  >
                    <div className="label">{k.replace(/_/g, " ")}</div>
                    <div
                      className="value"
                      style={isNestedStat(v) ? { fontSize: "0.85rem", lineHeight: 1.5 } : undefined}
                    >
                      {formatStatValue(v)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty">No dataset prepared. Run the pipeline first.</p>
            )}
          </div>

          <div className="card">
            <h2>Artifacts</h2>
            <div className="artifact-list">
              {status?.artifacts.map((a) => (
                <div key={a.name} className="artifact">
                  <span style={{ display: "flex", alignItems: "center" }}>
                    <span className={`dot ${a.exists ? "ok" : "missing"}`} />
                    {a.name}
                  </span>
                  <span className="mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                    {a.exists ? "ready" : "missing"}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h2>Quick Start</h2>
            <p style={{ color: "var(--text-muted)", marginBottom: 16, fontSize: "0.9rem" }}>
              Download real Amazon data and run the full pipeline, then test recommendations in the Shop tab.
            </p>
            <button className="btn btn-primary btn-run-all" onClick={runFullPipeline} disabled={loading}>
              {loading ? `Running: ${runningTask}...` : "Run Full Pipeline"}
            </button>
          </div>
        </div>
      )}

      {tab === "pipeline" && (
        <div className="grid grid-2">
          <div className="card">
            <h2>Pipeline Steps</h2>
            <div className="pipeline-grid">
              {PIPELINE_STEPS.map((step) => (
                <div key={step.task} className="pipeline-step">
                  <div className="info">
                    <div className="title">{step.title}</div>
                    <div className="desc">{step.desc}</div>
                  </div>
                  <button
                    className="btn btn-primary"
                    onClick={() => runTask(step.task)}
                    disabled={runningTask !== null}
                  >
                    {runningTask === step.task ? "Running..." : "Run"}
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>Recent Jobs</h2>
            {jobs.length === 0 ? (
              <p className="empty">No jobs yet</p>
            ) : (
              <div className="jobs-list">
                {jobs.map((job) => (
                  <div key={job.id} className="job">
                    <div className="job-header">
                      <strong>{job.task}</strong>
                      <span className={`status ${job.status}`}>{job.status}</span>
                    </div>
                    <div className="mono" style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                      {job.id}
                    </div>
                    {job.result?.message != null && (
                      <div style={{ marginTop: 6, fontSize: "0.8rem" }}>{String(job.result.message)}</div>
                    )}
                    {job.error && <div style={{ marginTop: 6, color: "var(--danger)", fontSize: "0.8rem" }}>{job.error}</div>}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {tab === "recommend" && (
        <RecommendShop
          status={status}
          users={users}
          usersError={usersError}
          selectedUser={selectedUser}
          onUserChange={setSelectedUser}
          onRefreshUsers={loadUsers}
        />
      )}

      {tab === "metrics" && (
        <div className="grid grid-2">
          <div className="card">
            <h2>Off-Policy Evaluation</h2>
            {metrics?.ope ? (
              <div className="stat-grid">
                {Object.entries(metrics.ope).map(([k, v]) => (
                  <div key={k} className="stat">
                    <div className="label">{k}</div>
                    <div className="value">{v.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty">Run simulator and evaluate first</p>
            )}
          </div>

          <div className="card">
            <h2>Simulator Summary</h2>
            {metrics?.simulator_summary ? (
              <div className="stat-grid">
                {Object.entries(metrics.simulator_summary).map(([k, v]) => (
                  <div key={k} className="stat">
                    <div className="label">{k.replace(/_/g, " ")}</div>
                    <div className="value">
                      {typeof v === "number"
                        ? v < 1 && v > 0
                          ? v.toFixed(4)
                          : v.toLocaleString()
                        : formatStatValue(v)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty">No simulator data</p>
            )}
          </div>

          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h2>Feature Skew (Batch vs Online)</h2>
            {metrics?.skew && metrics.skew.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Batch Mean</th>
                      <th>Online Mean</th>
                      <th>Abs Delta</th>
                      <th>Rel Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.skew.map((row) => (
                      <tr key={String(row.feature)}>
                        <td>{row.feature}</td>
                        <td className="mono">{Number(row.batch_mean).toFixed(3)}</td>
                        <td className="mono">{Number(row.online_mean).toFixed(3)}</td>
                        <td className="mono">{Number(row.absolute_delta).toFixed(3)}</td>
                        <td className="mono">{(Number(row.relative_delta) * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty">Run measure skew step first</p>
            )}
          </div>
        </div>
      )}

      {tab === "simulator" && (
        <div className="card">
          <h2>Simulator Logs ({logs?.total ?? 0} impressions)</h2>
          {logs && logs.rows.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Item</th>
                    <th>Pos</th>
                    <th>Clicked</th>
                    <th>Purchased</th>
                    <th>Propensity</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {logs.rows.map((row, i) => (
                    <tr key={i}>
                      <td className="mono">{String(row.user_id)}</td>
                      <td className="mono">{String(row.item_id)}</td>
                      <td>{String(row.position)}</td>
                      <td>{row.clicked ? "✓" : "–"}</td>
                      <td>{row.purchased ? "✓" : "–"}</td>
                      <td className="mono">{Number(row.propensity).toFixed(3)}</td>
                      <td className="mono">{Number(row.score).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="empty">No simulator logs. Run the simulator from the Pipeline tab.</p>
          )}
        </div>
      )}
    </div>
  );
}
