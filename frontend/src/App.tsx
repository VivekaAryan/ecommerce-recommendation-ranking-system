import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type {
  JobDetail,
  MetricsResponse,
  RecommendResponse,
  SimulatorLogs,
  SystemStatus,
  UserInfo,
} from "./types";
import "./App.css";

type Tab = "overview" | "pipeline" | "recommend" | "metrics" | "simulator";

const PIPELINE_STEPS = [
  { task: "download", title: "Prepare Dataset", desc: "Download or generate synthetic Amazon Electronics data" },
  { task: "features", title: "Build Features", desc: "Compute batch user and item feature tables" },
  { task: "measure_skew", title: "Measure Skew", desc: "Compare batch vs online feature distributions" },
  { task: "train_retrieval", title: "Train Retrieval", desc: "Two-tower model + FAISS ANN index" },
  { task: "train_ranker", title: "Train Ranker", desc: "Sequential multi-task + LightGBM hybrid" },
  { task: "simulator", title: "Run Simulator", desc: "Marketplace simulation with propensity logging" },
  { task: "evaluate", title: "Evaluate", desc: "IPS, doubly-robust OPE, interference experiments" },
];

const FULL_PIPELINE = ["download", "features", "measure_skew", "train_retrieval", "train_ranker", "simulator", "evaluate"];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [jobs, setJobs] = useState<JobDetail[]>([]);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [logs, setLogs] = useState<SimulatorLogs | null>(null);
  const [recommendation, setRecommendation] = useState<RecommendResponse | null>(null);
  const [selectedUser, setSelectedUser] = useState("");
  const [synthetic, setSynthetic] = useState(true);
  const [loading, setLoading] = useState(false);
  const [runningTask, setRunningTask] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      if (u.length && !selectedUser) setSelectedUser(u[0].user_id);
    } catch {
      setUsers([]);
    }
  }, [selectedUser]);

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
      await api.runJob(task, synthetic);
      await refresh();
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
        const { id } = await api.runJob(task, synthetic);
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

  const handleRecommend = async () => {
    if (!selectedUser) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.recommend(selectedUser);
      setRecommendation(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Recommendation failed");
    } finally {
      setLoading(false);
    }
  };

  const budgetMs = { retrieval: 20, prerank: 10, ranking: 50, reranking: 30, total: 100 };

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
            {t.charAt(0).toUpperCase() + t.slice(1)}
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
                  <div key={k} className="stat">
                    <div className="label">{k.replace(/_/g, " ")}</div>
                    <div className="value">{typeof v === "number" ? v.toLocaleString() : v}</div>
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
              Run the full pipeline with synthetic data, then test recommendations.
            </p>
            <button className="btn btn-primary btn-run-all" onClick={runFullPipeline} disabled={loading}>
              {loading ? `Running: ${runningTask}...` : "Run Full Pipeline (Synthetic)"}
            </button>
          </div>
        </div>
      )}

      {tab === "pipeline" && (
        <div className="grid grid-2">
          <div className="card">
            <h2>Pipeline Steps</h2>
            <label className="toggle">
              <input type="checkbox" checked={synthetic} onChange={(e) => setSynthetic(e.target.checked)} />
              Use synthetic data (fast local dev)
            </label>
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
        <div className="grid grid-2">
          <div className="card" style={{ gridColumn: "1 / -1" }}>
            <h2>Live Recommendation</h2>
            <div className="recommend-form">
              <select className="select" value={selectedUser} onChange={(e) => setSelectedUser(e.target.value)}>
                <option value="">Select user...</option>
                {users.map((u) => (
                  <option key={u.user_id} value={u.user_id}>
                    {u.user_id} ({u.interaction_count} interactions)
                  </option>
                ))}
              </select>
              <button className="btn btn-primary" onClick={handleRecommend} disabled={!selectedUser || loading}>
                {loading ? "Generating..." : "Get Recommendations"}
              </button>
              <button className="btn btn-secondary" onClick={loadUsers}>
                Refresh Users
              </button>
            </div>

            {recommendation && (
              <>
                {recommendation.user_history.length > 0 && (
                  <div>
                    <h3>Recent History</h3>
                    <div className="history-chips">
                      {recommendation.user_history.map((id) => (
                        <span key={id} className="chip">{id}</span>
                      ))}
                    </div>
                  </div>
                )}

                <h3 style={{ marginTop: 24 }}>Recommended Slate</h3>
                <div className="slate-grid">
                  {recommendation.slate.map((item) => (
                    <div key={item.item_id} className="slate-item">
                      <div className="rank">#{item.position + 1}</div>
                      <div>
                        <div className="title">{item.title}</div>
                        <div className="meta">
                          {item.category} · {item.item_id}
                          {item.price != null && ` · $${item.price.toFixed(2)}`}
                        </div>
                      </div>
                      <div className="score">{item.score.toFixed(3)}</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          {recommendation && (
            <div className="card">
              <h2>Latency Profile</h2>
              <div className="latency-bars">
                {(["retrieval", "prerank", "ranking", "reranking", "total"] as const).map((stage) => {
                  const ms = recommendation.latency[`${stage}_ms` as keyof typeof recommendation.latency] as number;
                  const budget = budgetMs[stage];
                  const pct = Math.min((ms / budget) * 100, 100);
                  const ok = recommendation.latency.within_budget[stage];
                  return (
                    <div key={stage} className="latency-row">
                      <span>{stage}</span>
                      <div className="latency-bar">
                        <div className={`fill ${ok ? "" : "over"}`} style={{ width: `${pct}%` }} />
                      </div>
                      <span className="mono">{ms.toFixed(1)}ms</span>
                      <span>{ok ? "✓" : "✗"}</span>
                    </div>
                  );
                })}
              </div>
              <p style={{ marginTop: 12, fontSize: "0.8rem", color: "var(--text-muted)" }}>
                Budget: 100ms total (20 + 10 + 50 + 30)
              </p>
            </div>
          )}
        </div>
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
                    <div className="value">{typeof v === "number" && v < 1 ? v.toFixed(4) : v}</div>
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
