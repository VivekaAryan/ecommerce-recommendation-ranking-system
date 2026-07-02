import type {
  JobDetail,
  MetricsResponse,
  RecommendResponse,
  SimulatorLogs,
  SystemStatus,
  UserInfo,
} from "./types";

const API = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string }>("/health"),
  status: () => request<SystemStatus>("/status"),
  users: (limit = 50) => request<UserInfo[]>(`/users?limit=${limit}`),
  recommend: (userId: string, slateSize = 10) =>
    request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, slate_size: slateSize }),
    }),
  runJob: (task: string, synthetic = true, extra: Record<string, unknown> = {}) =>
    request<{ id: string; task: string; status: string }>("/jobs", {
      method: "POST",
      body: JSON.stringify({ task, synthetic, synthetic_size: 10000, ...extra }),
    }),
  jobs: () => request<JobDetail[]>("/jobs"),
  job: (id: string) => request<JobDetail>(`/jobs/${id}`),
  metrics: () => request<MetricsResponse>("/metrics"),
  simulatorLogs: (limit = 50, offset = 0) =>
    request<SimulatorLogs>(`/simulator/logs?limit=${limit}&offset=${offset}`),
};
