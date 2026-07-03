import type {
  CatalogResponse,
  JobDetail,
  MetricsResponse,
  ProductCard,
  RecommendResponse,
  SimulatorLogs,
  SystemStatus,
  UserHistoryResponse,
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
  catalog: (limit = 48, offset = 0, category?: string) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (category) params.set("category", category);
    return request<CatalogResponse>(`/catalog?${params}`);
  },
  userHistory: (userId: string, limit = 12) =>
    request<UserHistoryResponse>(`/users/${encodeURIComponent(userId)}/history?limit=${limit}`),
  recommend: (userId: string, contextItemId: string, slateSize = 8) =>
    request<RecommendResponse>("/recommend", {
      method: "POST",
      body: JSON.stringify({
        user_id: userId,
        slate_size: slateSize,
        context_item_id: contextItemId,
      }),
    }),
  runJob: (task: string, extra: Record<string, unknown> = {}) =>
    request<{ id: string; task: string; status: string }>("/jobs", {
      method: "POST",
      body: JSON.stringify({ task, ...extra }),
    }),
  jobs: () => request<JobDetail[]>("/jobs"),
  job: (id: string) => request<JobDetail>(`/jobs/${id}`),
  metrics: () => request<MetricsResponse>("/metrics"),
  simulatorLogs: (limit = 50, offset = 0) =>
    request<SimulatorLogs>(`/simulator/logs?limit=${limit}&offset=${offset}`),
};

export type { ProductCard };
