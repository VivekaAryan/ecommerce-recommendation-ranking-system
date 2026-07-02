export interface ArtifactStatus {
  name: string;
  exists: boolean;
  path: string;
  detail: string;
}

export interface SystemStatus {
  ready: boolean;
  artifacts: ArtifactStatus[];
  dataset: Record<string, number> | null;
}

export interface JobDetail {
  id: string;
  task: string;
  status: string;
  message: string;
  result?: Record<string, unknown>;
  error?: string;
}

export interface SlateItem {
  item_id: string;
  title: string;
  category: string;
  price: number | null;
  score: number;
  position: number;
}

export interface RecommendResponse {
  user_id: string;
  slate: SlateItem[];
  latency: {
    retrieval_ms: number;
    prerank_ms: number;
    ranking_ms: number;
    reranking_ms: number;
    total_ms: number;
    within_budget: Record<string, boolean>;
  };
  user_history: string[];
}

export interface UserInfo {
  user_id: string;
  interaction_count: number;
}

export interface MetricsResponse {
  skew?: Array<Record<string, number | string>>;
  ope?: Record<string, number>;
  simulator_summary?: Record<string, number>;
}

export interface SimulatorLogs {
  total: number;
  rows: Array<Record<string, unknown>>;
}
