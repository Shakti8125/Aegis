export type HealthStatus = "healthy" | "degraded" | "critical";

export interface ServiceSnapshot {
  id: string;
  name: string;
  tier: string;
  health: number;
  status: HealthStatus;
  cpu_pct: number;
  mem_pct: number;
  p99_latency_ms: number;
  error_rate: number;
  replicas: number;
  ready_replicas: number;
  isolated: boolean;
  sla_violating: boolean;
}

export interface NodeSnapshot {
  id: string;
  name: string;
  cpu_pct: number;
  mem_pct: number;
  pod_count: number;
  pod_capacity: number;
  health: number;
}

export interface EdgeSnapshot {
  source: string;
  target: string;
  relation: string;
  p99_latency_ms: number | null;
  error_rate: number | null;
  traffic_share: number | null;
}

export interface FaultSnapshot {
  fault_type: string;
  target: string;
  tick_start: number;
  duration: number;
  details?: Record<string, unknown>;
}

export interface ClusterSnapshot {
  tick: number;
  services: ServiceSnapshot[];
  nodes: NodeSnapshot[];
  edges: EdgeSnapshot[];
  active_faults: FaultSnapshot[];
  sla_violation_rate: number;
  mean_health: number;
  min_health: number;
}

export interface ActionEvent {
  tick: number;
  agent_id: string;
  action: string;
  target_service: string;
  narration: string;
  was_vetoed: boolean;
  veto_reason: string;
  veto_policy: string;
  reward_components?: Record<string, number>;
}

export interface EpisodeSummary {
  episode_id: string;
  seed: number;
  scenario: string;
  length: number;
  recovered: boolean;
  collapsed: boolean;
  terminal_reason: string;
  ttr: number;
  sla_service_ticks: number;
  mean_health: number;
  total_reward: number;
  action_counts?: Record<string, number>;
  reward_components?: Record<string, number>;
}

export type WsFrameType = "tick" | "action" | "veto" | "episode_end" | "error" | "connected";

export interface WsFrame {
  type: WsFrameType;
  tick?: number;
  cluster?: ClusterSnapshot;
  actions?: ActionEvent[];
  message?: string;
  episode_summary?: EpisodeSummary;
}

export interface ScenarioInfo {
  name: string;
  enabled_faults: string[];
  n_faults_range: [number, number];
}
