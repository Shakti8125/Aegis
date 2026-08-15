export type HealthStatus = "healthy" | "degraded" | "critical";
export type AutonomyLevel = 0 | 1 | 2 | 3 | 4;

export interface ServiceSnapshot {
  id: string;
  name: string;
  tier: "front" | "edge" | "mid" | "back" | string;
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
  position3d?: [number, number, number];
  embedding?: number[];
}

export interface NodeSnapshot {
  id: string;
  name: string;
  cpu_pct: number;
  mem_pct: number;
  pod_count: number;
  pod_capacity: number;
  health: number;
  position3d?: [number, number, number];
}

export interface EdgeSnapshot {
  source: string;
  target: string;
  relation: string;
  p99_latency_ms: number | null;
  error_rate: number | null;
  traffic_share: number | null;
  bandwidth_mbps?: number;
}

export interface FaultSnapshot {
  fault_type: string;
  target: string;
  tick_start: number;
  duration: number;
  intensity?: number;
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
  total_cpu_pct?: number;
  total_mem_pct?: number;
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
  cited_edge_source?: string;
  cited_edge_target?: string;
  reward_components?: Record<string, number>;
  autonomy_level?: AutonomyLevel;
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

// ------------------------------------------------------------------
// 3D Spatial & HUD Command Center Data Structures
// ------------------------------------------------------------------

export interface CameraState3D {
  position: [number, number, number];
  target: [number, number, number];
  focusedNodeId: string | null;
  zoom: number;
}

export interface SpatialNode3D {
  id: string;
  name: string;
  type: "service" | "node" | "ingress";
  position: [number, number, number];
  health: number;
  status: HealthStatus;
  cpu_pct: number;
  mem_pct: number;
  p99_latency_ms: number;
  error_rate: number;
  tierLevel: number; // +1: Ingress, 0: Microservices, -1: Physical Nodes
  isolated: boolean;
  sla_violating: boolean;
  activeFaults: string[];
}

export interface TierPlane3D {
  id: string;
  name: string;
  yLevel: number;
  color: string;
  gridSize: number;
  nodeCount: number;
}

export interface TrafficStream3D {
  id: string;
  sourceId: string;
  targetId: string;
  sourcePos: [number, number, number];
  targetPos: [number, number, number];
  p99_latency_ms: number;
  error_rate: number;
  traffic_share: number;
  isHighlighted: boolean;
}

export type ChaosType = "pod_kill" | "cpu_stress" | "mem_exhaustion" | "network_partition" | "latency_injection";

export interface ChaosTrigger {
  id: string;
  type: ChaosType;
  target: string;
  duration: number;
  intensity: number; // 0.0 - 1.0
  status: "pending" | "active" | "resolved";
}

export interface PartitionPlaneState {
  active: boolean;
  positionY: number;
  positionZ: number;
  angleDegrees: number;
  blockedServices: string[];
}

export interface LatentRadarMetrics {
  latency_impact: number;
  error_cascade: number;
  dependency_centrality: number;
  resource_stress: number;
  anomaly_score: number;
}

export interface GNNEmbeddingData {
  service_id: string;
  name: string;
  latent_vector: number[]; // 16D vector
  anomaly_score: number;
  metrics: LatentRadarMetrics;
  coords2d: [number, number];
}

export interface RewardBreakdown {
  r_health: number; // Availability / health score reward (+ component)
  r_cost: number;   // Resource cost penalty (- component)
  r_churn: number;  // Action instability penalty (- component)
  r_veto: number;   // Safety supervisor veto penalty (- component)
  r_sla: number;    // SLA violation penalty (- component)
  r_total: number;  // Net reward scalar
}

export interface TimelineState {
  currentTick: number;
  maxTick: number;
  isPlaying: boolean;
  playbackSpeed: number; // 0.5x, 1x, 2x, 5x
  snapshotBuffer: ClusterSnapshot[];
  diffTick: number | null;
}
