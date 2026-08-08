import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { ActionEvent, ClusterSnapshot, EpisodeSummary, WsFrame } from "../types";

interface SimulationHistoryPoint {
  tick: number;
  mean_health: number;
  min_health: number;
  sla_violation_rate: number;
  active_faults_count: number;
}

interface WebSocketContextType {
  status: "connecting" | "connected" | "disconnected" | "error";
  cluster: ClusterSnapshot | null;
  history: SimulationHistoryPoint[];
  actions: ActionEvent[];
  summary: EpisodeSummary | null;
  scenario: string;
  seed: number;
  tickDelayMs: number;
  isPlaying: boolean;
  selectedNodeId: string | null;
  highlightedEdge: { source: string; target: string } | null;
  setSelectedNodeId: (id: string | null) => void;
  setScenario: (s: string) => void;
  setSeed: (s: number) => void;
  setTickDelayMs: (ms: number) => void;
  startSimulation: (opts?: { scenario?: string; seed?: number; tickDelayMs?: number }) => void;
  stopSimulation: () => void;
  resetStream: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | null>(null);

// Fallback initial cluster for offline / initial state
const MOCK_CLUSTER: ClusterSnapshot = {
  tick: 0,
  sla_violation_rate: 0,
  mean_health: 1.0,
  min_health: 1.0,
  active_faults: [],
  services: Array.from({ length: 12 }, (_, i) => ({
    id: `svc-${i.toString().padStart(2, "0")}`,
    name: i === 0 ? "svc-00-gateway" : i < 3 ? `svc-${i.toString().padStart(2, "0")}-edge` : i < 8 ? `svc-${i.toString().padStart(2, "0")}-mid` : `svc-${i.toString().padStart(2, "0")}-data`,
    tier: i === 0 ? "front" : i < 3 ? "edge" : i < 8 ? "mid" : "back",
    health: 1.0,
    status: "healthy",
    cpu_pct: 0.25 + (i * 0.04) % 0.4,
    mem_pct: 0.35 + (i * 0.05) % 0.3,
    p99_latency_ms: 12.5 + i * 2.1,
    error_rate: 0.0,
    replicas: 2,
    ready_replicas: 2,
    isolated: false,
    sla_violating: false,
  })),
  nodes: Array.from({ length: 6 }, (_, j) => ({
    id: `node-${j}`,
    name: `node-${j}`,
    cpu_pct: 0.20 + j * 0.08,
    mem_pct: 0.30 + j * 0.05,
    pod_count: 4,
    pod_capacity: 8,
    health: 0.95,
  })),
  edges: [
    { source: "svc-00", target: "svc-01", relation: "CALLS", p99_latency_ms: 15.2, error_rate: 0.0, traffic_share: 0.5 },
    { source: "svc-00", target: "svc-02", relation: "CALLS", p99_latency_ms: 18.0, error_rate: 0.0, traffic_share: 0.5 },
    { source: "svc-01", target: "svc-03", relation: "CALLS", p99_latency_ms: 22.1, error_rate: 0.0, traffic_share: 0.6 },
    { source: "svc-01", target: "svc-04", relation: "CALLS", p99_latency_ms: 24.5, error_rate: 0.0, traffic_share: 0.4 },
    { source: "svc-02", target: "svc-05", relation: "CALLS", p99_latency_ms: 19.8, error_rate: 0.0, traffic_share: 1.0 },
    { source: "svc-03", target: "svc-08", relation: "CALLS", p99_latency_ms: 35.0, error_rate: 0.0, traffic_share: 0.7 },
    { source: "svc-04", target: "svc-09", relation: "CALLS", p99_latency_ms: 41.2, error_rate: 0.0, traffic_share: 0.8 },
    { source: "svc-05", target: "svc-10", relation: "CALLS", p99_latency_ms: 28.9, error_rate: 0.0, traffic_share: 0.9 },
    { source: "svc-06", target: "svc-11", relation: "CALLS", p99_latency_ms: 32.4, error_rate: 0.0, traffic_share: 1.0 },
  ],
};

export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [status, setStatus] = useState<WebSocketContextType["status"]>("connecting");
  const [cluster, setCluster] = useState<ClusterSnapshot | null>(MOCK_CLUSTER);
  const [history, setHistory] = useState<SimulationHistoryPoint[]>([]);
  const [actions, setActions] = useState<ActionEvent[]>([]);
  const [summary, setSummary] = useState<EpisodeSummary | null>(null);
  const [scenario, setScenario] = useState("mixed");
  const [seed, setSeed] = useState(42);
  const [tickDelayMs, setTickDelayMs] = useState(100);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedEdge, setHighlightedEdge] = useState<{ source: string; target: string } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    const wsUrl = `ws://${window.location.hostname}:8000/ws/live`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      setStatus("connected");
    };

    socket.onmessage = (event) => {
      try {
        const frame: WsFrame = JSON.parse(event.data);
        if (frame.type === "tick" && frame.cluster) {
          setCluster(frame.cluster);
          setHistory((prev) => [
            ...prev.slice(-120),
            {
              tick: frame.cluster!.tick,
              mean_health: frame.cluster!.mean_health,
              min_health: frame.cluster!.min_health,
              sla_violation_rate: frame.cluster!.sla_violation_rate,
              active_faults_count: frame.cluster!.active_faults.length,
            },
          ]);
          if (frame.actions && frame.actions.length > 0) {
            setActions((prev) => [...frame.actions!, ...prev].slice(0, 100));
            // Edge trace animation for narrated dependencies
            const lastAction = frame.actions[0];
            if (lastAction && lastAction.target_service) {
              const src = lastAction.target_service;
              const edgeMatch = cluster?.edges.find((e) => e.source === src || e.target === src);
              if (edgeMatch) {
                setHighlightedEdge({ source: edgeMatch.source, target: edgeMatch.target });
                setTimeout(() => setHighlightedEdge(null), 1800);
              }
            }
          }
        } else if (frame.type === "episode_end" && frame.episode_summary) {
          setSummary(frame.episode_summary);
          setIsPlaying(false);
        }
      } catch (err) {
        console.error("Failed to parse WS frame", err);
      }
    };

    socket.onerror = () => {
      setStatus("error");
    };

    socket.onclose = () => {
      setStatus("disconnected");
      setIsPlaying(false);
      // Auto-reconnect after 3s
      setTimeout(connect, 3000);
    };

    wsRef.current = socket;
  }, [cluster?.edges]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const startSimulation = useCallback(
    (opts?: { scenario?: string; seed?: number; tickDelayMs?: number }) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const sc = opts?.scenario || scenario;
        const sd = opts?.seed !== undefined ? opts.seed : seed;
        const delay = opts?.tickDelayMs !== undefined ? opts.tickDelayMs : tickDelayMs;

        setActions([]);
        setHistory([]);
        setSummary(null);
        setIsPlaying(true);

        wsRef.current.send(
          JSON.stringify({
            command: "start",
            scenario: sc,
            seed: sd,
            tick_delay_ms: delay,
            max_cycles: 200,
          })
        );
      }
    },
    [scenario, seed, tickDelayMs]
  );

  const stopSimulation = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command: "stop" }));
      setIsPlaying(false);
    }
  }, []);

  const resetStream = useCallback(() => {
    stopSimulation();
    setHistory([]);
    setActions([]);
    setSummary(null);
  }, [stopSimulation]);

  return (
    <WebSocketContext.Provider
      value={{
        status,
        cluster,
        history,
        actions,
        summary,
        scenario,
        seed,
        tickDelayMs,
        isPlaying,
        selectedNodeId,
        highlightedEdge,
        setSelectedNodeId,
        setScenario,
        setSeed,
        setTickDelayMs,
        startSimulation,
        stopSimulation,
        resetStream,
      }}
    >
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
};
