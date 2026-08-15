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



export const WebSocketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [status, setStatus] = useState<WebSocketContextType["status"]>("connecting");
  const [cluster, setCluster] = useState<ClusterSnapshot | null>(null);
  const [history, setHistory] = useState<SimulationHistoryPoint[]>([]);
  const [actions, setActions] = useState<ActionEvent[]>([]);
  const [summary, setSummary] = useState<EpisodeSummary | null>(null);
  const [scenario, setScenario] = useState("mixed");
  const [seed, setSeed] = useState(42);
  const [tickDelayMs, setTickDelayMs] = useState(100);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedEdge, setHighlightedEdge] = useState<{ source: string; target: string } | null>(null);

  const clusterRef = useRef<ClusterSnapshot | null>(cluster);
  useEffect(() => {
    clusterRef.current = cluster;
  }, [cluster]);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelayRef = useRef(1000);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = import.meta.env.VITE_WS_URL || `${protocol}//${window.location.host}/ws/live`;
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      setStatus("connected");
      reconnectDelayRef.current = 1000;
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
            if (lastAction) {
              if (lastAction.cited_edge_source && lastAction.cited_edge_target) {
                setHighlightedEdge({ source: lastAction.cited_edge_source, target: lastAction.cited_edge_target });
                setTimeout(() => setHighlightedEdge(null), 1800);
              } else if (lastAction.target_service) {
                const src = lastAction.target_service;
                const edgeMatch = clusterRef.current?.edges.find((e) => e.source === src || e.target === src);
                if (edgeMatch) {
                  setHighlightedEdge({ source: edgeMatch.source, target: edgeMatch.target });
                  setTimeout(() => setHighlightedEdge(null), 1800);
                }
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
      // Auto-reconnect with exponential backoff up to 30s
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      const delay = reconnectDelayRef.current;
      reconnectDelayRef.current = Math.min(delay * 2, 30000);
      reconnectTimeoutRef.current = setTimeout(connect, delay);
    };

    wsRef.current = socket;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
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
