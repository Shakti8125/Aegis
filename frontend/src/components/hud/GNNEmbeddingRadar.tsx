import React from "react";
import { Cpu, Activity, Info, Eye } from "lucide-react";
import { ClusterSnapshot, GNNEmbeddingData } from "../../types";

interface GNNEmbeddingRadarProps {
  cluster: ClusterSnapshot | null;
  selectedNodeId: string | null;
  onSelectService?: (id: string) => void;
}

export const GNNEmbeddingRadar: React.FC<GNNEmbeddingRadarProps> = ({
  cluster,
  selectedNodeId,
  onSelectService,
}) => {
  const services = cluster?.services || [];
  const selectedSvc = services.find((s) => s.id === selectedNodeId) || services[0];

  // Derive simulated GNN embedding metrics from service telemetry
  const metrics = selectedSvc
    ? {
        latency_impact: Math.min(1.0, selectedSvc.p99_latency_ms / 100),
        error_cascade: Math.min(1.0, selectedSvc.error_rate * 5),
        dependency_centrality: selectedSvc.tier === "front" ? 0.95 : selectedSvc.tier === "edge" ? 0.75 : 0.45,
        resource_stress: Math.max(selectedSvc.cpu_pct, selectedSvc.mem_pct),
        anomaly_score: Math.min(1.0, (1 - selectedSvc.health) * 1.5),
      }
    : {
        latency_impact: 0.15,
        error_cascade: 0.05,
        dependency_centrality: 0.6,
        resource_stress: 0.3,
        anomaly_score: 0.1,
      };

  const axes = [
    { key: "latency_impact", label: "Latency Impact", val: metrics.latency_impact },
    { key: "error_cascade", label: "Error Cascade", val: metrics.error_cascade },
    { key: "dependency_centrality", label: "Centrality", val: metrics.dependency_centrality },
    { key: "resource_stress", label: "Resource Stress", val: metrics.resource_stress },
    { key: "anomaly_score", label: "Anomaly Score", val: metrics.anomaly_score },
  ];

  // SVG Polar Radar math
  const center = 100;
  const radius = 65;
  const nAxes = axes.length;

  const points = axes.map((axis, i) => {
    const angle = (i / nAxes) * Math.PI * 2 - Math.PI / 2;
    const r = axis.val * radius;
    const x = center + r * Math.cos(angle);
    const y = center + r * Math.sin(angle);
    return { x, y, labelX: center + (radius + 18) * Math.cos(angle), labelY: center + (radius + 18) * Math.sin(angle) };
  });

  const polygonPath = points.map((p) => `${p.x},${p.y}`).join(" ");

  // Grid concentric rings
  const rings = [0.25, 0.5, 0.75, 1.0];

  return (
    <div className="flex flex-col bg-slate-950/85 backdrop-blur-md border border-cyan-900/40 rounded-2xl p-4 w-80 text-slate-100 shadow-2xl text-xs space-y-3">
      {/* Title */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>GNN LATENT EMBEDDING PROBE</span>
        </div>
        <span className="text-[10px] font-mono text-cyan-300/80 bg-cyan-950 border border-cyan-800/60 px-2 py-0.5 rounded-full">
          GraphSAGE 16D
        </span>
      </div>

      {/* Selected Node Indicator */}
      <div className="flex items-center justify-between bg-slate-900/60 p-2 rounded-xl border border-slate-800">
        <span className="text-slate-400">Target Microservice:</span>
        <span className="font-mono font-bold text-cyan-300">{selectedSvc ? selectedSvc.name : "N/A"}</span>
      </div>

      {/* SVG Radar Chart */}
      <div className="relative flex justify-center py-1">
        <svg width="200" height="200" className="overflow-visible">
          {/* Concentric Grid Rings */}
          {rings.map((rPct, idx) => (
            <circle
              key={idx}
              cx={center}
              cy={center}
              r={radius * rPct}
              fill="none"
              stroke="#334155"
              strokeDasharray="3,3"
              strokeWidth="1"
            />
          ))}

          {/* Spokes */}
          {axes.map((_, i) => {
            const angle = (i / nAxes) * Math.PI * 2 - Math.PI / 2;
            const x2 = center + radius * Math.cos(angle);
            const y2 = center + radius * Math.sin(angle);
            return <line key={i} x1={center} y1={center} x2={x2} y2={y2} stroke="#334155" strokeWidth="1" />;
          })}

          {/* Polygon Area */}
          <polygon
            points={polygonPath}
            fill="rgba(6, 182, 212, 0.3)"
            stroke="#06B6D4"
            strokeWidth="2"
            className="transition-all duration-300"
          />

          {/* Data Vertex Dots */}
          {points.map((p, idx) => (
            <circle key={idx} cx={p.x} cy={p.y} r="3.5" fill="#38BDF8" stroke="#0F172A" strokeWidth="1.5" />
          ))}

          {/* Axis Labels */}
          {axes.map((axis, i) => {
            const pt = points[i];
            return (
              <text
                key={i}
                x={pt.labelX}
                y={pt.labelY}
                fill="#94A3B8"
                fontSize="9"
                fontFamily="sans-serif"
                textAnchor="middle"
                dominantBaseline="middle"
              >
                {axis.label}
              </text>
            );
          })}
        </svg>
      </div>

      {/* Metrics Legend */}
      <div className="grid grid-cols-2 gap-1.5 pt-1 text-[10px] font-mono">
        <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800 flex justify-between">
          <span className="text-slate-400">Anomaly:</span>
          <span className="text-rose-400 font-bold">{(metrics.anomaly_score * 100).toFixed(0)}%</span>
        </div>
        <div className="bg-slate-900/60 p-1.5 rounded-lg border border-slate-800 flex justify-between">
          <span className="text-slate-400">Stress:</span>
          <span className="text-amber-400 font-bold">{(metrics.resource_stress * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
};
