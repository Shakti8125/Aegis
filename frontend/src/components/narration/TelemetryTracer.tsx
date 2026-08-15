import React from "react";
import { Activity, ArrowRight, Zap, AlertTriangle, ShieldCheck } from "lucide-react";
import { ClusterSnapshot } from "../../types";

interface TelemetryTracerProps {
  cluster: ClusterSnapshot | null;
  selectedNodeId: string | null;
  onSelectNode?: (id: string) => void;
}

export const TelemetryTracer: React.FC<TelemetryTracerProps> = ({
  cluster,
  selectedNodeId,
  onSelectNode,
}) => {
  const services = cluster?.services || [];
  const edges = cluster?.edges || [];

  // Build dependency trace path starting from selectedNodeId or gateway
  const rootId = selectedNodeId || "svc-00";
  const tracePath: string[] = [rootId];

  let curr = rootId;
  for (let i = 0; i < 4; i++) {
    const nextEdge = edges.find((e) => e.source === curr);
    if (nextEdge && !tracePath.includes(nextEdge.target)) {
      tracePath.push(nextEdge.target);
      curr = nextEdge.target;
    } else {
      break;
    }
  }

  const traceNodes = tracePath.map((id) => services.find((s) => s.id === id)).filter(Boolean);

  return (
    <div className="flex flex-col bg-slate-950/85 backdrop-blur-md border border-cyan-900/40 rounded-2xl p-4 w-96 text-slate-100 shadow-2xl text-xs space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span>CASCADE TELEMETRY TRACER</span>
        </div>
        <span className="text-[10px] font-mono text-cyan-300 bg-cyan-950 border border-cyan-800/80 px-2 py-0.5 rounded-full">
          eBPF TRACE LINKAGE
        </span>
      </div>

      {/* Trace Path Flow Visualizer */}
      <div className="space-y-2 pt-1">
        <span className="text-slate-400 text-[10px] font-medium block">
          Incident Propagation Path ({traceNodes.length} Hops)
        </span>

        <div className="flex items-center space-x-1.5 overflow-x-auto py-2 px-1">
          {traceNodes.map((node, idx) => {
            if (!node) return null;
            const isSelected = selectedNodeId === node.id;
            const isFaulty = node.status === "critical" || node.sla_violating;

            return (
              <React.Fragment key={node.id}>
                {idx > 0 && (
                  <ArrowRight className="w-3.5 h-3.5 text-slate-600 shrink-0" />
                )}
                <button
                  onClick={() => onSelectNode && onSelectNode(node.id)}
                  className={`flex flex-col p-2 rounded-xl border shrink-0 transition-all text-left ${
                    isSelected
                      ? "bg-cyan-950/80 border-cyan-500 text-white ring-2 ring-cyan-500/40"
                      : isFaulty
                      ? "bg-rose-950/40 border-rose-800 text-rose-200"
                      : "bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700"
                  }`}
                >
                  <span className="font-mono font-bold text-[11px]">{node.name}</span>
                  <div className="flex items-center space-x-2 text-[10px] font-mono mt-1">
                    <span className={node.p99_latency_ms > 40 ? "text-rose-400" : "text-emerald-400"}>
                      {node.p99_latency_ms.toFixed(1)}ms
                    </span>
                    <span className="text-slate-500">•</span>
                    <span className={node.cpu_pct > 0.8 ? "text-amber-400" : "text-slate-400"}>
                      {Math.round(node.cpu_pct * 100)}% CPU
                    </span>
                  </div>
                </button>
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
};
