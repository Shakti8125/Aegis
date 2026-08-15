import React from "react";
import { ClusterSnapshot } from "../types";
import { X, ShieldAlert, Cpu, HardDrive, Activity, Layers, CornerDownRight } from "lucide-react";

interface NodeInspectorProps {
  selectedNodeId: string | null;
  cluster: ClusterSnapshot | null;
  onClose: () => void;
}

export const NodeInspector: React.FC<NodeInspectorProps> = ({
  selectedNodeId,
  cluster,
  onClose,
}) => {
  if (!selectedNodeId || !cluster) return null;

  const service = cluster.services.find((s) => s.id === selectedNodeId);
  if (!service) return null;

  const outgoingEdges = cluster.edges.filter((e) => e.source === selectedNodeId);
  const incomingEdges = cluster.edges.filter((e) => e.target === selectedNodeId);

  const getHealthBadge = (health: number) => {
    if (health >= 0.85) return <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-[#3DDC97]/20 text-[#3DDC97] border border-[#3DDC97]/40">Healthy</span>;
    if (health >= 0.40) return <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-[#F5A623]/20 text-[#F5A623] border border-dashed border-[#F5A623]/40">Degraded</span>;
    return <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-[#E5484D]/20 text-[#E5484D] border border-dotted border-[#E5484D]/40">Critical</span>;
  };

  return (
    <div className="absolute right-6 top-20 w-80 glass-panel rounded-xl border border-[#232B3E] p-4 shadow-2xl z-40 text-xs font-mono animate-in slide-in-from-right duration-200">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#232B3E] pb-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-[#38BDF8]" />
          <h3 className="font-bold text-sm text-[#F1F5F9]">{service.id}</h3>
          {getHealthBadge(service.health)}
        </div>
        <button
          onClick={onClose}
          className="text-[#7C89A3] hover:text-[#F1F5F9] transition-colors p-1 rounded hover:bg-[#232B3E]"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        <div className="glass-card p-2.5 rounded-lg border border-[#232B3E]">
          <div className="text-[10px] text-[#7C89A3] flex items-center gap-1 mb-1">
            <Activity className="w-3 h-3 text-[#3DDC97]" />
            <span>HEALTH SCORE</span>
          </div>
          <div className="text-base font-bold text-[#F1F5F9]">{(service.health * 100).toFixed(1)}%</div>
        </div>

        <div className="glass-card p-2.5 rounded-lg border border-[#232B3E]">
          <div className="text-[10px] text-[#7C89A3] flex items-center gap-1 mb-1">
            <Cpu className="w-3 h-3 text-[#38BDF8]" />
            <span>P99 LATENCY</span>
          </div>
          <div className="text-base font-bold text-[#F1F5F9]">{service.p99_latency_ms.toFixed(1)} ms</div>
        </div>

        <div className="glass-card p-2.5 rounded-lg border border-[#232B3E]">
          <div className="text-[10px] text-[#7C89A3] flex items-center gap-1 mb-1">
            <HardDrive className="w-3 h-3 text-[#F5A623]" />
            <span>CPU UTILIZATION</span>
          </div>
          <div className="text-base font-bold text-[#F1F5F9]">{(service.cpu_pct * 100).toFixed(0)}%</div>
        </div>

        <div className="glass-card p-2.5 rounded-lg border border-[#232B3E]">
          <div className="text-[10px] text-[#7C89A3] flex items-center gap-1 mb-1">
            <ShieldAlert className="w-3 h-3 text-[#E5484D]" />
            <span>ERROR RATE</span>
          </div>
          <div className="text-base font-bold text-[#F1F5F9]">{(service.error_rate * 100).toFixed(2)}%</div>
        </div>
      </div>

      {/* Service Metadata */}
      <div className="glass-card p-3 rounded-lg border border-[#232B3E] mb-4 space-y-1.5 text-[11px]">
        <div className="flex justify-between text-[#7C89A3]">
          <span>Service Tier:</span>
          <span className="text-[#F1F5F9] font-semibold">{service.tier || "mid"}</span>
        </div>
        <div className="flex justify-between text-[#7C89A3]">
          <span>Replicas Ready:</span>
          <span className="text-[#F1F5F9] font-semibold">{service.ready_replicas} / {service.replicas}</span>
        </div>
        <div className="flex justify-between text-[#7C89A3]">
          <span>Isolated State:</span>
          <span className={service.isolated ? "text-[#E5484D] font-bold" : "text-[#3DDC97]"}>
            {service.isolated ? "ISOLATED" : "Normal"}
          </span>
        </div>
        <div className="flex justify-between text-[#7C89A3]">
          <span>SLA Status:</span>
          <span className={service.sla_violating ? "text-[#E5484D] font-bold" : "text-[#3DDC97]"}>
            {service.sla_violating ? "VIOLATING SLA" : "Compliant"}
          </span>
        </div>
      </div>

      {/* Outgoing Call Dependencies */}
      <div>
        <div className="text-[10px] text-[#7C89A3] font-bold tracking-wider mb-1.5 flex items-center gap-1">
          <CornerDownRight className="w-3 h-3 text-[#38BDF8]" />
          <span>OUTGOING CALL EDGES ({outgoingEdges.length})</span>
        </div>
        {outgoingEdges.length === 0 ? (
          <p className="text-[10px] text-[#7C89A3] italic">No outgoing dependencies</p>
        ) : (
          <div className="space-y-1 max-h-32 overflow-y-auto pr-1">
            {outgoingEdges.map((e, idx) => (
              <div key={idx} className="flex justify-between items-center p-1.5 rounded bg-[#131822] border border-[#232B3E] text-[10px]">
                <span className="text-[#38BDF8]">&rarr; {e.target}</span>
                <span className="text-[#7C89A3]">{e.p99_latency_ms?.toFixed(0)} ms</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
