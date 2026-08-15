import React, { useState } from "react";
import { Flame, Skull, Cpu, HardDrive, WifiOff, Clock, Play, XCircle, ChevronRight, Sliders } from "lucide-react";
import { ChaosTrigger, ChaosType, ClusterSnapshot } from "../../types";

interface ChaosWarfareStudioProps {
  cluster: ClusterSnapshot | null;
  selectedNodeId: string | null;
  onInjectChaos: (trigger: Omit<ChaosTrigger, "id" | "status">) => void;
  onAbortChaos?: (faultTarget: string) => void;
  onClose?: () => void;
}

const FAULT_TYPES: Array<{
  type: ChaosType;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
}> = [
  {
    type: "pod_kill",
    label: "Pod Crash / Kill",
    description: "Terminates microservice pod replicas abruptly",
    icon: Skull,
    color: "text-rose-400 bg-rose-950/60 border-rose-800/80",
  },
  {
    type: "cpu_stress",
    label: "CPU Load Spike",
    description: "Injects artificial compute load up to 100%",
    icon: Cpu,
    color: "text-amber-400 bg-amber-950/60 border-amber-800/80",
  },
  {
    type: "mem_exhaustion",
    label: "Memory Leak",
    description: "Causes heap leak memory pressure",
    icon: HardDrive,
    color: "text-purple-400 bg-purple-950/60 border-purple-800/80",
  },
  {
    type: "network_partition",
    label: "Subnet Partition",
    description: "Severs inter-service network packets",
    icon: WifiOff,
    color: "text-cyan-400 bg-cyan-950/60 border-cyan-800/80",
  },
  {
    type: "latency_injection",
    label: "Latency Injection",
    description: "Adds 200ms-800ms RPC delay cascades",
    icon: Clock,
    color: "text-emerald-400 bg-emerald-950/60 border-emerald-800/80",
  },
];

export const ChaosWarfareStudio: React.FC<ChaosWarfareStudioProps> = ({
  cluster,
  selectedNodeId,
  onInjectChaos,
  onAbortChaos,
  onClose,
}) => {
  const [selectedType, setSelectedType] = useState<ChaosType>("pod_kill");
  const [targetId, setTargetId] = useState<string>(selectedNodeId || "svc-03");
  const [duration, setDuration] = useState<number>(30);
  const [intensity, setIntensity] = useState<number>(0.8);

  const services = cluster?.services || [];
  const activeFaults = cluster?.active_faults || [];

  const handleInject = () => {
    onInjectChaos({
      type: selectedType,
      target: targetId,
      duration,
      intensity,
    });
  };

  return (
    <div className="flex flex-col bg-slate-950/95 backdrop-blur-xl border border-rose-900/40 rounded-2xl w-96 text-slate-100 shadow-2xl overflow-hidden max-h-[85vh]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-rose-950/80 to-slate-900 border-b border-rose-900/50">
        <div className="flex items-center space-x-2 text-rose-400 font-bold">
          <Flame className="w-5 h-5 animate-pulse text-rose-500" />
          <span className="tracking-wide">CHAOS WARFARE STUDIO</span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <XCircle className="w-5 h-5" />
          </button>
        )}
      </div>

      <div className="p-4 overflow-y-auto space-y-4 text-xs">
        {/* Target Selector */}
        <div>
          <label className="block text-slate-400 font-medium mb-1">
            Target Service / Node
          </label>
          <select
            value={targetId}
            onChange={(e) => setTargetId(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-slate-200 focus:outline-none focus:border-rose-500 font-mono"
          >
            {services.map((svc) => (
              <option key={svc.id} value={svc.id}>
                {svc.name} ({svc.tier} tier - {svc.status.toUpperCase()})
              </option>
            ))}
          </select>
        </div>

        {/* Fault Type Selection Grid */}
        <div>
          <label className="block text-slate-400 font-medium mb-1.5">
            Select Chaos Injection Vector
          </label>
          <div className="grid grid-cols-1 gap-2">
            {FAULT_TYPES.map((fault) => {
              const Icon = fault.icon;
              const isSelected = selectedType === fault.type;
              return (
                <button
                  key={fault.type}
                  onClick={() => setSelectedType(fault.type)}
                  className={`flex items-start space-x-3 p-2.5 rounded-xl border transition-all text-left ${
                    isSelected
                      ? `${fault.color} ring-2 ring-rose-500/50 shadow-lg`
                      : "bg-slate-900/60 border-slate-800 hover:border-slate-700"
                  }`}
                >
                  <div className="p-1.5 rounded-lg bg-slate-950/60 shrink-0">
                    <Icon className="w-4 h-4" />
                  </div>
                  <div className="flex-1">
                    <div className="font-semibold text-slate-200 flex items-center justify-between">
                      <span>{fault.label}</span>
                      {isSelected && <ChevronRight className="w-4 h-4 text-rose-400" />}
                    </div>
                    <p className="text-[11px] text-slate-400 leading-tight">
                      {fault.description}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Intensity & Duration Controls */}
        <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 space-y-3">
          <div>
            <div className="flex justify-between text-slate-300 font-medium mb-1">
              <span className="flex items-center space-x-1">
                <Sliders className="w-3.5 h-3.5 text-rose-400" />
                <span>Fault Intensity</span>
              </span>
              <span className="font-mono text-rose-400 font-bold">
                {Math.round(intensity * 100)}%
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={intensity}
              onChange={(e) => setIntensity(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-rose-500"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-300 font-medium mb-1">
              <span>Duration (Ticks)</span>
              <span className="font-mono text-amber-400 font-bold">{duration} Ticks</span>
            </div>
            <input
              type="range"
              min="5"
              max="100"
              step="5"
              value={duration}
              onChange={(e) => setDuration(parseInt(e.target.value, 10))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-amber-500"
            />
          </div>
        </div>

        {/* Trigger Button */}
        <button
          onClick={handleInject}
          className="w-full py-2.5 bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-bold rounded-xl shadow-xl shadow-rose-950/50 flex items-center justify-center space-x-2 transition-all transform active:scale-98"
        >
          <Play className="w-4 h-4 fill-current" />
          <span>EXECUTE CHAOS INJECTION</span>
        </button>

        {/* Active Faults Monitor */}
        {activeFaults.length > 0 && (
          <div className="border-t border-slate-800 pt-3 space-y-2">
            <span className="text-slate-400 font-medium block">
              Active Chaos Injections ({activeFaults.length})
            </span>
            <div className="space-y-1.5 max-h-32 overflow-y-auto pr-1">
              {activeFaults.map((fault, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-rose-950/40 border border-rose-900/60 p-2 rounded-lg text-[11px]"
                >
                  <div>
                    <span className="font-bold text-rose-300 font-mono">
                      {fault.fault_type.toUpperCase()}
                    </span>
                    <span className="text-slate-400 ml-1.5">on {fault.target}</span>
                  </div>
                  {onAbortChaos && (
                    <button
                      onClick={() => onAbortChaos(fault.target)}
                      className="text-slate-400 hover:text-rose-400 transition-colors"
                      title="Abort Fault"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
