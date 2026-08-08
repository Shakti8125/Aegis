import React from "react";
import { useWebSocket } from "../context/WebSocketContext";
import { Play, Square, RefreshCw, Zap, ShieldCheck, Activity, Cpu } from "lucide-react";

export const Header: React.FC = () => {
  const {
    status,
    cluster,
    scenario,
    seed,
    tickDelayMs,
    isPlaying,
    setScenario,
    setSeed,
    setTickDelayMs,
    startSimulation,
    stopSimulation,
    resetStream,
  } = useWebSocket();

  const scenarios = [
    { id: "mixed", label: "Mixed Faults" },
    { id: "pod_crash", label: "Pod Crashes" },
    { id: "node_spike", label: "Node CPU/Mem Spike" },
    { id: "partition", label: "Network Partition" },
    { id: "cascading_latency", label: "Cascading Latency" },
  ];

  return (
    <header className="h-16 px-6 glass-panel border-b border-[#232B3E] flex items-center justify-between z-30 select-none">
      {/* Brand & Connection Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#38BDF8] to-[#3DDC97] p-0.5 flex items-center justify-center shadow-[0_0_15px_rgba(56,189,248,0.3)]">
            <div className="w-full h-full bg-[#0B0E14] rounded-[6px] flex items-center justify-center">
              <Zap className="w-4 h-4 text-[#38BDF8]" />
            </div>
          </div>
          <div>
            <h1 className="font-mono font-bold text-sm tracking-wider text-[#F1F5F9] flex items-center gap-2">
              AEGIS <span className="text-[10px] font-normal px-1.5 py-0.5 rounded bg-[#232B3E] text-[#38BDF8]">MARL OPS</span>
            </h1>
            <p className="text-[10px] font-mono text-[#7C89A3]">Autonomous Kubernetes Self-Healing System</p>
          </div>
        </div>

        {/* Live WS Status Indicator */}
        <div className="h-4 w-px bg-[#232B3E] mx-1" />
        <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#131822] border border-[#232B3E] text-xs font-mono">
          <span
            className={`w-2 h-2 rounded-full ${
              status === "connected"
                ? "bg-[#3DDC97] animate-pulse shadow-[0_0_8px_#3DDC97]"
                : status === "connecting"
                ? "bg-[#F5A623] animate-ping"
                : "bg-[#E5484D]"
            }`}
          />
          <span className="text-[#7C89A3] capitalize">{status}</span>
        </div>
      </div>

      {/* Control Panel: Scenario Selector, Seed, Speed, Play/Stop */}
      <div className="flex items-center gap-4">
        {/* Scenario Selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-mono text-[#7C89A3]">Scenario:</label>
          <select
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            disabled={isPlaying}
            className="bg-[#1A202C] text-[#F1F5F9] border border-[#232B3E] rounded px-3 py-1.5 text-xs font-mono focus:outline-none focus:border-[#38BDF8] disabled:opacity-50"
          >
            {scenarios.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </div>

        {/* Seed Input */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-mono text-[#7C89A3]">Seed:</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
            disabled={isPlaying}
            className="w-16 bg-[#1A202C] text-[#F1F5F9] border border-[#232B3E] rounded px-2 py-1 text-xs font-mono text-center focus:outline-none focus:border-[#38BDF8] disabled:opacity-50"
          />
        </div>

        {/* Speed Slider */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-mono text-[#7C89A3]">Delay:</label>
          <input
            type="range"
            min="20"
            max="500"
            step="10"
            value={tickDelayMs}
            onChange={(e) => setTickDelayMs(parseInt(e.target.value))}
            className="w-20 accent-[#38BDF8] cursor-pointer"
          />
          <span className="text-xs font-mono text-[#F1F5F9] w-10">{tickDelayMs}ms</span>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          {!isPlaying ? (
            <button
              onClick={() => startSimulation()}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-[#38BDF8] hover:bg-[#0284C7] text-[#0B0E14] font-mono font-semibold text-xs transition-colors shadow-[0_0_12px_rgba(56,189,248,0.4)]"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Start Sim</span>
            </button>
          ) : (
            <button
              onClick={stopSimulation}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-[#E5484D] hover:bg-[#B91C1C] text-white font-mono font-semibold text-xs transition-colors shadow-[0_0_12px_rgba(229,72,77,0.4)]"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop</span>
            </button>
          )}

          <button
            onClick={resetStream}
            className="p-1.5 rounded-lg bg-[#1A202C] hover:bg-[#232B3E] text-[#7C89A3] hover:text-[#F1F5F9] border border-[#232B3E] transition-colors"
            title="Reset Stream"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Cluster Metrics Banner */}
      <div className="flex items-center gap-6">
        <div className="text-right">
          <div className="text-[10px] font-mono text-[#7C89A3]">TICK</div>
          <div className="font-mono text-sm font-bold text-[#38BDF8]">
            {cluster?.tick ?? 0}
          </div>
        </div>

        <div className="text-right">
          <div className="text-[10px] font-mono text-[#7C89A3]">MEAN HEALTH</div>
          <div
            className={`font-mono text-sm font-bold ${
              (cluster?.mean_health ?? 1.0) >= 0.85
                ? "text-[#3DDC97]"
                : (cluster?.mean_health ?? 1.0) >= 0.4
                ? "text-[#F5A623]"
                : "text-[#E5484D]"
            }`}
          >
            {((cluster?.mean_health ?? 1.0) * 100).toFixed(0)}%
          </div>
        </div>

        <div className="text-right">
          <div className="text-[10px] font-mono text-[#7C89A3]">SLA VIOLATIONS</div>
          <div
            className={`font-mono text-sm font-bold ${
              (cluster?.sla_violation_rate ?? 0) > 0 ? "text-[#E5484D]" : "text-[#3DDC97]"
            }`}
          >
            {((cluster?.sla_violation_rate ?? 0) * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </header>
  );
};
