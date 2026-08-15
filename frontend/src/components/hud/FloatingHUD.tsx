import React from "react";
import {
  ShieldCheck,
  Flame,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Box,
  Layers,
  Scissors,
  Clock,
  Zap,
} from "lucide-react";
import { AutonomyLevel, ClusterSnapshot } from "../../types";

interface FloatingHUDProps {
  status: "connecting" | "connected" | "disconnected" | "error";
  cluster: ClusterSnapshot | null;
  isPlaying: boolean;
  autonomyLevel: AutonomyLevel;
  tickDelayMs: number;
  viewMode: "3d" | "2d";
  showChaosStudio: boolean;
  showPartitionSlicer: boolean;
  showTimelineScrubber: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  onSetTickDelay: (ms: number) => void;
  onToggleViewMode: () => void;
  onToggleChaosStudio: () => void;
  onTogglePartitionSlicer: () => void;
  onToggleTimelineScrubber: () => void;
  onSetAutonomyLevel?: (lvl: AutonomyLevel) => void;
}

export const FloatingHUD: React.FC<FloatingHUDProps> = ({
  status,
  cluster,
  isPlaying,
  autonomyLevel,
  tickDelayMs,
  viewMode,
  showChaosStudio,
  showPartitionSlicer,
  showTimelineScrubber,
  onTogglePlay,
  onReset,
  onSetTickDelay,
  onToggleViewMode,
  onToggleChaosStudio,
  onTogglePartitionSlicer,
  onToggleTimelineScrubber,
  onSetAutonomyLevel,
}) => {
  const meanHealth = cluster ? Math.round(cluster.mean_health * 100) : 100;
  const slaViolationPct = cluster ? Math.round(cluster.sla_violation_rate * 100) : 0;
  const currentTick = cluster ? cluster.tick : 0;

  const autonomyLabels: Record<AutonomyLevel, string> = {
    0: "L0: Manual Advisory",
    1: "L1: Human-in-the-Loop",
    2: "L2: Auto-Healing (Override)",
    3: "L3: High Autonomy",
    4: "L4: Fully Autonomous",
  };

  return (
    <header className="absolute top-3 left-4 right-4 z-40 flex items-center justify-between bg-slate-950/80 backdrop-blur-xl border border-slate-800/80 px-4 py-2.5 rounded-2xl shadow-2xl text-slate-100">
      {/* Brand & Connection Status */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <div className="p-1.5 rounded-xl bg-gradient-to-tr from-cyan-600 to-emerald-500 shadow-lg shadow-cyan-950/50">
            <ShieldCheck className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-extrabold text-sm tracking-wider bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
              AEGIS 3D COMMAND CENTER
            </div>
            <div className="text-[10px] text-slate-400 font-mono flex items-center space-x-1.5">
              <span
                className={`inline-block w-2 h-2 rounded-full ${
                  status === "connected"
                    ? "bg-emerald-400 animate-pulse"
                    : status === "connecting"
                    ? "bg-amber-400"
                    : "bg-rose-500"
                }`}
              />
              <span>{status.toUpperCase()}</span>
              <span>•</span>
              <span>TICK: {currentTick}</span>
            </div>
          </div>
        </div>

        {/* Autonomy Level Badge */}
        <div className="hidden md:flex items-center space-x-1 bg-slate-900/80 border border-slate-800 px-2.5 py-1 rounded-xl text-xs font-mono">
          <Zap className="w-3.5 h-3.5 text-cyan-400" />
          <select
            value={autonomyLevel}
            onChange={(e) =>
              onSetAutonomyLevel && onSetAutonomyLevel(parseInt(e.target.value, 10) as AutonomyLevel)
            }
            className="bg-transparent text-cyan-300 font-semibold focus:outline-none cursor-pointer"
          >
            {[0, 1, 2, 3, 4].map((lvl) => (
              <option key={lvl} value={lvl} className="bg-slate-900 text-slate-100">
                {autonomyLabels[lvl as AutonomyLevel]}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Cluster Health Metrics */}
      <div className="hidden lg:flex items-center space-x-6">
        {/* Mean Health Indicator */}
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-emerald-400" />
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider">Cluster Health</span>
            <div className="flex items-center space-x-2">
              <div className="w-24 h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all duration-300 ${
                    meanHealth > 80 ? "bg-emerald-500" : meanHealth > 50 ? "bg-amber-500" : "bg-rose-500"
                  }`}
                  style={{ width: `${meanHealth}%` }}
                />
              </div>
              <span className="font-mono text-xs font-bold text-slate-200">{meanHealth}%</span>
            </div>
          </div>
        </div>

        {/* SLA Violation Rate */}
        <div className="flex flex-col">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider">SLA Violation Rate</span>
          <span
            className={`font-mono text-xs font-bold ${
              slaViolationPct > 15 ? "text-rose-400" : slaViolationPct > 0 ? "text-amber-400" : "text-emerald-400"
            }`}
          >
            {slaViolationPct}%
          </span>
        </div>
      </div>

      {/* Action Controls & Toggles */}
      <div className="flex items-center space-x-2">
        {/* Play/Pause Button */}
        <button
          onClick={onTogglePlay}
          className={`p-2 rounded-xl border font-bold text-xs flex items-center space-x-1.5 transition-all ${
            isPlaying
              ? "bg-amber-500/20 border-amber-500/40 text-amber-300 hover:bg-amber-500/30"
              : "bg-emerald-500/20 border-emerald-500/40 text-emerald-300 hover:bg-emerald-500/30"
          }`}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          <span className="hidden sm:inline">{isPlaying ? "PAUSE" : "START"}</span>
        </button>

        {/* Speed Selector */}
        <select
          value={tickDelayMs}
          onChange={(e) => onSetTickDelay(parseInt(e.target.value, 10))}
          className="bg-slate-900 border border-slate-800 text-slate-300 text-xs rounded-xl px-2 py-1.5 font-mono focus:outline-none"
        >
          <option value={50}>50ms (20 FPS)</option>
          <option value={100}>100ms (10 FPS)</option>
          <option value={250}>250ms (4 FPS)</option>
          <option value={500}>500ms (2 FPS)</option>
        </select>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-xl text-slate-300 transition-colors"
          title="Reset Stream"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        <div className="h-4 w-px bg-slate-800 my-auto" />

        {/* View Mode Toggle (3D / 2D) */}
        <button
          onClick={onToggleViewMode}
          className={`p-2 rounded-xl border text-xs font-semibold flex items-center space-x-1 transition-all ${
            viewMode === "3d"
              ? "bg-cyan-950/60 border-cyan-700/60 text-cyan-300"
              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200"
          }`}
          title="Toggle 3D / 2D Canvas"
        >
          <Box className="w-4 h-4" />
          <span className="hidden md:inline">{viewMode.toUpperCase()}</span>
        </button>

        {/* Chaos Studio Toggle */}
        <button
          onClick={onToggleChaosStudio}
          className={`p-2 rounded-xl border text-xs font-semibold flex items-center space-x-1 transition-all ${
            showChaosStudio
              ? "bg-rose-950/80 border-rose-700/80 text-rose-300 shadow-lg shadow-rose-950/50"
              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-rose-400"
          }`}
          title="Toggle Chaos Warfare Studio"
        >
          <Flame className="w-4 h-4" />
          <span className="hidden lg:inline">CHAOS</span>
        </button>

        {/* Partition Slicer Toggle */}
        <button
          onClick={onTogglePartitionSlicer}
          className={`p-2 rounded-xl border text-xs font-semibold flex items-center space-x-1 transition-all ${
            showPartitionSlicer
              ? "bg-cyan-950/80 border-cyan-700/80 text-cyan-300 shadow-lg shadow-cyan-950/50"
              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-cyan-400"
          }`}
          title="Toggle Partition Slicer"
        >
          <Scissors className="w-4 h-4" />
        </button>

        {/* Timeline Replay Toggle */}
        <button
          onClick={onToggleTimelineScrubber}
          className={`p-2 rounded-xl border text-xs font-semibold flex items-center space-x-1 transition-all ${
            showTimelineScrubber
              ? "bg-purple-950/80 border-purple-700/80 text-purple-300 shadow-lg shadow-purple-950/50"
              : "bg-slate-900 border-slate-800 text-slate-400 hover:text-purple-400"
          }`}
          title="Toggle Timeline Replay Scrubber"
        >
          <Clock className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
};
