import React from "react";
import { useWebSocket } from "../context/WebSocketContext";
import { ShieldAlert, Activity, AlertTriangle, Info, ShieldX } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const IncidentFeed: React.FC = () => {
  const { actions, cluster, status } = useWebSocket();

  const getStatusBadge = () => {
    switch (status) {
      case "connected":
        return (
          <div className="flex items-center gap-2 text-xs font-mono text-[#3DDC97]">
            <span className="w-2 h-2 rounded-full bg-[#3DDC97] animate-pulse"></span>
            <span>LIVE</span>
          </div>
        );
      case "connecting":
        return (
          <div className="flex items-center gap-2 text-xs font-mono text-[#F5A623]">
            <span className="w-2 h-2 rounded-full bg-[#F5A623] animate-ping"></span>
            <span>CONNECTING</span>
          </div>
        );
      default:
        return (
          <div className="flex items-center gap-2 text-xs font-mono text-[#E5484D]">
            <span className="w-2 h-2 rounded-full bg-[#E5484D]"></span>
            <span>OFFLINE</span>
          </div>
        );
    }
  };

  return (
    <div className="flex h-full flex-col bg-[#0B0E14] border-l border-[#7C89A3]/20 w-80">
      <div className="p-4 border-b border-[#7C89A3]/20 flex items-center justify-between">
        <h2 className="text-sm font-mono uppercase tracking-wider text-[#7C89A3]">Incident Feed</h2>
        {getStatusBadge()}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {actions.length === 0 ? (
          <div className="text-center text-[#7C89A3] text-sm font-sans mt-10">
            No actions recorded yet.
          </div>
        ) : (
          actions.map((act, i) => (
            <div
              key={`${act.tick}-${act.agent_id}-${i}`}
              className={cn(
                "p-3 rounded border text-sm font-sans relative overflow-hidden",
                act.was_vetoed
                  ? "bg-[#E5484D]/10 border-[#E5484D]/30"
                  : act.action === "no_op"
                  ? "bg-[#7C89A3]/5 border-[#7C89A3]/10"
                  : "bg-[#3DDC97]/10 border-[#3DDC97]/30"
              )}
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  {act.was_vetoed ? (
                    <ShieldX size={14} className="text-[#E5484D]" />
                  ) : act.action === "no_op" ? (
                    <Info size={14} className="text-[#7C89A3]" />
                  ) : (
                    <Activity size={14} className="text-[#3DDC97]" />
                  )}
                  <span className="font-mono text-xs text-[#F1F5F9] font-semibold">
                    {act.target_service}
                  </span>
                </div>
                <span className="font-mono text-xs text-[#7C89A3]">t={act.tick}</span>
              </div>

              {/* Action type */}
              <div className="flex items-center gap-2 mb-2">
                <span
                  className={cn(
                    "px-2 py-0.5 rounded text-xs font-mono uppercase",
                    act.was_vetoed
                      ? "bg-[#E5484D]/20 text-[#E5484D]"
                      : "bg-[#7C89A3]/20 text-[#F1F5F9]"
                  )}
                >
                  {act.action}
                </span>
                {act.was_vetoed && (
                  <span className="text-xs font-bold text-[#E5484D]">VETOED</span>
                )}
              </div>

              {/* Narration prose */}
              <p className="text-[#F1F5F9]/90 text-xs leading-relaxed">
                {act.narration}
              </p>

              {/* Veto reason if applicable */}
              {act.was_vetoed && (
                <div className="mt-2 p-2 bg-[#0B0E14]/50 rounded border border-[#E5484D]/20">
                  <p className="text-[#E5484D] text-xs">
                    <span className="font-semibold">Reason:</span> {act.veto_reason}
                  </p>
                  <p className="text-[#7C89A3] text-[10px] font-mono mt-1">
                    Policy: {act.veto_policy}
                  </p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
