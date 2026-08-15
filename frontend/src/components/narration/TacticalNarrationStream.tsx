import React, { useRef, useEffect } from "react";
import { MessageSquare, ShieldAlert, CheckCircle2, Navigation, AlertTriangle, Cpu } from "lucide-react";
import { ActionEvent } from "../../types";

interface TacticalNarrationStreamProps {
  actions: ActionEvent[];
  onFlyToNode?: (serviceId: string) => void;
}

export const TacticalNarrationStream: React.FC<TacticalNarrationStreamProps> = ({
  actions,
  onFlyToNode,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [actions]);

  return (
    <div className="flex flex-col bg-slate-950/90 backdrop-blur-xl border border-slate-800 rounded-2xl w-96 text-slate-100 shadow-2xl h-full max-h-[85vh] overflow-hidden text-xs">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 border-b border-slate-800 shrink-0">
        <div className="flex items-center space-x-2 text-cyan-400 font-bold">
          <MessageSquare className="w-4 h-4 text-cyan-400" />
          <span>TACTICAL LLM INCIDENT STREAM</span>
        </div>
        <span className="text-[10px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800/80 px-2 py-0.5 rounded-full">
          GROUNDED RCA
        </span>
      </div>

      {/* Feed Stream */}
      <div ref={containerRef} className="flex-1 overflow-y-auto p-3 space-y-3">
        {actions.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500 space-y-2">
            <Cpu className="w-8 h-8 opacity-40" />
            <p className="text-center italic text-xs">Awaiting agent policy execution & LLM narrations...</p>
          </div>
        ) : (
          actions.map((act, idx) => {
            const isVetoed = act.was_vetoed;

            return (
              <div
                key={idx}
                className={`p-3 rounded-xl border transition-all space-y-2 ${
                  isVetoed
                    ? "bg-rose-950/30 border-rose-900/60 text-slate-200"
                    : "bg-slate-900/70 border-slate-800 hover:border-slate-700"
                }`}
              >
                {/* Top Badge Info */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span className="font-mono text-[10px] bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded">
                      TICK {act.tick}
                    </span>
                    <span className="font-mono font-semibold text-cyan-400">
                      {act.agent_id}
                    </span>
                  </div>

                  {/* Status Indicator */}
                  {isVetoed ? (
                    <span className="flex items-center space-x-1 text-rose-400 font-semibold text-[10px] bg-rose-950/80 border border-rose-800 px-2 py-0.5 rounded-full">
                      <ShieldAlert className="w-3 h-3" />
                      <span>VETOED</span>
                    </span>
                  ) : (
                    <span className="flex items-center space-x-1 text-emerald-400 font-semibold text-[10px] bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded-full">
                      <CheckCircle2 className="w-3 h-3" />
                      <span>EXECUTED</span>
                    </span>
                  )}
                </div>

                {/* Action Command & Target */}
                <div className="flex items-center justify-between bg-slate-950/60 p-2 rounded-lg border border-slate-800 font-mono text-[11px]">
                  <div>
                    <span className="text-slate-400">Action: </span>
                    <span className="font-bold text-amber-300">{act.action}</span>
                    <span className="text-slate-400"> → </span>
                    <span className="font-bold text-cyan-300">{act.target_service}</span>
                  </div>

                  {/* Fly To Node Button */}
                  {onFlyToNode && (
                    <button
                      onClick={() => onFlyToNode(act.target_service)}
                      className="p-1 bg-cyan-950 hover:bg-cyan-900 border border-cyan-800 text-cyan-300 rounded-md transition-colors flex items-center space-x-1 text-[10px]"
                      title="Fly camera to node in 3D scene"
                    >
                      <Navigation className="w-3 h-3" />
                      <span>FLY</span>
                    </button>
                  )}
                </div>

                {/* LLM Narration Text */}
                <p className="text-slate-300 text-xs leading-relaxed">
                  {act.narration}
                </p>

                {/* Cited Graph Facts */}
                {(act.cited_edge_source || act.cited_edge_target) && (
                  <div className="flex items-center space-x-1 text-[10px] font-mono text-purple-300 bg-purple-950/40 border border-purple-800/60 p-1.5 rounded-lg">
                    <span className="font-bold text-purple-400">CITED GRAPH FACT:</span>
                    <span>
                      {act.cited_edge_source} → {act.cited_edge_target}
                    </span>
                  </div>
                )}

                {/* Veto Reason if applicable */}
                {isVetoed && act.veto_reason && (
                  <div className="flex items-start space-x-1.5 text-[10px] text-rose-300 bg-rose-950/60 border border-rose-800/80 p-2 rounded-lg">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold">Veto Policy ({act.veto_policy}): </span>
                      <span>{act.veto_reason}</span>
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
