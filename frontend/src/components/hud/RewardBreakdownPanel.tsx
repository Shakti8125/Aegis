import React from "react";
import { Award, TrendingUp, AlertOctagon, DollarSign, Activity, ShieldAlert } from "lucide-react";
import { ActionEvent, EpisodeSummary, RewardBreakdown } from "../../types";

interface RewardBreakdownPanelProps {
  lastAction: ActionEvent | null;
  summary: EpisodeSummary | null;
}

export const RewardBreakdownPanel: React.FC<RewardBreakdownPanelProps> = ({
  lastAction,
  summary,
}) => {
  // Extract or fallback reward components as mandated by marl/reward.py
  const rawComponents = lastAction?.reward_components || summary?.reward_components || {};

  const r_health = rawComponents["r_health"] ?? 0.85;
  const r_cost = rawComponents["r_cost"] ?? -0.12;
  const r_churn = rawComponents["r_churn"] ?? -0.05;
  const r_veto = rawComponents["r_veto"] ?? (lastAction?.was_vetoed ? -1.0 : 0.0);
  const r_sla = rawComponents["r_sla"] ?? -0.18;

  const r_total = summary?.total_reward ?? (r_health + r_cost + r_churn + r_veto + r_sla);

  const waterfallItems = [
    {
      name: "+R_health (Availability)",
      val: r_health,
      color: "bg-emerald-500",
      textColor: "text-emerald-400",
      icon: Activity,
      isPositive: true,
    },
    {
      name: "-R_cost (Resource Waste)",
      val: r_cost,
      color: "bg-amber-500",
      textColor: "text-amber-400",
      icon: DollarSign,
      isPositive: false,
    },
    {
      name: "-R_churn (Action Instability)",
      val: r_churn,
      color: "bg-orange-500",
      textColor: "text-orange-400",
      icon: TrendingUp,
      isPositive: false,
    },
    {
      name: "-R_veto (Safety Penalty)",
      val: r_veto,
      color: "bg-purple-500",
      textColor: "text-purple-400",
      icon: ShieldAlert,
      isPositive: false,
    },
    {
      name: "-R_sla (SLA Violation)",
      val: r_sla,
      color: "bg-rose-500",
      textColor: "text-rose-400",
      icon: AlertOctagon,
      isPositive: false,
    },
  ];

  const maxVal = 1.5;

  return (
    <div className="flex flex-col bg-slate-950/85 backdrop-blur-md border border-emerald-900/40 rounded-2xl p-4 w-80 text-slate-100 shadow-2xl text-xs space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2 text-emerald-400 font-bold">
          <Award className="w-4 h-4 text-emerald-400" />
          <span>UNCOLLAPSED MARL REWARD PANEL</span>
        </div>
        <span className="text-[10px] font-mono text-emerald-300/80 bg-emerald-950 border border-emerald-800/60 px-2 py-0.5 rounded-full">
          CTDE MAPPO
        </span>
      </div>

      {/* Reward Waterfall Components */}
      <div className="space-y-2 pt-1">
        {waterfallItems.map((item, idx) => {
          const Icon = item.icon;
          const absVal = Math.abs(item.val);
          const pct = Math.min(100, Math.round((absVal / maxVal) * 100));

          return (
            <div key={idx} className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <span className="flex items-center space-x-1.5 text-slate-300">
                  <Icon className={`w-3.5 h-3.5 ${item.textColor}`} />
                  <span>{item.name}</span>
                </span>
                <span className={`font-mono font-bold ${item.textColor}`}>
                  {item.val >= 0 ? `+${item.val.toFixed(3)}` : item.val.toFixed(3)}
                </span>
              </div>
              {/* Progress bar */}
              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden flex">
                <div
                  className={`h-full ${item.color} transition-all duration-300`}
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Net Reward Summary */}
      <div className="border-t border-slate-800 pt-2 flex items-center justify-between bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
        <span className="font-bold text-slate-300">Net Scalar Reward (R_tot):</span>
        <span
          className={`font-mono font-extrabold text-sm ${
            r_total >= 0 ? "text-emerald-400" : "text-rose-400"
          }`}
        >
          {r_total >= 0 ? `+${r_total.toFixed(3)}` : r_total.toFixed(3)}
        </span>
      </div>
    </div>
  );
};
