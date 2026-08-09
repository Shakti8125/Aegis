import React from "react";
import { useWebSocket } from "../context/WebSocketContext";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export const MetricsPanel: React.FC = () => {
  const { history, cluster } = useWebSocket();

  if (!cluster || history.length === 0) {
    return (
      <div className="flex h-64 flex-col bg-[#0B0E14] border-t border-[#7C89A3]/20 w-full p-4">
        <h2 className="text-sm font-mono uppercase text-[#7C89A3] mb-4">Live Metrics</h2>
        <div className="flex-1 flex items-center justify-center text-[#7C89A3] text-sm font-sans">
          Waiting for simulation history...
        </div>
      </div>
    );
  }

  // Format tick labels
  const formatTick = (tick: number) => `t=${tick}`;

  return (
    <div className="flex h-64 bg-[#0B0E14] border-t border-[#7C89A3]/20 w-full p-4 gap-8">
      {/* Chart 1: Cluster Health */}
      <div className="flex-1 flex flex-col">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-xs font-mono uppercase text-[#7C89A3]">Cluster Health</h3>
          <span className="text-xs font-mono font-bold text-[#3DDC97]">
            Avg: {(cluster.mean_health * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex-1 min-h-0 relative">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#7C89A3" opacity={0.2} vertical={false} />
              <XAxis
                dataKey="tick"
                tickFormatter={formatTick}
                stroke="#7C89A3"
                fontSize={10}
                tickMargin={8}
                minTickGap={30}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                stroke="#7C89A3"
                fontSize={10}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0B0E14",
                  border: "1px solid rgba(124, 137, 163, 0.2)",
                  borderRadius: "4px",
                  fontSize: "12px",
                }}
                itemStyle={{ color: "#F1F5F9" }}
                labelStyle={{ color: "#7C89A3", fontFamily: "monospace" }}
              />
              <Line
                type="monotone"
                dataKey="mean_health"
                name="Mean Health"
                stroke="#3DDC97"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="stepAfter"
                dataKey="min_health"
                name="Min Health"
                stroke="#F5A623"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Chart 2: SLA Violations */}
      <div className="flex-1 flex flex-col">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-xs font-mono uppercase text-[#7C89A3]">SLA Violation Rate</h3>
          <span className="text-xs font-mono font-bold text-[#E5484D]">
            {(cluster.sla_violation_rate * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex-1 min-h-0 relative">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="slaGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#E5484D" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#E5484D" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#7C89A3" opacity={0.2} vertical={false} />
              <XAxis
                dataKey="tick"
                tickFormatter={formatTick}
                stroke="#7C89A3"
                fontSize={10}
                tickMargin={8}
                minTickGap={30}
              />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(val) => `${(val * 100).toFixed(0)}%`}
                stroke="#7C89A3"
                fontSize={10}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0B0E14",
                  border: "1px solid rgba(124, 137, 163, 0.2)",
                  borderRadius: "4px",
                  fontSize: "12px",
                }}
                itemStyle={{ color: "#E5484D" }}
                labelStyle={{ color: "#7C89A3", fontFamily: "monospace" }}
              />
              <Area
                type="monotone"
                dataKey="sla_violation_rate"
                name="SLA Violations"
                stroke="#E5484D"
                fillOpacity={1}
                fill="url(#slaGradient)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
