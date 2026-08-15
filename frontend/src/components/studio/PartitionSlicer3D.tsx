import React, { useState } from "react";
import { Scissors, Zap, AlertTriangle, X, Sliders } from "lucide-react";
import { ClusterSnapshot, PartitionPlaneState } from "../../types";

interface PartitionSlicer3DProps {
  cluster: ClusterSnapshot | null;
  partitionState: PartitionPlaneState;
  onUpdatePartition: (state: PartitionPlaneState) => void;
  onTriggerPartitionFault: (blockedServices: string[]) => void;
  onClose?: () => void;
}

export const PartitionSlicer3D: React.FC<PartitionSlicer3DProps> = ({
  cluster,
  partitionState,
  onUpdatePartition,
  onTriggerPartitionFault,
  onClose,
}) => {
  const [sliceHeight, setSliceHeight] = useState(partitionState.positionY);
  const [sliceAngle, setSliceAngle] = useState(partitionState.angleDegrees);

  const services = cluster?.services || [];

  // Compute which services are affected by current slicing plane position
  const affectedServices = services.filter((svc) => {
    // Slicing threshold evaluation
    const inUpperSubnet = (svc.position3d ? svc.position3d[1] : 0) >= sliceHeight;
    return inUpperSubnet;
  });

  const affectedIds = affectedServices.map((s) => s.id);

  const handleToggleActive = () => {
    onUpdatePartition({
      ...partitionState,
      active: !partitionState.active,
      positionY: sliceHeight,
      angleDegrees: sliceAngle,
      blockedServices: !partitionState.active ? affectedIds : [],
    });
  };

  const handleApplyChanges = () => {
    onUpdatePartition({
      ...partitionState,
      positionY: sliceHeight,
      angleDegrees: sliceAngle,
      blockedServices: affectedIds,
    });
  };

  const handleExecuteFault = () => {
    onTriggerPartitionFault(affectedIds);
  };

  return (
    <div className="flex flex-col bg-slate-900/90 backdrop-blur-md border border-slate-700/70 rounded-xl p-4 w-80 text-slate-100 shadow-2xl text-xs space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2 text-cyan-400 font-semibold text-sm">
          <Scissors className="w-4 h-4" />
          <span>3D Network Partition Slicer</span>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Slicer Activation Toggle */}
      <div className="flex items-center justify-between bg-slate-800/50 p-2 rounded-lg border border-slate-700/40">
        <span className="font-medium text-slate-300">Spatial Slicing Plane</span>
        <button
          onClick={handleToggleActive}
          className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
            partitionState.active
              ? "bg-rose-600 text-white shadow-lg shadow-rose-600/30"
              : "bg-slate-700 text-slate-300 hover:bg-slate-600"
          }`}
        >
          {partitionState.active ? "SLICER ACTIVE" : "ENABLE PLANE"}
        </button>
      </div>

      {/* Sliders */}
      {partitionState.active && (
        <div className="space-y-3 pt-1">
          <div>
            <div className="flex justify-between text-slate-400 mb-1">
              <span className="flex items-center space-x-1">
                <Sliders className="w-3 h-3 text-cyan-400" />
                <span>Cutting Plane Y-Offset</span>
              </span>
              <span className="font-mono text-cyan-300">{sliceHeight.toFixed(1)}m</span>
            </div>
            <input
              type="range"
              min="-4.0"
              max="4.0"
              step="0.2"
              value={sliceHeight}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setSliceHeight(val);
                handleApplyChanges();
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
          </div>

          <div>
            <div className="flex justify-between text-slate-400 mb-1">
              <span>Plane Rotation Angle</span>
              <span className="font-mono text-amber-300">{sliceAngle}°</span>
            </div>
            <input
              type="range"
              min="0"
              max="180"
              step="5"
              value={sliceAngle}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                setSliceAngle(val);
                handleApplyChanges();
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
          </div>

          {/* Severed Dependencies Preview */}
          <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 space-y-1.5">
            <div className="flex items-center justify-between text-[11px] text-slate-400">
              <span className="flex items-center space-x-1 text-rose-400 font-medium">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Severed Subnet Nodes ({affectedIds.length})</span>
              </span>
            </div>
            <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto pr-1">
              {affectedServices.length > 0 ? (
                affectedServices.map((svc) => (
                  <span
                    key={svc.id}
                    className="bg-rose-950/70 border border-rose-800/60 text-rose-300 px-1.5 py-0.5 rounded text-[10px] font-mono"
                  >
                    {svc.name}
                  </span>
                ))
              ) : (
                <span className="text-slate-500 italic text-[10px]">No nodes crossed by cutting plane</span>
              )}
            </div>
          </div>

          {/* Trigger Fault Button */}
          <button
            onClick={handleExecuteFault}
            disabled={affectedIds.length === 0}
            className="w-full py-2 bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-bold rounded-lg shadow-lg flex items-center justify-center space-x-2 transition-all"
          >
            <Zap className="w-4 h-4 fill-current" />
            <span>INJECT NETWORK PARTITION</span>
          </button>
        </div>
      )}
    </div>
  );
};
