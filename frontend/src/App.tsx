import React, { useState } from "react";
import { useWebSocket } from "./context/WebSocketContext";
import { OpsCenterCanvas } from "./components/3d/OpsCenterCanvas";
import { FloatingHUD } from "./components/hud/FloatingHUD";
import { GNNEmbeddingRadar } from "./components/hud/GNNEmbeddingRadar";
import { RewardBreakdownPanel } from "./components/hud/RewardBreakdownPanel";
import { TacticalNarrationStream } from "./components/narration/TacticalNarrationStream";
import { TelemetryTracer } from "./components/narration/TelemetryTracer";
import { ChaosWarfareStudio } from "./components/studio/ChaosWarfareStudio";
import { PartitionSlicer3D } from "./components/studio/PartitionSlicer3D";
import { TimelineScrubber } from "./components/replay/TimelineScrubber";
import { NodeInspector } from "./components/NodeInspector";
import { AutonomyLevel, ChaosTrigger, PartitionPlaneState, TimelineState } from "./types";

export const App: React.FC = () => {
  const {
    status,
    cluster,
    actions,
    summary,
    isPlaying,
    tickDelayMs,
    selectedNodeId,
    highlightedEdge,
    setSelectedNodeId,
    setTickDelayMs,
    startSimulation,
    stopSimulation,
    resetStream,
  } = useWebSocket();

  // State management for 3D HUD & Studio features
  const [viewMode, setViewMode] = useState<"3d" | "2d">("3d");
  const [autonomyLevel, setAutonomyLevel] = useState<AutonomyLevel>(2);

  const [showChaosStudio, setShowChaosStudio] = useState(false);
  const [showPartitionSlicer, setShowPartitionSlicer] = useState(false);
  const [showTimelineScrubber, setShowTimelineScrubber] = useState(false);

  const [partitionState, setPartitionState] = useState<PartitionPlaneState>({
    active: false,
    positionY: 0.0,
    positionZ: 0.0,
    angleDegrees: 0,
    blockedServices: [],
  });

  const [timelineState, setTimelineState] = useState<TimelineState>({
    currentTick: cluster?.tick ?? 0,
    maxTick: 200,
    isPlaying: isPlaying,
    playbackSpeed: 1.0,
    snapshotBuffer: [],
    diffTick: null,
  });

  const handleTogglePlay = () => {
    if (isPlaying) {
      stopSimulation();
    } else {
      startSimulation({ tickDelayMs });
    }
  };

  const handleInjectChaos = (trigger: Omit<ChaosTrigger, "id" | "status">) => {
    console.log("Injecting Chaos Trigger into Aegis Simulator:", trigger);
    // Future backend trigger dispatch via WebSocket / REST
  };

  const handleTriggerPartitionFault = (blockedServices: string[]) => {
    console.log("Triggering Network Partition Fault on subnets:", blockedServices);
  };

  const handleSeekTick = (targetTick: number) => {
    setTimelineState((prev) => ({ ...prev, currentTick: targetTick }));
  };

  const handleFlyToNode = (serviceId: string) => {
    setSelectedNodeId(serviceId);
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0B0E14] text-[#F1F5F9] overflow-hidden relative select-none">
      {/* 1. Sleek Glassmorphism Header HUD */}
      <FloatingHUD
        status={status}
        cluster={cluster}
        isPlaying={isPlaying}
        autonomyLevel={autonomyLevel}
        tickDelayMs={tickDelayMs}
        viewMode={viewMode}
        showChaosStudio={showChaosStudio}
        showPartitionSlicer={showPartitionSlicer}
        showTimelineScrubber={showTimelineScrubber}
        onTogglePlay={handleTogglePlay}
        onReset={resetStream}
        onSetTickDelay={setTickDelayMs}
        onToggleViewMode={() => setViewMode(viewMode === "3d" ? "2d" : "3d")}
        onToggleChaosStudio={() => setShowChaosStudio(!showChaosStudio)}
        onTogglePartitionSlicer={() => setShowPartitionSlicer(!showPartitionSlicer)}
        onToggleTimelineScrubber={() => setShowTimelineScrubber(!showTimelineScrubber)}
        onSetAutonomyLevel={setAutonomyLevel}
      />

      {/* 2. Main Hero 3D Spatial Canvas Container */}
      <main className="flex-1 relative w-full h-full">
        <OpsCenterCanvas
          cluster={cluster}
          actions={actions}
          selectedNodeId={selectedNodeId}
          highlightedEdge={highlightedEdge}
          viewMode={viewMode}
          partitionState={partitionState}
          onSelectNode={setSelectedNodeId}
        />

        {/* Floating Left Dock: GNN Embedding Radar + Reward Waterfall */}
        <div className="absolute top-20 left-4 z-30 flex flex-col space-y-4 max-h-[calc(100vh-140px)] overflow-y-auto pr-1">
          <GNNEmbeddingRadar
            cluster={cluster}
            selectedNodeId={selectedNodeId}
            onSelectService={setSelectedNodeId}
          />
          <RewardBreakdownPanel
            lastAction={actions[0] || null}
            summary={summary}
          />
        </div>

        {/* Floating Right Dock: LLM Incident Stream + Telemetry Tracer */}
        <div className="absolute top-20 right-4 z-30 flex flex-col space-y-4 max-h-[calc(100vh-140px)] overflow-y-auto pl-1">
          <TacticalNarrationStream
            actions={actions}
            onFlyToNode={handleFlyToNode}
          />
          <TelemetryTracer
            cluster={cluster}
            selectedNodeId={selectedNodeId}
            onSelectNode={setSelectedNodeId}
          />
        </div>

        {/* Floating Studio Drawers */}
        {showChaosStudio && (
          <div className="absolute top-20 left-[340px] z-40">
            <ChaosWarfareStudio
              cluster={cluster}
              selectedNodeId={selectedNodeId}
              onInjectChaos={handleInjectChaos}
              onClose={() => setShowChaosStudio(false)}
            />
          </div>
        )}

        {showPartitionSlicer && (
          <div className="absolute top-20 left-[340px] z-40">
            <PartitionSlicer3D
              cluster={cluster}
              partitionState={partitionState}
              onUpdatePartition={setPartitionState}
              onTriggerPartitionFault={handleTriggerPartitionFault}
              onClose={() => setShowPartitionSlicer(false)}
            />
          </div>
        )}

        {/* Node Details Inspector Overlay Drawer */}
        {selectedNodeId && (
          <NodeInspector
            selectedNodeId={selectedNodeId}
            cluster={cluster}
            onClose={() => setSelectedNodeId(null)}
          />
        )}
      </main>

      {/* 3. Bottom Time-Travel Replay Scrubber */}
      {showTimelineScrubber && (
        <TimelineScrubber
          timelineState={timelineState}
          cluster={cluster}
          onSeekTick={handleSeekTick}
          onTogglePlay={handleTogglePlay}
          onSetSpeed={(speed) => setTimelineState((p) => ({ ...p, playbackSpeed: speed }))}
          onClose={() => setShowTimelineScrubber(false)}
        />
      )}
    </div>
  );
};

export default App;
