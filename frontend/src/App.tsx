import React from "react";
import { Header } from "./components/Header";
import { ClusterGraph } from "./components/ClusterGraph";
import { NodeInspector } from "./components/NodeInspector";
import { useWebSocket } from "./context/WebSocketContext";

export const App: React.FC = () => {
  const {
    cluster,
    actions,
    selectedNodeId,
    highlightedEdge,
    setSelectedNodeId,
  } = useWebSocket();

  return (
    <div className="flex flex-col h-screen w-screen bg-[#0B0E14] text-[#F1F5F9] overflow-hidden">
      {/* Top Navigation & Controls */}
      <Header />

      {/* Main Hero Canvas: ClusterGraph */}
      <main className="flex-1 relative p-4 bg-[#0B0E14]">
        <ClusterGraph
          cluster={cluster}
          actions={actions}
          selectedNodeId={selectedNodeId}
          highlightedEdge={highlightedEdge}
          onSelectNode={setSelectedNodeId}
        />

        {/* Selected Node Details Drawer */}
        <NodeInspector
          selectedNodeId={selectedNodeId}
          cluster={cluster}
          onClose={() => setSelectedNodeId(null)}
        />
      </main>
    </div>
  );
};

export default App;
