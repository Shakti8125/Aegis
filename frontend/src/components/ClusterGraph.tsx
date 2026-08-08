import React, { useEffect, useRef, useState, useMemo } from "react";
import * as d3 from "d3";
import { ActionEvent, ClusterSnapshot, EdgeSnapshot, ServiceSnapshot } from "../types";
import { ZoomIn, ZoomOut, RefreshCw, Layers, ShieldAlert, Activity } from "lucide-react";

interface ClusterGraphProps {
  cluster: ClusterSnapshot | null;
  actions: ActionEvent[];
  selectedNodeId: string | null;
  highlightedEdge: { source: string; target: string } | null;
  onSelectNode: (id: string | null) => void;
}

interface GraphNode extends d3.SimulationNodeDatum, ServiceSnapshot {
  x?: number;
  y?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  source: string | GraphNode;
  target: string | GraphNode;
  relation: string;
  p99_latency_ms: number | null;
  error_rate: number | null;
  traffic_share: number | null;
}

export const ClusterGraph: React.FC<ClusterGraphProps> = ({
  cluster,
  actions,
  selectedNodeId,
  highlightedEdge,
  onSelectNode,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);
  const [pulseNodeId, setPulseNodeId] = useState<string | null>(null);
  const [pulseIsVeto, setPulseIsVeto] = useState(false);

  // Monitor latest action for pulse trigger
  useEffect(() => {
    if (actions.length > 0) {
      const last = actions[0];
      if (last && last.target_service) {
        setPulseNodeId(last.target_service);
        setPulseIsVeto(last.was_vetoed);
        const t = setTimeout(() => setPulseNodeId(null), 1500);
        return () => clearTimeout(t);
      }
    }
  }, [actions]);

  // Color helper matching PLAN.md §9 token system
  const getNodeColor = (health: number, isolated: boolean) => {
    if (isolated) return "#7C89A3"; // Cool gray when isolated
    if (health >= 0.85) return "#3DDC97"; // Muted teal (healthy)
    if (health >= 0.40) return "#F5A623"; // Amber (degraded)
    return "#E5484D"; // Controlled red (critical)
  };

  // Node data mapping
  const { nodesData, linksData } = useMemo(() => {
    if (!cluster) return { nodesData: [], linksData: [] };

    const nodes: GraphNode[] = cluster.services.map((s) => ({
      ...s,
    }));

    const links: GraphLink[] = cluster.edges.map((e) => ({
      source: e.source,
      target: e.target,
      relation: e.relation,
      p99_latency_ms: e.p99_latency_ms,
      error_rate: e.error_rate,
      traffic_share: e.traffic_share,
    }));

    return { nodesData: nodes, linksData: links };
  }, [cluster]);

  // D3 force simulation render
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || nodesData.length === 0) return;

    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 600;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Definitions for markers and glow filters
    const defs = svg.append("defs");

    // Glow filter
    const filter = defs.append("filter").attr("id", "glow").attr("x", "-20%").attr("y", "-20%").attr("width", "140%").attr("height", "140%");
    filter.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "blur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Directional Arrow Markers
    defs.append("marker")
      .attr("id", "arrow-normal")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 28)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#232B3E");

    defs.append("marker")
      .attr("id", "arrow-active")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 28)
      .attr("refY", 0)
      .attr("markerWidth", 8)
      .attr("markerHeight", 8)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#38BDF8");

    // Main graph container for zoom/pan
    const g = svg.append("g").attr("class", "main-group");

    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoom);

    // Tier Y positions
    const getTierY = (tier: string) => {
      const t = tier.toLowerCase();
      if (t === "front" || t === "tier_0") return height * 0.15;
      if (t === "edge") return height * 0.32;
      if (t.includes("mid")) return height * 0.52;
      return height * 0.78; // back/data
    };

    // Force simulation setup
    const simulation = d3.forceSimulation<GraphNode, GraphLink>(nodesData)
      .force("link", d3.forceLink<GraphNode, GraphLink>(linksData).id((d) => d.id).distance(120))
      .force("charge", d3.forceManyBody().strength(-400))
      .force("collide", d3.forceCollide().radius(45))
      .force("x", d3.forceX(width / 2).strength(0.15))
      .force("y", d3.forceY<GraphNode>((d: GraphNode) => getTierY(d.tier)).strength(0.6));

    simulationRef.current = simulation;

    // Render Edges
    const linkGroup = g.append("g").attr("class", "links");
    const link = linkGroup
      .selectAll("path")
      .data(linksData)
      .enter()
      .append("path")
      .attr("class", "edge-path")
      .attr("stroke", (d) => {
        const srcId = typeof d.source === "string" ? d.source : d.source.id;
        const tgtId = typeof d.target === "string" ? d.target : d.target.id;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "#38BDF8";
        }
        return "#232B3E";
      })
      .attr("stroke-width", (d) => {
        const srcId = typeof d.source === "string" ? d.source : d.source.id;
        const tgtId = typeof d.target === "string" ? d.target : d.target.id;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return 3;
        }
        return 1.5;
      })
      .attr("stroke-dasharray", (d) => {
        const srcId = typeof d.source === "string" ? d.source : d.source.id;
        const tgtId = typeof d.target === "string" ? d.target : d.target.id;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "6, 4";
        }
        return "none";
      })
      .attr("fill", "none")
      .attr("marker-end", (d) => {
        const srcId = typeof d.source === "string" ? d.source : d.source.id;
        const tgtId = typeof d.target === "string" ? d.target : d.target.id;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "url(#arrow-active)";
        }
        return "url(#arrow-normal)";
      });

    // Render Nodes
    const nodeGroup = g.append("g").attr("class", "nodes");
    const node = nodeGroup
      .selectAll("g")
      .data(nodesData)
      .enter()
      .append("g")
      .attr("class", "node-group cursor-pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectNode(d.id);
      })
      .call(
        d3.drag<SVGGElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Pulse Ring for Actions / Vetoes / Selection
    node
      .filter((d) => d.id === pulseNodeId || d.id === selectedNodeId)
      .append("circle")
      .attr("r", 34)
      .attr("fill", "none")
      .attr("stroke", (d) => {
        if (d.id === pulseNodeId && pulseIsVeto) return "#E5484D";
        if (d.id === pulseNodeId) return "#38BDF8";
        return "#3DDC97";
      })
      .attr("stroke-width", 3)
      .attr("opacity", 0.8)
      .attr("class", "animate-ping");

    // Outer glow ring for SLA violators
    node
      .filter((d) => d.sla_violating)
      .append("circle")
      .attr("r", 28)
      .attr("fill", "none")
      .attr("stroke", "#E5484D")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "4, 2")
      .attr("filter", "url(#glow)");

    // Base Node Circle
    node
      .append("circle")
      .attr("r", 22)
      .attr("fill", "#131822")
      .attr("stroke", (d) => getNodeColor(d.health, d.isolated))
      .attr("stroke-width", (d) => (d.id === selectedNodeId ? 3.5 : 2))
      .attr("filter", (d) => (d.health < 0.85 || d.id === selectedNodeId ? "url(#glow)" : "none"));

    // Inner Health Indicator Fill
    node
      .append("circle")
      .attr("r", 8)
      .attr("fill", (d) => getNodeColor(d.health, d.isolated));

    // Service Name Label
    node
      .append("text")
      .text((d) => d.id)
      .attr("y", 36)
      .attr("text-anchor", "middle")
      .attr("fill", (d) => (d.id === selectedNodeId ? "#38BDF8" : "#F1F5F9"))
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("font-size", "11px")
      .attr("font-weight", "600");

    // Replica Count Badge
    node
      .append("text")
      .text((d) => `${d.ready_replicas}/${d.replicas}`)
      .attr("y", 4)
      .attr("text-anchor", "middle")
      .attr("fill", "#0B0E14")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("font-size", "9px")
      .attr("font-weight", "700");

    // Simulation tick handler
    simulation.on("tick", () => {
      link.attr("d", (d) => {
        const source = d.source as GraphNode;
        const target = d.target as GraphNode;
        const dx = (target.x || 0) - (source.x || 0);
        const dy = (target.y || 0) - (source.y || 0);
        const dr = Math.sqrt(dx * dx + dy * dy) * 1.2;
        return `M${source.x},${source.y}A${dr},${dr} 0 0,1 ${target.x},${target.y}`;
      });

      node.attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodesData, linksData, selectedNodeId, pulseNodeId, pulseIsVeto, highlightedEdge, onSelectNode]);

  return (
    <div ref={containerRef} className="relative w-full h-full bg-[#0B0E14] overflow-hidden rounded-xl border border-[#232B3E]">
      {/* Tier Zone Labels */}
      <div className="absolute left-4 top-4 flex flex-col gap-2 z-10 pointer-events-none">
        <div className="flex items-center gap-2 text-xs font-mono text-[#7C89A3] bg-[#131822]/80 px-2.5 py-1 rounded border border-[#232B3E]">
          <Layers className="w-3.5 h-3.5 text-[#38BDF8]" />
          <span>Tier 0 (Gateway / Front)</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-[#7C89A3] bg-[#131822]/80 px-2.5 py-1 rounded border border-[#232B3E] mt-12">
          <Activity className="w-3.5 h-3.5 text-[#3DDC97]" />
          <span>Mid Tier Services</span>
        </div>
        <div className="flex items-center gap-2 text-xs font-mono text-[#7C89A3] bg-[#131822]/80 px-2.5 py-1 rounded border border-[#232B3E] mt-24">
          <ShieldAlert className="w-3.5 h-3.5 text-[#F5A623]" />
          <span>Data / Storage Tier</span>
        </div>
      </div>

      {/* Health Legend */}
      <div className="absolute right-4 top-4 flex items-center gap-3 bg-[#131822]/90 backdrop-blur border border-[#232B3E] px-3 py-1.5 rounded-lg z-10 text-xs font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#3DDC97] shadow-[0_0_8px_#3DDC97]"></span>
          <span className="text-[#F1F5F9]">Healthy (&ge;85%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#F5A623] shadow-[0_0_8px_#F5A623]"></span>
          <span className="text-[#F1F5F9]">Degraded</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5484D] shadow-[0_0_8px_#E5484D]"></span>
          <span className="text-[#F1F5F9]">Critical (&lt;40%)</span>
        </div>
      </div>

      {/* SVG Canvas */}
      <svg ref={svgRef} className="w-full h-full min-h-[550px] cursor-grab active:cursor-grabbing" />
    </div>
  );
};
