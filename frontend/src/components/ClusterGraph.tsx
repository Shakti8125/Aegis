import React, { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { ActionEvent, ClusterSnapshot } from "../types";
import { Layers, ShieldAlert, Activity, WifiOff } from "lucide-react";

interface ClusterGraphProps {
  cluster: ClusterSnapshot | null;
  actions: ActionEvent[];
  selectedNodeId: string | null;
  highlightedEdge: { source: string; target: string } | null;
  onSelectNode: (id: string | null) => void;
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  name: string;
  tier: string;
  health: number;
  status: "healthy" | "degraded" | "critical";
  cpu_pct: number;
  mem_pct: number;
  p99_latency_ms: number;
  error_rate: number;
  replicas: number;
  ready_replicas: number;
  isolated: boolean;
  sla_violating: boolean;
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
  const nodesMapRef = useRef<Map<string, GraphNode>>(new Map());
  const linksRef = useRef<GraphLink[]>([]);
  const isInitializedRef = useRef(false);

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

  // Accessibility stroke pattern helper (§1.5)
  const getNodeStrokeDash = (health: number, isolated: boolean) => {
    if (isolated) return "6, 6";
    if (health >= 0.85) return "none";
    if (health >= 0.40) return "5, 3";
    return "2, 2";
  };

  // Initialize SVG structure and D3 force simulation ONCE
  useEffect(() => {
    if (!svgRef.current || !containerRef.current || isInitializedRef.current) return;

    const width = containerRef.current.clientWidth || 800;
    const height = containerRef.current.clientHeight || 600;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Defs
    const defs = svg.append("defs");

    // Glow filter
    const filter = defs
      .append("filter")
      .attr("id", "glow")
      .attr("x", "-20%")
      .attr("y", "-20%")
      .attr("width", "140%")
      .attr("height", "140%");
    filter.append("feGaussianBlur").attr("stdDeviation", "4").attr("result", "blur");
    const feMerge = filter.append("feMerge");
    feMerge.append("feMergeNode").attr("in", "blur");
    feMerge.append("feMergeNode").attr("in", "SourceGraphic");

    // Arrow Markers
    defs
      .append("marker")
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

    defs
      .append("marker")
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

    // Root Group for Zoom
    const g = svg.append("g").attr("class", "main-group");
    g.append("g").attr("class", "links");
    g.append("g").attr("class", "nodes");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.5, 3])
      .on("zoom", (event) => {
        g.attr("transform", event.transform);
      });

    svg.call(zoom);

    const getTierY = (tier: string) => {
      const t = tier.toLowerCase();
      if (t === "front" || t === "tier_0") return height * 0.15;
      if (t === "edge") return height * 0.32;
      if (t.includes("mid")) return height * 0.52;
      return height * 0.78;
    };

    // Force simulation
    const simulation = d3
      .forceSimulation<GraphNode, GraphLink>([])
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphLink>([])
          .id((d) => d.id)
          .distance(120)
      )
      .force("charge", d3.forceManyBody().strength(-400))
      .force("collide", d3.forceCollide().radius(45))
      .force("x", d3.forceX(width / 2).strength(0.15))
      .force("y", d3.forceY<GraphNode>((d: GraphNode) => getTierY(d.tier)).strength(0.6));

    simulation.on("tick", () => {
      const svgEl = d3.select(svgRef.current);
      svgEl
        .selectAll<SVGPathElement, GraphLink>("path.edge-path")
        .attr("d", (d) => {
          const source = d.source as GraphNode;
          const target = d.target as GraphNode;
          if (!source.x || !target.x) return "";
          const dx = (target.x || 0) - (source.x || 0);
          const dy = (target.y || 0) - (source.y || 0);
          const dr = Math.sqrt(dx * dx + dy * dy) * 1.2;
          return `M${source.x},${source.y}A${dr},${dr} 0 0,1 ${target.x},${target.y}`;
        });

      svgEl
        .selectAll<SVGGElement, GraphNode>("g.node-group")
        .attr("transform", (d) => `translate(${d.x || 0},${d.y || 0})`);
    });

    simulationRef.current = simulation;
    isInitializedRef.current = true;

    return () => {
      simulation.stop();
      isInitializedRef.current = false;
    };
  }, []);

  // Update telemetry and DOM attributes without full remount or zoom reset
  useEffect(() => {
    if (!cluster || !svgRef.current || !simulationRef.current) return;

    const simulation = simulationRef.current;
    const currentNodesMap = nodesMapRef.current;
    let topologyChanged = false;

    // 1. Sync Nodes
    const incomingServiceIds = new Set(cluster.services.map((s) => s.id));

    // Remove deleted nodes
    for (const [id] of currentNodesMap) {
      if (!incomingServiceIds.has(id)) {
        currentNodesMap.delete(id);
        topologyChanged = true;
      }
    }

    // Add or update existing nodes in place
    cluster.services.forEach((s) => {
      let existing = currentNodesMap.get(s.id);
      if (!existing) {
        existing = { ...s };
        currentNodesMap.set(s.id, existing);
        topologyChanged = true;
      } else {
        // In-place attribute update
        Object.assign(existing, s);
      }
    });

    const nodesArray = Array.from(currentNodesMap.values());

    // 2. Sync Links
    const linksArray: GraphLink[] = cluster.edges.map((e) => ({
      source: e.source,
      target: e.target,
      relation: e.relation,
      p99_latency_ms: e.p99_latency_ms,
      error_rate: e.error_rate,
      traffic_share: e.traffic_share,
    }));

    if (linksArray.length !== linksRef.current.length) {
      topologyChanged = true;
    }
    linksRef.current = linksArray;

    // Restart simulation only if topology (IDs / edge count) changed
    if (topologyChanged) {
      simulation.nodes(nodesArray);
      const linkForce = simulation.force<d3.ForceLink<GraphNode, GraphLink>>("link");
      if (linkForce) {
        linkForce.links(linksArray);
      }
      simulation.alpha(0.3).restart();
    }

    // 3. Update DOM selections via Data Join
    const svg = d3.select(svgRef.current);
    const linkGroup = svg.select<SVGGElement>("g.links");
    const nodeGroup = svg.select<SVGGElement>("g.nodes");

    // Edges Join
    const linkJoin = linkGroup
      .selectAll<SVGPathElement, GraphLink>("path.edge-path")
      .data(linksArray, (d) => {
        const srcId = typeof d.source === "object" ? d.source.id : d.source;
        const tgtId = typeof d.target === "object" ? d.target.id : d.target;
        return `${srcId}->${tgtId}`;
      });

    linkJoin.exit().remove();

    const linkEnter = linkJoin
      .enter()
      .append("path")
      .attr("class", "edge-path")
      .attr("fill", "none");

    const linkMerge = linkEnter.merge(linkJoin);

    linkMerge
      .attr("stroke", (d) => {
        const srcId = typeof d.source === "object" ? d.source.id : d.source;
        const tgtId = typeof d.target === "object" ? d.target.id : d.target;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "#38BDF8";
        }
        return "#232B3E";
      })
      .attr("stroke-width", (d) => {
        const srcId = typeof d.source === "object" ? d.source.id : d.source;
        const tgtId = typeof d.target === "object" ? d.target.id : d.target;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return 3;
        }
        return 1.5;
      })
      .attr("stroke-dasharray", (d) => {
        const srcId = typeof d.source === "object" ? d.source.id : d.source;
        const tgtId = typeof d.target === "object" ? d.target.id : d.target;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "6, 4";
        }
        return "none";
      })
      .attr("marker-end", (d) => {
        const srcId = typeof d.source === "object" ? d.source.id : d.source;
        const tgtId = typeof d.target === "object" ? d.target.id : d.target;
        if (highlightedEdge && highlightedEdge.source === srcId && highlightedEdge.target === tgtId) {
          return "url(#arrow-active)";
        }
        return "url(#arrow-normal)";
      });

    // Nodes Join
    const nodeJoin = nodeGroup
      .selectAll<SVGGElement, GraphNode>("g.node-group")
      .data(nodesArray, (d) => d.id);

    nodeJoin.exit().remove();

    const nodeEnter = nodeJoin
      .enter()
      .append("g")
      .attr("class", "node-group cursor-pointer")
      .on("click", (event, d) => {
        event.stopPropagation();
        onSelectNode(d.id);
      })
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
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

    // Append child elements to enter selection
    nodeEnter.append("circle").attr("class", "pulse-ring");
    nodeEnter.append("circle").attr("class", "sla-glow");
    nodeEnter.append("circle").attr("class", "base-circle").attr("r", 22).attr("fill", "#131822");
    nodeEnter.append("circle").attr("class", "health-fill").attr("r", 8);
    nodeEnter
      .append("text")
      .attr("class", "name-label")
      .attr("y", 36)
      .attr("text-anchor", "middle")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("font-size", "11px")
      .attr("font-weight", "600");
    nodeEnter
      .append("text")
      .attr("class", "replica-label")
      .attr("y", 4)
      .attr("text-anchor", "middle")
      .attr("font-family", "IBM Plex Mono, monospace")
      .attr("font-size", "9px")
      .attr("font-weight", "700");
    nodeEnter.append("title");

    const nodeMerge = nodeEnter.merge(nodeJoin);

    // Update Pulse Ring
    nodeMerge
      .select<SVGCircleElement>("circle.pulse-ring")
      .attr("r", (d) => (d.id === pulseNodeId || d.id === selectedNodeId ? 34 : 0))
      .attr("fill", "none")
      .attr("stroke", (d) => {
        if (d.id === pulseNodeId && pulseIsVeto) return "#E5484D";
        if (d.id === pulseNodeId) return "#38BDF8";
        return "#3DDC97";
      })
      .attr("stroke-width", 3)
      .attr("opacity", (d) => (d.id === pulseNodeId || d.id === selectedNodeId ? 0.8 : 0))
      .attr("class", (d) =>
        d.id === pulseNodeId || d.id === selectedNodeId ? "pulse-ring animate-ping" : "pulse-ring"
      );

    // Update SLA Glow
    nodeMerge
      .select<SVGCircleElement>("circle.sla-glow")
      .attr("r", (d) => (d.sla_violating ? 28 : 0))
      .attr("fill", "none")
      .attr("stroke", "#E5484D")
      .attr("stroke-width", 2)
      .attr("stroke-dasharray", "4, 2")
      .attr("filter", (d) => (d.sla_violating ? "url(#glow)" : "none"));

    // Update Base Circle with color + accessibility stroke patterns (§1.5)
    nodeMerge
      .select<SVGCircleElement>("circle.base-circle")
      .attr("stroke", (d) => getNodeColor(d.health, d.isolated))
      .attr("stroke-dasharray", (d) => getNodeStrokeDash(d.health, d.isolated))
      .attr("stroke-width", (d) => (d.id === selectedNodeId ? 3.5 : 2))
      .attr("filter", (d) => (d.health < 0.85 || d.id === selectedNodeId ? "url(#glow)" : "none"));

    // Update Inner Health Fill
    nodeMerge
      .select<SVGCircleElement>("circle.health-fill")
      .attr("fill", (d) => getNodeColor(d.health, d.isolated));

    // Update Name Label
    nodeMerge
      .select<SVGTextElement>("text.name-label")
      .text((d) => d.id)
      .attr("fill", (d) => (d.id === selectedNodeId ? "#38BDF8" : "#F1F5F9"));

    // Update Replica Label
    nodeMerge
      .select<SVGTextElement>("text.replica-label")
      .text((d) => `${d.ready_replicas}/${d.replicas}`)
      .attr("fill", "#F1F5F9");

    // Update Accessible Title Tooltip (§1.5)
    nodeMerge
      .select<SVGTitleElement>("title")
      .text(
        (d) =>
          `${d.id} [${d.status.toUpperCase()}]\nHealth: ${(d.health * 100).toFixed(
            0
          )}%\nCPU: ${(d.cpu_pct * 100).toFixed(0)}%\nReplicas: ${d.ready_replicas}/${d.replicas}`
      );
  }, [cluster, selectedNodeId, pulseNodeId, pulseIsVeto, highlightedEdge, onSelectNode]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full bg-[#0B0E14] overflow-hidden rounded-xl border border-[#232B3E]"
    >
      {/* Empty / Disconnected Overlay */}
      {!cluster && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0B0E14]/90 z-20 font-mono text-[#7C89A3] gap-3">
          <WifiOff className="w-8 h-8 text-[#E5484D] animate-bounce" />
          <span className="text-sm font-semibold text-[#F1F5F9]">Disconnected from Telemetry</span>
          <span className="text-xs text-[#7C89A3]">Start a simulation or wait for connection...</span>
        </div>
      )}

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

      {/* Health Legend with Accessible Shapes */}
      <div className="absolute right-4 top-4 flex items-center gap-3 bg-[#131822]/90 backdrop-blur border border-[#232B3E] px-3 py-1.5 rounded-lg z-10 text-xs font-mono">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#3DDC97] shadow-[0_0_8px_#3DDC97]"></span>
          <span className="text-[#F1F5F9]">Healthy (&ge;85%)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#F5A623] border border-dashed border-[#F5A623]"></span>
          <span className="text-[#F1F5F9]">Degraded</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-[#E5484D] border border-dotted border-[#E5484D]"></span>
          <span className="text-[#F1F5F9]">Critical (&lt;40%)</span>
        </div>
      </div>

      {/* SVG Canvas */}
      <svg ref={svgRef} className="w-full h-full min-h-[550px] cursor-grab active:cursor-grabbing" />
    </div>
  );
};
