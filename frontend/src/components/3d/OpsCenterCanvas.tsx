import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import {
  ClusterSnapshot,
  PartitionPlaneState,
  SpatialNode3D,
  HealthStatus,
  ActionEvent,
} from "../../types";
import { TierPlanesRenderer } from "./TierPlanes";
import { SpatialNodesRenderer, computeSpatialPositions } from "./SpatialNodes";
import { TrafficStreamsRenderer, computeTrafficStreams } from "./TrafficStreams";
import { CameraControllerHelper, CameraPreset } from "./CameraController";

interface OpsCenterCanvasProps {
  cluster: ClusterSnapshot | null;
  actions: ActionEvent[];
  selectedNodeId: string | null;
  highlightedEdge: { source: string; target: string } | null;
  viewMode: "3d" | "2d";
  partitionState: PartitionPlaneState;
  onSelectNode: (id: string | null) => void;
}

export const OpsCenterCanvas: React.FC<OpsCenterCanvasProps> = ({
  cluster,
  actions,
  selectedNodeId,
  highlightedEdge,
  viewMode,
  partitionState,
  onSelectNode,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const canvas2dRef = useRef<HTMLCanvasElement>(null);

  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [tooltipData, setTooltipData] = useState<{
    node: SpatialNode3D;
    x: number;
    y: number;
  } | null>(null);
  const [webglError, setWebglError] = useState(false);

  // References for Three.js instance
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const animFrameId = useRef<number | null>(null);

  const tierPlanesRendererRef = useRef<TierPlanesRenderer | null>(null);
  const spatialNodesRendererRef = useRef<SpatialNodesRenderer | null>(null);
  const trafficStreamsRendererRef = useRef<TrafficStreamsRenderer | null>(null);
  const cameraControllerRef = useRef<CameraControllerHelper | null>(null);
  const partitionPlaneMeshRef = useRef<THREE.Mesh | null>(null);

  // Raycaster for 3D node picking
  const raycasterRef = useRef(new THREE.Raycaster());
  const mouseRef = useRef(new THREE.Vector2());

  // ------------------------------------------------------------------
  // 1. WebGL Three.js Scene Setup & Loop
  // ------------------------------------------------------------------
  useEffect(() => {
    if (viewMode !== "3d" || !mountRef.current) return;

    try {
      const width = mountRef.current.clientWidth;
      const height = mountRef.current.clientHeight;

      // Create Scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color("#0B0E14");
      scene.fog = new THREE.FogExp2("#0B0E14", 0.018);
      sceneRef.current = scene;

      // Perspective Camera
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      cameraRef.current = camera;

      // Renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      rendererRef.current = renderer;

      // Clear previous children
      while (mountRef.current.firstChild) {
        mountRef.current.removeChild(mountRef.current.firstChild);
      }
      mountRef.current.appendChild(renderer.domElement);

      // Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);

      const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
      dirLight1.position.set(15, 25, 15);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0xa855f7, 0.8);
      dirLight2.position.set(-15, -10, -15);
      scene.add(dirLight2);

      // Instantiate Sub-Renderers
      const tierPlanes = new TierPlanesRenderer();
      scene.add(tierPlanes.getGroup());
      tierPlanesRendererRef.current = tierPlanes;

      const spatialNodes = new SpatialNodesRenderer();
      scene.add(spatialNodes.getGroup());
      spatialNodesRendererRef.current = spatialNodes;

      const trafficStreams = new TrafficStreamsRenderer();
      scene.add(trafficStreams.getGroup());
      trafficStreamsRendererRef.current = trafficStreams;

      // Camera Controller
      const camController = new CameraControllerHelper(camera, renderer.domElement);
      cameraControllerRef.current = camController;

      // Partition Slicer Plane Mesh
      const sliceGeo = new THREE.PlaneGeometry(24, 24);
      const sliceMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color("#EF4444"),
        transparent: true,
        opacity: 0.25,
        side: THREE.DoubleSide,
      });
      const sliceMesh = new THREE.Mesh(sliceGeo, sliceMat);
      sliceMesh.rotation.x = Math.PI / 2;
      sliceMesh.visible = false;
      scene.add(sliceMesh);
      partitionPlaneMeshRef.current = sliceMesh;

      // Resize Handler
      const handleResize = () => {
        if (!mountRef.current || !renderer || !camera) return;
        const w = mountRef.current.clientWidth;
        const h = mountRef.current.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener("resize", handleResize);

      // Animation Loop
      let startTime = performance.now();
      const animate = () => {
        const time = (performance.now() - startTime) * 0.001;

        tierPlanes.update(time);
        camController.update();

        renderer.render(scene, camera);
        animFrameId.current = requestAnimationFrame(animate);
      };
      animate();

      return () => {
        window.removeEventListener("resize", handleResize);
        if (animFrameId.current) cancelAnimationFrame(animFrameId.current);
        camController.detachEvents();
        tierPlanes.dispose();
        spatialNodes.dispose();
        trafficStreams.dispose();
        renderer.dispose();
      };
    } catch (err) {
      console.error("WebGL 3D Context Init failed, falling back to 2D Canvas:", err);
      setWebglError(true);
    }
  }, [viewMode]);

  // ------------------------------------------------------------------
  // 2. Update 3D Objects on Cluster / Data Prop Changes
  // ------------------------------------------------------------------
  useEffect(() => {
    if (viewMode !== "3d" || webglError) return;

    const spatialNodesData = computeSpatialPositions(cluster);
    const time = performance.now() * 0.001;

    // Update Nodes
    if (spatialNodesRendererRef.current) {
      spatialNodesRendererRef.current.updateNodes(
        spatialNodesData,
        selectedNodeId,
        hoveredNodeId,
        time
      );
    }

    // Update Traffic Streams
    if (trafficStreamsRendererRef.current) {
      const edgesData = cluster?.edges || [];
      const streams = computeTrafficStreams(edgesData, spatialNodesData, highlightedEdge);
      trafficStreamsRendererRef.current.updateStreams(streams, time);
    }

    // Update Partition Plane Slicer
    if (partitionPlaneMeshRef.current) {
      partitionPlaneMeshRef.current.visible = partitionState.active;
      if (partitionState.active) {
        partitionPlaneMeshRef.current.position.y = partitionState.positionY;
        partitionPlaneMeshRef.current.rotation.z = (partitionState.angleDegrees * Math.PI) / 180;
      }
    }
  }, [cluster, selectedNodeId, hoveredNodeId, highlightedEdge, partitionState, viewMode, webglError]);

  // ------------------------------------------------------------------
  // 3. Raycasting Mouse Interactivity (Hover & Click Node)
  // ------------------------------------------------------------------
  const handleMouseMove3D = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!mountRef.current || !cameraRef.current || !spatialNodesRendererRef.current) return;

    const rect = mountRef.current.getBoundingClientRect();
    mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    raycasterRef.current.setFromCamera(mouseRef.current, cameraRef.current);
    const group = spatialNodesRendererRef.current.getGroup();
    const intersects = raycasterRef.current.intersectObjects(group.children, true);

    if (intersects.length > 0) {
      let object: THREE.Object3D | null = intersects[0].object;
      while (object && !object.userData?.nodeId && object.parent) {
        object = object.parent;
      }

      if (object && object.userData?.nodeData) {
        const node: SpatialNode3D = object.userData.nodeData;
        setHoveredNodeId(node.id);
        setTooltipData({
          node,
          x: e.clientX - rect.left + 15,
          y: e.clientY - rect.top + 15,
        });
        return;
      }
    }

    setHoveredNodeId(null);
    setTooltipData(null);
  };

  const handleClick3D = (e: React.MouseEvent<HTMLDivElement>) => {
    if (hoveredNodeId) {
      onSelectNode(hoveredNodeId);

      // Smooth camera fly to node
      const spatialNodesData = computeSpatialPositions(cluster);
      const targetNode = spatialNodesData.find((n) => n.id === hoveredNodeId);
      if (targetNode && cameraControllerRef.current) {
        cameraControllerRef.current.flyToNode(targetNode);
      }
    } else {
      onSelectNode(null);
    }
  };

  // ------------------------------------------------------------------
  // 4. Fallback 2D Canvas Renderer (when viewMode === "2d" or WebGL error)
  // ------------------------------------------------------------------
  useEffect(() => {
    if (viewMode !== "2d" && !webglError) return;
    const canvas = canvas2dRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    const render2D = () => {
      const width = (canvas.width = canvas.parentElement?.clientWidth || 800);
      const height = (canvas.height = canvas.parentElement?.clientHeight || 600);

      ctx.fillStyle = "#0B0E14";
      ctx.fillRect(0, 0, width, height);

      // Draw Grid
      ctx.strokeStyle = "rgba(51, 65, 85, 0.3)";
      ctx.lineWidth = 1;
      const gridSize = 40;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const services = cluster?.services || [];
      const edges = cluster?.edges || [];
      const nodePosMap = new Map<string, { x: number; y: number }>();

      // Position 2D hierarchy
      const cx = width / 2;
      const cy = height / 2;
      const radius = Math.min(width, height) * 0.35;

      services.forEach((svc, idx) => {
        const angle = (idx / services.length) * Math.PI * 2;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        nodePosMap.set(svc.id, { x, y });
      });

      // Draw Edges
      edges.forEach((edge) => {
        const p1 = nodePosMap.get(edge.source);
        const p2 = nodePosMap.get(edge.target);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle =
            edge.p99_latency_ms && edge.p99_latency_ms > 40
              ? "rgba(245, 158, 11, 0.6)"
              : "rgba(6, 182, 212, 0.4)";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });

      // Draw Nodes
      services.forEach((svc) => {
        const pos = nodePosMap.get(svc.id);
        if (!pos) return;

        const isSelected = selectedNodeId === svc.id;
        let color = "#10B981";
        if (svc.status === "degraded") color = "#F59E0B";
        if (svc.status === "critical") color = "#EF4444";

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, isSelected ? 18 : 12, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.strokeStyle = isSelected ? "#38BDF8" : "rgba(255, 255, 255, 0.3)";
        ctx.lineWidth = isSelected ? 3 : 1;
        ctx.stroke();

        ctx.font = "Bold 11px sans-serif";
        ctx.fillStyle = "#F8FAFC";
        ctx.textAlign = "center";
        ctx.fillText(svc.name, pos.x, pos.y + 26);
      });

      animId = requestAnimationFrame(render2D);
    };

    render2D();

    return () => cancelAnimationFrame(animId);
  }, [cluster, selectedNodeId, viewMode, webglError]);

  return (
    <div className="relative w-full h-full bg-[#0B0E14] overflow-hidden select-none">
      {/* 3D WebGL Container */}
      {viewMode === "3d" && !webglError ? (
        <div
          ref={mountRef}
          onMouseMove={handleMouseMove3D}
          onClick={handleClick3D}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        />
      ) : (
        /* 2D Fallback Canvas */
        <canvas ref={canvas2dRef} className="w-full h-full block" />
      )}

      {/* Node Metadata Tooltip Overlay */}
      {tooltipData && (
        <div
          style={{ left: tooltipData.x, top: tooltipData.y }}
          className="absolute z-50 pointer-events-none bg-slate-950/90 backdrop-blur-md border border-cyan-500/50 p-3 rounded-xl shadow-2xl text-xs space-y-1 text-slate-100 min-w-44"
        >
          <div className="font-bold text-cyan-300 font-mono flex items-center justify-between">
            <span>{tooltipData.node.name}</span>
            <span
              className={`text-[9px] px-1.5 py-0.5 rounded uppercase ${
                tooltipData.node.status === "healthy"
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-800"
                  : tooltipData.node.status === "degraded"
                  ? "bg-amber-950 text-amber-300 border border-amber-800"
                  : "bg-rose-950 text-rose-300 border border-rose-800"
              }`}
            >
              {tooltipData.node.status}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] font-mono pt-1 text-slate-300">
            <div>
              CPU: <span className="text-amber-400 font-bold">{Math.round(tooltipData.node.cpu_pct * 100)}%</span>
            </div>
            <div>
              MEM: <span className="text-purple-400 font-bold">{Math.round(tooltipData.node.mem_pct * 100)}%</span>
            </div>
            <div>
              Latency: <span className="text-cyan-400 font-bold">{tooltipData.node.p99_latency_ms.toFixed(1)}ms</span>
            </div>
            <div>
              Error: <span className="text-rose-400 font-bold">{(tooltipData.node.error_rate * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
