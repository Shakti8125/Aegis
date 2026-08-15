import * as THREE from "three";
import { ClusterSnapshot, HealthStatus, NodeSnapshot, ServiceSnapshot, SpatialNode3D } from "../../types";

export function computeSpatialPositions(cluster: ClusterSnapshot | null): SpatialNode3D[] {
  if (!cluster) return [];

  const spatialNodes: SpatialNode3D[] = [];

  // 1. Services
  const services = cluster.services || [];
  const nServices = services.length;
  const radius = Math.max(4.5, nServices * 0.45);

  services.forEach((svc, idx) => {
    let yLevel = 0.0;
    let tierLvl = 0;
    let angle = (idx / nServices) * Math.PI * 2;
    let r = radius;

    if (svc.tier === "front" || svc.id.includes("gateway")) {
      yLevel = 3.5;
      tierLvl = 1;
      r = radius * 0.4;
      angle = (idx / Math.max(1, services.filter((s) => s.tier === "front").length)) * Math.PI * 2;
    } else if (svc.tier === "back" || svc.id.includes("data")) {
      yLevel = 0.0;
      tierLvl = 0;
      r = radius * 1.1;
    }

    const x = Math.cos(angle) * r;
    const z = Math.sin(angle) * r;

    // Collect active faults targeting this service
    const activeFaults = (cluster.active_faults || [])
      .filter((f) => f.target === svc.id || f.target === svc.name)
      .map((f) => f.fault_type);

    spatialNodes.push({
      id: svc.id,
      name: svc.name,
      type: svc.tier === "front" ? "ingress" : "service",
      position: [x, yLevel, z],
      health: svc.health,
      status: svc.status,
      cpu_pct: svc.cpu_pct,
      mem_pct: svc.mem_pct,
      p99_latency_ms: svc.p99_latency_ms,
      error_rate: svc.error_rate,
      tierLevel: tierLvl,
      isolated: svc.isolated,
      sla_violating: svc.sla_violating,
      activeFaults,
    });
  });

  // 2. Infrastructure Nodes
  const nodes = cluster.nodes || [];
  const nNodes = nodes.length;
  const nodeRadius = Math.max(5.0, nNodes * 0.8);

  nodes.forEach((nd, idx) => {
    const angle = (idx / nNodes) * Math.PI * 2 + Math.PI / 6;
    const x = Math.cos(angle) * nodeRadius;
    const z = Math.sin(angle) * nodeRadius;

    const nodeHealth: HealthStatus = nd.health > 0.8 ? "healthy" : nd.health > 0.5 ? "degraded" : "critical";

    spatialNodes.push({
      id: nd.id,
      name: nd.name,
      type: "node",
      position: [x, -3.5, z],
      health: nd.health,
      status: nodeHealth,
      cpu_pct: nd.cpu_pct,
      mem_pct: nd.mem_pct,
      p99_latency_ms: 0,
      error_rate: 0,
      tierLevel: -1,
      isolated: false,
      sla_violating: false,
      activeFaults: [],
    });
  });

  return spatialNodes;
}

export class SpatialNodesRenderer {
  private group: THREE.Group;
  private nodeMeshes: Map<string, THREE.Mesh> = new Map();
  private glowRings: Map<string, THREE.Mesh> = new Map();
  private labelSprites: Map<string, THREE.Sprite> = new Map();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = "SpatialNodesGroup";
  }

  public getGroup(): THREE.Group {
    return this.group;
  }

  public getMeshByNodeId(id: string): THREE.Mesh | undefined {
    return this.nodeMeshes.get(id);
  }

  public updateNodes(nodes: SpatialNode3D[], selectedId: string | null, hoveredId: string | null, time: number) {
    const currentIds = new Set(nodes.map((n) => n.id));

    // Remove obsolete nodes
    this.nodeMeshes.forEach((mesh, id) => {
      if (!currentIds.has(id)) {
        this.group.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
        this.nodeMeshes.delete(id);

        const ring = this.glowRings.get(id);
        if (ring) {
          this.group.remove(ring);
          ring.geometry.dispose();
          (ring.material as THREE.Material).dispose();
          this.glowRings.delete(id);
        }

        const sprite = this.labelSprites.get(id);
        if (sprite) {
          this.group.remove(sprite);
          sprite.material.dispose();
          this.labelSprites.delete(id);
        }
      }
    });

    // Create or update current nodes
    nodes.forEach((node) => {
      let mesh = this.nodeMeshes.get(node.id);
      const isSelected = selectedId === node.id;
      const isHovered = hoveredId === node.id;

      // Color mapping
      let colorHex = "#10B981"; // Emerald
      if (node.status === "degraded") colorHex = "#F59E0B"; // Amber
      if (node.status === "critical") colorHex = "#EF4444"; // Crimson Red
      if (node.isolated) colorHex = "#64748B"; // Slate grey
      if (node.sla_violating) colorHex = "#DC2626"; // Bright Red

      if (!mesh) {
        // Geometry based on type
        let geo: THREE.BufferGeometry;
        if (node.type === "node") {
          geo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
        } else if (node.type === "ingress") {
          geo = new THREE.OctahedronGeometry(0.65, 1);
        } else {
          geo = new THREE.SphereGeometry(0.55, 24, 24);
        }

        const mat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(colorHex),
          roughness: 0.2,
          metalness: 0.5,
          emissive: new THREE.Color(colorHex).multiplyScalar(0.3),
          emissiveIntensity: 0.6,
        });

        mesh = new THREE.Mesh(geo, mat);
        mesh.userData = { nodeId: node.id, nodeData: node };
        mesh.position.set(...node.position);
        this.group.add(mesh);
        this.nodeMeshes.set(node.id, mesh);

        // Pulsing / Selection Glow Ring
        const ringGeo = new THREE.RingGeometry(0.7, 0.85, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(colorHex),
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.5,
        });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        ringMesh.rotation.x = Math.PI / 2;
        ringMesh.position.set(...node.position);
        ringMesh.position.y -= 0.05;
        this.group.add(ringMesh);
        this.glowRings.set(node.id, ringMesh);

        // Label Sprite
        const sprite = this.createLabelSprite(node.name);
        sprite.position.set(node.position[0], node.position[1] + 0.85, node.position[2]);
        this.group.add(sprite);
        this.labelSprites.set(node.id, sprite);
      } else {
        // Update position & colors
        mesh.position.set(...node.position);
        mesh.userData.nodeData = node;

        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.color.set(colorHex);

        const isFaulty = node.activeFaults.length > 0 || node.status === "critical";
        const pulse = Math.sin(time * 5 + node.health * 10) * 0.3 + 0.7;

        mat.emissive.set(colorHex).multiplyScalar(isFaulty ? pulse * 0.8 : 0.3);

        const scale = isSelected ? 1.35 : isHovered ? 1.15 : 1.0;
        mesh.scale.set(scale, scale, scale);

        // Update Ring
        const ring = this.glowRings.get(node.id);
        if (ring) {
          ring.position.set(node.position[0], node.position[1] - 0.05, node.position[2]);
          const ringMat = ring.material as THREE.MeshBasicMaterial;
          ringMat.color.set(colorHex);
          ring.scale.set(scale, scale, scale);
          ringMat.opacity = isSelected ? 0.9 : isFaulty ? pulse * 0.8 : 0.4;
          ring.rotation.z = time * 0.5;
        }

        // Update Sprite position
        const sprite = this.labelSprites.get(node.id);
        if (sprite) {
          sprite.position.set(node.position[0], node.position[1] + 0.85 * scale, node.position[2]);
        }
      }
    });
  }

  private createLabelSprite(text: string): THREE.Sprite {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
      ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
      ctx.lineWidth = 2;
      ctx.roundRect(4, 4, 248, 56, 8);
      ctx.fill();
      ctx.stroke();

      ctx.font = "Bold 22px Inter, sans-serif";
      ctx.fillStyle = "#F8FAFC";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(text, 128, 32);
    }

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.0, 0.5, 1.0);
    return sprite;
  }

  public dispose() {
    this.nodeMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.glowRings.forEach((ring) => {
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
    });
    this.labelSprites.forEach((sprite) => {
      sprite.material.dispose();
    });
    this.group.clear();
  }
}
