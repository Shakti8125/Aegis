import * as THREE from "three";
import { SpatialNode3D, TrafficStream3D } from "../../types";

export function computeTrafficStreams(
  edges: Array<{
    source: string;
    target: string;
    p99_latency_ms: number | null;
    error_rate: number | null;
    traffic_share: number | null;
  }>,
  nodes: SpatialNode3D[],
  highlightedEdge: { source: string; target: string } | null
): TrafficStream3D[] {
  const nodeMap = new Map<string, SpatialNode3D>();
  nodes.forEach((n) => nodeMap.set(n.id, n));

  const streams: TrafficStream3D[] = [];

  edges.forEach((edge, idx) => {
    const srcNode = nodeMap.get(edge.source);
    const tgtNode = nodeMap.get(edge.target);

    if (srcNode && tgtNode) {
      const isHighlighted =
        highlightedEdge !== null &&
        ((highlightedEdge.source === edge.source && highlightedEdge.target === edge.target) ||
          (highlightedEdge.source === edge.target && highlightedEdge.target === edge.source));

      streams.push({
        id: `stream-${idx}-${edge.source}-${edge.target}`,
        sourceId: edge.source,
        targetId: edge.target,
        sourcePos: srcNode.position,
        targetPos: tgtNode.position,
        p99_latency_ms: edge.p99_latency_ms ?? 15,
        error_rate: edge.error_rate ?? 0,
        traffic_share: edge.traffic_share ?? 0.5,
        isHighlighted,
      });
    }
  });

  return streams;
}

export class TrafficStreamsRenderer {
  private group: THREE.Group;
  private lineGeometries: Map<string, THREE.BufferGeometry> = new Map();
  private lineMeshes: Map<string, THREE.Line> = new Map();
  private particleGroups: Map<string, THREE.Points> = new Map();

  constructor() {
    this.group = new THREE.Group();
    this.group.name = "TrafficStreamsGroup";
  }

  public getGroup(): THREE.Group {
    return this.group;
  }

  public updateStreams(streams: TrafficStream3D[], time: number) {
    const activeIds = new Set(streams.map((s) => s.id));

    // Cleanup obsolete streams
    this.lineMeshes.forEach((mesh, id) => {
      if (!activeIds.has(id)) {
        this.group.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
        this.lineMeshes.delete(id);

        const pts = this.particleGroups.get(id);
        if (pts) {
          this.group.remove(pts);
          pts.geometry.dispose();
          (pts.material as THREE.Material).dispose();
          this.particleGroups.delete(id);
        }
      }
    });

    // Create or update active streams
    streams.forEach((stream) => {
      let lineMesh = this.lineMeshes.get(stream.id);

      // Color computation based on latency and error rate
      let streamColor = "#06B6D4"; // Cyan
      if (stream.p99_latency_ms > 40) streamColor = "#F59E0B"; // Amber
      if (stream.p99_latency_ms > 80 || stream.error_rate > 0.05) streamColor = "#EF4444"; // Crimson Red
      if (stream.isHighlighted) streamColor = "#A855F7"; // Purple glow

      const vSource = new THREE.Vector3(...stream.sourcePos);
      const vTarget = new THREE.Vector3(...stream.targetPos);
      const midPoint = new THREE.Vector3()
        .addVectors(vSource, vTarget)
        .multiplyScalar(0.5);
      // Lift middle control point to form arc curve
      const distance = vSource.distanceTo(vTarget);
      midPoint.y += Math.max(0.6, distance * 0.25);

      const curve = new THREE.QuadraticBezierCurve3(vSource, midPoint, vTarget);
      const points = curve.getPoints(32);

      if (!lineMesh) {
        const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
        const lineMat = new THREE.LineBasicMaterial({
          color: new THREE.Color(streamColor),
          transparent: true,
          opacity: stream.isHighlighted ? 0.9 : 0.4,
          linewidth: stream.isHighlighted ? 3 : 1,
        });

        lineMesh = new THREE.Line(lineGeo, lineMat);
        this.group.add(lineMesh);
        this.lineMeshes.set(stream.id, lineMesh);

        // Flow Particles along Curve
        const pCount = Math.max(8, Math.floor(stream.traffic_share * 20));
        const pGeo = new THREE.BufferGeometry();
        const pPositions = new Float32Array(pCount * 3);

        for (let i = 0; i < pCount; i++) {
          const t = i / pCount;
          const pos = curve.getPoint(t);
          pPositions[i * 3] = pos.x;
          pPositions[i * 3 + 1] = pos.y;
          pPositions[i * 3 + 2] = pos.z;
        }

        pGeo.setAttribute("position", new THREE.BufferAttribute(pPositions, 3));

        const pMat = new THREE.PointsMaterial({
          color: new THREE.Color(streamColor),
          size: stream.isHighlighted ? 0.18 : 0.12,
          transparent: true,
          opacity: 0.9,
          depthWrite: false,
        });

        const particles = new THREE.Points(pGeo, pMat);
        particles.userData = { curve, pCount, speed: Math.max(0.2, 100 / Math.max(1, stream.p99_latency_ms)) };
        this.group.add(particles);
        this.particleGroups.set(stream.id, particles);
      } else {
        // Update line material & color
        const mat = lineMesh.material as THREE.LineBasicMaterial;
        mat.color.set(streamColor);
        mat.opacity = stream.isHighlighted ? 0.95 : 0.4;

        // Animate particles along Bezier curve
        const particles = this.particleGroups.get(stream.id);
        if (particles) {
          const pMat = particles.material as THREE.PointsMaterial;
          pMat.color.set(streamColor);
          pMat.size = stream.isHighlighted ? 0.22 : 0.13;

          const pGeo = particles.geometry as THREE.BufferGeometry;
          const pPositions = pGeo.attributes.position.array as Float32Array;
          const pCount = particles.userData.pCount;
          const speed = Math.max(0.15, 60 / Math.max(1, stream.p99_latency_ms));

          for (let i = 0; i < pCount; i++) {
            let t = (i / pCount + time * speed * 0.2) % 1.0;
            if (t < 0) t += 1.0;
            const pos = curve.getPoint(t);
            pPositions[i * 3] = pos.x;
            pPositions[i * 3 + 1] = pos.y;
            pPositions[i * 3 + 2] = pos.z;
          }
          pGeo.attributes.position.needsUpdate = true;
        }
      }
    });
  }

  public dispose() {
    this.lineMeshes.forEach((mesh) => {
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
    });
    this.particleGroups.forEach((pts) => {
      pts.geometry.dispose();
      (pts.material as THREE.Material).dispose();
    });
    this.group.clear();
  }
}
