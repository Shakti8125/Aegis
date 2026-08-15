import * as THREE from "three";
import { TierPlane3D } from "../../types";

export const DEFAULT_TIER_PLANES: TierPlane3D[] = [
  {
    id: "ingress-tier",
    name: "INGRESS & GATEWAYS TIER",
    yLevel: 3.5,
    color: "#3B82F6", // Blue
    gridSize: 14,
    nodeCount: 1,
  },
  {
    id: "microservices-tier",
    name: "MICROSERVICES MESH TIER",
    yLevel: 0.0,
    color: "#10B981", // Emerald
    gridSize: 18,
    nodeCount: 12,
  },
  {
    id: "physical-tier",
    name: "PHYSICAL K8S NODES TIER",
    yLevel: -3.5,
    color: "#8B5CF6", // Purple
    gridSize: 22,
    nodeCount: 6,
  },
];

export class TierPlanesRenderer {
  private group: THREE.Group;
  private planes: THREE.Mesh[] = [];
  private gridHelpers: THREE.GridHelper[] = [];

  constructor() {
    this.group = new THREE.Group();
    this.group.name = "TierPlanesGroup";
    this.initPlanes();
  }

  public getGroup(): THREE.Group {
    return this.group;
  }

  private initPlanes() {
    DEFAULT_TIER_PLANES.forEach((tier) => {
      // Translucent Grid Plane
      const planeGeo = new THREE.PlaneGeometry(tier.gridSize, tier.gridSize);
      const planeMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(tier.color),
        transparent: true,
        opacity: 0.05,
        side: THREE.DoubleSide,
        depthWrite: false,
      });

      const planeMesh = new THREE.Mesh(planeGeo, planeMat);
      planeMesh.rotation.x = Math.PI / 2;
      planeMesh.position.y = tier.yLevel;
      this.group.add(planeMesh);
      this.planes.push(planeMesh);

      // Grid Lines Helper
      const gridHelper = new THREE.GridHelper(
        tier.gridSize,
        14,
        new THREE.Color(tier.color).multiplyScalar(1.2),
        new THREE.Color(tier.color).multiplyScalar(0.4)
      );
      gridHelper.position.y = tier.yLevel;
      // Fade out grid lines slightly
      const materials = Array.isArray(gridHelper.material)
        ? gridHelper.material
        : [gridHelper.material];
      materials.forEach((mat) => {
        mat.transparent = true;
        mat.opacity = 0.25;
      });

      this.group.add(gridHelper);
      this.gridHelpers.push(gridHelper);

      // Wireframe Edge Outline
      const edges = new THREE.EdgesGeometry(planeGeo);
      const lineMat = new THREE.LineBasicMaterial({
        color: new THREE.Color(tier.color),
        transparent: true,
        opacity: 0.5,
      });
      const wireframe = new THREE.LineSegments(edges, lineMat);
      wireframe.rotation.x = Math.PI / 2;
      wireframe.position.y = tier.yLevel;
      this.group.add(wireframe);
    });
  }

  public update(time: number) {
    // Subtle pulse opacity on grid planes for dynamic visual effect
    this.planes.forEach((plane, idx) => {
      const mat = plane.material as THREE.MeshBasicMaterial;
      const baseOpacity = 0.04 + 0.02 * Math.sin(time * 1.5 + idx);
      mat.opacity = baseOpacity;
    });
  }

  public dispose() {
    this.planes.forEach((p) => {
      p.geometry.dispose();
      (p.material as THREE.Material).dispose();
    });
    this.gridHelpers.forEach((g) => g.dispose());
    this.group.clear();
  }
}
