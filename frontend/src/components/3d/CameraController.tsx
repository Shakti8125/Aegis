import * as THREE from "three";
import { SpatialNode3D } from "../../types";

export type CameraPreset = "isometric" | "top_down" | "front";

export class CameraControllerHelper {
  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;

  private isDragging: boolean = false;
  private isRightDragging: boolean = false;
  private previousMousePosition = { x: 0, y: 0 };

  // Spherical coordinates relative to target
  private radius: number = 18;
  private theta: number = Math.PI / 4; // Azimuthal angle
  private phi: number = Math.PI / 3.5; // Polar angle

  private target: THREE.Vector3 = new THREE.Vector3(0, 0, 0);
  private desiredTarget: THREE.Vector3 = new THREE.Vector3(0, 0, 0);
  private desiredCameraPos: THREE.Vector3 = new THREE.Vector3();

  private lerpSpeed: number = 0.08;

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera;
    this.domElement = domElement;
    this.updateCameraPosInstant();
    this.attachEvents();
  }

  private attachEvents() {
    this.domElement.addEventListener("mousedown", this.onMouseDown);
    window.addEventListener("mouseup", this.onMouseUp);
    window.addEventListener("mousemove", this.onMouseMove);
    this.domElement.addEventListener("wheel", this.onWheel, { passive: false });
  }

  public detachEvents() {
    this.domElement.removeEventListener("mousedown", this.onMouseDown);
    window.removeEventListener("mouseup", this.onMouseUp);
    window.removeEventListener("mousemove", this.onMouseMove);
    this.domElement.removeEventListener("wheel", this.onWheel);
  }

  private onMouseDown = (e: MouseEvent) => {
    if (e.button === 0) {
      this.isDragging = true;
    } else if (e.button === 2) {
      this.isRightDragging = true;
    }
    this.previousMousePosition = { x: e.clientX, y: e.clientY };
  };

  private onMouseUp = () => {
    this.isDragging = false;
    this.isRightDragging = false;
  };

  private onMouseMove = (e: MouseEvent) => {
    const deltaX = e.clientX - this.previousMousePosition.x;
    const deltaY = e.clientY - this.previousMousePosition.y;

    if (this.isDragging) {
      this.theta -= deltaX * 0.005;
      this.phi = Math.max(0.1, Math.min(Math.PI / 2.05, this.phi - deltaY * 0.005));
    } else if (this.isRightDragging) {
      const panSpeed = 0.015;
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();

      this.camera.matrix.extractBasis(right, up, new THREE.Vector3());

      this.desiredTarget.addScaledVector(right, -deltaX * panSpeed);
      this.desiredTarget.addScaledVector(up, deltaY * panSpeed);
    }

    this.previousMousePosition = { x: e.clientX, y: e.clientY };
  };

  private onWheel = (e: WheelEvent) => {
    e.preventDefault();
    this.radius = Math.max(5, Math.min(45, this.radius + e.deltaY * 0.015));
  };

  public flyToNode(node: SpatialNode3D) {
    const [x, y, z] = node.position;
    this.desiredTarget.set(x, y, z);
    this.radius = 10;
    this.phi = Math.PI / 4;
  }

  public setPreset(preset: CameraPreset) {
    this.desiredTarget.set(0, 0, 0);
    if (preset === "isometric") {
      this.radius = 18;
      this.theta = Math.PI / 4;
      this.phi = Math.PI / 3.5;
    } else if (preset === "top_down") {
      this.radius = 22;
      this.theta = 0;
      this.phi = 0.01;
    } else if (preset === "front") {
      this.radius = 18;
      this.theta = 0;
      this.phi = Math.PI / 2.1;
    }
  }

  private updateCameraPosInstant() {
    const x = this.target.x + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    const y = this.target.y + this.radius * Math.cos(this.phi);
    const z = this.target.z + this.radius * Math.sin(this.phi) * Math.cos(this.theta);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.target);
  }

  public update() {
    // Lerp Target
    this.target.lerp(this.desiredTarget, this.lerpSpeed);

    // Compute desired camera position
    const x = this.target.x + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    const y = this.target.y + this.radius * Math.cos(this.phi);
    const z = this.target.z + this.radius * Math.sin(this.phi) * Math.cos(this.theta);

    this.desiredCameraPos.set(x, y, z);
    this.camera.position.lerp(this.desiredCameraPos, this.lerpSpeed);
    this.camera.lookAt(this.target);
  }
}
