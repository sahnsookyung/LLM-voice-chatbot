import {Object3D, Quaternion, Vector3} from 'three';
const epsilon = 1e-6;
/** Two-bone solve in world space. Pole selects the knee's forward bend plane. */
export function kneePosition(hip: Vector3, target: Vector3, pole: Vector3, upper: number, lower: number, out = new Vector3()) {
  const axis = target.clone().sub(hip);
  const distance = Math.max(epsilon, Math.min(axis.length(), upper + lower - epsilon));
  axis.normalize();
  const bend = pole.clone().addScaledVector(axis, -pole.dot(axis)).normalize();
  const along = (upper * upper - lower * lower + distance * distance) / (2 * distance);
  const height = Math.sqrt(Math.max(0, upper * upper - along * along));
  return out.copy(hip).addScaledVector(axis, along).addScaledVector(bend, height);
}

export class PlantedLeg {
  private rest: Quaternion[];
  private target = new Vector3();
  private ankle = new Vector3();
  private hip = new Vector3(); private knee = new Vector3(); private current = new Vector3();
  private desired = new Vector3(); private pole = new Vector3();
  private rootRotation = new Quaternion(); private rotation = new Quaternion(); private parentRotation = new Quaternion();
  private footRotation: Quaternion;
  private upperLength: number; private lowerLength: number;
  constructor(private root: Object3D, private upper: Object3D, private lower: Object3D, private foot: Object3D) {
    root.updateMatrixWorld(true);
    this.rest = [upper,lower,foot].map(bone => bone.quaternion.clone());
    upper.getWorldPosition(this.hip); lower.getWorldPosition(this.knee); foot.getWorldPosition(this.ankle);
    this.upperLength = this.hip.distanceTo(this.knee); this.lowerLength = this.knee.distanceTo(this.ankle);
    root.worldToLocal(this.ankle);
    this.footRotation = root.getWorldQuaternion(new Quaternion()).invert().multiply(foot.getWorldQuaternion(new Quaternion()));
  }
  private aim(bone: Object3D, child: Object3D, destination: Vector3) {
    bone.getWorldPosition(this.current);
    child.getWorldPosition(this.desired).sub(this.current).normalize();
    this.current.subVectors(destination,this.current).normalize();
    this.rotation.setFromUnitVectors(this.desired,this.current).multiply(bone.getWorldQuaternion(this.parentRotation));
    bone.parent!.getWorldQuaternion(this.parentRotation).invert();
    bone.quaternion.copy(this.parentRotation.multiply(this.rotation));
    bone.updateWorldMatrix(false,true);
  }
  solve(lift = 0, forward = 0) {
    [this.upper,this.lower,this.foot].forEach((bone,i) => bone.quaternion.copy(this.rest[i]));
    // Only this leg and its ancestors changed. Walking the whole avatar here
    // repeats face, hair and mesh work twice per simulation step.
    this.upper.updateWorldMatrix(true,true);
    this.target.copy(this.ankle); this.target.y += lift; this.target.z += forward;
    this.root.localToWorld(this.target);
    this.upper.getWorldPosition(this.hip);
    this.root.getWorldQuaternion(this.rootRotation);
    this.pole.set(0,0,1).applyQuaternion(this.rootRotation);
    kneePosition(this.hip,this.target,this.pole,this.upperLength,this.lowerLength,this.knee);
    this.aim(this.upper,this.lower,this.knee);
    this.aim(this.lower,this.foot,this.target);
    this.foot.parent!.getWorldQuaternion(this.parentRotation).invert();
    this.foot.quaternion.copy(this.parentRotation.multiply(this.rootRotation).multiply(this.footRotation));
    this.foot.updateWorldMatrix(false,true);
  }
  error() { return this.foot.getWorldPosition(this.current).distanceTo(this.target); }
}
