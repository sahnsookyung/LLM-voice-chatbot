import test from 'node:test';
import assert from 'node:assert/strict';
import {Object3D,Vector3,Quaternion} from 'three';
import {PlantedLeg,kneePosition} from '../.test-build/leg-ik.js';
import {MotionController} from '../.test-build/motion.js';
test('two-bone knee solve preserves limb lengths and bends forwards',()=>{
  const hip=new Vector3(.02,.9,0), foot=new Vector3(.05,.04,0);
  const knee=kneePosition(hip,foot,new Vector3(0,0,1),.44,.44);
  assert.ok(Math.abs(hip.distanceTo(knee)-.44)<1e-8);
  assert.ok(Math.abs(foot.distanceTo(knee)-.44)<1e-8);
  assert.ok(knee.z>0);
});
test('feet remain planted through weight shifts, steps, turning and gesture interruption',()=>{
  const root=new Object3D(),hips=new Object3D();root.add(hips);hips.position.y=.9;
  const legs=['left','right'].map((side,i)=>{
    const upper=new Object3D(),lower=new Object3D(),foot=new Object3D();hips.add(upper);upper.position.x=i?.07:-.07;upper.add(lower);lower.position.y=-.42;lower.add(foot);foot.position.y=-.42;
    return {foot,ik:new PlantedLeg(root,upper,lower,foot)};
  });
  const m=new MotionController(()=>.3);
  for(let i=0;i<1200;i++) {
    if(i%50===0)m.update({gesture:['shift','step','wave','explain'][Math.floor(i/50)%4]});
    const w=m.step(1/60);root.rotation.y=Math.sin(i*.01)*2;hips.position.set(w.hipX,.9-w.hipDrop,0);
    legs[0].ik.solve(w.leftFootLift,w.leftFootForward);legs[1].ik.solve();
    for(const {foot,ik} of legs){assert.ok(ik.error()<1e-5,`foot drift ${ik.error()}`);assert.ok(foot.getWorldQuaternion(new Quaternion()).angleTo(root.quaternion)<1e-5);}
  }
});
test('reduced motion settles both arms, hips and leg movement',()=>{
  const m=new MotionController(()=>.2);m.update({gesture:'step'});for(let i=0;i<40;i++)m.step(1/30);
  m.update({reducedMotion:true});let w;for(let i=0;i<100;i++)w=m.step(1/30);
  for(const key of ['leftArm','leftForearm','leftHand','hipX','hipDrop','bodyLean','leftFootLift','leftFootForward'])assert.ok(Math.abs(w[key])<1e-6,key);
});
test('leg solve follows changed ancestors without traversing unrelated avatar branches',()=>{
  const scene=new Object3D(),root=new Object3D(),hips=new Object3D(),upper=new Object3D(),lower=new Object3D(),foot=new Object3D();
  scene.add(root);root.add(hips);hips.position.y=.9;hips.add(upper);upper.add(lower);lower.position.y=-.42;lower.add(foot);foot.position.y=-.42;
  const unrelated=new Object3D();root.add(unrelated);
  const ik=new PlantedLeg(root,upper,lower,foot);
  let visits=0;
  unrelated.updateMatrixWorld=unrelated.updateWorldMatrix=()=>{visits++;};
  for(let i=0;i<120;i++){
    scene.position.set(.1+i*.001,.2,-.3);scene.rotation.y=i*.02;
    root.rotation.y=-i*.03;hips.position.x=Math.sin(i*.1)*.02;
    ik.solve(.02,.03);
    assert.ok(ik.error()<1e-5,'changed ancestor transforms must reach the foot target');
  }
  assert.equal(visits,0,'leg IK must not scale with hair/mesh tree size');
});
