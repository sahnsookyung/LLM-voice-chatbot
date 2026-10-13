import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Mesh,Vector3,Box3} from 'three';
import {AccessoryRig,parseAccessories,compatibleAccessories} from '../.test-build/accessories.js';

test('accessory choices validate independently',()=>{
  for(const glasses of ['none','round'])for(const headwear of ['none','beret'])assert.deepEqual(parseAccessories({glasses,headwear}),{glasses,headwear});
  for(const value of [null,{}, {glasses:'round',headwear:'unknown'},{glasses:'unknown',headwear:'none'}])assert.throws(()=>parseAccessories(value));
  assert.deepEqual(compatibleAccessories({glasses:'round',headwear:'beret'},'double-buns'),{glasses:'round',headwear:'none'});
  assert.deepEqual(compatibleAccessories({glasses:'round',headwear:'beret'},'bob'),{glasses:'round',headwear:'beret'});
});

test('beret opening sits below the scalp apex and follows the head without drift',()=>{
  const head=new Group();head.position.set(0,1.43,-.02);head.updateMatrixWorld(true);
  const rig=new AccessoryRig({humanoid:{getRawBoneNode:()=>head}});rig.set({glasses:'round',headwear:'beret'});head.updateMatrixWorld(true);
  const root=head.children[0],cap=root.children.find(n=>n instanceof Mesh&&n.geometry.attributes.position.count>6000);
  assert.ok(cap,'shaped cloth crown present');
  const box=new Box3().setFromObject(cap);
  assert.ok(box.min.y<1.51&&box.max.y>1.63&&box.max.y<1.68,'tilted opening wraps the hair beneath the fuller reference crown');
  const p=new Vector3().fromBufferAttribute(cap.geometry.attributes.position,0),before=cap.localToWorld(p.clone());
  head.rotation.set(.12,.25,-.08);head.updateMatrixWorld(true);
  const after=cap.localToWorld(p.clone()),local=head.worldToLocal(after.clone());
  assert.ok(after.distanceTo(before)>.005);assert.ok(local.distanceTo(p.clone().applyMatrix4(root.matrix))<1e-6,'head-local attachment is stable');
  rig.dispose();assert.equal(head.children.length,0);
});

test('repeated accessory swaps release every retired geometry and material',()=>{
  const head=new Group();head.updateMatrixWorld(true);const rig=new AccessoryRig({humanoid:{getRawBoneNode:()=>head}});
  for(let i=0;i<8;i++){
    rig.set({glasses:'round',headwear:'beret'});
    let resources=0,released=0;head.traverse(n=>{if(n instanceof Mesh)for(const resource of [n.geometry,n.material]){resources++;resource.addEventListener('dispose',()=>released++);}});
    assert.ok(resources>8);rig.set({glasses:'none',headwear:'none'});assert.equal(released,resources);assert.equal(head.children[0].children.length,0);
  }
  rig.dispose();
});
