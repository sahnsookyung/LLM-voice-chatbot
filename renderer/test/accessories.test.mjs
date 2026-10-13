import test from 'node:test';
import assert from 'node:assert/strict';
import {Group,Mesh,Vector3,Box3} from 'three';
import {AccessoryRig,parseAccessories,compatibleAccessories} from '../.test-build/accessories.js';
import {outfits} from '../.test-build/wardrobe.js';
import {outfitPalettes} from '../.test-build/wardrobe-palette.js';

test('every outfit gives the beret its dominant fabric and a coordinated secondary ribbon',()=>{
  const crowns=new Set();
  for(const outfit of outfits){
    const head=new Group();head.updateMatrixWorld(true);
    const rig=new AccessoryRig({humanoid:{getRawBoneNode:()=>head}},outfit);
    for(let repeat=0;repeat<2;repeat++){
      rig.set({glasses:'round',headwear:'beret'});
      const color=name=>head.getObjectByName(`Ari beret ${name}`).material.color.toArray();
      const palette=outfitPalettes[outfit],main=palette[palette.dominant];
      assert.deepEqual(color('crown'),main,`${outfit} crown uses linear garment color`);
      assert.deepEqual(color('ribbon'),palette.accent,`${outfit} ribbon uses the secondary fabric`);
      assert.notDeepEqual(color('band'),main,'band remains visibly distinct');
      for(const value of Object.values(rig.palette()).flat())assert.ok(Number.isFinite(value)&&value>=0&&value<=1);
      crowns.add(JSON.stringify(color('crown')));
      rig.set({glasses:'round',headwear:'none'});
      assert.equal(head.getObjectByName('Ari beret crown'),undefined);
    }
    rig.dispose();assert.equal(head.children.length,0);
  }
  assert.ok(crowns.size>=12,'catalog no longer uses one fixed beret color');
});

test('outfit changes recolor the beret without tinting glasses or changing hat geometry',()=>{
  const signatures=[];
  for(const outfit of ['pink-sailor','forest-pinafore']){
    const head=new Group();head.updateMatrixWorld(true);
    const rig=new AccessoryRig({humanoid:{getRawBoneNode:()=>head}},outfit);
    rig.set({glasses:'round',headwear:'none'});
    const glasses=[];head.traverse(n=>{if(n instanceof Mesh)glasses.push(n.material.color.toArray());});
    rig.set({glasses:'round',headwear:'beret'});
    const cap=head.getObjectByName('Ari beret crown');
    signatures.push({glasses,vertices:Array.from(cap.geometry.attributes.position.array),crown:cap.material.color.toArray()});
    rig.dispose();
  }
  assert.deepEqual(signatures[0].glasses,signatures[1].glasses);
  assert.deepEqual(signatures[0].vertices,signatures[1].vertices);
  assert.notDeepEqual(signatures[0].crown,signatures[1].crown);
});

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
  assert.ok(box.min.y<1.51&&box.max.y>1.61&&box.max.y<1.635,'lower crown still clears the scalp and wraps the hairline');
  assert.ok(Math.abs(box.min.x+box.max.x)<.006,'crown is centered over the head');
  assert.ok(box.max.x-box.min.x<.23,'overhang does not flare into a mushroom cap');
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
