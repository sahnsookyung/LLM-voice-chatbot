import test from 'node:test';
import assert from 'node:assert/strict';
import {BufferGeometry,Float32BufferAttribute,Group,Mesh,MeshStandardMaterial} from 'three';
import {GarmentCorrectives,garmentDelta} from '../.test-build/garment-correctives.js';

test('garment corrections follow each arm independently and leave neutral mesh data unchanged',()=>{
  const geometry=new BufferGeometry();geometry.setAttribute('position',new Float32BufferAttribute([-.14,1.2,.09,.14,1.2,.09,-.13,1,.08,.13,1,.08],3));geometry.setIndex([0,2,1,1,2,3]);geometry.computeVertexNormals();
  const material=new MeshStandardMaterial();material.name='Tops_01';
  const a=new Mesh(geometry,material),b=new Mesh(geometry,material),root=new Group();root.add(a,b);
  const original=geometry.attributes.position.array.slice(),normal=geometry.attributes.normal.array.slice();
  const rig=new GarmentCorrectives(root);assert.equal(rig.meshes,2);
  assert.equal(geometry.morphAttributes.position.length,2,'shared geometry receives only two targets');
  for(const attribute of [...geometry.morphAttributes.position,...geometry.morphAttributes.normal])assert.ok([...attribute.array].every(Number.isFinite));
  rig.update({leftArm:2,upperArm:0});assert.deepEqual(a.morphTargetInfluences,[1,0]);assert.deepEqual(b.morphTargetInfluences,[1,0]);
  assert.equal(geometry.morphAttributes.position[0].getY(0),0,'left correction does not lift the opposite sleeve');
  assert.ok(geometry.morphAttributes.position[0].getY(1)>.01,'lift gathers the underarm');
  rig.update({leftArm:0,upperArm:2});assert.deepEqual(a.morphTargetInfluences,[0,1]);
  rig.update({});assert.deepEqual(a.morphTargetInfluences,[0,0]);
  assert.deepEqual(geometry.attributes.position.array,original);assert.deepEqual(geometry.attributes.normal.array,normal);
  geometry.dispose();material.dispose();
});

test('pose folds are bounded and do not move hands or lower legs',()=>{
  for(const side of [-1,1]){
    for(const p of [[.54,1.28,-.025],[.08,.45,0]])assert.ok(garmentDelta(...p,side).every(v=>v===0));
    for(let x=-.3;x<.3;x+=.025)for(let y=.7;y<1.4;y+=.025){
      const a=garmentDelta(x,y,.1,side);
      assert.ok(a.every(v=>Number.isFinite(v)&&Math.abs(v)<.03));
    }
  }
});
