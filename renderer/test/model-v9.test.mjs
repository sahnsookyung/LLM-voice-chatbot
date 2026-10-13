import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
async function load(version){const b=await readFile(new URL(`../../characters/ari/model/ari-vroid-${version}.vrm`,import.meta.url)),size=b.readUInt32LE(12);
  const g=JSON.parse(b.subarray(20,20+size)),bin=b.subarray(28+size);
  const view=i=>{const v=g.bufferViews[i];return bin.subarray(v.byteOffset??0,(v.byteOffset??0)+v.byteLength);};
  const values=i=>{const a=g.accessors[i],b=view(a.bufferView),stride=g.bufferViews[a.bufferView].byteStride??12;
    return Array.from({length:a.count},(_,j)=>[0,1,2].map(k=>b.readFloatLE((a.byteOffset??0)+j*stride+k*4)));};
  return {g,view,values};}
const before=await load('v8'),after=await load('v9'),{g}=after;
test('v9 exposes ears through a local hair tuck, preserving ear size, face, body, textures and skinning',()=>{
  assert.deepEqual(g.extensions.VRMC_vrm.humanoid,before.g.extensions.VRMC_vrm.humanoid);
  assert.deepEqual(g.extensions.VRMC_vrm.expressions,before.g.extensions.VRMC_vrm.expressions);
  assert.deepEqual(g.nodes,before.g.nodes);assert.deepEqual(g.skins,before.g.skins);
  const attrs=g.meshes[2].primitives[0].attributes,changed=[attrs.POSITION,attrs.NORMAL].map(i=>g.accessors[i].bufferView);
  for(let i=0;i<g.bufferViews.length;i++)if(!changed.includes(i))assert.ok(after.view(i).equals(before.view(i)),`unchanged buffer ${i}`);
  const old=before.values(attrs.POSITION),oldNormals=before.values(attrs.NORMAL),positions=after.values(attrs.POSITION),normals=after.values(attrs.NORMAL);let count=0,largest=0;
  positions.forEach((p,i)=>{const q=old[i],displacement=q[2]-p[2];assert.ok(Math.abs(q[0])-Math.abs(p[0])>=-1e-8&&Math.abs(q[0])-Math.abs(p[0])<=.0091);assert.equal(q[1],p[1]);
    assert.ok(displacement>=0&&displacement<=.0161);assert.ok(Math.abs(Math.hypot(...normals[i])-1)<.00001);
    if(q[1]>=1.465){assert.deepEqual(p,q,'preserve upper temple coverage');assert.deepEqual(normals[i],oldNormals[i]);}
    if(displacement>1e-7){count++;largest=Math.max(largest,displacement);assert.ok(Math.abs(q[0])>.058&&Math.abs(q[0])<.116&&q[1]>1.369&&q[1]<1.465&&q[2]>-.067&&q[2]<.083);}
  });assert.ok(count>4000&&count<6000);assert.ok(largest>.014);
});
test('fixed-step tuning preserves world-space inertia and fitted collision groups',()=>{
  assert.equal(g.asset.extras.ariPhysicsHz,60);
  const springs=g.extensions.VRMC_springBone.springs;
  springs.forEach((spring,i)=>{const old=before.g.extensions.VRMC_springBone.springs[i];assert.deepEqual(spring.colliderGroups,old.colliderGroups);
    if(spring.name==='Hair')assert.equal(spring.center,undefined);
    spring.joints.forEach((joint,j)=>{if(joint.stiffness===undefined)return;assert.ok(joint.stiffness>0&&joint.stiffness<=old.joints[j].stiffness*.5+1e-9);
      if(joint.gravityPower!==undefined)assert.equal(joint.gravityPower,old.joints[j].gravityPower*.5);
      const drag=old.joints[j].dragForce;
      assert.ok(joint.dragForce>=0&&joint.dragForce<=drag);
      if(drag>0&&drag<1)assert.ok(joint.dragForce<drag);
      else assert.equal(joint.dragForce,drag,'zero/full damping endpoints stay fixed');});
  });
});
