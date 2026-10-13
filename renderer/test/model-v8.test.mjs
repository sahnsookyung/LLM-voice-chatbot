import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
async function model(version) {
  const bytes=await readFile(new URL(`../../characters/ari/model/ari-vroid-${version}.vrm`,import.meta.url));
  const size=bytes.readUInt32LE(12),g=JSON.parse(bytes.toString('utf8',20,20+size)),bin=bytes.subarray(28+size);
  const view=i=>{const v=g.bufferViews[i];return bin.subarray(v.byteOffset??0,(v.byteOffset??0)+v.byteLength);};
  const values=i=>{const a=g.accessors[i],v=g.bufferViews[a.bufferView],data=view(a.bufferView);
    const width={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16}[a.type];
    const [bytes,read]={5123:[2,'readUInt16LE'],5125:[4,'readUInt32LE'],5126:[4,'readFloatLE']}[a.componentType];
    return Array.from({length:a.count},(_,n)=>Array.from({length:width},(_,k)=>data[read]((a.byteOffset??0)+n*(v.byteStride??width*bytes)+k*bytes)));};
  return {g,bytes,view,values,image:i=>view(g.images[i].bufferView)};
}
const source=await model('v7'),current=await model('v8'),{g,values}=current;

test('v8 shrinks only ear regions while preserving expressions, body and garment alpha',()=>{
  assert.equal(g.extensions.VRMC_vrm.meta.version,'0.8');
  assert.deepEqual(g.extensions.VRMC_vrm.humanoid,source.g.extensions.VRMC_vrm.humanoid);
  assert.deepEqual(g.extensions.VRMC_vrm.expressions,source.g.extensions.VRMC_vrm.expressions);
  assert.ok(current.bytes.length<28*1024*1024);
  const old=source.values(0),next=values(0),moved=[];
  for(let i=0;i<old.length;i++)if(old[i].some((v,k)=>v!==next[i][k]))moved.push(i);
  assert.equal(moved.length,540);
  for(const i of moved){
    assert.ok(Math.abs(old[i][0])>.055&&old[i][1]>1.39&&old[i][1]<1.47);
    assert.ok(Math.abs(next[i][0])<=Math.abs(old[i][0])+1e-7);
    assert.ok(Math.abs(next[i][1]-1.436)<=Math.abs(old[i][1]-1.436)+1e-7);
  }
  for(const side of [-1,1]){
    const outer=moved.filter(i=>old[i][0]*side>.082);
    assert.ok(outer.length>10);
    assert.ok(outer.every(i=>Math.abs(next[i][0])<Math.abs(old[i][0])-.003));
  }
  const body=g.meshes[1],coatVertices=new Set(values(body.primitives.find(p=>p.material===13).indices).flat());
  const bodyPos=values(body.primitives[0].attributes.POSITION),oldBody=source.values(body.primitives[0].attributes.POSITION);
  bodyPos.forEach((p,i)=>{
    if(!coatVertices.has(i))assert.deepEqual(p,oldBody[i]);
    else {assert.equal(p[0],oldBody[i][0]);assert.equal(p[1],oldBody[i][1]);assert.ok(p[2]<=oldBody[i][2]+1e-7&&p[2]>=oldBody[i][2]-.0301);}
  });
  for(const mesh of [1,3])for(const p of g.meshes[mesh].primitives)for(const [name,a] of Object.entries(p.attributes))if(mesh!==1||!['POSITION','NORMAL'].includes(name))assert.deepEqual(values(a),source.values(a));
  for(let i=0;i<source.g.images.length;i++)if(![3,19].includes(i))assert.ok(current.image(i).equals(source.image(i)),`preserved image ${i}`);
  for(const target of g.meshes[0].primitives[0].targets){
    const delta=values(target.POSITION),norm=values(1),dn=values(target.NORMAL);
    for(let i=0;i<next.length;i++){
      assert.ok(delta[i].every(Number.isFinite));
      assert.ok(Math.abs(Math.hypot(...norm[i].map((v,k)=>v+dn[i][k]))-1)<.001);
    }
  }
});

test('finer hair has valid outward geometry, tiny tapered tips, and normalized skinning',()=>{
  const primitive=g.meshes[2].primitives[0],a=primitive.attributes;
  const pos=values(a.POSITION),norm=values(a.NORMAL),weights=values(a.WEIGHTS_0),joints=values(a.JOINTS_0),indices=values(primitive.indices).flat();
  const skin=g.skins[g.nodes.find(n=>n.mesh===2).skin];
  assert.ok(pos.length>60000&&pos.length<70000);
  assert.ok(indices.length/3<115000);
  for(let i=0;i<pos.length;i++){
    assert.ok(pos[i].every(Number.isFinite));
    assert.ok(Math.abs(Math.hypot(...norm[i])-1)<.001);
    assert.ok(Math.abs(weights[i].reduce((sum,w)=>sum+w,0)-1)<.00001);
    assert.ok(weights[i].every(w=>w>=0));assert.ok(joints[i].every(j=>skin.joints[j]!==undefined));
  }
  // A new disconnected strip starts whenever the index range increases beyond
  // the previous connected run. Its final seven-vertex ring is a fine tip.
  const starts=[];let max=-1;
  for(let i=0;i<indices.length;i+=3){const tri=indices.slice(i,i+3);assert.ok(tri.every(v=>v>=0&&v<pos.length));
    if(Math.min(...tri)>max)starts.push(Math.min(...tri));max=Math.max(max,...tri);
    const [a,b,c]=tri.map(v=>pos[v]),u=b.map((v,k)=>v-a[k]),v=c.map((v,k)=>v-a[k]);
    const cross=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]];
    const average=[0,1,2].map(k=>tri.reduce((sum,i)=>sum+norm[i][k],0));
    assert.ok(cross.reduce((sum,v,k)=>sum+v*average[k],0)>-1e-9,'outward triangle winding');
  }
  assert.equal(starts.length,176);
  for(let i=0;i<starts.length;i++){
    const end=starts[i+1]??pos.length,ring=pos.slice(end-7,end),center=[0,1,2].map(k=>ring.slice(0,6).reduce((s,p)=>s+p[k],0)/6);
    assert.ok(ring.every(p=>Math.hypot(...p.map((v,k)=>v-center[k]))<.00008),'fine non-bulbous tips');
  }
});

test('all hair chains use fitted body capsules with stable spring settings',()=>{
  const s=g.extensions.VRMC_springBone,group=s.colliderGroups.findIndex(g=>g.name==='Ari fitted clothing contact');
  assert.ok(group>=0);assert.equal(s.colliderGroups[group].colliders.length,7);
  for(const index of s.colliderGroups[group].colliders){const c=s.colliders[index];assert.ok(g.nodes[c.node]);assert.ok(c.shape.capsule.radius>=.07&&c.shape.capsule.radius<.16);}
  const springs=s.springs.filter(s=>s.name==='Hair');assert.equal(springs.length,10);
  for(const spring of springs){assert.ok(spring.colliderGroups.includes(group));assert.equal(spring.center,undefined,'whole-body turns retain hair inertia');
    for(const joint of spring.joints){assert.ok(g.nodes[joint.node]);if(joint.stiffness===undefined)continue;
      assert.ok(joint.stiffness>=.44&&joint.stiffness<=.86);assert.ok(joint.dragForce>=.23&&joint.dragForce<=.35);
    }
  }
  const hair=g.materials[14];assert.equal(hair.extras.ariHairContact,true);
  assert.equal(hair.emissiveTexture,undefined);assert.equal(hair.normalTexture,undefined);
  for(const texture of [3,19]){
    assert.ok(!current.image(g.textures[texture].source).equals(source.image(source.g.textures[texture].source)));
    assert.equal(g.samplers[g.textures[texture].sampler].minFilter,9987,'fine detail uses mipmap filtering');
  }
});
