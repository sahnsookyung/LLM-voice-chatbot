import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Matrix4,Quaternion,Vector3} from 'three';

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
const source=await model('v6'),current=await model('v7'),{g,values}=current;
const oldPos=source.values(0),pos=values(0);

test('v7 is self-contained and preserves the humanoid, body fit, and garment cutouts',()=>{
  assert.equal(g.extensions.VRMC_vrm.meta.version,'0.7');
  assert.ok(current.bytes.length<24*1024*1024,'bounded texture budget');
  assert.ok(g.buffers.every(b=>!b.uri)&&g.images.every(i=>i.bufferView!==undefined&&!i.uri));
  assert.deepEqual(g.extensions.VRMC_vrm.humanoid,source.g.extensions.VRMC_vrm.humanoid);
  assert.deepEqual(g.extensions.VRMC_vrm.expressions,source.g.extensions.VRMC_vrm.expressions);
  for(const meshIndex of [1,3])for(const p of g.meshes[meshIndex].primitives){
    for(const a of Object.values(p.attributes))assert.deepEqual(values(a),source.values(a),'body, clothing and skin weights unchanged');
  }
  for(let i=0;i<source.g.images.length;i++)if(![3,5].includes(i))assert.ok(current.image(i).equals(source.image(i)),`original image ${i}, including alpha`);
  for(const i of [3,5])assert.ok(!current.image(i).equals(source.image(i)),'face and iris detail actually replaced');
  const coat=g.materials[13];
  assert.deepEqual(coat.pbrMetallicRoughness.baseColorTexture,source.g.materials[13].pbrMetallicRoughness.baseColorTexture);
  assert.equal(coat.extras.ariColorFromShade,true);
  const color=g.textures[coat.extensions.VRMC_materials_mtoon.shadeMultiplyTexture.index].source;
  assert.ok(color>=source.g.images.length && current.image(color).length>100000);
});

test('v7 lengthens the brows, refines the chin and exposes narrower ears',()=>{
  const vertices=m=>[...new Set(values(g.meshes[0].primitives.find(p=>p.material===m).indices).flat())];
  for(const side of [-1,1]){
    const brow=vertices(5).filter(i=>oldPos[i][0]*side>0);
    const extent=p=>Math.max(...brow.map(i=>Math.abs(p[i][0])));
    assert.ok(extent(pos)-extent(oldPos)>.008,'longer eyebrow tails');
    const ears=vertices(3).filter(i=>oldPos[i][0]*side>.09 && oldPos[i][1]<1.47);
    assert.ok(ears.length>30);
    for(const i of ears){assert.ok(Math.abs(pos[i][0])<Math.abs(oldPos[i][0]));assert.ok(pos[i][2]>oldPos[i][2]+.03,'ear turns forward');}
  }
  const chin=oldPos.map((p,i)=>[p,i]).filter(([p])=>p[1]<1.365&&p[2]>.045);
  assert.ok(chin.length>10);
  assert.ok(chin.every(([p,i])=>pos[i][1]<p[1]-.002),'lower contour is visibly refined');
  for(const i of vertices(1))assert.deepEqual(pos[i],oldPos[i],'iris placement preserved');
});

test('all 57 facial morph endpoints remain finite with normalized normals',()=>{
  const p=g.meshes[0].primitives[0],normal=values(p.attributes.NORMAL);
  assert.equal(p.targets.length,57);
  for(const n of normal)assert.ok(Math.abs(Math.hypot(...n)-1)<.001);
  for(const target of p.targets){
    const delta=values(target.POSITION),oldDelta=source.values(target.POSITION),dn=values(target.NORMAL);
    for(let i=0;i<pos.length;i++){
      assert.ok(delta[i].every(Number.isFinite));
      assert.ok(Math.hypot(...delta[i])<Math.hypot(...oldDelta[i])+.015,'bounded expression deformation');
      assert.ok(Math.abs(Math.hypot(...normal[i].map((v,k)=>v+dn[i][k]))-1)<.001,'normalized posed normals');
    }
  }
});

test('temple hair remains skinned in its new rest pose without moving the crown',()=>{
  const old=source.values(138),next=values(138);let moved=0;
  next.forEach((p,i)=>{if(old[i][1]>=1.527)assert.deepEqual(p,old[i]);if(Math.hypot(...p.map((v,k)=>v-old[i][k]))>.001)moved++;});
  assert.ok(moved>2000);
  const worlds=nodes=>{const parents=new Map(),cache=new Map();nodes.forEach((n,i)=>(n.children??[]).forEach(c=>parents.set(c,i)));
    const world=i=>{if(cache.has(i))return cache.get(i);const n=nodes[i],m=new Matrix4().compose(new Vector3(...n.translation??[0,0,0]),new Quaternion(...n.rotation??[0,0,0,1]),new Vector3(...n.scale??[1,1,1]));
      if(parents.has(i))m.premultiply(world(parents.get(i)));cache.set(i,m);return m;};return world;};
  const now=worlds(g.nodes),before=worlds(source.g.nodes);
  for(const skin of g.skins){const binds=values(skin.inverseBindMatrices),oldBinds=source.values(skin.inverseBindMatrices);
    skin.joints.forEach((node,i)=>{const a=now(node).clone().multiply(new Matrix4().fromArray(binds[i])),b=before(node).clone().multiply(new Matrix4().fromArray(oldBinds[i]));
      a.elements.forEach((v,k)=>assert.ok(Math.abs(v-b.elements[k])<.000002,`rest matrix ${node}`));});}
});

test('gold hoops follow the head and new geometry has valid normals, indices and skin weights',()=>{
  let triangles=0;
  for(const node of g.nodes.filter(n=>n.mesh!==undefined)){
    const skin=g.skins[node.skin];
    for(const p of g.meshes[node.mesh].primitives){
      const positions=values(p.attributes.POSITION),indices=values(p.indices).flat();triangles+=indices.length/3;
      assert.ok(indices.every(i=>i>=0&&i<positions.length));
      for(const n of values(p.attributes.NORMAL))assert.ok(Math.abs(Math.hypot(...n)-1)<.001);
      for(const w of values(p.attributes.WEIGHTS_0))assert.ok(w.every(v=>v>=0)&&Math.abs(w.reduce((a,b)=>a+b,0)-1)<.0001);
      for(const joints of values(p.attributes.JOINTS_0))assert.ok(joints.every(j=>skin.joints[j]!==undefined));
      if(g.meshes[node.mesh].name==='Ari small gold hoops'){
        for(const joints of values(p.attributes.JOINTS_0))assert.equal(skin.joints[joints[0]],g.extensions.VRMC_vrm.humanoid.humanBones.head.node);
        for(const point of positions)assert.ok(point[1]>1.394&&point[1]<1.411&&Math.abs(point[0])>.068&&Math.abs(point[0])<.079,'attached at the lower ear');
      }
    }
  }
  assert.ok(triangles<50000,'geometry remains modest for the desktop companion');
});
