import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Matrix3, Matrix4, Quaternion, Vector3, BufferGeometry, Float32BufferAttribute, Mesh, MeshBasicMaterial, DoubleSide, Raycaster} from '../renderer/node_modules/three/build/three.module.js';

// Offline art finish: keeps the original VRoid export and v5 fit reproducible.
// Bone rest positions and inverse binds follow the same warp as the hair mesh.
const root = new URL('../characters/ari/model/', import.meta.url);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
async function load(name, expected) {
  const raw = await readFile(new URL(name, root));
  assert.equal(hash(raw), expected, `Reinspect changed source ${name} before editing this profile`);
  const length = raw.readUInt32LE(12);
  return {json:JSON.parse(raw.toString('utf8',20,20+length)), bin:Buffer.from(raw.subarray(28+length))};
}
const source = await load('ari-vroid-v6-raw.vrm','01199d6e3e043c5c78dcc090f620f096a4955a89c283ca15d110ca7bf36700d7');
const fitted = await load('ari-vroid-v5.vrm','e4f5e4a0f9c08fd2a6679c924d12b60391ff1606fbaa041035669390c447e08e');
const gltf = source.json;
let bin = source.bin;
const widths = {SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
function accessor(index, model=source) {
  const a=model.json.accessors[index],v=model.json.bufferViews[a.bufferView];
  const width=widths[a.type], bytes=a.componentType===5123?2:4;
  const read={5123:'readUInt16LE',5125:'readUInt32LE',5126:'readFloatLE'}[a.componentType];
  assert.ok(read && width && !a.sparse);
  const data=model===source?bin:model.bin;
  const offset=i=>(v.byteOffset??0)+(a.byteOffset??0)+i*(v.byteStride??width*bytes);
  return {a,get:i=>Array.from({length:width},(_,k)=>data[read](offset(i)+k*bytes)),
    set:(i,values)=>{assert.equal(a.componentType,5126);values.forEach((x,k)=>data.writeFloatLE(x,offset(i)+k*bytes));}};
}
function bounds(a) {
  const points=Array.from({length:a.a.count},(_,i)=>a.get(i));
  a.a.min=[0,1,2].map(k=>Math.min(...points.map(p=>p[k])));
  a.a.max=[0,1,2].map(k=>Math.max(...points.map(p=>p[k])));
}
assert.deepEqual(gltf.meshes,fitted.json.meshes);
assert.deepEqual(gltf.nodes,fitted.json.nodes);
for (const name of ['POSITION','NORMAL']) {
  const index=gltf.meshes[1].primitives[0].attributes[name];
  const to=accessor(index),from=accessor(index,fitted);
  for(let i=0;i<to.a.count;i++)to.set(i,from.get(i));
  if(name==='POSITION')bounds(to);
}

const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
function hairWarp([x,y,z]) {
  // Roots/fringe remain untouched. Ease into length below the jaw, then push
  // front and back locks away from the torso while retaining the authored waves.
  const lower=Math.max(0,1.405-y);
  const length=.72*lower*smooth(lower/.12);
  const depth=smooth(lower/.17);
  const front=smooth((z+.035)/.08);
  return [x*(1+.04*depth),y-length,z+depth*(-.065+.145*front)];
}
const hair=gltf.meshes[2].primitives[0];
const hp=accessor(hair.attributes.POSITION),hn=accessor(hair.attributes.NORMAL);
let changedHairVertices=0;
for(let i=0;i<hp.a.count;i++) {
  const p=hp.get(i),next=hairWarp(p);
  if(Math.hypot(...p.map((v,k)=>v-next[k]))<1e-9)continue;
  const columns=p.map((_,axis)=>{
    const a=p.slice(),b=p.slice();a[axis]-=1e-5;b[axis]+=1e-5;
    const lo=hairWarp(a),hi=hairWarp(b);return hi.map((v,k)=>(v-lo[k])/2e-5);
  });
  const jacobian=new Matrix3().set(...[0,1,2].flatMap(k=>columns.map(c=>c[k])));
  assert.ok(jacobian.determinant()>0);
  hn.set(i,new Vector3(...hn.get(i)).applyMatrix3(jacobian.invert().transpose()).normalize().toArray());
  hp.set(i,next);changedHairVertices++;
}
bounds(hp);
const parents=new Map();
gltf.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>parents.set(c,i)));
function worldMatrices() {
  const cache=new Map();
  function world(i) {
    if(cache.has(i))return cache.get(i);
    const n=gltf.nodes[i];
    const m=n.matrix?new Matrix4().fromArray(n.matrix):new Matrix4().compose(
      new Vector3(...n.translation??[0,0,0]),new Quaternion(...n.rotation??[0,0,0,1]),new Vector3(...n.scale??[1,1,1]));
    if(parents.has(i))m.premultiply(world(parents.get(i)));
    cache.set(i,m);return m;
  }
  gltf.nodes.forEach((_,i)=>world(i));return cache;
}
const before=worldMatrices(), desired=new Map(before);
const hairNodes=new Set(gltf.extensions.VRMC_springBone.springs.filter(s=>s.name==='Hair').flatMap(s=>s.joints.map(j=>j.node)));
for(const i of hairNodes) {
  const m=before.get(i).clone();m.setPosition(new Vector3(...hairWarp(new Vector3().setFromMatrixPosition(m).toArray())));
  desired.set(i,m);
}
for(const i of hairNodes) {
  const local=desired.get(i).clone().premultiply(desired.get(parents.get(i)).clone().invert());
  gltf.nodes[i].translation=new Vector3().setFromMatrixPosition(local).toArray();
}
const after=worldMatrices();
for(const skin of gltf.skins) {
  const binds=accessor(skin.inverseBindMatrices);
  skin.joints.forEach((node,i)=>{
    if(!hairNodes.has(node))return;
    // Preserve any bind-shape correction carried by the original export.
    const correction=before.get(node).clone().multiply(new Matrix4().fromArray(binds.get(i)));
    binds.set(i,after.get(node).clone().invert().multiply(correction).toArray());
  });
}
for(const spring of gltf.extensions.VRMC_springBone.springs.filter(s=>s.name==='Hair')) {
  for(const joint of spring.joints)if(joint.dragForce!==undefined) {
    joint.dragForce=.55; // Longer locks settle instead of oscillating after a turn.
    joint.stiffness=Math.max(joint.stiffness,.65);
  }
}
// Reduce the stock broad silver highlight without changing the hair texture UVs.
gltf.materials[14].emissiveFactor=gltf.materials[14].emissiveFactor.map(v=>v*.45);

function append(data,type,componentType=5126,target=34962) {
  const bytes=componentType===5123?2:4,values=data.flat(),buffer=Buffer.alloc(values.length*bytes);
  values.forEach((v,i)=>componentType===5123?buffer.writeUInt16LE(v,i*bytes):buffer.writeFloatLE(v,i*bytes));
  const offset=Math.ceil(bin.length/4)*4;
  bin=Buffer.concat([bin,Buffer.alloc(offset-bin.length),buffer]);
  const view=gltf.bufferViews.push({buffer:0,byteOffset:offset,byteLength:buffer.length,target})-1;
  const index=gltf.accessors.push({bufferView:view,componentType,count:data.length,type})-1;
  if(type==='VEC3')bounds(accessor(index));
  return index;
}
// A fitted mock-neck overlaps the crew-neck's alpha-cut texture edge.
// Its base inherits shirt weights and its top follows the neck.
const body=gltf.meshes[1],shirt=body.primitives.find(p=>p.material===11);
const bp=accessor(shirt.attributes.POSITION),bj=accessor(shirt.attributes.JOINTS_0),bw=accessor(shirt.attributes.WEIGHTS_0);
const ring=[8373,8372,8375,8377,8376,8466,8467,8472,8473,8475,8477,8487,8493,8494,8496,8515,8508,8506,8500,8498,8497,8589,8588,8586];
assert.equal(ring.length,24);
const position=[],uv=[],joints=[],weights=[],indices=[];
const slices=96,rows=6,neckJoint=gltf.skins[1].joints.indexOf(17);
const skinPrimitive=body.primitives.find(p=>p.material===8),skinIndices=accessor(skinPrimitive.indices);
const skinGeometry=new BufferGeometry();
skinGeometry.setAttribute('position',new Float32BufferAttribute(Array.from({length:bp.a.count},(_,i)=>bp.get(i)).flat(),3));
skinGeometry.setIndex(Array.from({length:skinIndices.a.count},(_,i)=>skinIndices.get(i)[0]));
const skinSurface=new Mesh(skinGeometry,new MeshBasicMaterial({side:DoubleSide}));skinSurface.updateMatrixWorld();
const clearanceRay=new Raycaster();
for(let row=0;row<=rows;row++)for(let s=0;s<=slices;s++) {
  const segment=(s%96)/4,index=Math.floor(segment),f=segment-index;
  const a=bp.get(ring[index]),b=bp.get(ring[(index+1)%24]);
  const base=a.map((v,k)=>v+(b[k]-v)*f),t=row/rows;
  base[0]*=1.12;base[1]-=.012;base[2]=-.027+(base[2]+.027)*1.12;
  const angle=Math.atan2(base[0],base[2]+.023);
  const top=[Math.sin(angle)*.045,1.342,-.027+Math.cos(angle)*.050];
  const point=new Vector3(...base.map((v,k)=>v+(top[k]-v)*t));
  const origin=new Vector3(0,point.y,-.027),radial=point.clone().sub(origin),radius=radial.length();
  clearanceRay.set(origin,radial.normalize());
  const hit=clearanceRay.intersectObject(skinSurface)[0];
  if(hit && hit.distance+.002>radius)point.copy(origin).addScaledVector(radial,hit.distance+.002);
  position.push(point.toArray());
  uv.push([.28+s/slices*.4,.36-t*.09]);
  const influences=new Map([[neckJoint,t]]);
  for(const [vertex,scale] of [[ring[index],(1-f)*(1-t)],[ring[(index+1)%24],f*(1-t)]]) {
    const js=bj.get(vertex),ws=bw.get(vertex);js.forEach((j,k)=>influences.set(j,(influences.get(j)??0)+ws[k]*scale));
  }
  const inf=[...influences].filter(([,w])=>w>1e-8).sort((a,b)=>b[1]-a[1]).slice(0,4);
  const total=inf.reduce((sum,[,w])=>sum+w,0);while(inf.length<4)inf.push([0,0]);
  joints.push(inf.map(([j])=>j));weights.push(inf.map(([,w])=>w/total));
}
skinGeometry.dispose();skinSurface.material.dispose();
for(let r=0;r<rows;r++)for(let s=0;s<slices;s++) {
  const a=r*(slices+1)+s,b=a+1,c=a+slices+1,d=c+1;
  indices.push([a,c,b],[b,c,d]);
}
const geometry=new BufferGeometry();
geometry.setAttribute('position',new Float32BufferAttribute(position.flat(),3));geometry.setIndex(indices.flat());geometry.computeVertexNormals();
// Verify outward winding once, independently of the clockwise boundary list.
const pa=new Vector3(...position[indices[0][0]]),pb=new Vector3(...position[indices[0][1]]),pc=new Vector3(...position[indices[0][2]]);
if(pb.sub(pa).cross(pc.sub(pa)).dot(new Vector3(pa.x,0,pa.z+.027))<0){
  indices.forEach(t=>[t[1],t[2]]=[t[2],t[1]]);geometry.setIndex(indices.flat());geometry.computeVertexNormals();
}
const normals=Array.from({length:position.length},(_,i)=>[0,1,2].map(k=>geometry.attributes.normal.array[i*3+k]));
const attributes={POSITION:append(position,'VEC3'),NORMAL:append(normals,'VEC3'),TEXCOORD_0:append(uv,'VEC2'),
  JOINTS_0:append(joints,'VEC4',5123),WEIGHTS_0:append(weights,'VEC4')};
const mesh=gltf.meshes.push({name:'Ari fitted mock-neck',primitives:[{attributes,indices:append(indices.flat().map(i=>[i]),'SCALAR',5123,34963),material:11,mode:4}]})-1;
const node=gltf.nodes.push({name:'Ari fitted mock-neck',mesh,skin:1})-1;gltf.scenes[0].nodes.push(node);geometry.dispose();
gltf.asset.generator+=' + Ari v6 hair rest-pose and knit neckline finish';
gltf.buffers[0].byteLength=bin.length;
bin=Buffer.concat([bin,Buffer.alloc((4-bin.length%4)%4)]);
const bytes=Buffer.from(JSON.stringify(gltf)),json=Buffer.alloc(Math.ceil(bytes.length/4)*4,0x20);bytes.copy(json);
const output=Buffer.alloc(28+json.length+bin.length);
output.write('glTF');output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);
output.writeUInt32LE(json.length,12);output.writeUInt32LE(0x4e4f534a,16);json.copy(output,20);
output.writeUInt32LE(bin.length,20+json.length);output.writeUInt32LE(0x004e4942,24+json.length);bin.copy(output,28+json.length);
await writeFile(new URL('ari-vroid-v6.vrm',root),output);
console.log(JSON.stringify({sha256:hash(output),bytes:output.length,changedHairVertices,hairBounds:{min:hp.a.min,max:hp.a.max},hairNodes:hairNodes.size,collarTriangles:indices.length},null,2));
