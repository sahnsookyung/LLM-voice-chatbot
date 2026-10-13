import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Matrix3,Matrix4,Quaternion,Vector3,TorusGeometry,SphereGeometry} from '../renderer/node_modules/three/build/three.module.js';

// A versioned offline art pass. The untouched v6 is the rollback/source asset.
const root=new URL('../characters/ari/',import.meta.url);
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const raw=await readFile(new URL('model/ari-vroid-v6.vrm',root));
assert.equal(hash(raw),'4a80172fdfd9307a0bfebe9f750f1b003f86f198155faf8af5f0fe8c5e108045');
const size=raw.readUInt32LE(12),gltf=JSON.parse(raw.toString('utf8',20,20+size)),bin=raw.subarray(28+size);
const views=gltf.bufferViews.map(v=>Buffer.from(bin.subarray(v.byteOffset??0,(v.byteOffset??0)+v.byteLength)));
const widths={SCALAR:1,VEC2:2,VEC3:3,VEC4:4,MAT4:16};
function accessor(index){
  const a=gltf.accessors[index],v=gltf.bufferViews[a.bufferView],data=views[a.bufferView],width=widths[a.type];
  const [bytes,read,write]={5123:[2,'readUInt16LE','writeUInt16LE'],5125:[4,'readUInt32LE','writeUInt32LE'],5126:[4,'readFloatLE','writeFloatLE']}[a.componentType];
  assert.ok(!a.sparse && width);
  const offset=i=>(a.byteOffset??0)+i*(v.byteStride??bytes*width);
  return {a,get:i=>Array.from({length:width},(_,k)=>data[read](offset(i)+k*bytes)),
    set:(i,p)=>p.forEach((value,k)=>{assert.ok(Number.isFinite(value));data[write](value,offset(i)+k*bytes);})};
}
function all(index){const a=accessor(index);return Array.from({length:a.a.count},(_,i)=>a.get(i));}
function bounds(index){const a=accessor(index),p=all(index);a.a.min=[0,1,2].map(k=>Math.min(...p.map(v=>v[k])));a.a.max=[0,1,2].map(k=>Math.max(...p.map(v=>v[k])));}
function append(data,type,componentType=5126,target=34962){
  const bytes=componentType===5123?2:4,buffer=Buffer.alloc(data.length*widths[type]*bytes);
  data.flat().forEach((v,i)=>componentType===5123?buffer.writeUInt16LE(v,i*bytes):buffer.writeFloatLE(v,i*bytes));
  const bufferView=gltf.bufferViews.push({buffer:0,byteLength:buffer.length,target})-1;views.push(buffer);
  const index=gltf.accessors.push({bufferView,componentType,count:data.length,type})-1;
  if(type==='VEC3')bounds(index);return index;
}
const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
const bell=(v,c,r)=>Math.abs(v-c)>=r?0:Math.pow(1-Math.pow((v-c)/r,2),2);
const face=gltf.meshes[0],materialVertices=material=>new Set(all(face.primitives.find(p=>p.material===material).indices).flat());
const brows=materialVertices(5),skin=materialVertices(3);
// Ears are four disconnected front/back UV islands, separate from the head.
const adjacency=new Map([...skin].map(i=>[i,new Set()]));
const skinIndices=all(face.primitives.find(p=>p.material===3).indices).flat();
for(let i=0;i<skinIndices.length;i+=3){const triangle=skinIndices.slice(i,i+3);for(const a of triangle)for(const b of triangle)adjacency.get(a).add(b);}
const remaining=new Set(skin),components=[];
while(remaining.size){const queue=[remaining.values().next().value],group=[];remaining.delete(queue[0]);
  while(queue.length){const i=queue.pop();group.push(i);for(const j of adjacency.get(i))if(remaining.delete(j))queue.push(j);}components.push(group);}
const originalFace=all(face.primitives[0].attributes.POSITION);
const earVertices=new Set(components.filter(group=>group.length<200&&group.every(i=>{const [x,y]=originalFace[i];return Math.abs(x)>.055&&y>1.39&&y<1.47;})).flat());
assert.equal(earVertices.size,540);
function earWarp([x,y,z]){
  const sign=Math.sign(x),w=smooth((Math.abs(x)-.059)/.017),angle=-.42*w;
  const dx=Math.abs(x)-.059,dz=z+.027;
  return [sign*(.059+(Math.cos(angle)*dx+Math.sin(angle)*dz)*.68),
    y+(y-1.432)*.025*w,-.027-Math.sin(angle)*dx+Math.cos(angle)*dz+.033*w];
}
function faceWarp(p,i){
  if(earVertices.has(i))return earWarp(p);
  let [x,y,z]=p;
  if(brows.has(i)){
    const tail=smooth((Math.abs(x)-.019)/.044);
    return [x+Math.sign(x)*.009*tail,y-.0007*tail,z-.0035*tail*tail];
  }
  const front=smooth((z-.005)/.035),lower=smooth((1.39-y)/.035)*front;
  x*=1-.045*bell(y,1.378,.035)*front;
  y-=.003*lower;
  return [x,y,z];
}
function normalAt(point,normal,warp,i){
  const columns=point.map((_,axis)=>{
    const a=point.slice(),b=point.slice();a[axis]-=1e-5;b[axis]+=1e-5;
    const lo=warp(a,i),hi=warp(b,i);return hi.map((v,k)=>(v-lo[k])/2e-5);
  });
  const j=new Matrix3().set(...[0,1,2].flatMap(k=>columns.map(c=>c[k])));
  assert.ok(j.determinant()>.05,'deformation must not invert the surface');
  return new Vector3(...normal).applyMatrix3(j.invert().transpose()).normalize().toArray();
}
function deform(mesh,warp){
  const attrs=mesh.primitives[0].attributes,pa=accessor(attrs.POSITION),na=accessor(attrs.NORMAL);
  const positions=all(attrs.POSITION),normals=all(attrs.NORMAL),next=positions.map(warp);
  let changed=0;
  positions.forEach((p,i)=>{
    if(Math.hypot(...p.map((v,k)=>v-next[i][k]))<1e-9)return;
    changed++;pa.set(i,next[i]);na.set(i,normalAt(p,normals[i],warp,i));
  });
  // Transform the endpoints of every shape key, not just the neutral mesh.
  const targets=mesh.primitives[0].targets??[];
  for(const target of targets){
    const dp=accessor(target.POSITION),dn=target.NORMAL===undefined?null:accessor(target.NORMAL);
    positions.forEach((p,i)=>{
      const posed=p.map((v,k)=>v+dp.get(i)[k]),moved=warp(posed,i);
      dp.set(i,moved.map((v,k)=>v-next[i][k]));
      if(dn){const old=normals[i].map((v,k)=>v+dn.get(i)[k]),n=normalAt(posed,old,warp,i),base=na.get(i);dn.set(i,n.map((v,k)=>v-base[k]));}
    });
    bounds(target.POSITION);
  }
  bounds(attrs.POSITION);return changed;
}
const faceVertices=deform(face,faceWarp);

function hairWarp([x,y,z]){
  // Sweep temple locks behind the ears. The crown and central fringe stay put.
  const w=smooth((Math.abs(x)-.051)/.026)*bell(y,1.435,.092)*smooth((z+.06)/.12);
  return [x+Math.sign(x)*.009*w,y,z-.060*w];
}
const hairVertices=deform(gltf.meshes[2],hairWarp);
const parents=new Map();gltf.nodes.forEach((n,i)=>(n.children??[]).forEach(c=>parents.set(c,i)));
function worlds(){
  const cache=new Map();
  function world(i){if(cache.has(i))return cache.get(i);const n=gltf.nodes[i];
    const m=new Matrix4().compose(new Vector3(...n.translation??[0,0,0]),new Quaternion(...n.rotation??[0,0,0,1]),new Vector3(...n.scale??[1,1,1]));
    if(parents.has(i))m.premultiply(world(parents.get(i)));cache.set(i,m);return m;}
  gltf.nodes.forEach((_,i)=>world(i));return cache;
}
const before=worlds(),desired=new Map(before);
const hairNodes=new Set(gltf.extensions.VRMC_springBone.springs.filter(s=>s.name==='Hair').flatMap(s=>s.joints.map(j=>j.node)));
for(const i of hairNodes){const m=before.get(i).clone();m.setPosition(new Vector3(...hairWarp(new Vector3().setFromMatrixPosition(m).toArray())));desired.set(i,m);}
for(const i of hairNodes){const local=desired.get(i).clone().premultiply(desired.get(parents.get(i)).clone().invert());gltf.nodes[i].translation=new Vector3().setFromMatrixPosition(local).toArray();}
const after=worlds();
for(const skin of gltf.skins){const binds=accessor(skin.inverseBindMatrices);skin.joints.forEach((node,i)=>{
  if(hairNodes.has(node))binds.set(i,after.get(node).clone().invert().multiply(before.get(node)).multiply(new Matrix4().fromArray(binds.get(i))).toArray());
});}

// Paint is replaced in its original image slot. Clothing retains its exact
// authored alpha; the new color is the standard MToon shade-multiply map.
for(const [image,file] of [[5,'face-v7-detail.png'],[3,'iris-v7-smoke-gray.png']]){
  views[gltf.images[image].bufferView]=await readFile(new URL(`design/textures/${file}`,root));
}
const cardigan=await readFile(new URL('design/textures/cardigan-v7-color.png',root));
const imageView=gltf.bufferViews.push({buffer:0,byteLength:cardigan.length})-1;views.push(cardigan);
const imageIndex=gltf.images.push({name:'Ari v7 ivory knit color',bufferView:imageView,mimeType:'image/png'})-1;
const texture=gltf.textures.push({source:imageIndex,sampler:gltf.textures[0].sampler})-1;
const coat=gltf.materials[13];coat.pbrMetallicRoughness.baseColorFactor=[1,1,1,1];
coat.extensions.VRMC_materials_mtoon.shadeMultiplyTexture={index:texture};
coat.extensions.VRMC_materials_mtoon.shadeColorFactor=[.72,.70,.67];
coat.extras={...coat.extras,ariColorFromShade:true};

// Small gold hoops reproduce the reference and follow the head's skin joint.
const gold=gltf.materials.push({name:'Ari warm gold hoops',pbrMetallicRoughness:{baseColorFactor:[.72,.43,.13,1],metallicFactor:.7,roughnessFactor:.28},doubleSided:false,
  extensions:{KHR_materials_unlit:{},VRMC_materials_mtoon:{specVersion:'1.0',shadeColorFactor:[.3,.12,.025],shadingToonyFactor:.55,shadingShiftFactor:0,giEqualizationFactor:.9,
    parametricRimColorFactor:[.3,.18,.07],parametricRimFresnelPowerFactor:3,parametricRimLiftFactor:0,rimLightingMixFactor:1,outlineWidthMode:'worldCoordinates',outlineWidthFactor:.0001,outlineColorFactor:[.08,.03,.005]}}})-1;
const pos=[],norm=[],uv=[],indices=[];
const earringCenters=[];
for(const sign of [-1,1]){
  const lobe=earWarp([sign*.07900418,1.4079062,-.03387256]);
  const center=[lobe[0],lobe[1]-.0045,lobe[2]+.006];earringCenters.push(center);
  const hoop=new TorusGeometry(.0037,.00065,8,32);hoop.scale(.85,1.35,1);hoop.rotateY(sign*.25);hoop.translate(...center);
  const stud=new SphereGeometry(.00085,8,6);stud.translate(lobe[0],lobe[1],lobe[2]+.006);
  for(const geometry of [hoop,stud]){
    const base=pos.length;
    for(let i=0;i<geometry.attributes.position.count;i++){
      pos.push([0,1,2].map(k=>geometry.attributes.position.array[i*3+k]));
      norm.push([0,1,2].map(k=>geometry.attributes.normal.array[i*3+k]));
      uv.push([geometry.attributes.uv.array[i*2],geometry.attributes.uv.array[i*2+1]]);
    }
    indices.push(...[...geometry.index.array].map(i=>[i+base]));geometry.dispose();
  }
}
const attributes={POSITION:append(pos,'VEC3'),NORMAL:append(norm,'VEC3'),TEXCOORD_0:append(uv,'VEC2'),
  JOINTS_0:append(pos.map(()=>[18,0,0,0]),'VEC4',5123),WEIGHTS_0:append(pos.map(()=>[1,0,0,0]),'VEC4')};
const mesh=gltf.meshes.push({name:'Ari small gold hoops',primitives:[{attributes,indices:append(indices,'SCALAR',5123,34963),material:gold,mode:4}]})-1;
gltf.scenes[0].nodes.push(gltf.nodes.push({name:'Ari small gold hoops',mesh,skin:0})-1);
gltf.extensions.VRMC_vrm.meta.version='0.7';gltf.asset.generator+=' + Ari v7 detail pass';

// Repack views so replaced PNGs do not leave obsolete copies in the file.
let offset=0;const parts=[];
views.forEach((bytes,i)=>{const pad=(4-offset%4)%4;parts.push(Buffer.alloc(pad),bytes);offset+=pad;
  gltf.bufferViews[i].byteOffset=offset;gltf.bufferViews[i].byteLength=bytes.length;offset+=bytes.length;});
gltf.buffers[0].byteLength=offset;parts.push(Buffer.alloc((4-offset%4)%4));const packed=Buffer.concat(parts);
const jsonBytes=Buffer.from(JSON.stringify(gltf)),json=Buffer.alloc(Math.ceil(jsonBytes.length/4)*4,0x20);jsonBytes.copy(json);
const output=Buffer.alloc(28+json.length+packed.length);output.write('glTF');output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);
output.writeUInt32LE(json.length,12);output.writeUInt32LE(0x4e4f534a,16);json.copy(output,20);
output.writeUInt32LE(packed.length,20+json.length);output.writeUInt32LE(0x004e4942,24+json.length);packed.copy(output,28+json.length);
await writeFile(new URL('model/ari-vroid-v7.vrm',root),output);
console.log(JSON.stringify({sha256:hash(output),bytes:output.length,faceVertices,hairVertices,earringCenters,earringTriangles:indices.length/3},null,2));
