import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {Matrix3,Vector3} from '../renderer/node_modules/three/build/three.module.js';

const root=new URL('../characters/ari/model/',import.meta.url);
const hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await readFile(new URL('ari-vroid-v8.vrm',root));
assert.equal(hash(raw),'d7bab078bc2e85745a5277bbe57d07b7885914e0c52bc94f202119cb10d45b0a');
const size=raw.readUInt32LE(12),g=JSON.parse(raw.toString('utf8',20,20+size)),bin=Buffer.from(raw.subarray(28+size));
function vectors(index){const a=g.accessors[index],v=g.bufferViews[a.bufferView];assert.equal(a.componentType,5126);assert.equal(a.type,'VEC3');
  const offset=i=>(v.byteOffset??0)+(a.byteOffset??0)+i*(v.byteStride??12);
  return {count:a.count,get:i=>[0,1,2].map(k=>bin.readFloatLE(offset(i)+k*4)),set:(i,p)=>p.forEach((x,k)=>{assert.ok(Number.isFinite(x));bin.writeFloatLE(x,offset(i)+k*4);})};}
const bell=(v,c,r)=>Math.abs(v-c)>=r?0:(1-((v-c)/r)**2)**2;
const smooth=t=>{t=Math.max(0,Math.min(1,t));return t*t*(3-2*t);};
// Move only the curtain over the helix behind the ear. Keep the ear geometry,
// face-framing wisps, crown and long ends exactly as they were in v8.
function tuck([x,y,z]){
  const amount=bell(Math.abs(x),.083,.033)*bell(y,1.442,.073)*bell(z,.008,.075)*smooth((Math.abs(x)-.058)/.016)*smooth((1.465-y)/.020);
  // The skull widens above the ear. Keep that upper curtain out of the scalp;
  // only the lower portion can safely fold inward toward the attachment.
  return [x-Math.sign(x)*.007*amount*bell(y,1.432,.033),y,z-(x>0?.016:.013)*amount];
}
const hair=g.meshes[2].primitives[0],positions=vectors(hair.attributes.POSITION),normals=vectors(hair.attributes.NORMAL);
let moved=0,maxDisplacement=0;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
for(let i=0;i<positions.count;i++){
  const p=positions.get(i),q=tuck(p),n=normals.get(i);
  if(p[2]!==q[2]){
    moved++;maxDisplacement=Math.max(maxDisplacement,Math.hypot(...p.map((v,k)=>v-q[k])));
    const columns=p.map((_,axis)=>{const a=p.slice(),b=p.slice();a[axis]-=1e-5;b[axis]+=1e-5;const l=tuck(a),h=tuck(b);return h.map((v,k)=>(v-l[k])/2e-5);});
    const jacobian=new Matrix3().set(...[0,1,2].flatMap(k=>columns.map(c=>c[k])));assert.ok(jacobian.determinant()>.35);
    normals.set(i,new Vector3(...n).applyMatrix3(jacobian.invert().transpose()).normalize().toArray());positions.set(i,q);
  }
  q.forEach((v,k)=>{lo[k]=Math.min(lo[k],v);hi[k]=Math.max(hi[k],v);});
}
g.accessors[hair.attributes.POSITION].min=lo;g.accessors[hair.attributes.POSITION].max=hi;
// three-vrm 3.5.5 applies drag once per update and stiffness * dt, rather
// than stiffness * dt². Convert the 30 Hz tuning for our fixed 60 Hz solver.
// Keep world-space history: root translation/rotation must excite the hair.
for(const spring of g.extensions.VRMC_springBone.springs){
  for(const [index,joint] of spring.joints.entries()){
    if(joint.stiffness===undefined)continue;
    const distal=index/Math.max(1,spring.joints.length-2);
    joint.stiffness*=.5*(spring.name==='Hair'?1-.12*distal:1);
    if(joint.gravityPower!==undefined)joint.gravityPower*=.5;
    joint.dragForce=1-Math.sqrt(1-joint.dragForce);
    if(spring.name==='Hair'){delete spring.center;joint.dragForce*=1-.08*distal;}
  }
}
g.extensions.VRMC_vrm.meta.version='0.9';
g.asset.generator+=' + Ari v9 temple tuck and fixed-step springs';
g.asset.extras={...g.asset.extras,ariPhysicsHz:60};
const bytes=Buffer.from(JSON.stringify(g)),json=Buffer.alloc(Math.ceil(bytes.length/4)*4,0x20);bytes.copy(json);
const output=Buffer.alloc(28+json.length+bin.length);output.write('glTF');output.writeUInt32LE(2,4);output.writeUInt32LE(output.length,8);
output.writeUInt32LE(json.length,12);output.writeUInt32LE(0x4e4f534a,16);json.copy(output,20);output.writeUInt32LE(bin.length,20+json.length);output.writeUInt32LE(0x004e4942,24+json.length);bin.copy(output,28+json.length);
await writeFile(new URL('ari-vroid-v9.vrm',root),output);
console.log(JSON.stringify({sha256:hash(output),bytes:output.length,movedHairVertices:moved,maxDisplacement},null,2));
